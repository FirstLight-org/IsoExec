"""Host-resident buffers shared by the engine ranks of one node, mapped into each rank's device address space.

A component may keep checkpoint buffers in host memory (``HOST_RESIDENT_BUFFERS``: tables too large for the
device that kernels read through mapped host memory, e.g. the DeepSeek V4.1 Engram tables, ~196 GB). Every
rank of a stage holds the same bytes, so one copy serves a node:

* construction (collective over the stage's CPU group): the lowest rank of each host (its leader) creates
  one shared memory segment in ``SHM_DIR`` holding every buffer page-aligned after a one-page header (magic,
  state, identity); the other ranks of the host map it read-only. The segment is unlinked as soon as every
  co-located rank has mapped it (the mappings persist), so no name outlives the load;
* checkpoint load: only the leader writes the buffers (the other ranks skip their names, ``skip_names``);
* ``finish`` (collective, after the load): the leader marks the segment ready; every rank checks the header
  (magic, ready, identity) and fails closed on a mismatch, then registers the segment with CUDA
  (``cudaHostRegister``, read-only where the device supports it) and gets device views of the same bytes
  (``device_views``). Host pages end read-only on every rank;
* ``release`` (engine shutdown, at the latest at process exit): this rank's registration is dropped.

Name: ``{PREFIX}{identity digest}-{leader pid}`` (the header holds the identity's SHA-256), where the identity covers the checkpoint fingerprint
(path, sizes and modification times of its files), the buffer layout and the stage. A leader removes
segments of the same digest whose creator no longer runs (a crashed load) before creating its own.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import mmap
import os
import socket
import warnings
import weakref
from dataclasses import dataclass

import numpy as np
import torch

SHM_DIR = "/dev/shm"
PREFIX = "isoexec-host-"
MAGIC = b"ISOEXEC-HOSTBUF1"
HEADER = 4096
FILLING, READY = 0, 1
#: Free space ``SHM_DIR`` must keep beyond the segment.
SHM_MARGIN = 1 << 30


def checkpoint_fingerprint(path: str | None) -> dict:
    """The checkpoint's identity as the ranks see it: resolved path and (name, size, mtime) of its files."""
    if not path or not os.path.isdir(path):
        return {"path": str(path)}
    root = os.path.realpath(path)
    files = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isfile(full):
            stat = os.stat(full)
            files.append((name, stat.st_size, stat.st_mtime_ns))
    return {"path": root, "files": files}


@dataclass(frozen=True)
class Entry:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    offset: int
    nbytes: int


def layout(taken) -> tuple[list[Entry], int]:
    """Page-aligned placement of the taken ``(module, name, meta buffer)`` after the header."""
    entries, offset = [], HEADER
    for _, name, meta in taken:
        nbytes = meta.numel() * meta.element_size()
        entries.append(Entry(name, tuple(meta.shape), meta.dtype, offset, nbytes))
        offset += -(-max(nbytes, 1) // mmap.PAGESIZE) * mmap.PAGESIZE
    return entries, offset


def identity_of(checkpoint: dict, entries: list[Entry], stage: str) -> dict:
    return {"checkpoint": checkpoint, "stage": stage,
            "layout": [(e.name, list(e.shape), str(e.dtype), e.offset, e.nbytes) for e in entries]}


def _digest(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def remove_stale(digest: str, directory: str = None) -> list[str]:
    """Remove this digest's segments whose creating process no longer runs (a load that crashed before unlink)."""
    directory = directory or SHM_DIR
    removed = []
    for name in os.listdir(directory):
        if not name.startswith(f"{PREFIX}{digest}-"):
            continue
        tail = name.rsplit("-", 1)[-1]
        if tail.isdecimal() and not _alive(int(tail)):
            try:
                os.unlink(os.path.join(directory, name))
                removed.append(name)
            except FileNotFoundError:
                pass
    return removed


def _libc():
    return ctypes.CDLL("libc.so.6", use_errno=True)


def _protect_read_only(address: int, nbytes: int) -> None:
    if _libc().mprotect(ctypes.c_void_p(address), ctypes.c_size_t(nbytes), mmap.PROT_READ) != 0:
        raise OSError(ctypes.get_errno(), "mprotect(PROT_READ) of the shared host buffers failed")


def identity_sha(identity: dict) -> str:
    """SHA-256 of the full identity (checkpoint fingerprint, layout, stage) as the header stores it."""
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _header(identity: dict, state: int) -> bytes:
    body = identity_sha(identity).encode()
    return MAGIC + state.to_bytes(8, "little") + len(body).to_bytes(8, "little") + body


def read_header(buffer) -> tuple[int, str]:
    """``(state, identity SHA-256)`` of a mapped segment; raises if the magic is not this format's."""
    raw = bytes(buffer[:HEADER])
    if raw[:16] != MAGIC:
        raise RuntimeError("shared host buffer segment has no IsoExec header")
    state = int.from_bytes(raw[16:24], "little")
    size = int.from_bytes(raw[24:32], "little")
    if not 0 < size <= HEADER - 32:
        raise RuntimeError("shared host buffer segment header is corrupt")
    return state, raw[32:32 + size].decode()


class NodeSharedHostBuffers:
    """The node-shared host copy of a stage's host-resident buffers (see the module docstring).

    ``taken``: ``(module, name, meta buffer)`` triples; their buffers are replaced by host views of the
    segment. ``group``: the stage's CPU process group (None: this process alone). ``read_only_register``:
    whether the device can register read-only host memory (``cudaDevAttrHostRegisterReadOnlySupported``);
    without it a non-leader maps the segment writable for registration and then protects its pages."""

    def __init__(self, taken, *, checkpoint: str | None, stage: str = "", group=None,
                 read_only_register: bool = False, directory: str | None = None) -> None:
        self.taken = list(taken)
        self.group = group
        self.directory = directory or SHM_DIR
        self.entries, self.total = layout(self.taken)
        # as the header stores it (JSON), so a mapped segment's identity compares equal
        self.identity = json.loads(json.dumps(identity_of(checkpoint_fingerprint(checkpoint), self.entries, stage),
                                              sort_keys=True))
        self.digest = _digest(self.identity)
        self.registered = None             # bases of the pieces registered with CUDA
        self.device_base = None
        self.ready = False
        host = socket.gethostname()
        peers = self._gather((host, self.digest))
        if any(d != self.digest for _, d in peers):
            raise RuntimeError("engine ranks disagree on the host buffers' checkpoint or layout")
        rank = self._rank()
        self.leader = rank == min(r for r, (h, _) in enumerate(peers) if h == host)
        path, error = None, None
        try:
            if self.leader:
                path = self._create()
        except Exception as exc:
            error = repr(exc)
        published = self._gather((path, error))
        error, leader_path = None, None
        try:
            if any(e for _, e in published):
                raise RuntimeError("; ".join(e for _, e in published if e))
            leader_path = next(p for (h, _), (p, _) in zip(peers, published) if h == host and p is not None)
            if not self.leader:
                self._map(leader_path, writable=not read_only_register)
                state, seen = read_header(self.array)
                if seen != identity_sha(self.identity):
                    raise RuntimeError(f"shared host buffer segment {leader_path} has another identity "
                                       f"(checkpoint, layout or stage) than this rank's")
        except Exception as exc:
            error = repr(exc)
        mapped = self._gather(error)
        if self.leader and path is not None:
            try:
                os.unlink(path)             # every co-located rank has mapped it (or failed); mappings persist
            except OSError:
                pass
        if any(mapped):
            raise RuntimeError("shared host buffers: " + "; ".join(e for e in mapped if e))
        self.path = leader_path
        self._views()

    # ------------------------------------------------------------------------- collectives
    def _rank(self) -> int:
        return 0 if self.group is None else torch.distributed.get_rank(self.group)

    def _gather(self, value):
        if self.group is None:
            return [value]
        out = [None] * torch.distributed.get_world_size(self.group)
        torch.distributed.all_gather_object(out, value, group=self.group)
        return out

    # ------------------------------------------------------------------------- segment
    def _create(self) -> str:
        remove_stale(self.digest, self.directory)
        stat = os.statvfs(self.directory)
        if stat.f_bavail * stat.f_frsize < self.total + SHM_MARGIN:
            raise RuntimeError(f"{self.directory} lacks {self.total} bytes (+{SHM_MARGIN} B margin) for the "
                               "shared host buffers")
        path = os.path.join(self.directory, f"{PREFIX}{self.digest}-{os.getpid()}")
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.ftruncate(fd, self.total)
            self.mapping = mmap.mmap(fd, self.total, flags=mmap.MAP_SHARED)
            self.array = np.frombuffer(self.mapping, dtype=np.uint8)
            head = _header(self.identity, FILLING)
            self.mapping[:len(head)] = head
        except BaseException:
            os.unlink(path)                 # never leave a partial segment behind
            raise
        finally:
            os.close(fd)
        self.writable = True
        return path

    def _map(self, path: str, *, writable: bool) -> None:
        fd = os.open(path, os.O_RDWR if writable else os.O_RDONLY)
        try:
            if os.fstat(fd).st_size != self.total:
                raise RuntimeError(f"shared host buffer segment {path} has another size than this rank's layout")
            prot = mmap.PROT_READ | (mmap.PROT_WRITE if writable else 0)
            self.mapping = mmap.mmap(fd, self.total, flags=mmap.MAP_SHARED, prot=prot)
        finally:
            os.close(fd)
        self.array = np.frombuffer(self.mapping, dtype=np.uint8)
        self.writable = writable

    def _views(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # read-only mappings of non-leaders
            flat = torch.from_numpy(self.array)
        for (module, name, _), e in zip(self.taken, self.entries):
            view = flat[e.offset:e.offset + e.nbytes].view(e.dtype).view(e.shape)
            module._buffers[name] = view
            module.__dict__.setdefault("_host_shared", {})[name] = self

    @property
    def base(self) -> int:
        return int(self.array.ctypes.data)

    def skip_names(self, names_of) -> set[str]:
        """The logical names this rank must not load (the leader alone writes the shared bytes)."""
        if self.leader and not self.ready:
            return set()
        return {names_of(module, name) for module, name, _ in self.taken}

    # ------------------------------------------------------------------------- after the load
    def finish(self, device) -> int:
        """Collective after the checkpoint load: publish, verify, register and map. Returns the buffers mapped."""
        error = None
        try:
            if self.ready:
                raise RuntimeError("the shared host buffers are already finished")
            if self.leader:
                self.mapping[16:24] = READY.to_bytes(8, "little")
        except Exception as exc:
            error = repr(exc)
        published = self._gather(error)
        error = None
        try:
            if any(published):
                raise RuntimeError("; ".join(e for e in published if e))
            state, seen = read_header(self.array)
            if seen != identity_sha(self.identity) or state != READY:
                raise RuntimeError("shared host buffer segment is not this rank's ready checkpoint copy")
            if torch.device(device).type == "cuda":
                self._register(device)
            if self.writable:
                _protect_read_only(self.base, self.total)
                self.writable = False
        except Exception as exc:
            error = repr(exc)
        outcomes = self._gather(error)
        if any(outcomes):
            self.release()
            raise RuntimeError("shared host buffers: " + "; ".join(e for e in outcomes if e))
        self.ready = True
        return len(self.taken)

    def _register(self, device) -> None:
        """Register the buffers in ``REGISTER_CHUNK`` pieces: one driver call over ~200 GB holds the driver lock
        long enough to stall every other client (``nvidia-smi`` beyond 10 s). The pieces must map to one
        contiguous device range (the device views are slices of one flat tensor)."""
        from cuda.bindings import runtime as rt
        flags = rt.cudaHostRegisterMapped | rt.cudaHostRegisterPortable
        if not self.writable:
            flags |= rt.cudaHostRegisterReadOnly
        chunks: list[int] = []
        self.registered = chunks
        weakref.finalize(self, _unregister, chunks)
        try:
            self.device_base = register_pieces(self.base, self.total, flags, chunks, "the shared host buffers")
        except RuntimeError:
            self.release()
            raise
        total = self.total

        class _Interface:
            __cuda_array_interface__ = {"shape": (total,), "typestr": "|u1", "data": (self.device_base, False),
                                        "version": 3, "strides": None}
        flat = torch.as_tensor(_Interface(), device=device)
        for (module, name, _), e in zip(self.taken, self.entries):
            views = module.__dict__.setdefault("_host_device_views", {})
            views[name] = flat[e.offset:e.offset + e.nbytes].view(e.dtype).view(e.shape)

    def release(self) -> None:
        """Drop this rank's CUDA registration (idempotent); the host pages stay mapped until process exit."""
        if self.registered is not None:
            _unregister(self.registered)
            self.registered = None
            for module, name, _ in self.taken:
                module.__dict__.get("_host_device_views", {}).pop(name, None)


#: Bytes per ``cudaHostRegister`` / ``cudaHostUnregister`` call (page-aligned pieces of a page-aligned range).
#: The driver holds its lock for the whole of one call and its time grows with the bytes: one call over a
#: ~92 GiB table, or the context teardown that drops such a registration at process exit, stalls every other
#: client of the driver (``nvidia-smi`` beyond 10 s); 2 GiB pieces let them in between calls.
REGISTER_CHUNK = 2 << 30


def register_pieces(base: int, total: int, flags: int, chunks: list[int], what: str) -> int:
    """Register the host range ``[base, base + total)`` (page-aligned) with CUDA in ``REGISTER_CHUNK`` pieces,
    appending each registered piece's base to ``chunks`` (the caller owns them: ``_unregister(chunks)`` drops
    them piece by piece, also on failure here). The pieces must map to one contiguous device range; returns
    its base."""
    from cuda.bindings import runtime as rt
    device_base = None
    for offset in range(0, total, REGISTER_CHUNK):
        n = min(REGISTER_CHUNK, total - offset)
        err, = rt.cudaHostRegister(base + offset, n, flags)
        if err != rt.cudaError_t.cudaSuccess:
            _unregister(chunks)
            raise RuntimeError(f"cudaHostRegister of {what} (+{offset}, {n} of {total} B) failed: {err}")
        chunks.append(base + offset)
        err, pointer = rt.cudaHostGetDevicePointer(base + offset, 0)
        if err != rt.cudaError_t.cudaSuccess:
            _unregister(chunks)
            raise RuntimeError(f"cudaHostGetDevicePointer failed: {err}")
        if device_base is None:
            device_base = int(pointer)
        elif int(pointer) != device_base + offset:
            _unregister(chunks)
            raise RuntimeError(f"the registered pieces of {what} are not one contiguous device range")
    return device_base


def _unregister(chunks: list[int]) -> None:
    """Unregister each piece once: the list is emptied, so a later call (release, then the finalizer) is a no-op."""
    while chunks:
        base = chunks.pop(0)
        try:
            from cuda.bindings import runtime as rt
            rt.cudaHostUnregister(base)
        except Exception:                            # pragma: no cover - interpreter shutdown
            pass


def read_only_registration_supported(device) -> bool:
    """Whether ``device`` can register read-only host memory (``cudaDevAttrHostRegisterReadOnlySupported``)."""
    try:
        from cuda.bindings import runtime as rt
        err, value = rt.cudaDeviceGetAttribute(rt.cudaDeviceAttr.cudaDevAttrHostRegisterReadOnlySupported,
                                               torch.device(device).index or 0)
    except Exception:
        return False
    return err == rt.cudaError_t.cudaSuccess and bool(value)


__all__ = ["HEADER", "MAGIC", "NodeSharedHostBuffers", "PREFIX", "SHM_DIR", "checkpoint_fingerprint",
           "identity_of", "identity_sha", "layout", "read_header", "read_only_registration_supported", "remove_stale"]
