"""Node-local NVLink realizations of the TP-sharded experts' two exchanges (``tp_experts``): the column all-gather of
``h`` ([R, I/tp] per rank -> [R, I]) and of the output slices ([T, H/tp] -> [T, H]), one launch each. Byte copies: the
same bytes in the same positions as ``NcclExchange``'s all-gather + reorder, so the experts' bits do not depend on the
realization.

``SymmExchange`` (pull; ``csrc/tp_exchange.cu`` ``gather_cols``, the PIK owner all-reduce's conventions: per-block
LOCAL epochs, per-block arrive / leave flags in every rank's symmetric array, st.release.sys / ld.acquire.sys; block b
publishes its row chunk to every peer, waits for every peer's chunk b, pulls them over NVLink, and leaves only when
every peer has pulled its chunk: two flag round trips per call). The producer may write the slice straight into the
region (``buffer``: in place, no copy); any other input is copied in by the same launch.

``MulticastExchange`` (push; ``lamport_gather_cols``, derived from vLLM's MNNVL Lamport all-gather, see
``csrc/PROVENANCE.md``): every rank stores its slice once through an NVLink SHARP multicast mapping into a 3-stage
region of every rank; the receiver spins on the words themselves (no flag round trip). Every 32-bit word carries one
16-bit payload unit and a per-call tag (never a payload sentinel: every bit pattern, NaN payloads included, travels
unchanged; 2x wire bytes). Slices above ``lamport_max_bytes`` take the pull kernel (same bytes). Needs multicast on
every rank of the group: otherwise construction raises ``MulticastUnavailableError`` on every rank (keep
``SymmExchange``). The multicast mapping is checked against the unicast one by a probe at construction.

Both: graph-capturable (state lives on the device, advanced by the kernels), no host value per call; collective at
construction and only there: every region (both pull regions; at world > 1 also both multicast regions, whatever slice
sizes later take which path) is created and rendezvoused eagerly on every rank, through ``moe.symm_rendezvous``
(a rank-skewed setup sequence raises ``RendezvousSkewError`` on every rank instead of hanging; take ``NcclExchange``).
No state is shared between exchange objects, so two groups of one size (or over the same ranks) never alias. At world
1 nothing is rendezvoused (plain device memory). A group spanning more than one host is refused by name
(``collectives.node_local.require_symmetric_peers``: also two ranks on one GPU); use ``NcclExchange`` there. Every rank must issue the same sequence of
calls with the same shapes (as NCCL's all-gather); a lost peer makes the others' bounded waits trap.
"""
from __future__ import annotations

from functools import lru_cache

import torch

MAX_WORLD, MAX_BLOCKS = 8, 256
HEADER = 4 * 2 * MAX_BLOCKS * MAX_WORLD              # arrive + leave flags
CHUNK_BYTES = 32 << 10                               # gathered bytes per block (latency vs bandwidth)
#: Rows past ``slots * max_tokens`` that a producer may ask ``h_buffer`` for: a swapped (decode) GEMM record reads one token
#: tile past the last row, so ``bi_tp`` asks for ``Tk + tile_n``; the widest swapped token tile of the vendored generator
#: is 64 (``SWAP_WIDE_TILE_N``; narrow tiles 8 / 16 / 32). Only the pull region holds them; nothing gathers them.
H_PAD_ROWS = 64


@lru_cache(None)
def _ext():
    from isoexec.ops.build_cache import build_sm100_extension
    return build_sm100_extension("deepseek_v41_tp_exchange", __file__, ("tp_exchange.cu",))


class _Region:
    def __init__(self, group, world, rank, device, slice_bytes):
        import torch.distributed as dist

        from isoexec.ops.moe import symm_rendezvous
        if world > MAX_WORLD:
            raise ValueError(f"tp_exchange: at most {MAX_WORLD} ranks")
        self.world, self.rank = world, rank
        self.slice_bytes = -(-slice_bytes // 4096) * 4096
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        with torch.cuda.device(device):
            if world == 1:                           # no peer: plain device memory, no rendezvous
                self.buf = torch.zeros(HEADER + self.slice_bytes, dtype=torch.int8, device=device)
                self.handle = None
                self.bases = [int(self.buf.data_ptr())]
            else:
                import torch.distributed._symmetric_memory as symm_mem
                self.buf = symm_mem.empty(HEADER + self.slice_bytes, dtype=torch.int8, device=device)
                self.handle = symm_rendezvous.rendezvous(self.buf, group, site="moe.tp_exchange.pull")
                self.buf.zero_()
                self.bases = [int(v) for v in self.handle.buffer_ptrs]
                # block b of every rank exchanges row chunk b: the block count is a group-wide value
                sms = min(v[0] for v in symm_rendezvous._gather([sms], group, torch.device(device)))
            self.epoch = torch.zeros(MAX_BLOCKS, dtype=torch.int32, device=device)
            torch.cuda.synchronize(device)
        if world > 1:
            dist.barrier(group=group)                # every rank's zeroed flags landed before any first publish
        self.sms = sms

    def blocks(self, rows, cols):
        nbytes = rows * cols * 2 * self.world
        return max(1, min(self.sms, MAX_BLOCKS, rows, -(-nbytes // CHUNK_BYTES)))

    def buffer(self, rows, cols):
        """This rank's slice inside the region, ``[rows, cols]`` bf16: a producer that writes its output here makes the
        next ``gather_cols`` of it in place (no copy). Valid until that gather returns."""
        if rows * cols * 2 > self.slice_bytes:
            raise ValueError("tp_exchange: slice exceeds the prepared capacity")
        return self.buf[HEADER:HEADER + rows * cols * 2].view(torch.bfloat16).view(rows, cols)

    def gather_cols(self, local: torch.Tensor) -> torch.Tensor:
        rows, cols = local.shape
        out = torch.empty((rows, self.world * cols), dtype=local.dtype, device=local.device)
        _ext().gather_cols(self.bases, self.epoch, self.rank, HEADER, self.slice_bytes, local.contiguous(), out,
                           self.blocks(rows, cols))
        return out


class SymmExchange:
    """``gather_h`` / ``gather_out`` of ``tp_experts`` over NVLink symmetric memory, for at most ``max_tokens`` rows
    per call (``7 max_tokens`` expert rows)."""

    def __init__(self, group, device, *, max_tokens: int, intermediate: int, hidden: int, slots: int = 7):
        import torch.distributed as dist

        from isoexec.ops.collectives.node_local import require_symmetric_peers
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SymmExchange must be created outside CUDA graph capture")
        group = group if group is not None else dist.group.WORLD
        self.world, self.rank = dist.get_world_size(group), dist.get_rank(group)
        if self.world > 1:
            require_symmetric_peers(group, "deepseek_v41 TP-sharded experts NVLink exchange (use NcclExchange across hosts)")
        Is, Hs = intermediate // self.world, hidden // self.world
        self.h = _Region(group, self.world, self.rank, device, (slots * max_tokens + H_PAD_ROWS) * Is * 2)
        self.out = _Region(group, self.world, self.rank, device, max_tokens * Hs * 2)

    def h_buffer(self, rows, cols):
        """Where GEMM1 may write this rank's h slice (``[rows, cols]``) for an in-place ``gather_h``."""
        return self.h.buffer(rows, cols)

    def out_buffer(self, rows, cols):
        """Where the combine may write this rank's output slice (``[rows, cols]``) for an in-place ``gather_out``."""
        return self.out.buffer(rows, cols)

    def gather_h(self, h_r):
        return self.h.gather_cols(h_r)

    def gather_out(self, y_r):
        return self.out.gather_cols(y_r)


# ---------------------------------------------------------------------------------------------------------------------
# Multicast (Lamport) realization

#: Default per-rank slice capacity of the multicast path by world size (vLLM's MNNVL all-gather limits); larger slices
#: take the pull kernel. Provisional until the crossover is measured.
# per-rank slice bytes up to which the Lamport push is used, else the pull: measured crossover ~1 MiB at world 2 (B200,
# v5a-q1 / r4q-q7: h push 12.3 vs pull 12.8 us at 1.03 MB, 20.2 vs 14.1 at 2.06 MB; out push 9.8 vs pull 12.6 at
# 0.66 MB, 24.7 vs 14.0 at 2.6 MB); world 4 / 8 provisional at the same value until measured. Placement only.
LAMPORT_MAX_BYTES = {2: 1 << 20, 4: 1 << 20, 8: 1 << 20}
LAMPORT_MAX_BYTES_DEFAULT = 1 << 20
LAMPORT_THREADS = 128
LAMPORT_STAGES = 3
_PROBE_MARK = 0x3F800000
_ERROR_FIELDS = ("set", "rank", "stage", "call", "tag_counter", "tag", "block", "thread", "source_rank", "vector",
                 "word_lo", "word_hi", "vectors", "world")


class MulticastUnavailableError(RuntimeError):
    """A rank of the group has no NVLink SHARP multicast mapping for torch symmetric memory (no NVSwitch / NVLS, or
    multicast disabled), or the mapping failed the construction probe. Raised on every rank by ``MulticastExchange``,
    with ``pull``: its already rendezvoused ``SymmExchange`` (keep it; building another would set up again)."""

    pull = None


def _multicast_supported(index: int) -> bool:
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory
        return bool(_SymmetricMemory.has_multicast_support(DeviceType.CUDA, index))
    except Exception:  # noqa: BLE001 - an older torch without the probe: the multicast pointer decides
        return True


class _LamportRegion:
    """3 stages x ``world`` slots of ``2 cap`` tagged bytes in symmetric memory with a multicast mapping, the device
    state, and a mapped host error record. Built in two steps so that every region of an exchange is rendezvoused
    before any admission verdict (``_admit_multicast``): a refusal then leaves every group at the same point of the
    process-wide setup sequence whatever region failed."""

    def __init__(self, group, world, rank, device, cap_bytes, *, timeout_s, pdl, label):
        import torch.distributed._symmetric_memory as symm_mem

        from isoexec.ops.moe import symm_rendezvous
        self.world, self.rank, self.label = world, rank, label
        self.device = torch.device(device)
        self.index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        cap_vecs = max(1, -(-int(cap_bytes) // 16))
        self.cap_bytes = cap_vecs * 16
        self.slot_vecs = 2 * cap_vecs                    # tagged 16-byte vectors per (stage, rank) slot
        self.stage_vecs = world * self.slot_vecs
        self.timeout_ns = int(timeout_s * 1e9)
        self.pdl = bool(pdl)
        self.status = []
        with torch.cuda.device(self.index):
            self.buf = symm_mem.empty(LAMPORT_STAGES * self.stage_vecs * 16, dtype=torch.uint8, device=self.device)
            self.handle = symm_rendezvous.rendezvous(self.buf, group, site="moe.tp_exchange.lamport")
            self.mc = int(getattr(self.handle, "multicast_ptr", 0) or 0)
            self.local = int(self.handle.buffer_ptrs[rank])

    def mapped(self) -> bool:
        return self.mc != 0 and self.local == self.buf.data_ptr()

    def admit(self, group):
        """Collective: the multicast probe, then the device state and the error record (after ``_admit_multicast``
        accepted every rank's mapping)."""
        import weakref

        import torch.distributed as dist
        index = self.index
        self._probe(group, index)
        self.state = torch.zeros(16, dtype=torch.int32, device=self.device)
        host, dev = _ext().error_record_alloc(index)
        self.err_host, self.err_dev = int(host), int(dev)
        weakref.finalize(self, _ext().error_record_free, self.err_host)
        with torch.cuda.device(index):
            torch.cuda.synchronize(index)
        dist.barrier(group=group)                    # zeroed stages and state on every rank before any first publish

    def _probe(self, group, index):
        """Every rank stores marks through the multicast address; every rank must read every rank's marks at the same
        offsets of its unicast buffer (first vector of stage 0, last vector of stage 2, each rank's slot)."""
        import torch.distributed as dist
        with torch.cuda.device(index):
            self.buf.zero_()
            torch.cuda.synchronize(index)
        dist.barrier(group=group)
        _ext().lamport_probe(self.mc, self.slot_vecs, self.world, self.rank, index)
        with torch.cuda.device(index):
            torch.cuda.synchronize(index)
        dist.barrier(group=group)
        words = self.buf.view(torch.int32)
        bad = []
        for t in range(self.world):
            m = _PROBE_MARK | (t << 8)
            first = t * self.slot_vecs * 4
            last = (2 * self.stage_vecs + t * self.slot_vecs + self.slot_vecs - 1) * 4
            got = [int(v) & 0xFFFFFFFF for v in torch.cat((words[first:first + 4], words[last:last + 4])).tolist()]
            if got != [m | k for k in range(1, 9)]:
                bad.append((t, [hex(v) for v in got]))
        oks = [None] * self.world
        dist.all_gather_object(oks, (self.rank, bad), group=group)
        with torch.cuda.device(index):
            self.buf.zero_()
            torch.cuda.synchronize(index)
        dist.barrier(group=group)
        failed = [o for o in oks if o[1]]
        if failed:
            raise MulticastUnavailableError(
                f"tp_exchange multicast ({self.label}): the multicast mapping does not deliver to the unicast buffers "
                f"(rank, [(source rank, words read)]): {failed}; use SymmExchange")

    def fits(self, x: torch.Tensor) -> bool:
        row_bytes = x.size(1) * x.element_size()
        return row_bytes % 16 == 0 and x.size(0) * row_bytes <= self.cap_bytes

    def gather_cols(self, x: torch.Tensor) -> torch.Tensor:
        rows, cols = x.shape
        out = torch.empty((rows, self.world * cols), dtype=x.dtype, device=x.device)
        nvec = rows * cols * x.element_size() // 16
        if nvec == 0:
            return out
        blocks = max(1, min(self.max_blocks, -(-nvec // LAMPORT_THREADS)))
        _ext().lamport_gather_cols(x.contiguous(), out, self.local, self.mc, self.state, self.err_dev, self.world,
                                   self.rank, self.slot_vecs, blocks, self.timeout_ns, self.pdl)
        return out

    def last_error(self):
        """The record a timed-out wait wrote (readable after the CUDA context is lost), or None."""
        rec = _ext().error_record_read(self.err_host)
        if not rec[0]:
            return None
        return dict(zip(_ERROR_FIELDS, rec), region=self.label)


def _admit_multicast(group, regions):
    """Collective over ``group``, after every region is rendezvoused: one gather of every rank's mappings (torch's
    support probe, multicast pointer, unicast pointer, SM count), then the probe of each region. Every verdict is a
    function of gathered values, so a refusal is raised on every rank at the same point."""
    import torch.distributed as dist
    first = regions[0]
    mine = {"rank": first.rank, "host_support": _multicast_supported(first.index),
            "multicast_ptr_nonnull": all(r.mc != 0 for r in regions),
            "local_is_buffer": all(r.local == r.buf.data_ptr() for r in regions),
            "sms": torch.cuda.get_device_properties(first.index).multi_processor_count}
    status = [None] * first.world
    dist.all_gather_object(status, mine, group=group)
    for r in regions:
        r.status = status
    bad = [s["rank"] for s in status if not (s["multicast_ptr_nonnull"] and s["local_is_buffer"])]
    if bad:
        raise MulticastUnavailableError(
            f"tp_exchange multicast: torch symmetric memory gives no multicast mapping on rank(s) {bad} of this group "
            f"({status}); NVLink SHARP multicast (NVSwitch, driver / fabric manager support, "
            "TORCH_SYMM_MEM_DISABLE_MULTICAST unset) is required; use SymmExchange")
    for r in regions:
        r.admit(group)
        r.max_blocks = min(s["sms"] for s in status)    # a group-wide launch width


class MulticastExchange:
    """``gather_h`` / ``gather_out`` of ``tp_experts`` over NVLink SHARP multicast (Lamport push, one launch, no flag
    round trip) for slices up to ``lamport_max_bytes`` per rank, the pull kernel (``SymmExchange``) above; both the
    bytes of ``NcclExchange``. ``h_buffer`` / ``out_buffer`` are the pull regions (a producer may write there; the
    multicast path reads its input locally, so any tensor serves). ``timeout_s`` bounds every wait (then the kernel
    records the failure, see ``last_error``, and traps). Raises ``MulticastUnavailableError`` on every rank when a
    rank has no multicast mapping (its ``pull`` is the already built ``SymmExchange``: every group is then at the same
    point of the setup sequence whichever check failed); ``NodeLocalityError`` when the group spans hosts (before any
    setup)."""

    def __init__(self, group, device, *, max_tokens: int, intermediate: int, hidden: int, slots: int = 7,
                 lamport_max_bytes: int | None = None, timeout_s: float = 60.0, pdl: bool = True):
        import torch.distributed as dist

        from isoexec.ops.collectives.node_local import require_symmetric_peers
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MulticastExchange must be created outside CUDA graph capture")
        group = group if group is not None else dist.group.WORLD
        self.world, self.rank = dist.get_world_size(group), dist.get_rank(group)
        if self.world > MAX_WORLD:
            raise ValueError(f"tp_exchange: at most {MAX_WORLD} ranks")
        if self.world > 1:
            require_symmetric_peers(group, "deepseek_v41 TP-sharded experts NVLink multicast exchange (use NcclExchange "
                                    "across hosts)")
        Is, Hs = intermediate // self.world, hidden // self.world
        cap = lamport_max_bytes if lamport_max_bytes is not None else LAMPORT_MAX_BYTES.get(
            self.world, LAMPORT_MAX_BYTES_DEFAULT)
        # Every region exists on every rank before any verdict: the pull regions, then (world > 1) both multicast
        # regions; the admission (one gather of the mappings, then the probes) refuses on every rank at the same
        # point of the setup sequence, and the refusal carries the pull exchange (``exc.pull``: no new rendezvous).
        self.pull = SymmExchange(group, device, max_tokens=max_tokens, intermediate=intermediate, hidden=hidden,
                                 slots=slots)
        self.h = self.out = None
        if self.world > 1:
            kw = {"timeout_s": timeout_s, "pdl": pdl}
            h = _LamportRegion(group, self.world, self.rank, device, min(cap, slots * max_tokens * Is * 2),
                               label="h", **kw)
            out = _LamportRegion(group, self.world, self.rank, device, min(cap, max_tokens * Hs * 2), label="out",
                                 **kw)
            try:
                _admit_multicast(group, (h, out))
            except MulticastUnavailableError as exc:
                exc.pull = self.pull
                raise
            self.h, self.out = h, out

    @property
    def multicast_status(self):
        """Per rank: torch's multicast support probe, whether the multicast pointer is non-null, and whether the
        unicast pointer is the local buffer (gathered at construction)."""
        return self.h.status if self.h is not None else []

    def path(self, x: torch.Tensor, which: str = "h") -> str:
        """``"multicast"`` or ``"pull"``: which kernel a gather of ``x`` on region ``which`` runs."""
        lam = self.h if which == "h" else self.out
        return "multicast" if lam is not None and lam.fits(x) else "pull"

    def h_buffer(self, rows, cols):
        return self.pull.h_buffer(rows, cols)

    def out_buffer(self, rows, cols):
        return self.pull.out_buffer(rows, cols)

    @staticmethod
    def _gather(lam, reg, x):
        if lam is not None and lam.fits(x):
            return lam.gather_cols(x)
        if x.dtype == torch.bfloat16:
            return reg.gather_cols(x)
        return reg.gather_cols(x.contiguous().view(torch.bfloat16)).view(x.dtype)   # a byte copy of 16-bit units

    def gather_h(self, h_r):
        return self._gather(self.h, self.pull.h, h_r)

    def gather_out(self, y_r):
        return self._gather(self.out, self.pull.out, y_r)

    def last_error(self):
        """Timeout records of the two multicast regions (None when no wait timed out)."""
        errs = [r.last_error() for r in (self.h, self.out) if r is not None]
        errs = [e for e in errs if e is not None]
        return errs or None


__all__ = ["H_PAD_ROWS", "LAMPORT_MAX_BYTES", "MulticastExchange", "MulticastUnavailableError", "SymmExchange"]
