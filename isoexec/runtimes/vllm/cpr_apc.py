"""vLLM prefix caching under the CPR GDN composition: boundary checkpoints, a shared index, scheduler admission.

CPR keeps a request's GDN state in a private record pool, not in vLLM's mamba blocks, so vLLM's own
prefix-cache resume (align-mode block state + ``prefill_has_initial_state``) would resume state CPR never
wrote. This module makes a vLLM prefix-cache hit resume from a CPR checkpoint scanned from exactly that
prefix, and otherwise recompute:

* **Checkpoints (workers).** At the start of each step the lifecycle driver captures, for every request
  whose computed prefix ``p`` (where its previous chunk ended) is a positive multiple of the checkpoint
  grid and lies inside its prefill tokens, the request's boundary record at ``p`` -- the encoded boundary
  state of every GDN layer, the conv windows and ``p`` (``gdn_cpr_sm90.engine_state.capture_boundaries``)
  -- into the boundary store (``gdn_cpr_state.CprBoundaryStore``) under the digest of tokens ``[0, p)``
  (``gdn_cpr_state.cpr_apc_key``). The grid is lcm(64, vLLM's mamba block size), and the GDN layer asks
  vLLM to make that block size a multiple of 64 (``_gdn.GDNStateLayer.get_prefill_checkpoint_alignment``),
  so every block boundary vLLM caches mamba state at is a checkpoint position.
* **Index (workers -> scheduler).** Each worker of an engine (every TP x PP rank) publishes the keys it
  holds in its own file under ``/dev/shm`` (flock-guarded), stamped with the prefix-cache epoch it belongs
  to.
* **Admission (scheduler).** ``KVCacheManager.get_computed_blocks`` keeps vLLM's hit of ``h`` tokens only if
  ``h`` is a multiple of 64 and every worker's index, at the current epoch, holds the key of the request's
  first ``h`` tokens; otherwise it asks vLLM for the next shorter hit, down to none. A truncated hit is a
  recompute, never a wrong resume.
* **Adoption (workers).** A request first scheduled with ``h > 0`` computed tokens takes a fresh row in every
  pool and the stored checkpoint is installed on it (``install_boundaries``) before the forward, whose
  prefill then continues from it as from any chunked-prefill continuation. The scheduler admitted ``h`` only
  because every worker advertised the key, and a key leaves the index ``APC_RETIRE_STEPS`` steps before its
  bytes are released, so the checkpoint is present; if it is not, the driver raises.
* **Epochs.** ``Scheduler.reset_prefix_cache`` (verl calls it at every wake, after a weight update) advances
  the epoch; each worker retires its whole store at its next step, and the scheduler admits nothing a worker
  has not re-advertised at the new epoch. Every worker sleep drops the store.

Bitwise: a checkpoint is the state the kernels left at ``p``; a row that adopts it is, to every kernel, the
row that scanned the ``p`` tokens itself (the chunked-prefill resume is split-point invariant).
"""

from __future__ import annotations

import atexit
import contextlib
import fcntl
import mmap
import os
import struct
import time

import numpy as np

_MAGIC = 0x4953_4F43_5052_4131  # b"ISOCPRA1", a format tag
_HEADER = struct.Struct("<QQQQ")  # magic, epoch, version, high-water mark
_KEY = 16
#: Keys one worker's index can hold. A full index refuses to store new checkpoints (a lost hit).
INDEX_SLOTS = 1 << 16
_DIR = "/dev/shm"


def _say(msg: str) -> None:
    """fd 1, not ``logger.info``: vLLM filters INFO out of engine subprocesses."""
    try:
        os.write(1, f"[ISOEXEC-CPR-APC] pid={os.getpid()} {msg}\n".encode())
    except Exception:  # noqa: BLE001, S110  # pragma: no cover - fd 1 closed
        pass


def engine_tag(vllm_config) -> str:
    """This engine's index namespace: vLLM's id of the engine instance plus its data-parallel rank."""
    return f"{vllm_config.instance_id}-dp{int(getattr(vllm_config.parallel_config, 'data_parallel_rank', 0) or 0)}"


def _unlink(path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


def _path(tag: str, name: str) -> str:
    return os.path.join(_DIR, f"isoexec-cprapc-{tag}-{name}")


class _Shm:
    """A fixed-size file under /dev/shm, mapped, with flock for every read-modify-write or snapshot.

    The process that creates a file removes it at exit (a reader keeps its open mapping)."""

    def __init__(self, path: str, size: int, create: bool):
        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        self.path = path
        existed = os.path.exists(path)
        self.fd = os.open(path, flags, 0o600)
        if create and os.fstat(self.fd).st_size < size:
            os.ftruncate(self.fd, size)
        self.size = os.fstat(self.fd).st_size
        self.mm = mmap.mmap(self.fd, self.size)
        if create and not existed:
            atexit.register(_unlink, path)

    def lock(self, exclusive: bool):
        fcntl.flock(self.fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)

    def unlock(self):
        fcntl.flock(self.fd, fcntl.LOCK_UN)

    def close(self):
        try:
            self.mm.close()
            os.close(self.fd)
        except Exception:  # noqa: BLE001, S110  # pragma: no cover - already closed
            pass


class Epoch:
    """The prefix-cache epoch: written by the scheduler (``advance``), read by every worker each step."""

    def __init__(self, tag: str, create: bool):
        self._shm = _Shm(_path(tag, "epoch"), 8, create)

    def read(self) -> int:
        self._shm.lock(False)
        try:
            return struct.unpack_from("<Q", self._shm.mm, 0)[0]
        finally:
            self._shm.unlock()

    def advance(self) -> int:
        self._shm.lock(True)
        try:
            e = struct.unpack_from("<Q", self._shm.mm, 0)[0] + 1
            struct.pack_into("<Q", self._shm.mm, 0, e)
            return e
        finally:
            self._shm.unlock()


class RankIndex:
    """One worker's published keys (writer side): a slot table of 16-byte keys under a header."""

    def __init__(self, tag: str, rank: int, slots: int = INDEX_SLOTS):
        self._shm = _Shm(_path(tag, f"r{rank}"), _HEADER.size + slots * _KEY, True)
        self.slots = slots
        self._where: dict[bytes, int] = {}
        self._free: list[int] = list(range(slots - 1, -1, -1))
        self._hwm = 0
        self._shm.lock(True)
        try:
            self._shm.mm[: self._shm.size] = bytes(self._shm.size)
            _HEADER.pack_into(self._shm.mm, 0, _MAGIC, 0, 1, 0)
        finally:
            self._shm.unlock()

    def _bump(self, epoch: int | None = None) -> None:
        magic, e, version, _ = _HEADER.unpack_from(self._shm.mm, 0)
        _HEADER.pack_into(self._shm.mm, 0, magic, e if epoch is None else epoch, version + 1, self._hwm)

    def full(self) -> bool:
        return not self._free

    def add(self, key: bytes) -> bool:
        if key in self._where:
            return True
        if not self._free:
            return False
        i = self._free.pop()
        self._where[key] = i
        self._hwm = max(self._hwm, i + 1)
        self._shm.lock(True)
        try:
            off = _HEADER.size + i * _KEY
            self._shm.mm[off:off + _KEY] = key
            self._bump()
        finally:
            self._shm.unlock()
        return True

    def remove(self, key: bytes) -> None:
        i = self._where.pop(key, None)
        if i is None:
            return
        self._free.append(i)
        self._shm.lock(True)
        try:
            off = _HEADER.size + i * _KEY
            self._shm.mm[off:off + _KEY] = bytes(_KEY)
            self._bump()
        finally:
            self._shm.unlock()

    def reset(self, epoch: int) -> None:
        """Retract every key and stamp the index with ``epoch``."""
        self._where.clear()
        self._free = list(range(self.slots - 1, -1, -1))
        self._shm.lock(True)
        try:
            self._shm.mm[_HEADER.size:_HEADER.size + self._hwm * _KEY] = bytes(self._hwm * _KEY)
            self._hwm = 0
            self._bump(epoch)
        finally:
            self._shm.unlock()


class IndexReader:
    """The scheduler's view of every worker's index: ``(epoch, keys)`` per rank, re-read when it changed."""

    def __init__(self, tag: str, ranks: int):
        self.tag, self.ranks = tag, ranks
        self._shm: dict[int, _Shm] = {}
        self._cache: dict[int, tuple[int, int, frozenset]] = {}

    def snapshot(self, rank: int) -> tuple[int, frozenset] | None:
        shm = self._shm.get(rank)
        if shm is None:
            try:
                shm = self._shm[rank] = _Shm(_path(self.tag, f"r{rank}"), 0, False)
            except FileNotFoundError:
                return None
        shm.lock(False)
        try:
            magic, epoch, version, hwm = _HEADER.unpack_from(shm.mm, 0)
            if magic != _MAGIC:
                return None
            hit = self._cache.get(rank)
            if hit is not None and hit[0] == version:
                return hit[1], hit[2]
            raw = shm.mm[_HEADER.size:_HEADER.size + hwm * _KEY]
        finally:
            shm.unlock()
        keys = frozenset(k for k in (raw[i:i + _KEY] for i in range(0, len(raw), _KEY)) if any(k))
        self._cache[rank] = (version, epoch, keys)
        return epoch, keys

    def holds(self, key: bytes, epoch: int) -> bool:
        """True iff every worker advertises ``key`` at ``epoch``."""
        for r in range(self.ranks):
            snap = self.snapshot(r)
            if snap is None or snap[0] != epoch or key not in snap[1]:
                return False
        return True


# ------------------------------------------------------------------------------------------------ scheduler


SCHED_STATS = {"admitted": 0, "admitted_tokens": 0, "truncated": 0, "refused": 0}


class SchedulerAdmission:
    """Keeps a vLLM prefix-cache hit only where every worker holds the checkpoint of exactly that prefix."""

    def __init__(self, vllm_config, tag: str):
        from isoexec.ops.gdn.shared.gdn_ops import fla_chunk_size

        par = vllm_config.parallel_config
        self.grid = int(fla_chunk_size())  # the CPR chunk: checkpoints exist only at its multiples
        self.ranks = int(par.tensor_parallel_size) * int(par.pipeline_parallel_size)
        self.tag = tag
        self.epoch = Epoch(tag, create=True)
        if self.epoch.read() == 0:
            self.epoch.advance()
        self.reader = IndexReader(tag, self.ranks)
        self._last_report = 0.0
        _say(f"scheduler admission on: index {_path(tag, 'r*')} over {self.ranks} worker(s), grid {self.grid}")

    def admits(self, request, num: int) -> bool:
        from isoexec.ops.gdn.shared.gdn_cpr_state import cpr_apc_key

        if num <= 0 or num % self.grid:
            return False
        # A prompt given as embeddings, with multimodal inputs or under an adapter is not identified by its token
        # ids alone; the workers capture no checkpoint for it either.
        if (getattr(request, "prompt_embeds", None) is not None or getattr(request, "mm_features", None)
                or getattr(request, "lora_request", None) is not None):
            return False
        key = cpr_apc_key(request.all_token_ids[:num], num)
        return self.reader.holds(key, self.epoch.read())

    def filter(self, manager, request, max_cache_hit_length, result, original):
        blocks, num, boundary = result
        first = num
        while num > 0 and not self.admits(request, num):
            SCHED_STATS["truncated"] += 1
            blocks, num, boundary = original(manager, request, num - 1)
        if num > 0:
            SCHED_STATS["admitted"] += 1
            SCHED_STATS["admitted_tokens"] += num
        elif first > 0:
            SCHED_STATS["refused"] += 1
        now = time.monotonic()
        if now - self._last_report > 30.0 and (SCHED_STATS["admitted"] or SCHED_STATS["refused"]):
            self._last_report = now
            _say(f"scheduler: {SCHED_STATS}")
        return blocks, num, boundary

    def advance_epoch(self) -> int:
        e = self.epoch.advance()
        _say(f"prefix cache reset: epoch {e}; every worker retires its checkpoints at its next step")
        return e


_scheduler_installed = False


def install_scheduler_admission() -> bool:
    """Wrap vLLM's scheduler and KV-cache manager (idempotent; inert unless the flag is on).

    * ``Scheduler.__init__`` attaches a :class:`SchedulerAdmission` to its KV-cache manager when CPR prefix
      caching is on for a model with mamba (GDN) layers, refusing a KV connector;
    * ``KVCacheManager.get_computed_blocks`` filters every hit through it;
    * ``Scheduler.reset_prefix_cache`` advances the epoch when the reset succeeded.
    """
    global _scheduler_installed
    if _scheduler_installed:
        return True
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

    from isoexec.ops.gdn.shared.gdn_cpr_state import cpr_apc_enabled

    from .gdn_engine_patch import enable_state_features

    orig_sched_init = Scheduler.__init__

    def sched_init(self, *args, **kw):
        orig_sched_init(self, *args, **kw)
        if not (self.cache_config.enable_prefix_caching and getattr(self.kv_cache_config, "has_mamba_layers", False)):
            return
        # The engine core may not have inherited the flags register_native exported (an engine started in another
        # Ray actor), so it turns them on from its own configuration, as each worker does (_binding._bind_gdn_core).
        enable_state_features(["enable_prefix_caching"])
        if not cpr_apc_enabled():
            return
        if getattr(self, "connector", None) is not None:
            raise ValueError("[isoexec-gdn] CPR prefix caching admits a hit only from the GDN boundary checkpoints "
                             "this engine's workers hold; a KV connector loads prefixes it never scanned. Disable "
                             "the KV connector or prefix caching.")
        self.kv_cache_manager._isoexec_cpr_apc = SchedulerAdmission(self.vllm_config, engine_tag(self.vllm_config))

    orig_get = KVCacheManager.get_computed_blocks

    def get_computed_blocks(self, request, max_cache_hit_length=None):
        result = orig_get(self, request, max_cache_hit_length)
        admission = getattr(self, "_isoexec_cpr_apc", None)
        if admission is None or result[1] <= 0:
            return result
        return admission.filter(self, request, max_cache_hit_length, result, orig_get)

    orig_reset = Scheduler.reset_prefix_cache

    def reset_prefix_cache(self, *args, **kw):
        ok = orig_reset(self, *args, **kw)
        admission = getattr(self.kv_cache_manager, "_isoexec_cpr_apc", None)
        if ok and admission is not None:
            admission.advance_epoch()
        return ok

    Scheduler.__init__ = sched_init
    KVCacheManager.get_computed_blocks = get_computed_blocks
    Scheduler.reset_prefix_cache = reset_prefix_cache
    _scheduler_installed = True
    return True


# ------------------------------------------------------------------------------------------------- workers


WORKER_STATS = {"captured": 0, "adopted": 0, "adopted_tokens": 0, "epochs": 0, "skipped_full": 0}


#: req id -> the request's prefill token ids (prompt, plus outputs on a resume after preemption), on the host.
REQUEST_TOKENS: dict[str, np.ndarray] = {}


def track_requests(sched_out) -> None:
    """Remember each new request's prefill tokens; forget finished and preempted ones. Every step, from the
    first (which runs before the GDN pools exist)."""
    if sched_out is None:
        return
    for rid in list(getattr(sched_out, "finished_req_ids", None) or ()) + \
            list(getattr(sched_out, "preempted_req_ids", None) or ()):
        REQUEST_TOKENS.pop(str(rid), None)
    for new in getattr(sched_out, "scheduled_new_reqs", None) or ():
        toks = getattr(new, "prefill_token_ids", None)
        if toks is None:
            toks = getattr(new, "prompt_token_ids", None)
        # The scheduler admits no hit for these either (SchedulerAdmission.admits).
        usable = (toks is not None and not getattr(new, "mm_features", None)
                  and getattr(new, "lora_request", None) is None and getattr(new, "prompt_embeds", None) is None)
        if usable:
            REQUEST_TOKENS[str(new.req_id)] = np.asarray(toks, dtype=np.int64)
        else:
            REQUEST_TOKENS.pop(str(new.req_id), None)


class WorkerApc:
    """One worker's checkpoint store, published index and request tokens; driven once per step by ``_drive``."""

    def __init__(self, runner, layers):
        from vllm.distributed import parallel_state as ps

        from isoexec.ops.gdn.shared import gdn_cpr_state as cs

        kinds = {type(ly) for ly in layers}
        if len(kinds) != 1 or not hasattr(next(iter(kinds)), "install_boundaries"):
            raise RuntimeError("[isoexec-gdn] CPR prefix caching needs a GDN pool that captures and installs "
                               "boundary checkpoints (gdn_cpr_sm90); this engine's GDN pools are "
                               f"{sorted(k.__name__ for k in kinds)}")
        self.pool_cls = next(iter(kinds))
        C = int(layers[0].chunk_size)
        tag = engine_tag(runner.vllm_config)
        install_worker_sleep_clear()
        tp = ps.get_tensor_model_parallel_rank()
        tp_size = ps.get_tensor_model_parallel_world_size()
        pp = ps.get_pp_group().rank_in_group
        self.rank = pp * tp_size + tp
        self.index = RankIndex(tag, self.rank)
        self.epoch_file = Epoch(tag, create=True)
        self.epoch = 0
        self.store = cs.cpr_apc_store()
        cs.CPR_APC_ON_STORE.append(self.index.add)
        cs.CPR_APC_ON_RETRACT.append(self.index.remove)
        cs.CPR_APC_ON_CLEAR.append(lambda: self.index.reset(self.epoch))
        from .gdn_engine_patch import _align_block_size

        bs = _align_block_size(runner)
        self.grid = int(np.lcm(C, bs)) if bs > 0 else C
        if bs > 0 and bs % C:
            _say(f"WARNING: vLLM's mamba block size {bs} is not a multiple of {C}; checkpoints only at "
                 f"multiples of {self.grid}, so most block-aligned hits will be recomputed")
        self.tokens = REQUEST_TOKENS
        self._last_report = 0.0
        _say(f"worker rank {self.rank}: boundary store {self.store.budget / 2**20:.0f} MiB, checkpoint grid "
             f"{self.grid} (mamba block {bs or 'n/a'}), index {self.index._shm.path}")

    # -- one step ---------------------------------------------------------------------------------------
    def step(self, req_ids, nct, slots, layers) -> None:
        """Retire on a new epoch; adopt checkpoints for admitted hits; capture this step's boundaries."""
        from isoexec.ops.gdn.shared.gdn_cpr_state import cpr_apc_key

        capture_boundaries, install_boundaries = type(layers[0]).capture_boundaries, type(layers[0]).install_boundaries

        epoch = self.epoch_file.read()
        if epoch != self.epoch:
            self.store.retire_all()
            self.index.reset(epoch)
            self.epoch = epoch
            WORKER_STATS["epochs"] += 1
        self.store.tick()
        l0 = layers[0]
        adopt: list[tuple[int, int, bytes]] = []   # (batch index, position, key)
        capture: list[tuple[int, int, bytes]] = []
        for i, rid in enumerate(req_ids):
            p = int(nct[i])
            if p <= 0:
                continue
            slot = int(slots[i])
            if slot not in l0._slot2row:
                # First scheduled with p computed tokens and no row: the scheduler admitted a prefix-cache
                # hit of p tokens. Its checkpoint must be here.
                toks = self.tokens.get(str(rid))
                if toks is None or toks.shape[0] < p:
                    raise RuntimeError(
                        f"[isoexec-gdn] request {rid} starts with {p} computed tokens and no GDN row, but this "
                        "worker has no prefill tokens for it; a prefix-cache hit cannot be resumed")
                adopt.append((i, p, cpr_apc_key(toks, p)))
            elif p % self.grid == 0:
                toks = self.tokens.get(str(rid))
                if toks is not None and p <= toks.shape[0]:
                    key = cpr_apc_key(toks, p)
                    if key not in self.store:
                        capture.append((i, p, key))
        if adopt:
            checkpoints = []
            for i, p, key in adopt:
                got = self.store.get(key)
                if got is None or got[0] != p:
                    raise RuntimeError(
                        f"[isoexec-gdn] the scheduler admitted a {p}-token prefix-cache hit for request "
                        f"{req_ids[i]} but worker rank {self.rank} holds no checkpoint of that prefix. "
                        "Admission requires every worker to advertise the key, and a key leaves the index "
                        "before its checkpoint is released, so the index and the store disagree; refusing to "
                        "resume a state this pool never scanned.")
                checkpoints.append((got[1], got[2]))
            adopt_slots = [int(slots[i]) for i, _p, _k in adopt]
            rows = None
            for ly in layers:
                got_rows = ly._assign_many(adopt_slots)
                if rows is None:
                    rows = got_rows
                elif got_rows != rows:
                    raise RuntimeError("[isoexec-gdn] CPR prefix caching: the GDN pools are not lockstepped "
                                       "(adoption gave different rows in different layers)")
            install_boundaries(layers, rows, [p for _i, p, _k in adopt], checkpoints)
            WORKER_STATS["adopted"] += len(adopt)
            WORKER_STATS["adopted_tokens"] += sum(p for _i, p, _k in adopt)
        if capture:
            rows = [l0._slot2row[int(slots[i])] for i, _p, _k in capture]
            taken = capture_boundaries(layers, rows, [p for _i, p, _k in capture])
            for (i, p, key), (rec, conv) in zip(capture, taken):
                if self.index.full():
                    WORKER_STATS["skipped_full"] += 1
                    continue
                if self.store.put(key, p, rec, conv):
                    WORKER_STATS["captured"] += 1
        now = time.monotonic()
        if now - self._last_report > 30.0 and (WORKER_STATS["captured"] or WORKER_STATS["adopted"]):
            self._last_report = now
            if self.rank == 0:
                _say(f"worker: {WORKER_STATS} store {self.store.stats()}")


_WORKER: list[WorkerApc] = []


def worker_apc(runner, layers) -> WorkerApc:
    if not _WORKER:
        _WORKER.append(WorkerApc(runner, layers))
    return _WORKER[0]


def reset_worker_for_tests() -> None:
    """Forget this process's worker state (a test that builds a second engine in one process)."""
    from isoexec.ops.gdn.shared import gdn_cpr_state as cs

    _WORKER.clear()
    REQUEST_TOKENS.clear()
    cs.CPR_APC_ON_STORE.clear()
    cs.CPR_APC_ON_RETRACT.clear()
    cs.CPR_APC_ON_CLEAR.clear()
    cs.CPR_APC_STORE = None


_sleep_installed = False


def install_worker_sleep_clear() -> None:
    """Drop the boundary store at every worker sleep (idempotent): the engine is idle there, the checkpoints
    must not hold device memory through training, and the weights may change before the next wake."""
    global _sleep_installed
    if _sleep_installed:
        return
    from vllm.v1.worker import gpu_worker

    orig = gpu_worker.Worker.sleep
    if getattr(orig, "_isoexec_cpr_apc", False):
        _sleep_installed = True
        return

    def sleep(self, level: int = 1):
        # (self, level): the CPR sleep hooks (gdn_cpr_sleep) check this exact surface before wrapping it.
        from isoexec.ops.gdn.shared.gdn_cpr_state import cpr_apc_store_invalidate

        cpr_apc_store_invalidate()
        return orig(self, level)

    sleep._isoexec_cpr_apc = True
    gpu_worker.Worker.sleep = sleep
    _sleep_installed = True
