"""Exact level-2 sleep for IsoExec models (``ISOEXEC_VLLM_EXACT_SLEEP``, default on).

vLLM's level-1 sleep copies every byte of the ``weights`` pool to pinned host memory and back; level 2
discards the pool and ``.cpu().clone()``s every ``named_buffer`` instead. Neither fits an IsoExec model:

* the weight stream rewrites every registered Parameter on each update (``VllmModel.finish_weight_update``
  refuses an update that misses one), so backing those bytes up (level 1) is pure waste -- 98% of the pool;
* the packed projection / packed expert buffers ARE the Parameters' storage, so stock level 2 copies them
  (pageable, synchronously) although the stream rewrites them, while bytes that no buffer names (tensors held
  by extensions, slot maps allocated outside a module, gaps of mixed segments) are lost.

This module makes level 2 exact. It wraps ``Worker.sleep``/``wake_up``: at a level-2 sleep it computes, per
``weights``-tagged cumem allocation, the bytes NOT covered by a contiguous registered Parameter of the
``VllmModel`` (the complement is derived from storage addresses, not from names) and copies exactly those to a
persistent pinned stash; vLLM's own buffer snapshot is suppressed. At the wake that maps ``weights`` back they
are restored, so every non-Parameter byte holds what a level-1 sleep would have restored, and every Parameter
byte is undefined until the next coverage-checked update rewrites it. Until that update finishes, the worker
refuses to run a forward over scheduled tokens (``check_weights_live``) instead of sampling from discarded
weights.

Allocations outside the ``weights`` tag keep vLLM's policy (``runtime``/``cudagraph`` backed up, ``kv_cache``,
``workspace`` and IsoExec's CPR arena discarded); PIK symmetric memory and NCCL buffers are not in the cumem pool.
"""

from __future__ import annotations

import os
import time
from typing import Any, Iterable, Sequence


FLAG = "ISOEXEC_VLLM_EXACT_SLEEP"

#: cumem tag whose allocations level 2 discards and the weight stream refills.
WEIGHTS_TAG = "weights"

#: Covered spans shorter than this between two backed-up ranges are copied along (one copy instead of two).
#: Over-copying a Parameter span is harmless: the stream rewrites it after the restore.
COALESCE_BYTES = 1 << 20

_installed = False
_discarded: set[int] = set()  # id() of models whose Parameter bytes are discarded (fast-path emptiness test)


class narrowed_named_buffers:
    """Temporarily narrow ``module.named_buffers`` (what stock level-2 sleep snapshots and restores).

    Composes with an enclosing narrowing (it filters whatever ``named_buffers`` currently yields) and restores
    the previous binding on exit.
    """

    _MISSING = object()

    def __init__(self, module, keep):
        self.module, self.keep = module, keep

    def __enter__(self):
        module, keep = self.module, self.keep
        self.prev = module.__dict__.get("named_buffers", self._MISSING)
        current = module.named_buffers

        def named_buffers(*args, **kwargs):
            return ((n, b) for n, b in current(*args, **kwargs) if keep(b))

        module.__dict__["named_buffers"] = named_buffers
        return self

    def __exit__(self, *exc):
        if self.prev is self._MISSING:
            self.module.__dict__.pop("named_buffers", None)
        else:
            self.module.__dict__["named_buffers"] = self.prev
        return False


def discard_only_storages() -> set[int]:
    """Storage bases of the CPR arenas (discard-at-sleep state owned by ``ops/gdn/shared/gdn_cpr_sleep``)."""
    from isoexec.ops.gdn.shared.gdn_cpr_sleep import _ARENAS

    return {flat.untyped_storage().data_ptr() for _, flat, _ in _ARENAS}


def exact_sleep_enabled() -> bool:
    """Whether level-2 sleep uses the exact complement backup (read at call time)."""
    return os.environ.get(FLAG, "1").lower() not in ("", "0", "false", "no")


def _say(msg: str) -> None:
    """fd 1 rather than ``logger.info``: vLLM filters INFO out of engine subprocesses."""
    try:
        os.write(1, f"[ISOEXEC-EXACT-SLEEP] pid={os.getpid()} {msg}\n".encode())
    except Exception:  # pragma: no cover - fd 1 closed
        pass


# ----------------------------------------------------------------------------------- range arithmetic
def merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorted union of half-open ``[lo, hi)`` ranges (empty ones dropped)."""
    out: list[list[int]] = []
    for lo, hi in sorted(r for r in ranges if r[1] > r[0]):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(lo, hi) for lo, hi in out]


def parameter_ranges(parameters: Iterable[Any]) -> list[tuple[int, int]]:
    """Byte ranges every update rewrites: contiguous CUDA Parameters only.

    A non-contiguous Parameter's span has holes the update does not write, so it covers nothing (its bytes
    are then backed up like any other).
    """
    ranges = []
    for p in parameters:
        if not getattr(p, "is_cuda", False) or p.numel() == 0 or not p.is_contiguous():
            continue
        lo = int(p.data_ptr())
        ranges.append((lo, lo + p.numel() * p.element_size()))
    return merge_ranges(ranges)


def backup_ranges(allocations: Sequence[tuple[int, int]], covered: Sequence[tuple[int, int]],
                  coalesce: int = COALESCE_BYTES) -> list[tuple[int, int]]:
    """Per allocation ``(ptr, size)``: the bytes no covered range owns, as half-open ranges.

    Two uncovered ranges separated by a covered span shorter than ``coalesce`` merge into one. Ranges never
    cross an allocation boundary (each allocation is mapped and unmapped on its own).
    """
    import bisect

    covered = merge_ranges(covered)
    starts = [lo for lo, _ in covered]
    out: list[tuple[int, int]] = []
    for base, size in sorted(allocations):
        end = base + size
        pieces: list[list[int]] = []
        cur = base
        i = max(bisect.bisect_right(starts, base) - 1, 0)
        while i < len(covered) and covered[i][0] < end:
            lo, hi = covered[i]
            i += 1
            if hi <= cur:
                continue
            if lo > cur:
                pieces.append([cur, lo])
            cur = max(cur, hi)
            if cur >= end:
                break
        if cur < end:
            pieces.append([cur, end])
        merged: list[list[int]] = []
        for lo, hi in pieces:
            if merged and lo - merged[-1][1] < coalesce:
                merged[-1][1] = hi
            else:
                merged.append([lo, hi])
        out.extend((lo, hi) for lo, hi in merged)
    return out


# ----------------------------------------------------------------------------------------- the stash
class _DeviceBytes:
    """A raw device byte range as a CUDA array (no ownership): lets torch copy it like a tensor."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False),
                                         "version": 3, "strides": None}


def _device_view(ptr: int, nbytes: int, device):
    import torch

    return torch.as_tensor(_DeviceBytes(ptr, nbytes), device=device)


class Stash:
    """A persistent pinned host buffer holding the backed-up ranges of one worker.

    Pinned once and reused across sleeps (pinning is the expensive part of a host copy); it only grows.
    """

    def __init__(self) -> None:
        self.host = None
        self.ranges: list[tuple[int, int]] = []
        self.allocations: dict[int, int] = {}  # allocation ptr -> size the ranges were taken from
        self.pending = False

    def _ensure(self, nbytes: int) -> None:
        import torch

        if self.host is None or self.host.numel() < nbytes:
            self.host = None
            self.host = torch.empty(max(nbytes, 1), dtype=torch.uint8, device="cpu", pin_memory=True)

    def save(self, ranges: Sequence[tuple[int, int]], allocations: dict[int, int], device) -> int:
        import torch

        total = sum(hi - lo for lo, hi in ranges)
        self._ensure(total)
        off = 0
        for lo, hi in ranges:
            n = hi - lo
            self.host[off:off + n].copy_(_device_view(lo, n, device), non_blocking=True)
            off += n
        torch.cuda.synchronize(device)
        self.ranges = list(ranges)
        self.allocations = dict(allocations)
        self.pending = True
        return total

    def restore(self, live: dict[int, tuple[int, bool]], device) -> tuple[int, int]:
        """Copy every range back whose allocation is mapped again with its old size.

        ``live`` maps allocation ptr -> (size, is_asleep). A range whose allocation was freed meanwhile has no
        reader (its contents died with it) and is skipped. Returns (bytes restored, ranges skipped).
        """
        import torch

        bases = sorted(self.allocations)
        import bisect

        off = done = skipped = 0
        for lo, hi in self.ranges:
            n = hi - lo
            base = bases[bisect.bisect_right(bases, lo) - 1]
            size, asleep = live.get(base, (None, True))
            if size != self.allocations[base] or asleep:
                skipped += 1
            else:
                _device_view(lo, n, device).copy_(self.host[off:off + n], non_blocking=True)
                done += n
            off += n
        torch.cuda.synchronize(device)
        self.pending = False
        self.ranges, self.allocations = [], {}
        return done, skipped


# ------------------------------------------------------------------------------------ the model contract
def _isoexec_model(worker):
    """The worker's VllmModel, or None (stock models keep stock sleep)."""
    try:
        from .model import VllmModel

        model = worker.model_runner.get_model()
    except Exception:  # pragma: no cover - runner without a model yet
        return None
    return model if isinstance(model, VllmModel) else None


def mark_discarded(model) -> None:
    model._isoexec_weights_discarded = True
    _discarded.add(id(model))


def mark_restored(model) -> None:
    """Called by ``finish_weight_update`` once a complete, coverage-checked update has rewritten every Parameter."""
    if getattr(model, "_isoexec_weights_discarded", False):
        model._isoexec_weights_discarded = False
    _discarded.discard(id(model))


def weights_discarded(model) -> bool:
    return bool(_discarded) and bool(getattr(model, "_isoexec_weights_discarded", False))


class WeightsDiscardedError(RuntimeError):
    pass


def check_weights_live(model) -> None:
    if weights_discarded(model):
        raise WeightsDiscardedError(
            "IsoExec engine: the Parameters were discarded by a level-2 sleep and no complete weight update has "
            "finished since; refusing to run a forward on undefined weights (start_weight_update -> load_weights "
            "-> finish_weight_update first)")


def _discard_only_storages() -> set[int]:
    """Storage bases of state that is discard-at-sleep by design (the CPR arena)."""
    try:
        return discard_only_storages()
    except Exception:  # pragma: no cover
        return set()


# ---------------------------------------------------------------------------------------------- hooks
def _weights_allocations(alloc) -> dict[int, int]:
    return {int(ptr): int(d.handle[1]) for ptr, d in alloc.pointer_to_data.items()
            if d.tag == WEIGHTS_TAG and not d.is_asleep}


def _exact_sleep(worker, orig_sleep, level: int):
    import gc

    import torch
    from vllm.device_allocator.cumem import CuMemAllocator

    model = _isoexec_model(worker)
    alloc = CuMemAllocator.instance
    stash: Stash = worker.__dict__.setdefault("_isoexec_stash", Stash())
    runner_model = worker.model_runner.model
    if stash.pending:
        # Weights already discarded and stashed; nothing mapped to back up again.
        with narrowed_named_buffers(runner_model, lambda b: False):
            return orig_sleep(worker, level)
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    # Release cached pool blocks first, so the census below sees exactly what the unmap will drop.
    gc.collect()
    torch.cuda.empty_cache()
    allocations = _weights_allocations(alloc)
    covered = parameter_ranges(p for _n, p in model.named_parameters())
    ranges = backup_ranges(list(allocations.items()), covered)
    device = torch.device("cuda", torch.cuda.current_device())
    saved = stash.save(ranges, allocations, device)
    t_save = time.perf_counter() - t0
    pool = sum(allocations.values())
    # vLLM's snapshot would .cpu().clone() buffers whose bytes the stash (or the stream) already owns.
    with narrowed_named_buffers(runner_model, lambda b: False):
        out = orig_sleep(worker, level)
    mark_discarded(model)
    # One line per sleep / wake: the receipt of what was discarded and what was kept.
    _say(f"level-2 sleep: weights pool {pool / 2**30:.3f} GiB in {len(allocations)} allocations; "
         f"{sum(h - l for l, h in covered) / 2**30:.3f} GiB Parameter-covered (rewritten by the next update, "
         f"discarded); {saved / 2**20:.1f} MiB in {len(ranges)} ranges backed up to a pinned stash in "
         f"{t_save:.3f}s; total sleep {time.perf_counter() - t0:.3f}s")
    return out


def _exact_wake(worker, out_tags) -> None:
    import torch
    from vllm.device_allocator.cumem import CuMemAllocator

    stash = worker.__dict__.get("_isoexec_stash")
    if stash is None or not stash.pending or not (out_tags is None or WEIGHTS_TAG in out_tags):
        return
    alloc = CuMemAllocator.instance
    live = {int(p): (int(d.handle[1]), bool(d.is_asleep)) for p, d in alloc.pointer_to_data.items()}
    t0 = time.perf_counter()
    device = torch.device("cuda", torch.cuda.current_device())
    done, skipped = stash.restore(live, device)
    _say(f"wake(weights): restored {done / 2**20:.1f} MiB of non-Parameter bytes in "
         f"{time.perf_counter() - t0:.3f}s ({skipped} ranges of freed allocations skipped); Parameters await "
         f"the weight update")


def _sleep_level(args, kwargs) -> int:
    return int(kwargs["level"]) if "level" in kwargs else int(args[0]) if args else 1


def _install_level2_guard(_gw) -> None:
    """``ISOEXEC_VLLM_EXACT_SLEEP=0`` (or a moved surface): an IsoExec model never runs stock level 2.

    Stock level 2 snapshots only ``named_buffers``; extension-held tensors, slot maps and gaps of the weights
    pool would be lost and the next forward would read them silently. A level-2 request for an IsoExec model
    therefore sleeps at level 1 (offload and restore the whole pool) -- what the rollout side picks with the
    flag off. The sleep level may be chosen in another process (verl's rollout server) whose environment
    differs from this worker's, so the guard is installed regardless of this process's flag.
    """
    if getattr(_gw.Worker, "_ix_level2_guard", False):
        return
    orig_sleep = _gw.Worker.sleep

    def sleep(self, *args, **kwargs):
        if _sleep_level(args, kwargs) == 2 and _isoexec_model(self) is not None:
            _say(f"level-2 sleep requested for an IsoExec model with exact sleep unavailable in this worker "
                 f"({FLAG}={os.environ.get(FLAG, '1')!r}); sleeping at level 1 instead")
            kwargs.pop("level", None)
            return orig_sleep(self, 1, *args[1:], **kwargs)
        return orig_sleep(self, *args, **kwargs)

    sleep.__wrapped__ = orig_sleep
    sleep.__name__, sleep.__qualname__, sleep.__doc__ = orig_sleep.__name__, orig_sleep.__qualname__, orig_sleep.__doc__
    _gw.Worker.sleep = sleep
    _gw.Worker._ix_level2_guard = True


def install_exact_sleep_hooks() -> bool:
    """Wrap ``Worker.sleep``/``wake_up``/``execute_model`` (idempotent; refuses loudly on a surface change).

    Called by every IsoExec model built with sleep mode on, whatever this process's flag says. With the flag
    off, only the level-2 guard is installed (an IsoExec model then sleeps at level 1) and False is returned.
    """
    global _installed
    if _installed:
        return True
    from vllm.v1.worker import gpu_worker as _gw

    if not exact_sleep_enabled():
        _install_level2_guard(_gw)
        return False
    try:
        import inspect

        from vllm.device_allocator.cumem import CuMemAllocator

        sig_sleep = list(inspect.signature(_gw.Worker.sleep).parameters)
        sig_wake = list(inspect.signature(_gw.Worker.wake_up).parameters)
        sig_exec = list(inspect.signature(_gw.Worker.execute_model).parameters)
        missing = [a for a in ("instance", "use_memory_pool") if not hasattr(CuMemAllocator, a)]
        if getattr(_gw.Worker, "_ix_level2_guard", False):
            raise RuntimeError("the level-2 guard of an earlier flag-off model is installed in this process")
        if sig_sleep != ["self", "level"] or sig_wake != ["self", "tags"] or sig_exec[:2] != ["self", "scheduler_output"]:
            raise RuntimeError(f"Worker.sleep{tuple(sig_sleep)} / wake_up{tuple(sig_wake)} / "
                               f"execute_model{tuple(sig_exec)} are not the surface these hooks wrap")
        if missing:
            raise RuntimeError(f"CuMemAllocator lacks {missing}")
    except Exception as e:
        raise RuntimeError(f"{FLAG}=1 but the vLLM worker/cumem surface moved ({e}); set {FLAG}=0 to sleep at "
                           "level 1 instead") from e
    if getattr(_gw.Worker, "_ix_exact_sleep", False):
        _installed = True
        return True

    orig_sleep, orig_wake, orig_exec = _gw.Worker.sleep, _gw.Worker.wake_up, _gw.Worker.execute_model

    def sleep(self, level: int = 1):
        if level != 2 or CuMemAllocator.instance is None or _isoexec_model(self) is None:
            if level == 2:
                if _isoexec_model(self) is not None:
                    # No cumem pool to census: never stock level 2 on an IsoExec model.
                    _say("level-2 sleep without a cumem allocator; sleeping at level 1 instead")
                    return orig_sleep(self, 1)
                # Stock level 2 restores every snapshotted buffer at the weights wake; discard-only arenas
                # are unmapped then, so they must stay out of the snapshot.
                gone = _discard_only_storages()
                if gone:
                    with narrowed_named_buffers(self.model_runner.model,
                                                lambda b: b.untyped_storage().data_ptr() not in gone):
                        return orig_sleep(self, level)
            return orig_sleep(self, level)
        return _exact_sleep(self, orig_sleep, level)

    def wake_up(self, tags=None):
        out = orig_wake(self, tags)
        _exact_wake(self, tags)
        return out

    def execute_model(self, scheduler_output, *args, **kwargs):
        if _discarded and getattr(scheduler_output, "total_num_scheduled_tokens", 0) > 0:
            check_weights_live(_isoexec_model(self))
        return orig_exec(self, scheduler_output, *args, **kwargs)

    for fn, orig in ((sleep, orig_sleep), (wake_up, orig_wake), (execute_model, orig_exec)):
        fn.__wrapped__ = orig
        fn.__name__, fn.__qualname__, fn.__doc__ = orig.__name__, orig.__qualname__, orig.__doc__
    _gw.Worker.sleep, _gw.Worker.wake_up, _gw.Worker.execute_model = sleep, wake_up, execute_model
    _gw.Worker._ix_exact_sleep = True
    _installed = True
    _say("INSTALLED: level-2 sleep keeps non-Parameter weights-pool bytes in a pinned stash; Parameters are "
         "discarded and must be rewritten by a coverage-checked update before the next forward")
    return True


__all__ = [
    "COALESCE_BYTES", "FLAG", "Stash", "WeightsDiscardedError", "backup_ranges", "check_weights_live",
    "exact_sleep_enabled", "install_exact_sleep_hooks", "mark_discarded", "mark_restored", "merge_ranges",
    "narrowed_named_buffers",
    "parameter_ranges", "weights_discarded",
]
