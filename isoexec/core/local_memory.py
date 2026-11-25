"""Keep the CUDA context's local-memory (per-thread stack) reservation at what selected kernels need.

The driver sizes each context's local-memory reservation as ``cudaLimitStackSize`` x the device's
resident-thread capacity (``max_threads_per_multi_processor`` x ``multi_processor_count``). A
launch that needs a larger per-thread stack raises the limit automatically, and the driver never
lowers it again on its own (``CU_CTX_LMEM_RESIZE_TO_MAX`` is deprecated because keeping the maximum
is now the only behaviour). The memory is outside PyTorch's allocator: ``empty_cache`` and every
offload leave it in place for the life of the process.

Triton autotuning is the common way to hit this. A sweep compiles and launches every candidate
config; a candidate that spills heavily (for example FLA's ``chunk_bwd_kernel_dqkwg`` at
``num_warps=2``: ~27 KB of stack per thread, 20x slower and never selected) inflates the
reservation to ``27 KB x 2048 x 132`` = ~6.8 GiB on an H100 even though the config that wins needs
under 1 KB. Whether a sweep runs depends only on the on-disk autotune cache, so a cold cache costs
that much device memory for the rest of the run.

``install_autotune_guard`` restores the pre-sweep limit as soon as a sweep finishes. The limit is
read from the device, never assumed, so the guard is architecture- and kernel-agnostic. Restoring
is correctness-neutral: a later launch that needs more stack grows the limit again by itself.
``release_local_memory`` is the same restore for callers that hand the device to another process
(colocated training), covering kernels whose selected config genuinely needs a large stack.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import os
import threading
from typing import Any

_CUDA_LIMIT_STACK_SIZE = 0  # cudaLimitStackSize
_CUDART: Any = None
_LOCK = threading.Lock()
_GUARD_INSTALLED = False


def _cudart() -> Any:
    """The CUDA runtime this process already loaded (PyTorch's), so limits act on its contexts."""
    global _CUDART
    if _CUDART is not None:
        return _CUDART
    candidates: list[str] = []
    try:
        with open("/proc/self/maps", encoding="utf-8") as maps:
            for line in maps:
                path = line.rsplit(" ", 1)[-1].strip()
                if os.path.basename(path).startswith("libcudart.so") and path not in candidates:
                    candidates.append(path)
    except OSError:
        pass
    found = ctypes.util.find_library("cudart")
    if found:
        candidates.append(found)
    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
            lib.cudaDeviceGetLimit.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
            lib.cudaDeviceSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
        except (OSError, AttributeError):
            continue
        _CUDART = lib
        return lib
    raise RuntimeError("no CUDA runtime library is loaded in this process")


def stack_limit() -> int:
    """The current device's per-thread stack limit in bytes."""
    value = ctypes.c_size_t(0)
    rc = _cudart().cudaDeviceGetLimit(ctypes.byref(value), _CUDA_LIMIT_STACK_SIZE)
    if rc != 0:
        raise RuntimeError(f"cudaDeviceGetLimit(cudaLimitStackSize) failed with CUDA error {rc}")
    return int(value.value)


def reserved_bytes(limit: int) -> int:
    """The local memory a per-thread stack of ``limit`` bytes reserves on the current device."""
    import torch

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return int(limit) * props.max_threads_per_multi_processor * props.multi_processor_count


def release_local_memory(limit: int, *, reason: str = "") -> int:
    """Lower the per-thread stack limit to ``limit`` bytes if a launch raised it; return bytes released.

    Synchronizes the device first (lowering the limit reallocates local memory, which must not race
    running kernels) and does nothing while the current stream is capturing a CUDA graph.
    """
    import torch

    if torch.cuda.is_current_stream_capturing():
        return 0
    with _LOCK:
        current = stack_limit()
        if current <= limit:
            return 0
        torch.cuda.synchronize()
        rc = _cudart().cudaDeviceSetLimit(_CUDA_LIMIT_STACK_SIZE, limit)
        if rc != 0:
            raise RuntimeError(f"cudaDeviceSetLimit(cudaLimitStackSize, {limit}) failed with CUDA error {rc}")
        released = reserved_bytes(current) - reserved_bytes(limit)
    print(f"[ISOEXEC-LMEM] pid={os.getpid()} {reason or 'released'}: per-thread stack {current} B -> {limit} B, "
          f"{released / 2**30:.2f} GiB of local memory returned", flush=True)
    return released


def install_autotune_guard() -> bool:
    """Restore the stack limit after every Triton autotune sweep in this process. Idempotent.

    Wraps ``triton.runtime.autotuner.Autotuner`` so subclasses (FLA's ``CachedAutotuner``, vLLM's
    copies) are covered. Only calls that actually benchmarked pay for the restore; cached
    selections take one attribute check.
    """
    global _GUARD_INSTALLED
    if _GUARD_INSTALLED:
        return True
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        return False
    if getattr(Autotuner, "_isoexec_lmem_guard", False):
        _GUARD_INSTALLED = True
        return True
    bench, run = Autotuner._bench, Autotuner.run

    @functools.wraps(bench)
    def _bench(self, *args, **kwargs):
        if getattr(self, "_isoexec_sweep_limit", None) is None:
            try:
                self._isoexec_sweep_limit = stack_limit()
            except RuntimeError:
                self._isoexec_sweep_limit = -1
        return bench(self, *args, **kwargs)

    @functools.wraps(run)
    def _run(self, *args, **kwargs):
        try:
            return run(self, *args, **kwargs)
        finally:
            limit = getattr(self, "_isoexec_sweep_limit", None)
            if limit is not None:
                self._isoexec_sweep_limit = None
                if limit >= 0:
                    release_local_memory(limit, reason=f"after autotuning {getattr(self, 'base_fn', self.fn).__name__}")

    Autotuner._bench, Autotuner.run = _bench, _run
    Autotuner._isoexec_lmem_guard = True
    _GUARD_INSTALLED = True
    return True


__all__ = ["install_autotune_guard", "release_local_memory", "reserved_bytes", "stack_limit"]
