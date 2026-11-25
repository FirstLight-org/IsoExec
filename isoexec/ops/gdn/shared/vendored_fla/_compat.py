"""The three vLLM utilities the vendored FLA kernels reach for, re-expressed without vLLM.

The vendored tree must import nothing from a runtime -- that is the whole point of vendoring it
(kernel-ownership.md 1). Upstream reaches outside ``fla/ops/`` for exactly two things, and both
are thin:

* ``vllm.triton_utils`` -- when Triton is installed it is literally ``import triton``,
  ``import triton.language as tl``, ``import triton.language.extra.libdevice as tldevice``; the
  rest of that module is a placeholder for builds without Triton, which cannot run these kernels
  at all. We take the real imports and let the ImportError speak for itself.
* ``vllm.platforms.current_platform.is_cuda_alike()`` -- consulted once, in ``utils.py``, to
  decide ``device = "cuda" if is_cuda_alike() else get_available_device()``. It is true for CUDA
  and for ROCm, which is what ``torch.cuda.is_available()`` reports on both (ROCm torch keeps the
  ``cuda`` namespace). The two answers can only differ on a CUDA build with no visible GPU, and
  there the value feeds only ``device_torch_lib`` -> ``get_all_max_shared_mem()``, which is
  already wrapped in ``except BaseException: return [-1]``. No kernel constant depends on it.

Keeping the substitution here rather than in the kernel bodies is deliberate: the recorded rewrite
rules stay one-line import redirects, so diffing against a newer upstream when re-vendoring shows
what UPSTREAM changed and never what we changed.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


class _CurrentPlatform:
    """``vllm.platforms.current_platform``, narrowed to the one predicate the kernels read."""

    @staticmethod
    def is_cuda_alike() -> bool:
        # CUDA or ROCm. torch reports both through `torch.cuda`.
        return torch.cuda.is_available()


current_platform = _CurrentPlatform()

__all__ = ["tl", "tldevice", "triton", "current_platform"]
