"""pik -- bitwise-identical LLM math across tensor-parallel sizes.

Determinism is a property of the reduction plan, not of the kernel: fix how K is cut into leaves and how
those leaves are combined, and every other knob (block sizes, warps, stages, grid order, split-K along leaf
boundaries, the all-reduce algorithm, the TP size itself) stays free.
"""

from .allreduce import tree_all_reduce
from .arch import ArchProfile, arch_tag
from .arch import current as current_arch
from .plan import DEFAULT_PLAN, ReductionPlan, combine_order


def __getattr__(name):
    # GEMM autotuning selects a CUDA architecture at import time. Keep plan and
    # code-generation utilities usable on CPU without initializing CUDA.
    from importlib import import_module

    modules = {
        "ti_gemm": ".gemm",
        "row_parallel_linear": ".linear",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "arch_tag",
    "current_arch",
    "ArchProfile",
    "ReductionPlan",
    "DEFAULT_PLAN",
    "combine_order",
    "ti_gemm",
    "tree_all_reduce",
    "row_parallel_linear",
]
