"""Architecture-neutral moe API; importing the facade does not load kernel providers."""

import importlib

from isoexec.ops._dispatch import facade as _facade

# One owner on every architecture, named relative to this package. Parameter-storage bookkeeping
# holds process-wide epochs and the injected buffer allocator, so it must be one module. The engine
# owner combine and shared-expert owner are what the SM100 engine bindings wrap (``sm100.py``); each
# serves a kernel only after a bitwise admission against the eager stage it replaces. The aten mm
# installer puts both hosts' ``torch.mm`` (the fp32 router GEMM included) on the vendored Triton GEMM.
_SHARED_EXPORTS = {
    "_install_moe_matmul_invariance": ("moe_sm90.moe_batch_invariant", "_install_moe_matmul_invariance"),
    "attach": ("shared.packed_experts", "attach"),
    "bump_epoch": ("shared.param_storage", "bump_epoch"),
    "bump_sync_epoch": ("shared.param_storage", "bump_sync_epoch"),
    "engine_combine_owner": ("moe_sm90.moe_pik_combine_owner", "engine_combine_owner"),
    "engine_shared_glu_epilogue": ("moe_sm90.moe_preamble_o12", "engine_shared_glu_epilogue"),
    "materialize_meta_experts": ("shared.packed_experts", "materialize_meta_experts"),
    "moe_batch_invariant": ("moe_sm90.moe_batch_invariant", None),
    "moe_pik_combine_owner": ("moe_sm90.moe_pik_combine_owner", None),
    "moe_preamble_o12": ("moe_sm90.moe_preamble_o12", None),
    "packed_experts": ("shared.packed_experts", None),
    "param_storage": ("shared.param_storage", None),
    "release_to_storage_owner": ("shared.packed_experts", "release_to_storage_owner"),
    "set_buffer_allocator": ("shared.param_storage", "set_buffer_allocator"),
    "stage_shared_partial": ("moe_sm90.moe_pik_combine_owner", "stage_shared_partial"),
}

# Served by the selected kernel package. The fused O2 router, the fused permute and the Triton
# trainer combine are SM90 recipes; SM100 binds its own router, dispatch and combine instead.
_EXPORTS = {
    "GroupedPlan": ("ops", "GroupedPlan"),
    "RoutePlan": ("ops", "RoutePlan"),
    "build_combine_rows": ("moe_combine_kernel", "build_combine_rows"),
    "combine": ("ops", "combine"),
    "dispatch": ("ops", "dispatch"),
    "engine_route_mechanics_o2": ("moe_router_o2_kernel", "engine_route_mechanics_o2"),
    "engine_router_o2_counts": ("moe_router_o2_kernel", "engine_router_o2_counts"),
    "experts": ("ops", "experts"),
    "fc1": ("ops", "fc1"),
    "fc1_glu": ("ops", "fc1_glu"),
    "fc2": ("ops", "fc2"),
    "fused_combine": ("moe_combine_backward", "fused_combine"),
    "integration": ("integration", None),
    "make_plan": ("ops", "make_plan"),
    "moe": ("ops", "moe"),
    "ops": ("ops", None),
    "route": ("ops", "route"),
    "route_to_permuted_rows": ("moe_fused_permute", "route_to_permuted_rows"),
    "stats": ("integration", "stats"),
}

__all__ = sorted(set(_EXPORTS) | set(_SHARED_EXPORTS))

_kernel_getattr, _kernel_dir = _facade("moe", _EXPORTS, globals())


def __getattr__(name):
    if name not in _SHARED_EXPORTS:
        return _kernel_getattr(name)
    from isoexec.ops._dispatch import selected

    selected("moe")  # shared helpers do not bypass the unsupported-architecture gate
    module, symbol = _SHARED_EXPORTS[name]
    provider = importlib.import_module(f"{__name__}.{module}")
    return provider if symbol is None else getattr(provider, symbol)


def __dir__():
    return sorted(set(_kernel_dir()) | set(_SHARED_EXPORTS))
