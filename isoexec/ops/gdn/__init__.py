"""Architecture-neutral gdn API; importing the facade does not load kernel providers."""

import importlib

from isoexec.ops._dispatch import facade as _facade

# State ownership, flags, metadata and reference adapters have one canonical owner.
# Resolve their live attributes so convolution installation and mutable state globals
# are shared by the trainer, engine, and CPU reference helpers.
_SHARED_EXPORTS = {
    "CPR_LAYERS": ("gdn_cpr_state", "CPR_LAYERS"),
    "GDN_CPR_MIN_STATE_SHAPES": ("gdn_ops", "GDN_CPR_MIN_STATE_SHAPES"),
    "RecurrentGDN": ("gdn_recurrent_state", "RecurrentGDN"),
    "_HOST_BOOKKEEPING_STATS": ("gdn_recurrent_state", "_HOST_BOOKKEEPING_STATS"),
    "causal_conv1d_metadata": ("packed_meta_cache", "causal_conv1d_metadata"),
    "chunked_prefill_enabled": ("gdn_recurrent_state", "chunked_prefill_enabled"),
    "cpr_mode": ("gdn_ops", "cpr_mode"),
    "defer_ab": ("gdn_fused_split", "defer_ab"),
    "fla_chunk_size": ("gdn_ops", "fla_chunk_size"),
    "flush_slot_maps": ("gdn_recurrent_state", "flush_slot_maps"),
    "assign_prefill_rows": ("gdn_recurrent_state", "assign_prefill_rows"),
    "fused_rows_census": ("gdn_cpr_rows", "fused_rows_census"),
    "fused_split_census": ("gdn_fused_split", "fused_split_census"),
    "gdn_cpr_min_pages": ("gdn_ops", "gdn_cpr_min_pages"),
    "gdn_cpr_state": ("gdn_cpr_state", None),
    "gdn_native_conv": ("gdn_ops", "gdn_native_conv"),
    "gdn_native_kernels_enabled": ("gdn_ops", "gdn_native_kernels_enabled"),
    "gdn_recurrent_state": ("gdn_recurrent_state", None),
    "layer_batched_resync": ("gdn_cpr_state", "layer_batched_resync"),
    "native_core_census": ("gdn_native_core_bv64", "native_core_census"),
    "native_cpr_entry_census": ("gdn_ops", "native_cpr_entry_census"),
    "native_state_enabled": ("gdn_recurrent_state", "native_state_enabled"),
    "pin_fla_autotune_configs": ("gdn_batch_invariant", "pin_fla_autotune_configs"),
    "pin_gdn_rmsnorm_rows_per_block": ("gdn_batch_invariant", "pin_gdn_rmsnorm_rows_per_block"),
    "recurrent_mode": ("gdn_ops", "recurrent_mode"),
}

_EXPORTS = {
    "IMPL": ("integration", "IMPL"),
    "decode": ("ops", "decode"),
    "integration": ("integration", None),
    "ops": ("ops", None),
    "prefill": ("ops", "prefill"),
}

# Execution tuning is provider-specific; it is
# deliberately not exported as part of the architecture-neutral family contract.

__all__ = list(_EXPORTS) + list(_SHARED_EXPORTS)
__all__ += ["build_state_pool", "gdn_core", "refuse_incompatible_flags"]


def gdn_core(q, k, v, alpha, beta, A_log, dt_bias, cu_seqlens, *, cu_seqlens_cpu=None):
    """Packed GDN core with the selected provider, or the existing reference composition."""
    import torch

    from isoexec.ops._dispatch import backend_module, selected

    if selected("gdn") is not None:
        metadata = {} if cu_seqlens_cpu is None else {"cu_seqlens_cpu": cu_seqlens_cpu}
        return backend_module("gdn", "").gdn_core(
            q, k, v, alpha, beta, A_log, dt_bias, cu_seqlens.to(torch.int32), **metadata)
    from isoexec.ops.gdn.shared import gdn_ops as reference
    hv, hk = v.shape[1], q.shape[1]
    rep = hv // hk
    if rep > 1:
        q = q.repeat_interleave(rep, dim=1)
        k = k.repeat_interleave(rep, dim=1)
    out = reference.gdn_native_cpr(
        q.unsqueeze(0).contiguous(),
        k.unsqueeze(0).contiguous(),
        v.unsqueeze(0).contiguous(),
        alpha.unsqueeze(0).contiguous(),
        beta.unsqueeze(0).contiguous(),
        A_log,
        dt_bias,
        cu_seqlens=cu_seqlens.to(torch.int32),
    )
    return out.squeeze(0)


def build_state_pool(**kwargs):
    """Construct the state implementation matching the selected GDN arithmetic."""
    from isoexec.ops._dispatch import backend_module, selected

    if selected("gdn") is not None:
        return backend_module("gdn", "").build_state_pool(**kwargs)
    from isoexec.ops.gdn.shared.gdn_cpr_state import build_cpr_gdn

    return build_cpr_gdn(**kwargs)


def refuse_incompatible_flags():
    """Validate selected-provider flags; the CPU/reference path installs no GPU state provider."""
    from isoexec.ops._dispatch import backend_module, selected

    if selected("gdn") is not None:
        backend_module("gdn", "").refuse_incompatible_flags()

_kernel_getattr, _kernel_dir = _facade("gdn", _EXPORTS, globals())


def __getattr__(name):
    if name not in _SHARED_EXPORTS:
        return _kernel_getattr(name)
    from isoexec.ops._dispatch import selected

    selected("gdn")  # shared helpers do not bypass the unsupported-architecture gate
    module, symbol = _SHARED_EXPORTS[name]
    provider = importlib.import_module(f"{__name__}.shared.{module}")
    return provider if symbol is None else getattr(provider, symbol)


def __dir__():
    return sorted(set(_kernel_dir()) | set(_SHARED_EXPORTS))
