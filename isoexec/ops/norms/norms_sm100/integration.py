"""SM100 norm callable factories and implementation identities."""

from __future__ import annotations

RMS_IMPL = "sm100_rms_v1"
GATED_IMPL = "sm100_gated_v1"
#: The fused residual join + RMSNorm, one identity per RMS recipe it runs on the rounded sum.
ADD_RMS_IMPL = "sm100_add_rms_v1"
ADD_RMS_IMPL_V2 = "sm100_add_rms_v2"
RMS_STAGES = ("norms.rms.pre_attn", "norms.rms.pre_mlp", "norms.rms.final")


def _check_placed(n: int, width: int | None) -> None:
    """The width and the active forward recipe are still the ones the binding was placed for."""
    from isoexec.ops.norms.norms_sm100.ops import active_recipe

    if width is not None and n != width:
        raise ValueError(f"RMSNorm width {n} differs from its placed width {width}")
    expected = 2 if rms_impl(n) == "sm100_rms_v2" else 1
    if active_recipe(n)[0] != expected:
        raise ValueError("RMSNorm recipe changed after placement; clear the benchmark override")


def rms(zero_centered: bool, width: int | None = None):
    """``norms.rms.*`` / ``norms.mla`` callable: ``fn(x, w, eps) -> y`` (same shape as ``x``)."""
    from isoexec.ops.norms.norms_sm100.ops import rms_norm

    def fn(x, w, eps):
        _check_placed(x.shape[-1], width)
        return rms_norm(x, w, eps, zero_centered)

    return fn


def add_rms(zero_centered: bool, width: int | None = None):
    """Fused residual join + ``norms.rms.*``: ``fn(x, residual, w, eps) -> (y, h)``.

    ``h = residual + x`` is the bf16 residual stream (one fp32 add, one RNE round -- what ``aten.add``
    on two bf16 tensors computes) and ``y`` is the width's RMS recipe on ``h``: bit-identical to the
    unfused ``h = residual + x; y = rms(h)``. Backward is the unfused program's backward
    (``ops._AddRMSNorm``).
    """
    from isoexec.ops.norms.norms_sm100.ops import add_rms_norm_fused

    def fn(x, residual, w, eps):
        _check_placed(x.shape[-1], width)
        return add_rms_norm_fused(x, residual, w, eps, zero_centered)

    return fn


def gated(zero_centered: bool = True):
    """``norms.gated_out`` callable: ``fn(x [T, hv, dv], z [T, hv, dv], w, eps) -> [T, hv*dv]``."""
    from isoexec.ops.norms.norms_sm100.ops import gated_rms_norm

    def fn(x, z, w, eps):
        return gated_rms_norm(x, z, w, eps, zero_centered)

    return fn


__all__ = ["ADD_RMS_IMPL", "GATED_IMPL", "RMS_IMPL", "add_rms", "add_rms_impl", "gated", "rms", "rms_impl"]


def rms_impl(width: int) -> str:
    """Mirror the shipped width-only recipe table in csrc/plan.h without compiling."""
    return "sm100_rms_v2" if width in (1536, 4096, 5120, 7168) else RMS_IMPL


def add_rms_impl(width: int) -> str:
    """The fused residual + RMS identity for ``width``: the add on top of :func:`rms_impl`'s recipe."""
    return ADD_RMS_IMPL_V2 if rms_impl(width) == "sm100_rms_v2" else ADD_RMS_IMPL
