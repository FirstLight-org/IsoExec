"""SM100 norm callable factories and implementation identities."""

from __future__ import annotations

RMS_IMPL = "sm100_rms_v3"
GATED_IMPL = "sm100_gated_v3"
#: The fused residual join + RMSNorm on the same schedule.
ADD_RMS_IMPL = "sm100_add_rms_v3"
RMS_STAGES = ("norms.rms.pre_attn", "norms.rms.pre_mlp", "norms.rms.final")


def _check_placed(n: int, width: int | None) -> None:
    """The width is still the one the binding was placed for."""
    if width is not None and n != width:
        raise ValueError(f"RMSNorm width {n} differs from its placed width {width}")


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
    on two bf16 tensors computes) and ``y`` is the RMS schedule on ``h``: bit-identical to the
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
    """The RMS identity for ``width``: one schedule, so one identity for every width."""
    return RMS_IMPL


def add_rms_impl(width: int) -> str:
    """The fused residual + RMS identity for ``width``."""
    return ADD_RMS_IMPL
