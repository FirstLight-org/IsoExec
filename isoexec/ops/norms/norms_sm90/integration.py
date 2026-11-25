"""SM90 norm callable factories and implementation identities.

ops.norms.resolve validates width and gamma convention before selecting these
factories. The RMS and gated recipes have their own rounding schedules; unsupported
configurations cannot fall back to a different norm implementation.
"""

from __future__ import annotations

RMS_IMPL = "sm90_rms_v1"
GATED_IMPL = "sm90_gated_v1"
ADD_RMS_IMPL = "sm90_add_rms_v1"
RMS_STAGES = ("norms.rms.pre_attn", "norms.rms.pre_mlp", "norms.rms.final")


def rms(zero_centered: bool):
    """``norms.rms.*`` / ``norms.mla`` callable: ``fn(x, w, eps) -> y`` (same shape as ``x``)."""
    from isoexec.ops.norms.norms_sm90.ops import rms_norm

    def fn(x, w, eps):
        return rms_norm(x, w, eps, zero_centered)

    return fn


def add_rms(zero_centered: bool):
    """Fused residual join + ``norms.rms.*``: ``fn(x, residual, w, eps) -> (y, h)``.

    ``h = residual + x`` is the bf16 residual stream (one fp32 add, one RNE round -- what ``aten.add`` on
    two bf16 tensors computes) and ``y`` is recipe v1 on ``h``: bit-identical to the unfused
    ``h = residual + x; y = rms(h)``. Backward is the unfused program's backward, op for op: the
    norm's own backward kernel on ``h`` (same rstd, same bands), then the gradient reaching ``h`` from
    its other users added in bf16 exactly as autograd accumulates it, passed through to both addends.
    Operands outside the kernel's domain (dtype/shape/device) run the unfused statement itself.
    """
    from isoexec.ops.norms.norms_sm90.ops import add_rms_norm_fused

    def fn(x, residual, w, eps):
        return add_rms_norm_fused(x, residual, w, eps, zero_centered)

    return fn


def gated(zero_centered: bool = True):
    """``norms.gated_out`` callable: ``fn(x [T, hv, dv], z [T, hv, dv], w, eps) -> [T, hv*dv]``."""
    from isoexec.ops.norms.norms_sm90.ops import gated_rms_norm

    def fn(x, z, w, eps):
        return gated_rms_norm(x, z, w, eps, zero_centered)

    return fn


__all__ = ["ADD_RMS_IMPL", "GATED_IMPL", "RMS_IMPL", "add_rms", "gated", "rms"]
