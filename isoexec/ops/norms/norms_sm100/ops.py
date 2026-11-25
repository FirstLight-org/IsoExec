"""SM100 norm APIs backed by graph-capturable C++ forward/autograd entry points without device synchronization."""

from __future__ import annotations

from isoexec.ops.norms.norms_sm100._extension import extension
from isoexec.ops.norms.norms_sm100.spec import RECIPE_VERSION


def supported_width(n: int) -> bool:
    """Widths the forward covers (mirrors ``csrc/plan.h::fwd_supported`` without building anything)."""
    return n > 0 and (n % 256 == 0 or n in (128, 384))


def rms_norm(x, weight, eps: float, zero_centered: bool):
    return extension().rms_norm(x, weight, eps, zero_centered)


def rms_norm_round_scale(x, weight, eps: float):
    """``weight * rms_norm(x, ones)`` bitwise (bf16(x * rstd) rounded, then the bf16 product rounded), forward only."""
    return extension().rms_norm_round_scale(x, weight, eps)


def add_rms_norm(x, residual, weight, eps: float, zero_centered: bool):
    return extension().add_rms_norm(x, residual, weight, eps, zero_centered)


class _AddRMSNorm:
    """Autograd for the fused residual join + RMSNorm whose backward is the UNFUSED program's backward.

    Forward is the fused kernel (``h = rn_bf16(x + residual)`` stored, then the RMS schedule on ``h``).
    Backward is ``raw_rms_bwd`` -- the kernel ``RMSNormFn`` runs, same schedule -- followed by
    the bf16 accumulation autograd does for the unfused ``h = residual + x; y = rms(h)``: the norm's
    input gradient rounded to bf16 first, then the residual gradient added in bf16.
    """

    _fn = None

    @classmethod
    def get(cls):
        if cls._fn is None:
            import torch

            class Fn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, residual, weight, eps, zero_centered):
                    out, h, rstd = extension().raw_rms_fwd(x, residual, weight, eps, zero_centered)
                    # An unconsumed output must reach backward as None, not as zeros: dx + 0 would turn
                    # a -0 gradient into +0, which the unfused program never does.
                    ctx.set_materialize_grads(False)
                    ctx.save_for_backward(h, weight, rstd)
                    ctx.zero_centered = zero_centered
                    return out, h

                @staticmethod
                def backward(ctx, dy, dh):
                    h, weight, rstd = ctx.saved_tensors
                    dw = None
                    if dy is not None:
                        dx, dw = extension().raw_rms_bwd(dy, h, rstd, weight, None, ctx.zero_centered)
                        dsum = dx if dh is None else dx + dh
                    else:
                        dsum = dh
                    if dsum is None:
                        return None, None, None, None, None
                    return dsum, dsum, (dw if ctx.needs_input_grad[2] else None), None, None

            cls._fn = Fn
        return cls._fn


def add_rms_norm_fused(x, residual, weight, eps: float, zero_centered: bool):
    """``(rms(residual + x), residual + x)`` with one kernel; bit-identical to the unfused pair.

    Returns ``(y, h)``. Operands outside the fused kernel's domain run the unfused statement
    (``residual + x`` then ``rms_norm``), the same program, so the result never depends on the branch.
    """
    import torch

    if not (
        isinstance(x, torch.Tensor) and isinstance(residual, torch.Tensor)
        and x.dtype == torch.bfloat16 and residual.dtype == torch.bfloat16
        and x.shape == residual.shape and x.is_cuda and residual.is_cuda and x.device == residual.device
        and x.dim() >= 1 and supported_width(int(x.shape[-1]))
    ):
        h = residual + x
        return rms_norm(h, weight, eps, zero_centered), h
    if torch.is_grad_enabled() and (x.requires_grad or residual.requires_grad or weight.requires_grad):
        return _AddRMSNorm.get().apply(x, residual, weight, eps, zero_centered)
    return extension().add_rms_norm(x, residual, weight, eps, zero_centered)


def gated_rms_norm(x, z, weight, eps: float, zero_centered: bool = True):
    return extension().gated_rms_norm(x, z, weight, eps, zero_centered)


def schedule():
    """``(hidden-axis segment length, row segment length)`` the loaded kernels were compiled with."""
    return extension().schedule()


def recipe(n: int):
    """``(fwd_lanes, fwd_chunks, bwd_lanes, bwd_chunks)`` for width ``n``: threads per row and 16-byte loads per
    thread. Execution geometry only; the numerical schedule (``schedule.py``) does not read it."""
    return tuple(extension().plan(n, 1)[:4])


def plan(n: int, rows: int):
    """Execution geometry at width ``n`` and ``rows`` rows (none of it numerical); see ``csrc/host.cpp::plan``."""
    return extension().plan(n, rows)


__all__ = ["RECIPE_VERSION", "add_rms_norm", "add_rms_norm_fused", "gated_rms_norm", "plan", "recipe",
           "rms_norm", "rms_norm_round_scale", "schedule", "supported_width"]
