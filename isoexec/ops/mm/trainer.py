"""Trainer GEMM dispatch and identity readback from the installed provider.
Recorded bindings pin that provider object so later installs cannot silently change their arithmetic."""

from __future__ import annotations

import torch

from isoexec.ops.mm.wgrad import fusable, weight_grad


def _mm():
    """Return the vendored batch-invariant matrix multiplication implementation."""
    from isoexec.ops.mm import mm_batch_invariant as V

    return V


class _MmFn(torch.autograd.Function):
    """Use the installed trainer GEMM for forward and matching deterministic gradient operations."""

    @staticmethod
    def setup_context_for_backward(ctx, x, w):
        """Leave ``ctx`` as :meth:`backward` reads it: saved ``(x, w)``, the gradient-accumulation sink
        ``wsink`` and the TP link ``tp_link``. Every subclass ``forward`` (one per GEMM provider) calls it
        on the operands the backward differentiates against, before any provider-specific padding."""
        ctx.save_for_backward(x, w)
        ctx.wsink = fusable(w) if w.requires_grad else None
        ctx.tp_link = _consume_tp_link(x)

    @staticmethod
    def forward(ctx, x, w):
        _MmFn.setup_context_for_backward(ctx, x, w)
        return _mm().matmul_persistent(x, w.t())

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        dx = torch.ops.aten.mm.out(dy, w, out=torch.empty(dy.shape[0], w.shape[1], dtype=dy.dtype, device=dy.device))
        link = getattr(ctx, "tp_link", None)
        if link is not None and link.uses == 1:
            # dgrad first, its fixed-tree TP sum on the comm lane, then wgrad under it (async TP all-reduce).
            link.launch(dx)
        dw = weight_grad(dy, x, w, ctx.wsink) if ctx.needs_input_grad[1] else None
        return dx, dw


def _consume_tp_link(x):
    """Register this GEMM as a consumer of the tp_copy that produced ``x`` (if any); see ``tp_bytes``."""
    # (Grad mode is always off inside Function.forward: the tag itself says a tp_copy ran with grad.)
    link = getattr(x, "_isoexec_tp_copy", None)
    if link is not None:
        link.uses += 1
    return link


def mm_trainer(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``mm``: y[T,N] = x[T,K] @ w[N,K]^T through ``bi.matmul_persistent(x, w.t())`` (the callable
    ``aten::mm`` -> ``mm_fwd_scope.mm_impl`` -> ``bi.mm_batch_invariant`` reaches on the trainer)."""
    _refuse_stale_record(_mm().matmul_persistent)
    x = x.contiguous()
    if x.requires_grad or w.requires_grad:
        return _MmFn.apply(x, w)
    return _mm().matmul_persistent(x, w.t())


def _vendored_kernel():
    """The vendored batch-invariant Triton ``matmul_persistent`` under every installed wrapper.

    ``mm_sm90`` and ``mm_sm100`` each record it as
    ``_isoexec_stock_matmul_persistent`` before rebinding ``matmul_persistent``; with none installed
    the binding is still the vendored function itself.
    """
    V = _mm()
    return getattr(V, "_isoexec_stock_matmul_persistent", V.matmul_persistent)


class _VendoredMmFn(_MmFn):
    @staticmethod
    def forward(ctx, x, w):
        _MmFn.setup_context_for_backward(ctx, x, w)
        return _vendored_kernel()(x, w.t())


def mm_vendored(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``y[T,N] = x[T,K] @ w[N,K]^T`` on the vendored Triton kernel itself, whatever is installed.

    For a GEMM whose provider must not depend on which host process it runs in: a host may wrap
    ``matmul_persistent`` with its own provider (the sm90 segment-tree GEMM, say) and another may not, and
    ``aten::mm`` resolves to whichever registration came last. This names one kernel on every host.
    The backward is :class:`_MmFn`'s.
    """
    x = x.contiguous()
    if x.requires_grad or w.requires_grad:
        return _VendoredMmFn.apply(x, w)
    return _vendored_kernel()(x, w.t())


def _live_impl_id(provider) -> str:
    if getattr(provider, "_isoexec_sm100", False):
        return "sm100_tcgen05_segtree"
    if getattr(provider, "_isoexec_sm90_segtree", False):
        return "sm90_wgmma_segtree"
    return "triton_batch_invariant"


def mm_impl_id() -> str:
    """Return the identity of the installed provider object."""
    return _live_impl_id(_mm().matmul_persistent)


# The latest recorded (provider object, impl ID); later placement supersedes earlier plan-only resolution.
_RECORDED: list[tuple[object, str]] = []


def pin_recorded_impl() -> str:
    """Read the installed implementation ID and pin its provider object for the record.
    Resolution may precede installation; the first GEMM refuses if the provider has changed since that record."""
    provider = _mm().matmul_persistent
    impl = mm_impl_id()
    _RECORDED[:] = [(provider, impl)]
    return impl


def _refuse_stale_record(provider) -> None:
    """Refuse a GEMM whose provider is not the one the last record named."""
    if not _RECORDED or _RECORDED[0][0] is provider:
        return
    live = _live_impl_id(provider)
    if live == _RECORDED[0][1]:
        return
    raise RuntimeError(
        f"mm was recorded as {_RECORDED[0][1]!r} and this call reaches {live!r}: the provider changed "
        f"after the composition was resolved, so the record a peer compares names a kernel this "
        f"forward does not run, and the two mm impls are not twins. Run install_trainer_kernels() "
        f"BEFORE apply_placement, as every shipped host does, or re-resolve after installing."
    )


__all__ = ["mm_impl_id", "mm_trainer", "mm_vendored", "pin_recorded_impl"]
