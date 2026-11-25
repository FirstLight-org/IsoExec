"""Expert-major packing of per-expert holders without changing a value.

``packed_slices`` places per-holder tensors (in stacking order) in one contiguous expert-major storage
and re-points each holder at its slice (``param.data = view``): one copy of the bytes, later
``load_state_dict`` copies land in the packed storage. It is idempotent: when the holders already are
adjacent slices of one storage the result is an alias (no copy). ``Stack`` is the differentiable
form: per-holder masters -> their packed alias, the gradient returned as per-holder slices.

Holders that another owner placed (a trainer's data-parallel buffer slices) are not re-pointed: ``stacked``
gives the operand as a stacked copy that carries its source parameters, and a training op saves it with
``save_operands`` (the parameters, not the copy; ``saved_operands`` re-stacks the same bytes in backward), so
no layer keeps a copy of its experts from its forward until its backward.
"""
from __future__ import annotations

import torch


def alias_slices(tensors, shape):
    """The tensor of ``shape`` whose consecutive slices are ``tensors`` when they already tile one
    storage contiguously, else None."""
    t0 = tensors[0]
    expect = t0.data_ptr()
    base = t0.untyped_storage().data_ptr()
    for t in tensors:
        if (not t.is_contiguous() or t.dtype != t0.dtype or t.device != t0.device
                or t.untyped_storage().data_ptr() != base or t.data_ptr() != expect):
            return None
        expect += t.numel() * t.element_size()
    stride, acc = [], 1
    for s in reversed(shape):
        stride.append(acc)
        acc *= s
    out = torch.empty(0, dtype=t0.dtype, device=t0.device)
    out.set_(t0.untyped_storage(), t0.storage_offset(), tuple(shape), tuple(reversed(stride)))
    return out


def packed_slices(params, shape):
    """Alias ``params`` (in stacking order) as one ``shape`` tensor, packing them first if needed."""
    out = alias_slices(params, shape)
    if out is not None:
        return out
    out = torch.empty(shape, dtype=params[0].dtype, device=params[0].device)
    flat = out.view(-1)
    offset = 0
    for p in params:
        n = p.numel()
        view = flat[offset:offset + n].view(p.shape)
        view.copy_(p.detach())
        p.data = view
        offset += n
    return alias_slices(params, shape)


class Stack(torch.autograd.Function):
    """Per-holder masters -> their packed alias; the gradient returns as per-holder slices."""

    @staticmethod
    def forward(ctx, shape, *params):
        ctx.shapes = [p.shape for p in params]
        out = alias_slices(params, shape)
        if out is None:
            raise RuntimeError("masters are not packed")
        return out

    @staticmethod
    def backward(ctx, grad):
        flat = grad.reshape(-1)
        out, offset = [], 0
        for s in ctx.shapes:
            n = s.numel()
            out.append(flat[offset:offset + n].view(s))
            offset += n
        return (None, *out)


#: Provenance of a stacked expert operand: per expert, the parameters it was stacked from, in stacking order (the
#: attribute Qwen's grouped experts set on their stacked operand, ``nn/moe.py`` ``expert_weight_views``).
EXPERT_PARAMS_ATTR = "_isoexec_expert_params"


def stacked(params, shape, per_expert):
    """``torch.stack(params).view(shape)`` (a copy; differentiable to ``params``), tagged with its source
    parameters (``per_expert`` consecutive parameters per expert) for ``save_operands``."""
    out = torch.stack(params).view(shape)
    setattr(out, EXPERT_PARAMS_ATTR,
            tuple(tuple(params[i:i + per_expert]) for i in range(0, len(params), per_expert)))
    return out


def save_operands(ctx, *tensors):
    """``ctx.save_for_backward(*tensors)``, except that a ``stacked`` copy is saved as its source parameters:
    the copy is freed with the forward's locals and ``saved_operands`` re-stacks the same bytes in backward
    (autograd's version check on the saved parameters refuses an in-place update in between)."""
    flat, layout = [], []
    for t in tensors:
        source = None if t is None else getattr(t, EXPERT_PARAMS_ATTR, None)
        if source is None:
            flat.append(t)
            layout.append(None)
        else:
            params = [p for expert in source for p in expert]
            flat += params
            layout.append((tuple(t.shape), len(params)))
    ctx.save_for_backward(*flat)
    ctx.operand_layout = tuple(layout)


def saved_operands(ctx):
    """The tensors given to ``save_operands``, a stacked copy re-stacked from its saved parameters."""
    saved, out, i = ctx.saved_tensors, [], 0
    for entry in ctx.operand_layout:
        if entry is None:
            out.append(saved[i])
            i += 1
        else:
            shape, n = entry
            with torch.no_grad():
                out.append(torch.stack(saved[i:i + n]).view(shape))
            i += n
    return out


__all__ = ["EXPERT_PARAMS_ATTR", "Stack", "alias_slices", "packed_slices", "save_operands", "saved_operands",
           "stacked"]
