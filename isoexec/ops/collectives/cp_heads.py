"""Head-wise context parallelism: re-cut a mixer from (local tokens, all heads) to (all tokens, local heads).

A recurrent mixer (GDN) cannot run on a CP rank's token cut: its state and its convolution halo cross the
chunk boundaries. Megatron's native GDN CP is head-wise, and so is this seam: an all-to-all hands every
rank the WHOLE packed sequence for ``1/cp`` of the heads, the per-head program (convolution, delta rule)
runs unchanged on it, and a second all-to-all returns every rank its own tokens for all heads. Each head's
arithmetic is a function of that head's operands only -- the property that makes the core's bits
tensor-parallel invariant -- so the result is the CP=1 result, in both directions: the movement is bytes,
and its adjoint is the reverse all-to-all, with no sum.

An operand group is a list of column ``segments`` ``(offset, heads, head_width)`` of the source; rank ``d``
owns heads ``[d * heads/cp, (d+1) * heads/cp)`` of every segment, so its columns are one contiguous range
per segment and packing is a strided 2-D copy per (rank, segment). Rows move between the zigzag local
order and the global packed order with the layout's permutations (``cp_gather.CpLayout``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isoexec.ops.collectives.cp_gather import CpLayout


class CpHeadsError(ValueError):
    """The operands do not split into whole heads over the CP group."""


def _all_to_all(x: torch.Tensor, group) -> torch.Tensor:
    """``x`` ``[cp, ...]``: block ``d`` goes to rank ``d``; returns ``[cp, ...]`` with block ``s`` from rank ``s``."""
    import torch.distributed as dist

    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group)
    return out


@dataclass(frozen=True)
class HeadGroup:
    """Source column ranges of one operand group: ``ranges[d]`` lists ``(src_start, width)`` for rank ``d``."""

    width: int  #: columns each rank receives
    ranges: tuple

    @staticmethod
    def of(segments: "list[tuple[int, int, int]]", cp: int) -> "HeadGroup":
        ranges = []
        for d in range(cp):
            mine = []
            for offset, heads, head_width in segments:
                if heads % cp:
                    raise CpHeadsError(f"{heads} heads do not split over cp={cp}")
                share = heads // cp * head_width
                mine.append((offset + d * share, share))
            ranges.append(tuple(mine))
        return HeadGroup(sum(w for _, w in ranges[0]), tuple(ranges))

    def pack(self, src: torch.Tensor) -> torch.Tensor:
        """``[cp, T, width]``: rank ``d``'s columns of every row of ``src``."""
        out = src.new_empty((len(self.ranges), src.shape[0], self.width))
        for d, mine in enumerate(self.ranges):
            at = 0
            for start, width in mine:
                out[d, :, at : at + width].copy_(src[:, start : start + width])
                at += width
        return out

    def unpack_into(self, dst: torch.Tensor, packed: torch.Tensor) -> None:
        """The adjoint of :meth:`pack`: write ``packed[d]`` back into rank ``d``'s columns of ``dst``."""
        for d, mine in enumerate(self.ranges):
            at = 0
            for start, width in mine:
                dst[:, start : start + width].copy_(packed[d, :, at : at + width])
                at += width


class _ToHeads(torch.autograd.Function):
    """``[T_local, C]`` (local tokens, every head) -> one ``[T_global, group.width]`` tensor per head group
    (global packed order, this rank's heads). Adjoint: the reverse all-to-alls, written into the source
    columns; columns no group reads get zeros."""

    @staticmethod
    def forward(ctx, x, groups: "tuple[HeadGroup, ...]", layout: CpLayout, group):  # noqa: D102
        outs = []
        for g in groups:
            received = _all_to_all(g.pack(x), group)  # [cp(source), T_local, w]: rank-ordered gathered rows
            outs.append(received.view(-1, g.width).index_select(0, layout.perm))
        ctx.groups, ctx.layout, ctx.group, ctx.shape = groups, layout, group, tuple(x.shape)
        return tuple(outs)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grads):  # noqa: D102
        layout = ctx.layout
        out = grads[0].new_zeros(ctx.shape)
        for g, grad in zip(ctx.groups, grads):
            if grad is None:
                continue
            ordered = grad.index_select(0, layout.inv_perm).view(layout.cp, layout.local, g.width)
            g.unpack_into(out, _all_to_all(ordered, ctx.group))
        return out, None, None, None


class _ToTokens(torch.autograd.Function):
    """``[T_global, w]`` (this rank's head range) -> ``[T_local, cp * w]`` (local tokens, heads in rank order).
    Adjoint: the reverse movement."""

    @staticmethod
    def forward(ctx, y, layout: CpLayout, group):  # noqa: D102
        received = _all_to_all(y.index_select(0, layout.inv_perm).view(layout.cp, layout.local, -1), group)
        ctx.layout, ctx.group = layout, group
        return received.transpose(0, 1).reshape(layout.local, -1)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):  # noqa: D102
        layout = ctx.layout
        send = grad.reshape(layout.local, layout.cp, -1).transpose(0, 1).contiguous()
        back = _all_to_all(send, ctx.group)  # [cp(token owner), T_local, w]
        return back.flatten(0, 1).index_select(0, layout.perm), None, None


def to_heads(x: torch.Tensor, groups: "tuple[HeadGroup, ...]", layout: CpLayout, group) -> "tuple[torch.Tensor, ...]":
    if x.ndim != 2 or x.shape[0] != layout.local:
        raise CpHeadsError(f"to_heads takes [T_local={layout.local}, C], got {tuple(x.shape)}")
    return _ToHeads.apply(x, groups, layout, group)


def to_tokens(y: torch.Tensor, layout: CpLayout, group) -> torch.Tensor:
    if y.ndim != 2 or y.shape[0] != layout.total:
        raise CpHeadsError(f"to_tokens takes [T_global={layout.total}, w], got {tuple(y.shape)}")
    return _ToTokens.apply(y, layout, group)


def make_gdn_region(placement, *, key_heads: int, value_heads: int, key_head_dim: int, value_head_dim: int):
    """The head-wise CP region of a GDN mixer at ``placement.cp``.

    ``region(proj, mix, conv_w, A_log, dt_bias, seq)`` takes the in-projection output of this rank's tokens
    (``q|k|v|z|beta|alpha`` columns, every head), moves ``q|k|v`` and ``beta|alpha`` to (all tokens, this
    rank's heads), calls ``mix(qkv, beta, alpha, conv_w, A_log, dt_bias, seq)`` with this rank's rows of the
    per-head parameters and the global boundaries, and returns its ``[T_global, heads, dv]`` output as
    ``[T_local, value_heads, dv]``. ``z`` never moves: the gated norm runs per token after the region.
    """
    from isoexec.ops.collectives.cp_gather import layout_for

    axis = placement.cp
    cp, rank, group = axis.size, axis.rank, axis.group
    if group is None:
        raise CpHeadsError("Placement.cp carries no process group; the host must hand in the live CP group")
    if key_heads % cp or value_heads % cp:
        raise CpHeadsError(f"GDN head-wise CP needs cp={cp} to divide {key_heads} key and {value_heads} value heads")
    kd, vd, hv = key_heads * key_head_dim, value_heads * value_head_dim, value_heads
    qkv_segments = [(0, key_heads, key_head_dim), (kd, key_heads, key_head_dim), (2 * kd, value_heads, value_head_dim)]
    groups = (HeadGroup.of(qkv_segments, cp),
              HeadGroup.of([(2 * kd + 2 * vd, hv, 1), (2 * kd + 2 * vd + hv, hv, 1)], cp))
    conv_rows = torch.cat([torch.arange(start, start + width) for start, width in HeadGroup.of(qkv_segments, cp).ranges[rank]])
    local_hv = hv // cp
    heads = slice(rank * local_hv, (rank + 1) * local_hv)
    rows_on: dict = {}

    def region(proj, mix, conv_w, A_log, dt_bias, seq):
        layout = layout_for(seq.cu_seqlens, cp, rank)
        qkv, ba = to_heads(proj, groups, layout, group)
        rows = rows_on.get(conv_w.device)
        if rows is None:
            rows = rows_on[conv_w.device] = conv_rows.to(conv_w.device)
        o = mix(qkv, ba[:, :local_hv], ba[:, local_hv:], conv_w.index_select(0, rows), A_log[heads], dt_bias[heads], seq)
        return to_tokens(o.flatten(1), layout, group).view(layout.local, value_heads, value_head_dim)

    return region


__all__ = ["CpHeadsError", "HeadGroup", "make_gdn_region", "to_heads", "to_tokens"]
