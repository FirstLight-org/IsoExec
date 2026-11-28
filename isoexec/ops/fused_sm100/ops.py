"""Python entry points for the SM100 fused glue kernels (graph-capturable: no host sync, no allocation
inside the extension; outputs are allocated here)."""

from __future__ import annotations

import torch

from isoexec.ops.fused_sm100._extension import extension


def scatter_rows(srcs, dsts, rows, cols, col_offset: int = 0):
    """``dst_t[rows[n], cols[n] + col_offset] = src_t[n]`` for 1..4 pairs in one launch.

    ``src_t [N, ...]``, ``dst_t [B, L, ...]`` contiguous, same dtype, equal row bytes (a multiple of 16). Bitwise
    the same as the torch advanced-index assignment ``dst[rows, cols + col_offset] = src`` (pure copies). Distinct
    ``(row, col)`` pairs are the caller's contract (a repeated pair is a data race in torch as well)."""
    extension().scatter_rows(list(srcs), list(dsts), rows, cols, int(col_offset))


def embed_expand(weight: torch.Tensor, ids: torch.Tensor, streams: int, *, one_hot_pre: bool = False):
    """``weight[ids]`` repeated over ``streams`` -> ``[*ids.shape, streams, H]``; with ``one_hot_pre`` also the
    fp32 ``pre [*ids.shape, streams]`` that is 1 on stream 0 and 0 elsewhere. Bitwise equal to
    ``embedding(ids, weight).unsqueeze(-2).expand(..., streams, H).contiguous()`` (pure copies)."""
    lead = tuple(ids.shape)
    flat = ids.reshape(-1)
    if flat.dtype != torch.long:
        flat = flat.long()
    out = torch.empty(flat.numel(), streams, weight.shape[1], dtype=weight.dtype, device=weight.device)
    pre = (torch.empty(flat.numel(), streams, dtype=torch.float32, device=weight.device) if one_hot_pre
           else torch.empty(0, dtype=torch.float32, device=weight.device))
    extension().embed_expand(weight, flat.contiguous(), out, pre)
    out = out.view(*lead, streams, weight.shape[1])
    return (out, pre.view(*lead, streams)) if one_hot_pre else out


def argmax_rows(x: torch.Tensor) -> torch.Tensor:
    """``x.argmax(-1)`` for a contiguous ``[R, V]`` float32/bfloat16/float16 tensor, bitwise (NaN wins, equal
    values keep the lowest index: the same total order as torch's reduction)."""
    R, V = x.shape
    out = torch.empty(R, dtype=torch.long, device=x.device)
    if R == 0:
        return out
    # slices per row: fill the machine (the winner does not depend on this choice: total order)
    S = max(1, min(64, 1024 // R, -(-V // 4096)))
    scratch = torch.empty(R, S, 2, dtype=torch.int32, device=x.device)
    extension().argmax_rows(x, out, scratch)
    return out


def mul_rows(y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``w * y`` for bf16 ``y [..., H]`` and ``w [H]`` (one rounding), bitwise equal to the torch product."""
    out = torch.empty_like(y)
    extension().mul_rows(y.contiguous(), w.contiguous(), out)
    return out


__all__ = ["argmax_rows", "embed_expand", "mul_rows", "scatter_rows"]
