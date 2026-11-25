"""Context-parallel trainer attention for gqa_sm100: every row's CP=1 bits, half the work per rank.

A CP rank holds the zigzag chunks ``r`` and ``2*cp - 1 - r`` of every packed sequence (SkyRL / Megatron
layout, ``ops/collectives/cp_layout.py``). This module runs the SAME numerical program as the CP=1 trainer
core on that cut, so each local row comes out with exactly the bits CP=1 gives it, in the forward and in
the backward:

* **Forward.** All-gather K and V (the only operands a local query reads from other ranks) into the
  global packed order. Each local chunk is one request of the packed forward: its queries, a key start at
  the sequence's first token and ``seqused_k`` = the chunk's end, so query ``i`` sits at its own logical
  position and attends the global causal prefix. The kernel's contract is that a row's bits are a
  function of (logical position, keys) only -- not of the batch, the request cut or the chunk boundary
  (README section 1.1, ``tests/test_seam_gpu.py``) -- so the local rows are the CP=1 rows.
* **Backward.** FA4's head-dim-256 backward owns every dQ tile (256 query rows) and every dK/dV tile (128
  keys) in one CTA pair that loops over the other operand's tiles in a fixed order, with no cross-tile
  reduction. A tile computed over the GLOBAL operands is therefore the CP=1 tile. This rank all-gathers Q,
  dO and the two per-row statistics (lse * log2 e, rowsum(O * dO)), and runs FA4's backward over the global
  packed problem with a tile mask: only the dQ tiles holding a local query and the dK/dV tiles holding a
  local key are computed. The local rows of those tiles are returned; nothing is summed across ranks.

Load balance comes from the zigzag cut (each rank owns one early and one late chunk of every sequence), so
a rank computes about half of the CP=1 tiles in both directions. Communication is byte movement only:
K|V forward, Q|dO|stats backward, all-gathered on the CP group.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from typing import Callable

import torch

from isoexec.ops.collectives.cp_layout import local_chunks, prefix_entries
from isoexec.schema import Placement

#: FA4 head-dim-256 backward tiles: a dQ tile is a CTA pair's 256 query rows, a dK/dV tile its 128 keys.
DQ_TILE_ROWS = 256
DKDV_TILE_KEYS = 128
#: The preprocess's statistics layout: each request's rows start at its offset rounded with one
#: 128-row pad per request (``seqlen_info.SeqlenInfoQK.padded_offset_q``).
STATS_TILE = 128
#: Tile-mask rows are padded to a multiple of 8 int32 entries: the compiled kernel assumes 16-byte strides.
_MASK_ALIGN = 8


class CpAttentionError(ValueError):
    """The cut, the operands and this core's head dim do not describe one context-parallel problem."""


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _stats_rows(cu: "list[int]") -> "tuple[list[int], int]":
    """Padded statistics slot of every packed row of ``cu``, and the padded total the FA4 interface sizes."""
    slots: list[int] = []
    for b, (start, stop) in enumerate(zip(cu, cu[1:])):
        base = (start + b * STATS_TILE) // STATS_TILE * STATS_TILE
        slots.extend(range(base, base + stop - start))
    return slots, (cu[-1] + len(cu) * STATS_TILE - 1) // STATS_TILE * STATS_TILE


def _tile_mask(lengths: "list[int]", ranges: "list[list[tuple[int, int]]]", tile: int) -> "list[list[int]]":
    """Per sequence, 1 for every ``tile``-row tile that holds a row of one of its local ``ranges``."""
    width = _round_up(max(1, max((n + tile - 1) // tile for n in lengths)), _MASK_ALIGN)
    rows = []
    for n, own in zip(lengths, ranges):
        row = [0] * width
        for a, e in own:
            for t in range(a // tile, (e + tile - 1) // tile):
                row[t] = 1
        rows.append(row)
    return rows


@dataclass(frozen=True)
class CpAttentionPlan:
    """One microbatch's cut, as the device tensors both directions launch with (built once per boundary
    tensor and shared by every attention layer)."""

    cp: int
    rank: int
    local: int  #: this rank's packed rows
    total: int  #: global packed rows
    max_chunk: int
    max_seqlen: int
    cu: torch.Tensor  #: int32 [B+1] global boundaries
    #: forward: one request per local chunk
    cu_q_local: torch.Tensor  #: int32 [2B+1]
    seqused_k: torch.Tensor  #: int32 [2B], the chunk's end inside its sequence
    k_starts: torch.Tensor  #: int32 [2B+1], the sequence's first global row
    #: movement
    perm: torch.Tensor  #: global row g = rank-ordered gathered row perm[g]
    mine: torch.Tensor  #: global row of every local row, in local order
    #: backward
    local_stats_slots: torch.Tensor  #: padded preprocess slot of every local row (local requests)
    local_stats_total: int
    global_stats_slots: torch.Tensor  #: padded slot of every global row (global requests)
    global_stats_total: int
    q_tile_mask: torch.Tensor  #: int32 [B, W] dQ tiles to compute
    kv_tile_mask: torch.Tensor  #: int32 [B, W] dK/dV tiles to compute


def build_plan(cu_seqlens: "list[int]", cp: int, rank: int, device) -> CpAttentionPlan:
    """The plan of ``rank`` for global boundaries ``cu_seqlens`` (every sequence a multiple of ``2*cp``)."""
    from isoexec.ops.collectives.cp_layout import gathered_to_global, local_to_global

    cu = [int(v) for v in cu_seqlens]
    entries = prefix_entries(cu, cp, rank)
    lengths = [b - a for a, b in zip(cu, cu[1:])]
    cu_q = [0]
    for e in entries:
        cu_q.append(cu_q[-1] + e.q_len)
    ranges = []
    for n in lengths:
        size = n // (2 * cp)
        ranges.append([(c * size, (c + 1) * size) for c in local_chunks(cp, rank)])
    local_slots, local_total = _stats_rows(cu_q)
    global_slots, global_total = _stats_rows(cu)

    def i32(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    def i64(values):
        return torch.tensor(values, dtype=torch.long, device=device)

    return CpAttentionPlan(
        cp=cp, rank=rank, local=cu_q[-1], total=cu[-1],
        max_chunk=max(e.q_len for e in entries), max_seqlen=max(lengths),
        cu=i32(cu), cu_q_local=i32(cu_q), seqused_k=i32([e.kv_len for e in entries]),
        k_starts=i32([e.kv_start for e in entries] + [cu[-1]]),
        perm=i64(gathered_to_global(cu, cp)), mine=i64(local_to_global(cu, cp, rank)),
        local_stats_slots=i64(local_slots), local_stats_total=local_total,
        global_stats_slots=i64(global_slots), global_stats_total=global_total,
        q_tile_mask=i32(_tile_mask(lengths, ranges, DQ_TILE_ROWS)),
        kv_tile_mask=i32(_tile_mask(lengths, ranges, DKDV_TILE_KEYS)),
    )


#: Plans by boundary-tensor identity (weak), so every attention layer of one microbatch shares one.
_PLANS: "dict[int, tuple[weakref.ref, CpAttentionPlan]]" = {}


def plan_for(cu_seqlens: torch.Tensor, cp: int, rank: int) -> CpAttentionPlan:
    entry = _PLANS.get(id(cu_seqlens))
    if entry is not None and entry[0]() is cu_seqlens and (entry[1].cp, entry[1].rank) == (cp, rank):
        return entry[1]
    plan = build_plan(cu_seqlens.tolist(), cp, rank, cu_seqlens.device)
    key = id(cu_seqlens)
    _PLANS[key] = (weakref.ref(cu_seqlens, lambda _ref, key=key: _PLANS.pop(key, None)), plan)
    return plan


# --------------------------------------------------------------------------- the arithmetic, per rank


def forward_rows(q: torch.Tensor, k_global: torch.Tensor, v_global: torch.Tensor, plan: CpAttentionPlan,
                 scale: float) -> "tuple[torch.Tensor, torch.Tensor]":
    """``(out [T_local, Hq, D], lse [T_local, Hq])`` of the local queries over the global keys."""
    from isoexec.ops.attention.gqa_sm100 import ops

    ops._check(q, k_global, v_global, "gqa_sm100.context_parallel")

    def run(qq):
        return ops.gqa_sm100_fwd(qq, k_global, v_global, plan.cu_q_local, plan.seqused_k, softmax_scale=float(scale),
                                 seg_tokens=ops.segment_span(q.shape[-1]), max_seqlen_q=plan.max_chunk,
                                 max_seqlen_k=plan.max_seqlen, cu_seqlens_k=plan.k_starts)

    grouped = ops._by_head_groups(q, k_global.shape[-2], None, run)
    return run(q.contiguous()) if grouped is None else grouped


def local_stats(out: torch.Tensor, dout: torch.Tensor, lse: torch.Tensor, plan: CpAttentionPlan) -> torch.Tensor:
    """FA4's backward preprocess on the local rows -> ``[T_local, 2, Hq]`` fp32 (lse * log2 e, rowsum(O dO))."""
    from isoexec.ops._vendor.flash_attn_cute.interface_bwd_sm100 import backward_stats_sm100

    lse_log2, dpsum = backward_stats_sm100(out, dout, lse.t().contiguous(), plan.cu_q_local)
    rows = plan.local_stats_slots
    return torch.stack((lse_log2.index_select(1, rows), dpsum.index_select(1, rows))).permute(2, 0, 1).contiguous()


def backward_rows(dout: torch.Tensor, q_global: torch.Tensor, k_global: torch.Tensor, v_global: torch.Tensor,
                  dout_global: torch.Tensor, stats_global: torch.Tensor, plan: CpAttentionPlan,
                  scale: float) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """``(dq, dk, dv)`` of the local rows: FA4's backward over the global problem, local tiles only.

    ``stats_global`` is ``[T, 2, Hq]`` from :func:`local_stats` of every rank, in global order.
    """
    from isoexec.ops._vendor.flash_attn_cute.interface_bwd_sm100 import flash_attn_varlen_bwd_sm100

    hq = q_global.shape[1]
    stats = q_global.new_zeros((2, hq, plan.global_stats_total), dtype=torch.float32)
    stats.index_copy_(2, plan.global_stats_slots, stats_global.permute(1, 2, 0))
    dq, dk, dv = flash_attn_varlen_bwd_sm100(
        q_global, k_global, v_global, None, dout_global, None, plan.cu, plan.cu, plan.max_seqlen, plan.max_seqlen,
        softmax_scale=float(scale), causal=True, deterministic=False, disable_2cta=False,
        stats=(stats[0], stats[1]), q_tile_mask=plan.q_tile_mask, kv_tile_mask=plan.kv_tile_mask,
    )
    mine = plan.mine
    return dq.index_select(0, mine), dk.index_select(0, mine), dv.index_select(0, mine)


# --------------------------------------------------------------------------- movement and autograd


def _gather_global(local: torch.Tensor, plan: CpAttentionPlan, group) -> torch.Tensor:
    """All-gather ``[T_local, ...]`` over the CP group and return it in the global packed order."""
    import torch.distributed as dist

    local = local.contiguous()
    gathered = local.new_empty((plan.cp * plan.local, *local.shape[1:]))
    dist.all_gather_into_tensor(gathered, local, group=group)
    return gathered.index_select(0, plan.perm)


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(t.shape[0], -1).view(torch.uint8)


class _CpAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, plan: CpAttentionPlan, group, scale: float):  # noqa: D102
        T, hk, d = k.shape
        kv = _gather_global(torch.cat((k.reshape(T, -1), v.reshape(T, -1)), dim=1), plan, group)
        k_global = kv[:, : hk * d].view(-1, hk, d)
        v_global = kv[:, hk * d :].view(-1, hk, d)
        out, lse = forward_rows(q, k_global, v_global, plan, scale)
        ctx.save_for_backward(q, k_global, v_global, out, lse)
        ctx.plan, ctx.group, ctx.scale = plan, group, scale
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout):  # noqa: D102
        q, k_global, v_global, out, lse = ctx.saved_tensors
        plan = ctx.plan
        dout = dout.contiguous()
        T, hq, d = q.shape
        stats = local_stats(out, dout, lse, plan)
        # One gather moves Q, dO and the fp32 statistics as bytes.
        packed = torch.cat((_as_bytes(q), _as_bytes(dout), _as_bytes(stats)), dim=1)
        packed = _gather_global(packed, plan, ctx.group)
        width = hq * d * q.element_size()
        q_global = packed[:, :width].contiguous().view(q.dtype).view(-1, hq, d)
        dout_global = packed[:, width : 2 * width].contiguous().view(q.dtype).view(-1, hq, d)
        stats_global = packed[:, 2 * width :].contiguous().view(torch.float32).view(-1, 2, hq)
        dq, dk, dv = backward_rows(dout, q_global, k_global, v_global, dout_global, stats_global, plan, ctx.scale)
        return dq, dk, dv, None, None, None


def make_trainer_core(placement: Placement) -> Callable[..., torch.Tensor]:
    """The trainer core at ``placement.cp``: ``core(q, k, v, cu_seqlens, max_q, scale)`` with this rank's
    rows of q/k/v and the GLOBAL packed boundaries, returning this rank's output rows."""
    axis = placement.cp
    if axis.group is None:
        raise CpAttentionError("Placement.cp carries no process group; the host must hand in the live CP group")
    cp, rank, group = axis.size, axis.rank, axis.group

    def core(q, k, v, cu_seqlens, max_q, scale):
        del max_q  # the plan reads the boundaries
        if q.shape[-1] != 256:
            raise CpAttentionError(
                f"gqa_sm100 context parallelism runs FA4's head-dim-256 backward, whose tiles are owned whole; "
                f"head dim {q.shape[-1]} is not served")
        if not math.isfinite(scale) or scale <= 0:
            raise CpAttentionError("gqa_sm100: softmax scale must be finite and positive")
        plan = plan_for(cu_seqlens, cp, rank)
        if q.shape[0] != plan.local or k.shape[0] != plan.local:
            raise CpAttentionError(f"q/k carry {q.shape[0]}/{k.shape[0]} rows; this rank's cut holds {plan.local}")
        return _CpAttention.apply(q, k, v, plan, group, float(scale))

    return core


__all__ = [
    "CpAttentionError",
    "CpAttentionPlan",
    "DKDV_TILE_KEYS",
    "DQ_TILE_ROWS",
    "backward_rows",
    "build_plan",
    "forward_rows",
    "local_stats",
    "make_trainer_core",
    "plan_for",
]
