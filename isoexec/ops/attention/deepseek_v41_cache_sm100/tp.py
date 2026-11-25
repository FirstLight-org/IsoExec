"""Tensor-parallel head sharding of the native V4.1 attention (recipe ``deepseek_v41_attn_sm100_v1``, TP-t).

Rank ``r`` of ``t`` owns attention heads ``[r H/t, (r+1) H/t)`` (``wq_b`` rows, ``attn_sink``) and output
groups ``[r G/t, (r+1) G/t)`` (``wo_a`` rows, ``wo_b`` columns). ``wq_a``, ``wkv``, the norms, the
compressor and the WHOLE indexer (index key path, ``indexer.wq_b``, ``weights_proj``) are replicated, so
every rank holds identical window/compressed KV and index keys and computes the identical index scores.

The one cross-rank reduction (``wo_b``) is **TP-degree invariant** (the pik ROOT/PUBLISH scheme): its K is
cut into ``spec.TP_LEAVES = 8`` fixed leaves (8 equal K column blocks = output groups), every leaf is an
FP32 partial, and one balanced adjacent-pair FP32 tree over the 8 leaves is rounded to BF16 once. Rank r
of t (t | 8) owns the aligned leaves ``[8r/t, 8(r+1)/t)``, evaluates their subtree and publishes the FP32
subtree root; the t roots complete the same tree (``leaf_tree`` kernel). TP = 1 evaluates the whole tree
locally. Hence TP = 1, 2, 4, 8 give identical bits (tests/test_tp_gpu.py).

The index scores are replicated: every rank evaluates all index heads itself (``ops.index_scores``, recipe
``spec.SUB_RECIPES["indexer"]``: DeepGEMM's per-key BF16 head fold), with no score collective at any TP degree.
Every rank applies the identical candidate filter and top-k to its identical scores.

Ranks are combined by rank id, never by arrival: ``all_gather`` then the local tree (distributed), or the
same kernel over the shards' roots (single-GPU simulation, used by the tests).
"""
from __future__ import annotations

import torch
from torch import nn

from . import cache as dc
from . import ops
from ._extension import extension

__all__ = ["shard_layer", "slice_projection", "tree_combine", "attention_chunk_tp", "attention_paged_tp", "attention_full_tp", "attention_decode_tp", "DistributedReducer"]


def tree_combine(parts):
    """Complete the fixed leaf tree over the ranks' FP32 subtree roots (rank order); BF16 result.
    One part (TP = 1) is already the BF16 root."""
    if len(parts) == 1:
        return parts[0]
    stacked = torch.stack([p.float() for p in parts], 0).contiguous()
    out = torch.empty(parts[0].shape, dtype=torch.bfloat16, device=parts[0].device)
    extension().leaf_tree(stacked, out)
    return out


class _Box(nn.Module):
    pass


def _rows(w, a, b):
    return nn.Parameter(w.detach()[a:b].clone(), requires_grad=w.requires_grad)


def slice_projection(module, *, rows=None, columns=None):
    """Copy a projection slice with its raw 32x32 FP8 scale tiles intact.

    No decoding or quantization. The same routine works on meta holders before
    checkpoint loading and concrete tensors for the existing TP simulators.
    """
    weight = module.weight
    if weight.ndim != 2:
        raise ValueError("projection weight must have shape [N,K]")
    n, k = weight.shape
    r0, r1 = (0, n) if rows is None else rows
    c0, c1 = (0, k) if columns is None else columns
    if not (0 <= r0 < r1 <= n and 0 <= c0 < c1 <= k):
        raise ValueError("projection slice must be nonempty and within its full shape")
    scale = getattr(module, "scale", None)
    fp8 = weight.dtype == torch.float8_e4m3fn
    if fp8:
        if scale is None or scale.dtype != torch.float8_e8m0fnu or scale.shape != ((n + 31) // 32, (k + 31) // 32):
            raise ValueError("FP8 projection requires its complete UE8M0 32x32 scale grid")
        if r0 % 32 or c0 % 32 or (r1 != n and r1 % 32) or (c1 != k and c1 % 32):
            raise ValueError("FP8 projection shard boundaries must align with 32x32 scale tiles")
    elif scale is not None:
        raise ValueError("scaled non-FP8 projection is not supported by this shard recipe")
    out = _Box()
    out.weight = nn.Parameter(weight.detach()[r0:r1, c0:c1].clone(), requires_grad=weight.requires_grad)
    out.out_features, out.in_features = r1 - r0, c1 - c0
    for name in ("storage", "scale_name"):
        if hasattr(module, name):
            setattr(out, name, getattr(module, name))
    if fp8:
        out.scale = nn.Parameter(scale.detach()[r0 // 32:(r1 + 31) // 32,
                                               c0 // 32:(c1 + 31) // 32].clone(), requires_grad=False)
    return out


def shard_layer(module, cfg, rank, world):
    """Rank ``rank`` of ``world``'s shard of one attention layer (replicated parts are shared)."""
    from .spec import TP_LEAVES
    if type(world) is not int or world < 1 or TP_LEAVES % world or type(rank) is not int or not 0 <= rank < world:
        raise ValueError("TP rank/degree must own aligned subtrees of the fixed eight-leaf recipe")
    H, G = cfg.num_attention_heads, cfg.o_groups
    if H % world or G % world:
        raise ValueError("TP degree must divide attention heads and output groups")
    D, R = cfg.head_dim, cfg.o_lora_rank
    h0, h1, g0, g1 = rank * H // world, (rank + 1) * H // world, rank * G // world, (rank + 1) * G // world
    s = _Box()
    for name in ("wq_a", "q_norm", "wkv", "kv_norm"):
        setattr(s, name, getattr(module, name))
    if hasattr(module, "compressor"):
        s.compressor = module.compressor
    s.wq_b = slice_projection(module.wq_b, rows=(h0 * D, h1 * D))
    s.attn_sink = _rows(module.attn_sink, h0, h1)
    s.wo_a = slice_projection(module.wo_a, rows=(g0 * R, g1 * R))
    s.wo_b = slice_projection(module.wo_b, columns=(g0 * R, g1 * R))
    if hasattr(module, "indexer"):
        s.indexer = module.indexer         # replicated: every rank scores all index heads (ops.index_scores)
    return s


def _lockstep(gens, order):
    """Advance the shards' generators to each reduction point (in ``order``), reduce by rank id."""
    t = len(gens)
    msgs, done = [None] * t, [None] * t
    pending = {r: None for r in range(t)}
    while True:
        for r in order:
            try:
                msgs[r] = gens[r].send(pending[r]) if pending[r] is not None else next(gens[r])
            except StopIteration as stop:
                done[r] = stop.value
        if all(d is not None for d in done):
            return done
        if any(d is not None for d in done):
            raise RuntimeError("TP shards diverged: one finished while another waits at a reduction")
        kinds = {m[0] for m in msgs}
        if kinds != {"scores"}:      # no composition yields since the indexer is replicated (ops.index_scores)
            raise RuntimeError(f"unexpected reduction points {kinds}")
        n = len(msgs[0][1])
        reduced = [tree_combine([msgs[r][1][b] for r in range(t)]) for b in range(n)]
        for r in range(t):
            pending[r] = [x.clone() for x in reduced] if t > 1 else reduced    # consumers mask in place


def attention_full_tp(shards, x, cfg, index, states, *, linear_fn, dequant_fn, recipe=None, order=None):
    """Single-process simulation of TP = len(shards); returns the reduced layer output."""
    t = len(shards)
    gens = [ops.attention_full_gen(s, x, cfg, index, states[r], linear_fn=linear_fn, dequant_fn=dequant_fn,
                                   recipe=recipe, tp=t) for r, s in enumerate(shards)]
    outs = _lockstep(gens, list(range(t)) if order is None else order)
    return tree_combine(outs)


@torch.no_grad()
def attention_decode_tp(shards, x, cfg, index, caches, *, linear_fn, dequant_fn, recipe=None, order=None):
    t = len(shards)
    gens = [dc.attention_decode_gen(s, x, cfg, index, caches[r], linear_fn=linear_fn, dequant_fn=dequant_fn,
                                    recipe=recipe, tp=t) for r, s in enumerate(shards)]
    outs = _lockstep(gens, list(range(t)) if order is None else order)
    return tree_combine(outs)


@torch.no_grad()
def attention_chunk_tp(shards, x, cfg, index, caches, *, rows, linear_fn, dequant_fn, recipe=None, order=None):
    t = len(shards)
    gens = [dc.attention_chunk_gen(s, x, cfg, index, caches[r], rows=rows, linear_fn=linear_fn,
                                   dequant_fn=dequant_fn, recipe=recipe, tp=t) for r, s in enumerate(shards)]
    outs = _lockstep(gens, list(range(t)) if order is None else order)
    return tree_combine(outs)


@torch.no_grad()
def attention_paged_tp(shards, x, cfg, index, pools, batches, *, linear_fn, dequant_fn, order=None):
    """Paged step at TP = len(shards): per-rank pools and per-rank batch objects (their ``step`` dicts)."""
    from . import paged
    t = len(shards)
    gens = [paged.attention_paged_gen(s, x, cfg, index, pools[r], batches[r], linear_fn=linear_fn,
                                      dequant_fn=dequant_fn, tp=t) for r, s in enumerate(shards)]
    outs = _lockstep(gens, list(range(t)) if order is None else order)
    return tree_combine(outs)


class DistributedReducer:
    """One shard per process: ``all_gather`` of FP32 subtree roots in rank order, then the same tree
    (no NCCL arithmetic)."""

    def __init__(self, group=None):
        import torch.distributed as dist
        self.dist, self.group = dist, group
        self.world = dist.get_world_size(group)

    def gather_sum(self, x):
        """Same transport and tree as before; the rank-ordered roots land directly in one stacked
        ``[world, ...]`` FP32 buffer (``all_gather_into_tensor``), so neither the per-rank copy-out of a
        tensor-list gather nor the stacking copy runs (graph glue)."""
        if x.dtype != torch.float32 or self.world == 1:
            parts = [torch.empty_like(x.contiguous()) for _ in range(self.world)]
            self.dist.all_gather(parts, x.contiguous(), group=self.group)
            return tree_combine(parts)
        x = x.contiguous()
        stacked = torch.empty((self.world, *x.shape), dtype=x.dtype, device=x.device)
        self.dist.all_gather_into_tensor(stacked, x, group=self.group)
        out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
        extension().leaf_tree(stacked, out)
        return out

    def run(self, gen):
        try:
            msg = next(gen)
            while True:
                msg = gen.send([self.gather_sum(s) for s in msg[1]])
        except StopIteration as stop:
            return self.gather_sum(stop.value)
