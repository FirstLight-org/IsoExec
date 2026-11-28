"""Engine-side GLM KDA layer: tensor-parallel head shards, a slot-indexed state pool, and one call for
mixed prefill / decode batches (vLLM / SGLang mamba-style state management).
Functional whole-sequence execution also supports native backward; mutable cache execution is inference-only.

**Tensor parallelism (TP = 1, 2, 4, 8; README "Tensor parallelism").** Heads are independent in every
stage, so rank ``r`` of ``t`` owns heads ``[r H/t, (r+1) H/t)``:

* column-parallel (rows of the weight): q / k / v projections and their depthwise convolutions, f_b, g_b,
  b, ``dt_bias``, ``A_log``; replicated: f_a, g_a (low rank), the output norm weight;
* every projection runs ``mm_sm100/general``'s ``sm100_tcgen05_chain_v1`` program (one ascending 16-wide
  chain per output element from a zeroed FP32 accumulator, one rounding): an output element's bits read
  only its own input row and weight row, never M, N or the tile, so a shard's columns are the TP = 1
  columns bitwise;
* o_proj is row-parallel: its K = H * 128 is cut into ``TP_LEAVES = 8`` fixed leaves of H / 8 heads; each
  leaf is one FP32 chain (``bmm`` with an FP32 root), and the leaves meet in ONE balanced adjacent-pair
  FP32 tree rounded once to bf16. Rank r holds the aligned leaves ``[8r/t, 8(r+1)/t)`` and publishes its
  FP32 subtree root; the t roots complete the same tree in rank order (``tree_combine``; distributed:
  ``all_gather`` then the same kernel -- NCCL never adds). TP = 1 evaluates the whole tree. Hence the
  layer output is TP-degree invariant (models/glm5_3_flash/tests/test_kda_engine_tp_gpu.py).

**State pool.** Per slot: the KDA engine records (``ops.KdaState``, ``[slots, H_local, rec_bytes]``
uint8; the slot stride is free, so an engine page may hold them) and the q/k/v convolution windows in
vLLM's dim-first layout ``[slots, 3 H_local 128, 3]`` (the model dtype). A batch names each sequence's
slot (``state_indices``) and whether it continues it (``has_initial_state``; False: a fresh sequence --
slot reuse needs no clearing, preemption is recompute from position 0, bitwise the same).

**Batches.** ``forward`` takes packed tokens ``[T, hidden]`` and ``cu_seqlens``. A batch with any
multi-token sequence runs the chunked path for all sequences (decode lanes included -- one-token prefill
and decode are bitwise the same row, README "Relations"); a pure one-token-per-sequence batch
(``decode=True``) runs the decode kernels: no host synchronization, fixed launch shapes, padded lanes
named by ``null_block_id``, so it is CUDA-graph capturable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isoexec.ops.kda import schedule
from isoexec.ops.kda.glm_sm100 import ops, record_layout
from isoexec.ops.kda.glm_sm100._extension import extension
from isoexec.ops.kda.glm_sm100.integration import forget_gate, gated_rms_norm, sigmoid_beta
from isoexec.ops.kda.glm_sm100.spec import HEAD_DIM
from isoexec.ops.mm.mm_sm100.segments import row_parallel_leaves

TP_DEGREES = (1, 2, 4, 8)
#: Leaves of the row-parallel o_proj reduction: the GEMM's statement of it (``mm_sm100`` ``row_parallel_leaves``, rule 3
#: of the schedule policy: d_max shards of G segments) at the largest TP degree. The tensor-core sum publishes no
#: shortest segment, so G = 1 at every K; it is evaluated at the smallest o_proj K the engine takes (8 heads) and
#: ``shard_layer`` re-derives it at the layer's own K.
TP_LEAVES = row_parallel_leaves(8 * HEAD_DIM, max(TP_DEGREES))


# ------------------------------------------------------------------------------------------ projections
def _gen():
    from isoexec.ops.mm.mm_sm100.general import ops as gen
    return gen


def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``x [M, K] @ w[N, K]^T`` in bf16 by ``sm100_tcgen05_chain_v1`` (M- and N-invariant bits)."""
    if torch.is_grad_enabled() and (x.requires_grad or w.requires_grad):
        from .engine_training import ChainLinear
        return ChainLinear.apply(x, w)
    return _gen().mm(x.contiguous(), w.t())


def o_proj_leaves(y: torch.Tensor, w_o: torch.Tensor, leaves: int, root_fp32: bool) -> torch.Tensor:
    """Row-parallel o_proj of this rank's ``leaves`` aligned leaves: ``y [T, leaves * Kl]`` (this rank's
    heads), ``w_o [hidden, leaves * Kl]``. Returns the subtree root (FP32) or, holding every leaf, the
    bf16 output."""
    if torch.is_grad_enabled() and (y.requires_grad or w_o.requires_grad):
        from .engine_training import LeafLinear
        return LeafLinear.apply(y, w_o, leaves, root_fp32)
    T, KL = y.shape
    N = w_o.shape[0]
    Kl = KL // leaves
    a3 = y.contiguous().view(T, leaves, Kl).permute(1, 0, 2)          # [leaves, T, Kl], K-major
    b3 = w_o.view(N, leaves, Kl).permute(1, 2, 0)                    # [leaves, Kl, N], K-major
    parts = _gen().bmm(a3, b3, out_dtype=torch.float32)               # one FP32 chain per leaf
    out = torch.empty(T, N, dtype=torch.float32 if root_fp32 else torch.bfloat16, device=y.device)
    extension().leaf_tree(parts.contiguous(), out)
    return out


def tree_combine(roots) -> torch.Tensor:
    """Complete the leaf tree over the ranks' FP32 subtree roots (rank order): the bf16 layer output. A
    single bf16 part (TP = 1) is already the output."""
    if len(roots) == 1 and roots[0].dtype == torch.bfloat16:
        return roots[0]
    stacked = torch.stack([r.float() for r in roots], 0).contiguous()
    out = torch.empty(roots[0].shape, dtype=torch.bfloat16, device=roots[0].device)
    extension().leaf_tree(stacked, out)
    return out


# ------------------------------------------------------------------------------------------------ shards
@dataclass
class KdaShard:
    """Rank ``rank`` of ``world``'s weights of one GLM KDA layer (bf16 GEMM operands, fp32 gates)."""
    rank: int
    world: int
    H: int                      # local heads
    hidden: int
    w_qkv: torch.Tensor         # [3 H 128, hidden]  (q rows, k rows, v rows)
    conv_w: torch.Tensor        # [3 H 128, 4] fp32
    w_fa: torch.Tensor          # [128, hidden]      replicated
    w_fb: torch.Tensor          # [H 128, 128]
    w_b: torch.Tensor           # [H, hidden]
    w_ga: torch.Tensor          # [128, hidden]      replicated
    w_gb: torch.Tensor          # [H 128, 128]
    w_o: torch.Tensor           # [hidden, H 128]    (columns of o_proj)
    dt_bias: torch.Tensor       # [H 128] fp32
    A_log: torch.Tensor         # [H] fp32
    norm_w: torch.Tensor        # [128]              replicated
    eps: float
    lower_bound: float
    # Optional fused decode operands (``fuse_decode_operands``): ``w_in`` [3 H 128 + 256 + H + pad, hidden] = the rows
    # of w_qkv, w_fa, w_ga, w_b in one storage (those four fields are then views of it), then ``pad`` zero rows up to
    # a multiple of ``FUSED_ROW_ALIGN``; ``w_fg`` [2, H 128, 128] = (w_fb, w_gb). ``sm100_tcgen05_chain_v1`` bits do
    # not depend on N or the batch, so the fused GEMMs give the separate GEMMs' bits.
    w_in: torch.Tensor | None = None
    w_fg: torch.Tensor | None = None

    @property
    def leaves(self) -> int:
        return TP_LEAVES // self.world


def _bf16(t, *, trainable=False):
    return (t if trainable else t.detach()).to(torch.bfloat16).contiguous()


def shard_layer(module, rank: int = 0, world: int = 1, *, trainable=False) -> KdaShard:
    """Shard an ``isoexec.nn.GLMKimiDeltaAttention`` (bf16 weights) for rank ``rank`` of ``world``."""
    cfg = module.cfg
    H, K = cfg.linear_num_heads, cfg.linear_head_dim
    if K != HEAD_DIM:
        raise ValueError("glm_sm100 engine requires linear_head_dim == 128")
    if world not in TP_DEGREES or H % (TP_LEAVES * 1) or (H // TP_LEAVES * K) % 64:
        raise ValueError(f"TP degree must be one of {TP_DEGREES} and the heads must form {TP_LEAVES} leaves")
    if row_parallel_leaves(H * K, max(TP_DEGREES)) != TP_LEAVES:
        raise ValueError(f"o_proj K = {H * K}: the GEMM's row-parallel tree has "
                         f"{row_parallel_leaves(H * K, max(TP_DEGREES))} leaves, the engine's has {TP_LEAVES}")
    if not 0 <= rank < world:
        raise ValueError("rank out of range")
    for name in ("q_proj", "k_proj", "v_proj", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj", "b_proj", "o_proj"):
        if getattr(module, name).weight.dtype == torch.float8_e4m3fn:
            raise ValueError("FP8 checkpoint projections: the TP-invariant GEMM for them is the mm seam's "
                             "(README 'Tensor parallelism'); the engine layer takes bf16 weights")
    Hl = H // world
    h0, h1 = rank * Hl, (rank + 1) * Hl
    c0, c1 = h0 * K, h1 * K
    # Training rebuilds these views/cats on each forward so optimizer updates and autograd
    # reach the live module parameters. Serving shards keep the existing detached layout.
    bf16 = lambda t: _bf16(t, trainable=trainable)
    source = lambda t: t if trainable else t.detach()
    rows = lambda w: bf16(w[c0:c1])
    w_qkv = torch.cat([rows(getattr(module, n + "_proj").weight) for n in ("q", "k", "v")], 0).contiguous()
    conv_w = torch.cat([source(getattr(module, n + "_conv1d").weight).float().reshape(H * K, -1)[c0:c1]
                        for n in ("q", "k", "v")], 0).contiguous()
    return KdaShard(
        rank=rank, world=world, H=Hl, hidden=cfg.hidden, w_qkv=w_qkv, conv_w=conv_w,
        w_fa=bf16(module.f_a_proj.weight), w_fb=rows(module.f_b_proj.weight), w_b=bf16(module.b_proj.weight[h0:h1]),
        w_ga=bf16(module.g_a_proj.weight), w_gb=rows(module.g_b_proj.weight),
        w_o=bf16(module.o_proj.weight[:, c0:c1]), dt_bias=source(module.dt_bias).float()[c0:c1].contiguous(),
        A_log=source(module.A_log).float()[h0:h1].contiguous(), norm_w=source(module.o_norm.weight).contiguous(),
        eps=float(cfg.rms_norm_eps), lower_bound=float(cfg.linear_gate_lower_bound))


@torch.no_grad()
def refresh_shard(module, shard):
    """Refresh copied projections after weight sync without moving graph-bound storage."""
    fresh = shard_layer(module, shard.rank, shard.world)
    if shard.w_in is not None:
        # Match the serving layout before copying. The old fields alias its fused buffers;
        # replacing them would leave captured decode graphs reading stale weights.
        fuse_decode_operands(fresh)
    for name, value in vars(fresh).items():
        old = getattr(shard, name)
        if isinstance(value, torch.Tensor):
            old.copy_(value)
        elif old != value:
            raise ValueError(f"engine shard structural field {name} changed during weight sync")
    return shard


# -------------------------------------------------------------------------------------------- state pool
class KdaStatePool:
    """Slot-indexed state of one layer shard: engine records and q/k/v convolution windows."""

    reducer = schedule.CHUNK

    def __init__(self, num_slots: int, local_heads: int, *, dtype=torch.bfloat16, device="cuda",
                 records: torch.Tensor | None = None, conv: torch.Tensor | None = None):
        self.H = local_heads
        self.rec = ops.KdaState(1, local_heads, device) if records is not None else ops.KdaState(num_slots, local_heads, device)
        if records is not None:   # an engine-allocated tensor (any 16-byte aligned slot stride)
            self.rec.data = records
        D = 3 * local_heads * HEAD_DIM
        self.conv = torch.zeros(num_slots, D, 3, dtype=dtype, device=device) if conv is None else conv
        if self.conv.shape[1:] != (D, 3):
            raise ValueError(f"conv state must be [slots, {D}, 3] (dim-first)")

    @staticmethod
    def state_shapes(world: int, num_heads: int, conv_kernel_size: int = 4):
        """Per-slot shapes, vLLM ``MambaStateShapeCalculator`` style: (conv dim-first, records)."""
        Hl = num_heads // world
        return (3 * Hl * HEAD_DIM, conv_kernel_size - 1), (Hl, record_layout.REC_BYTES)   # no build: the layout is declared in record_layout.py (CPU-checked against csrc/record.cuh)

    @staticmethod
    def state_dtypes(model_dtype=torch.bfloat16):
        return model_dtype, torch.uint8


#: The fused input weight's row count is padded with zero rows to a multiple of this, so the decode GEMM output's
#: row pitch is 16 bytes: ``mm_sm100/general`` reads its (fa x, ga x) columns in place and refuses a pitch that is
#: not (a local head count that is not a multiple of 8, e.g. 32 heads at TP 8). Nothing reads a padding column, and
#: no other column's bits depend on N.
FUSED_ROW_ALIGN = 8


def _fused_row_blocks(H: int) -> list[int]:
    """Row blocks of the fused input weight: q k v, f_a, g_a, b, zero padding."""
    n = [3 * H * HEAD_DIM, HEAD_DIM, HEAD_DIM, H]
    return n + [-sum(n) % FUSED_ROW_ALIGN]


def fused_input_weight(w_qkv, w_fa, w_ga, w_b):
    """The fused decode input projection rows ``[w_qkv; w_fa; w_ga; w_b; 0]`` (one new storage; the zero rows pad the
    row count to a multiple of ``FUSED_ROW_ALIGN``, none when the local head count is a multiple of 8)."""
    pad = _fused_row_blocks(w_b.shape[0])[-1]
    return torch.cat([w_qkv, w_fa, w_ga, w_b, w_b.new_zeros(pad, w_b.shape[1])], 0).contiguous()


def fuse_decode_operands(shard: KdaShard, w_in: torch.Tensor | None = None) -> KdaShard:
    """Give ``shard`` the fused decode operands. ``w_in``: an already-built :func:`fused_input_weight` whose row
    blocks the shard's w_qkv / w_fa / w_ga / w_b must equal (the callers re-point those fields as views, so the
    weights are stored once); None builds it and re-points the fields here. Inference only."""
    if w_in is None:
        w_in = fused_input_weight(shard.w_qkv, shard.w_fa, shard.w_ga, shard.w_b)
    n = _fused_row_blocks(shard.H)
    if w_in.shape != (sum(n), shard.hidden) or w_in.dtype != torch.bfloat16 or not w_in.is_contiguous():
        raise ValueError(f"fused KDA input weight must be bf16 [3 H 128 + 256 + H rounded up to a multiple of "
                         f"{FUSED_ROW_ALIGN}, hidden] = {[sum(n), shard.hidden]} (fused_input_weight)")
    shard.w_qkv, shard.w_fa, shard.w_ga, shard.w_b, _ = torch.split(w_in, n, 0)
    shard.w_in = w_in
    shard.w_fg = torch.stack([shard.w_fb, shard.w_gb], 0).contiguous()
    return shard


def _decode_fused(shard: KdaShard, x, batch, pool):
    """Decode lanes with the fused operands: one input GEMM, one batched second-stage GEMM (fb, gb), the
    convolution reading the GEMM's qkv columns in place, and the gate/beta stages inside the row kernel.
    Bit for bit the unfused decode path (same chains, same element-wise programs)."""
    from isoexec.ops.gdn.conv_sm100 import ops as conv_ops
    T, Hl, K = x.shape[0], shard.H, HEAD_DIM
    HK = Hl * K
    y = linear(x, shard.w_in)                                              # [T, 3HK + 256 + H + pad]
    mixed = conv_ops.update(y[:, :3 * HK], pool.conv, shard.conv_w, activation="silu",
                            cache_indices=batch.state_indices, null_block_id=batch.null_block_id)
    q, k, v = (mixed[:, i * HK:(i + 1) * HK].view(T, Hl, K) for i in range(3))
    low = y[:, 3 * HK:3 * HK + 2 * K].view(T, 2, K).transpose(0, 1)        # [2, T, 128]: (fa x, ga x)
    fg = _gen().bmm(low, shard.w_fg.transpose(1, 2))                       # [2, T, HK]: (f, gate)
    core = ops.decode_step_fused(q, k, v, fg[0], y[:, 3 * HK + 2 * K:3 * HK + 2 * K + Hl], shard.dt_bias, shard.A_log,
                                 shard.lower_bound, pool.rec, batch.state_indices, null_slot=batch.null_block_id)
    normalized = gated_rms_norm(core, fg[1].view(T, Hl, K), shard.norm_w, shard.eps)
    return o_proj_leaves(normalized.view(T, HK), shard.w_o, shard.leaves, root_fp32=shard.world > 1)


def batch_from_vllm(meta, null_block_id: int = 0) -> "EngineBatch":
    """An ``EngineBatch`` from vLLM's ``GDNAttentionMetadata`` (no vLLM import): the non-speculative
    sequences, decodes first, ``non_spec_state_indices_tensor`` as slots (CUDA-graph padding lanes carry
    ``NULL_BLOCK_ID``), ``has_initial_state`` as the continuation flags. A batch with no prefill is a
    decode batch. vLLM's conv state, when stored state-len-first, is passed as its ``transpose(-1, -2)``
    view (any strides)."""
    if getattr(meta, "spec_sequence_masks", None) is not None:
        raise NotImplementedError("speculative multi-token decode is not served by the glm_sm100 engine path")
    decode = meta.num_prefills == 0
    return EngineBatch(meta.non_spec_query_start_loc, meta.non_spec_state_indices_tensor,
                       None if decode else meta.has_initial_state, decode, null_block_id)


# ---------------------------------------------------------------------------------------- the layer call
@dataclass
class EngineBatch:
    """One call's sequences: packed tokens split by ``cu_seqlens`` (int32, device), each sequence's slot
    (``state_indices``) and whether it continues its slot's state (``has_initial_state``; ignored for a
    decode batch, whose live lanes always continue). ``decode``: every sequence has one token (the
    capturable path). Lanes whose slot is negative or ``null_block_id`` are padding (decode only)."""
    cu_seqlens: torch.Tensor
    state_indices: torch.Tensor
    has_initial_state: torch.Tensor | None = None
    decode: bool = False
    null_block_id: int = -1


def forward(shard: KdaShard, x: torch.Tensor, batch: EngineBatch | None = None, pool: KdaStatePool | None = None,
            *, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
    """This shard's part of the layer on packed tokens ``x [T, hidden]``: the bf16 output at TP = 1, else
    the FP32 o_proj subtree root (complete with ``tree_combine`` over the ranks, in rank order).

    With ``pool`` / ``batch``: the engine path (records and conv windows read and advanced). Without:
    functional whole sequences from position 0 (``cu_seqlens``)."""
    from isoexec.ops.gdn.conv_sm100 import ops as conv_ops

    needs_grad = torch.is_grad_enabled() and (x.requires_grad or any(
        isinstance(t, torch.Tensor) and t.requires_grad for t in vars(shard).values()))
    schedule.check_pool(schedule.CHUNK, pool)
    if needs_grad and pool is not None:
        raise ValueError("engine KDA backward requires functional whole sequences, not mutable cache state")
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != shard.hidden:
        raise ValueError(f"engine KDA expects bf16 [T, {shard.hidden}] tokens")
    T, Hl, K = x.shape[0], shard.H, HEAD_DIM
    HK = Hl * K
    x = x.contiguous()
    if pool is not None and batch.decode and shard.w_in is not None and not needs_grad:
        return _decode_fused(shard, x, batch, pool)
    qkv = linear(x, shard.w_qkv)                                           # [T, 3 H 128]
    if pool is None:
        cu = cu_seqlens.to(device=x.device, dtype=torch.int32)
        mixed = conv_ops.conv(qkv, shard.conv_w, cu_seqlens=cu, activation="silu")
    elif batch.decode:
        cu = None
        mixed = conv_ops.update(qkv, pool.conv, shard.conv_w, activation="silu", cache_indices=batch.state_indices,
                                null_block_id=batch.null_block_id)
    else:
        cu = batch.cu_seqlens
        hinit = batch.has_initial_state
        mixed = conv_ops.varlen(qkv, shard.conv_w, cu_seqlens=cu, activation="silu", conv_state=pool.conv,
                                cache_indices=batch.state_indices, has_initial_state=hinit,
                                null_block_id=batch.null_block_id)
    q, k, v = (mixed[:, i * HK:(i + 1) * HK].reshape(T, Hl, K) for i in range(3))
    log_decay = forget_gate(linear(linear(x, shard.w_fa), shard.w_fb), shard.dt_bias, shard.A_log,
                            shard.lower_bound).view(T, Hl, K)
    beta = sigmoid_beta(linear(x, shard.w_b))
    if pool is None:
        if needs_grad:
            from .integration import kda_core
            core = kda_core(q, k, v, log_decay, beta, cu_seqlens=cu)
        else:
            core, _ = ops.forward(q, k, v, log_decay, beta, cu, None)
    elif batch.decode:
        core = ops.decode_step(q, k, v, log_decay, beta, pool.rec, batch.state_indices,
                               null_slot=batch.null_block_id)
    else:
        idx = batch.state_indices.to(torch.int32)
        slot_in = torch.where(batch.has_initial_state.to(torch.bool), idx, torch.full_like(idx, -1))
        core = ops.prefill(q, k, v, log_decay, beta, cu, pool.rec, slot_in, idx)
    gate = linear(linear(x, shard.w_ga), shard.w_gb).view(T, Hl, K)
    normalized = gated_rms_norm(core, gate, shard.norm_w, shard.eps)
    return o_proj_leaves(normalized.view(T, HK), shard.w_o, shard.leaves, root_fp32=shard.world > 1)


# ------------------------------------------------------------------------------------- tensor parallel
def forward_tp(shards, x, batch=None, pools=None, *, cu_seqlens=None) -> torch.Tensor:
    """Single-process simulation of TP = len(shards) (rank order): the bf16 layer output."""
    roots = [forward(s, x, batch, None if pools is None else pools[r], cu_seqlens=cu_seqlens)
             for r, s in enumerate(shards)]
    return tree_combine(roots)


class DistributedReducer:
    """One shard per process: ``all_gather`` of the FP32 subtree roots in rank order, then the same tree
    (no NCCL arithmetic)."""

    def __init__(self, group=None):
        import torch.distributed as dist
        self.dist, self.group = dist, group
        self.world = dist.get_world_size(group)

    def __call__(self, root: torch.Tensor) -> torch.Tensor:
        if self.world == 1:
            return root
        root = root.contiguous()
        parts = [torch.empty_like(root) for _ in range(self.world)]
        self.dist.all_gather(parts, root, group=self.group)
        return tree_combine(parts)


__all__ = ["TP_LEAVES", "TP_DEGREES", "KdaShard", "KdaStatePool", "EngineBatch", "DistributedReducer", "batch_from_vllm", "forward",
           "forward_tp", "fuse_decode_operands", "fused_input_weight", "linear", "o_proj_leaves", "shard_layer",
           "tree_combine"]
