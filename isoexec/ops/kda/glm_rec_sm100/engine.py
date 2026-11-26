"""Engine-side GLM KDA layer for the recurrent recipe ``sm100_kda_rec_v1``.

Same composition as ``kda/glm_sm100/engine.py`` (tensor-parallel head shards, chain projections, q/k/v
convolutions, the TP-invariant 8-leaf o_proj tree) with exactly two differences: the core is the
per-token recurrent update (``ops``), and the per-slot state is vLLM's regular KDA cache -- an fp32
``[H_local, 128 (v), 128 (k)]`` matrix plus the q/k/v convolution window ``[3 H_local 128, 3]`` -- with no
custom record. ``state_shapes`` IS ``MambaStateShapeCalculator.kda_state_shape``.

Shards, projections, the o_proj leaf tree, ``tree_combine``, ``DistributedReducer``, ``EngineBatch`` and
``batch_from_vllm`` are glm_sm100's, unchanged (they carry no recipe-specific arithmetic).
"""

from __future__ import annotations

import torch

from isoexec.ops.kda import schedule
from isoexec.ops.kda.glm_rec_sm100 import ops
from isoexec.ops.kda.glm_rec_sm100._extension import extension as ops_extension
from isoexec.ops.kda.glm_rec_sm100.spec import HEAD_DIM
from isoexec.ops.kda.glm_sm100.engine import (TP_DEGREES, TP_LEAVES, DistributedReducer, EngineBatch, KdaShard,
                                               batch_from_vllm, fuse_decode_operands, fused_input_weight, linear,
                                               refresh_shard, shard_layer, tree_combine)
from isoexec.ops.kda.glm_sm100.engine import o_proj_leaves as _o_proj_leaves_cpr
from isoexec.ops.kda.glm_sm100.integration import gated_rms_norm


def _gen():
    from isoexec.ops.mm.mm_sm100.general import ops as gen
    return gen


class KdaStatePool:
    """Slot-indexed state of one layer shard: fp32 recurrent state and q/k/v convolution windows."""

    reducer = schedule.TOKEN

    def __init__(self, num_slots: int, local_heads: int, *, dtype=torch.bfloat16, device="cuda",
                 state: torch.Tensor | None = None, conv: torch.Tensor | None = None):
        self.H = local_heads
        self.state = (torch.zeros(num_slots, local_heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device)
                      if state is None else state)
        if self.state.shape[1:] != (local_heads, HEAD_DIM, HEAD_DIM) or self.state.dtype != torch.float32:
            raise ValueError(f"recurrent state must be fp32 [slots, {local_heads}, 128, 128] (v-first)")
        D = 3 * local_heads * HEAD_DIM
        self.conv = torch.zeros(num_slots, D, 3, dtype=dtype, device=device) if conv is None else conv
        if self.conv.shape[1:] != (D, 3):
            raise ValueError(f"conv state must be [slots, {D}, 3] (dim-first)")

    @staticmethod
    def state_shapes(world: int, num_heads: int, conv_kernel_size: int = 4):
        """``MambaStateShapeCalculator.kda_state_shape(world, num_heads, 128)``: (conv, recurrent), the conv window in the
        dim-first orientation (vLLM flips it to state-len-first when its layout switch says so)."""
        Hl = num_heads // world
        return (3 * Hl * HEAD_DIM, conv_kernel_size - 1), (Hl, HEAD_DIM, HEAD_DIM)

    @staticmethod
    def state_dtypes(model_dtype=torch.bfloat16):
        return model_dtype, torch.float32

    def state_bytes_view(self) -> torch.Tensor:
        """The state as ``[slots, H, 65536]`` bytes (what the shared slot-reset kernel zeroes)."""
        return self.state.view(torch.uint8).view(self.state.shape[0], self.state.shape[1], -1)


def o_proj_leaves(y: torch.Tensor, w_o: torch.Tensor, leaves: int, root_fp32: bool) -> torch.Tensor:
    """glm_sm100's o_proj leaf program (one FP32 chain per leaf, one FP32 adjacent-pair tree), the tree evaluated by the
    vectorized kernel of this package (same adds, same bits). Training (autograd) keeps glm_sm100's."""
    if torch.is_grad_enabled() and (y.requires_grad or w_o.requires_grad):
        return _o_proj_leaves_cpr(y, w_o, leaves, root_fp32)
    T, KL = y.shape
    N = w_o.shape[0]
    Kl = KL // leaves
    a3 = y.contiguous().view(T, leaves, Kl).permute(1, 0, 2)
    b3 = w_o.view(N, leaves, Kl).permute(1, 2, 0)
    parts = _gen().bmm(a3, b3, out_dtype=torch.float32).contiguous()
    out = torch.empty(T, N, dtype=torch.float32 if root_fp32 else torch.bfloat16, device=y.device)
    if (T * N) % 4 == 0 and leaves in (1, 2, 4, 8):
        ops_extension().leaf_tree(parts, out)
    else:
        return _o_proj_leaves_cpr(y, w_o, leaves, root_fp32)
    return out


def _decode_fused(shard: KdaShard, x, batch, pool):
    from isoexec.ops.gdn.conv_sm100 import ops as conv_ops
    T, Hl, K = x.shape[0], shard.H, HEAD_DIM
    HK = Hl * K
    y = linear(x, shard.w_in)                                              # [T, 3HK + 256 + H]
    mixed = conv_ops.update(y[:, :3 * HK], pool.conv, shard.conv_w, activation="silu",
                            cache_indices=batch.state_indices, null_block_id=batch.null_block_id)
    q, k, v = (mixed[:, i * HK:(i + 1) * HK].view(T, Hl, K) for i in range(3))
    low = y[:, 3 * HK:3 * HK + 2 * K].view(T, 2, K).transpose(0, 1)        # [2, T, 128]: (fa x, ga x)
    fg = _gen().bmm(low, shard.w_fg.transpose(1, 2))                       # [2, T, HK]: (f, gate)
    core = ops.decode_step_fused(q, k, v, fg[0], y[:, 3 * HK + 2 * K:], shard.dt_bias, shard.A_log,
                                 shard.lower_bound, pool.state, batch.state_indices, null_slot=batch.null_block_id)
    normalized = gated_rms_norm(core, fg[1].view(T, Hl, K), shard.norm_w, shard.eps)
    return o_proj_leaves(normalized.view(T, HK), shard.w_o, shard.leaves, root_fp32=shard.world > 1)


def forward(shard: KdaShard, x: torch.Tensor, batch: EngineBatch | None = None, pool: KdaStatePool | None = None,
            *, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
    """This shard's part of the layer on packed tokens ``x [T, hidden]``: the bf16 output at TP = 1, else the
    FP32 o_proj subtree root (complete with ``tree_combine`` over the ranks, in rank order). With ``pool`` /
    ``batch``: the engine path; without: functional whole sequences from position 0 (the trainer forward)."""
    from isoexec.ops.gdn.conv_sm100 import ops as conv_ops

    needs_grad = torch.is_grad_enabled() and (x.requires_grad or any(
        isinstance(t, torch.Tensor) and t.requires_grad for t in vars(shard).values()))
    schedule.check_pool(schedule.TOKEN, pool)
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
        mixed = conv_ops.varlen(qkv, shard.conv_w, cu_seqlens=cu, activation="silu", conv_state=pool.conv,
                                cache_indices=batch.state_indices, has_initial_state=batch.has_initial_state,
                                null_block_id=batch.null_block_id)
    q, k, v = (mixed[:, i * HK:(i + 1) * HK].reshape(T, Hl, K) for i in range(3))
    f = linear(linear(x, shard.w_fa), shard.w_fb)
    braw = linear(x, shard.w_b)
    if needs_grad:   # autograd gate / beta ops (glm_sm100's, the same bits as ops.forget_gate / sigmoid_beta)
        from isoexec.ops.kda.glm_sm100.integration import forget_gate, sigmoid_beta
        log_decay = forget_gate(f, shard.dt_bias, shard.A_log, shard.lower_bound).view(T, Hl, K)
        beta = sigmoid_beta(braw)
    else:
        log_decay = ops.forget_gate(f, shard.dt_bias, shard.A_log, shard.lower_bound).view(T, Hl, K)
        beta = ops.sigmoid_beta(braw)
    if pool is None and needs_grad:
        core, _ = ops.core(q, k, v, log_decay, beta, cu, None)
    elif pool is None:
        core, _ = ops.forward(q, k, v, log_decay, beta, cu, None, output_final_state=False)
    elif batch.decode:
        core = ops.decode_step(q, k, v, log_decay, beta, pool.state, batch.state_indices, null_slot=batch.null_block_id)
    else:
        idx = batch.state_indices.to(torch.int32)
        slot_in = torch.where(batch.has_initial_state.to(torch.bool), idx, torch.full_like(idx, -1))
        core = ops.prefill(q, k, v, log_decay, beta, cu, pool.state, slot_in, idx, null_slot=batch.null_block_id)
    gate = linear(linear(x, shard.w_ga), shard.w_gb).view(T, Hl, K)
    normalized = gated_rms_norm(core, gate, shard.norm_w, shard.eps)
    return o_proj_leaves(normalized.view(T, HK), shard.w_o, shard.leaves, root_fp32=shard.world > 1)


def forward_tp(shards, x, batch=None, pools=None, *, cu_seqlens=None) -> torch.Tensor:
    """Single-process simulation of TP = len(shards) (rank order): the bf16 layer output."""
    roots = [forward(s, x, batch, None if pools is None else pools[r], cu_seqlens=cu_seqlens) for r, s in enumerate(shards)]
    return tree_combine(roots)


__all__ = ["TP_LEAVES", "TP_DEGREES", "KdaShard", "KdaStatePool", "EngineBatch", "DistributedReducer", "batch_from_vllm",
           "forward", "forward_tp", "fuse_decode_operands", "fused_input_weight", "linear", "o_proj_leaves", "shard_layer",
           "refresh_shard", "tree_combine"]
