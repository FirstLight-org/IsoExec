"""SM100 GLM Kimi Delta Attention, recurrent recipe ``sm100_kda_rec_v1`` (csrc/recipe.cuh).

One kernel for everything: a sequence of T tokens is the per-token update applied T times. The state is
an FP32 ``[slots, H, 128 (v), 128 (k)]`` tensor -- vLLM's native KDA state layout -- and the functional form
uses the same layout with one "slot" per sequence.

    o, ht = ops.forward(q, k, v, g, beta, cu_seqlens, h0, output_final_state=True)   # packed [T, H, 128]
    o = ops.prefill(q, k, v, g, beta, cu, state, slot_in, slot_out)                  # engine: continue slots
    o = ops.decode_step(q, k, v, g, beta, state, slots, null_slot=0)                 # one token per sequence
    o = ops.decode_step_fused(q, k, v, f, b, dt_bias, A_log, lb, state, slots, ...)  # gate + beta in-kernel

Glue here is allocation, views and dtype/shape checks. Launch shape (value columns per CTA / thread) is
execution only: columns never meet, so it never enters the bits.
"""

from __future__ import annotations

import torch

from isoexec.ops.kda import _checks
from isoexec.ops.kda.glm_rec_sm100._extension import extension
from isoexec.ops.kda.glm_rec_sm100.spec import HEAD_DIM

#: Packed fp32x2 instructions (bit-identical to the scalar reference path; measured no faster, so off).
PACKED = False

_BETA_MODE = {torch.bfloat16: 0, torch.float32: 1}

#: CUDA's limit of gridDim.z, the axis the recurrent kernel puts sequences (decode lanes) on: a call with more is
#: refused before the launch (and before any allocation) instead of launching nothing.
MAX_SEQUENCES = 65535


def _check_count(n: int, what: str = "sequences"):
    if n > MAX_SEQUENCES:
        raise ValueError(f"kda glm_rec_sm100: {n} {what} in one call, more than the {MAX_SEQUENCES} the kernel's grid "
                         f"z axis holds; split the call")


def _shape(N: int, H: int, decode: bool):
    """(lanes per column, columns per lane group NC, columns per CTA VB): tuned for the work available. Execution
    only: columns never meet and the 8/16/32-lane mappings are bit-identical (recipe.cuh)."""
    if decode:
        # N counts the padded graph batch (null lanes still launch CTAs): the in-model GLM decode is N = 128, 58 of them null
        return (8, 4, 32) if N * H >= 2048 else (8, 4, 16) if N * H >= 128 else (8, 2, 8)
    return (8, 4, 64) if N * H >= 64 else (8, 2, 32)


def _check_qkv(q, k, v):
    if q.ndim != 3 or q.shape[-1] != HEAD_DIM or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"kda glm_rec_sm100 expects q, k, v [T, H, {HEAD_DIM}]")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("kda glm_rec_sm100 requires bf16 q, k, v")
    for t in (q, k, v):
        if t.stride(2) != 1 or t.stride(1) != HEAD_DIM:
            raise ValueError("q, k, v need 128-contiguous heads (one common token stride per tensor)")
    if not q.is_cuda or k.device != q.device or v.device != q.device:
        raise ValueError("kda glm_rec_sm100 requires CUDA tensors on one device")


def _check_device(q, *ts):
    if any(t is not None and t.device != q.device for t in ts):
        raise ValueError("kda glm_rec_sm100 requires all inputs on one CUDA device")


def _check_state(state, H: int, device):
    if (not isinstance(state, torch.Tensor) or state.dtype != torch.float32 or state.dim() != 4
            or state.shape[1:] != (H, HEAD_DIM, HEAD_DIM) or state.device != device):
        raise ValueError(f"kda glm_rec_sm100: state must be fp32 [slots, {H}, 128, 128] on {device}")


def _check_out(out, N: int, H: int, device):
    if out is None:
        return torch.empty(N, H, HEAD_DIM, dtype=torch.bfloat16, device=device)
    if (not isinstance(out, torch.Tensor) or out.shape != (N, H, HEAD_DIM) or out.dtype != torch.bfloat16
            or out.device != device or out.stride(2) != 1 or out.stride(1) != HEAD_DIM):
        raise ValueError(f"kda glm_rec_sm100: out must be bf16 [{N}, {H}, {HEAD_DIM}] on {device} with "
                         f"128-contiguous heads")
    return out


def _slots(name, x, n, device, state, null_slot):
    return _checks.slots("kda glm_rec_sm100", name, x, n, device, state.shape[0], null_slot)


def _call(q, k, v, gate, beta, cu, slot_in, slot_out, null_slot, state, o, N, *, fused=False, dt_bias=None,
          A_log=None, lb=0.0, beta_mode=None, decode=False, shape=None, packed=True, ckpt=None, cp_base=None, ws=None):
    lanes, nc, vb = shape or _shape(N, q.shape[1], decode)
    mode = _BETA_MODE[beta.dtype] if beta_mode is None else beta_mode
    extension().rec(q, k, v, gate, beta, dt_bias, A_log, float(lb), bool(fused), int(mode), cu, slot_in, slot_out,
                    int(null_slot), state, o, int(N), lanes, nc, vb, bool(packed and PACKED), bool(not decode if ws is None else ws), ckpt, cp_base)
    return o


def forward(q, k, v, g, beta, cu_seqlens, initial_state=None, output_final_state=True, *, shape=None, ws=None):
    """Functional whole-sequence form (the trainer's forward). ``initial_state`` / the returned state are fp32
    ``[N, H, 128 (v), 128 (k)]``; no initial state = +0. Returns ``(o [T, H, 128] bf16, final | None)``."""
    _check_count(cu_seqlens.numel() - 1)
    _check_qkv(q, k, v)
    T, H = q.shape[:2]
    if g.shape != q.shape or g.dtype != torch.float32:
        raise ValueError("kda glm_rec_sm100 forward expects fp32 log decay [T, H, 128]")
    if beta.shape != (T, H) or beta.dtype not in _BETA_MODE:
        raise ValueError("kda glm_rec_sm100 expects beta [T, H] bf16 or fp32 (post-sigmoid)")
    _check_device(q, g, beta, initial_state)
    cu = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    N = cu.numel() - 1
    o = torch.empty(T, H, HEAD_DIM, dtype=torch.bfloat16, device=q.device)
    if initial_state is not None:
        if initial_state.shape != (N, H, HEAD_DIM, HEAD_DIM) or initial_state.dtype != torch.float32:
            raise ValueError(f"initial_state must be fp32 [{N}, {H}, 128, 128] (v-first)")
        state = initial_state.clone()
        slot_in = torch.arange(N, dtype=torch.int32, device=q.device)
    else:
        state = torch.empty(N, H, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=q.device)
        slot_in = torch.full((N,), -1, dtype=torch.int32, device=q.device)
    slot_out = torch.arange(N, dtype=torch.int32, device=q.device)
    if N:
        _call(q, k, v, g, beta, cu, slot_in, slot_out, -1, state, o, N, shape=shape, ws=ws)
    return o, (state if output_final_state else None)


def checkpoint_layout(cu_seqlens, T: int):
    """(``cp_base`` int32 [N], upper bound of the number of 64-token checkpoints) for a packed batch: sequence n's
    checkpoint b (the state entering its token 64 b) is slot ``cp_base[n] + b``."""
    cu = cu_seqlens.to(torch.int32)
    lens = cu[1:] - cu[:-1]
    nblk = (lens + 63) // 64
    base = (torch.cumsum(nblk, 0) - nblk).to(torch.int32)
    return base.contiguous(), T // 64 + lens.numel()


def forward_train(q, k, v, g, beta, cu_seqlens, initial_state=None, *, shape=None):
    """:func:`forward` that also keeps the state entering every 64th token of each sequence (exec-only extra stores,
    the trainer backward's checkpoints). Returns ``(o, final_state, ckpt, cp_base)``; ``o`` and the state are bit for
    bit :func:`forward`'s."""
    _check_count(cu_seqlens.numel() - 1)
    _check_qkv(q, k, v)
    T, H = q.shape[:2]
    if g.shape != q.shape or g.dtype != torch.float32:
        raise ValueError("kda glm_rec_sm100 forward expects fp32 log decay [T, H, 128]")
    if beta.shape != (T, H) or beta.dtype not in _BETA_MODE:
        raise ValueError("kda glm_rec_sm100 expects beta [T, H] bf16 or fp32 (post-sigmoid)")
    N = cu_seqlens.numel() - 1
    if initial_state is not None and (initial_state.shape != (N, H, HEAD_DIM, HEAD_DIM)
                                      or initial_state.dtype != torch.float32):
        raise ValueError(f"initial_state must be fp32 [{N}, {H}, 128, 128] (v-first)")
    _check_device(q, g, beta, initial_state)
    cu = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    cp_base, nck = checkpoint_layout(cu, T)
    ckpt = torch.empty(nck, H, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=q.device)
    o = torch.empty(T, H, HEAD_DIM, dtype=torch.bfloat16, device=q.device)
    state = torch.empty(N, H, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=q.device) if initial_state is None else initial_state.clone()
    slot_in = torch.arange(N, dtype=torch.int32, device=q.device) if initial_state is not None else torch.full((N,), -1, dtype=torch.int32, device=q.device)
    slot_out = torch.arange(N, dtype=torch.int32, device=q.device)
    if N:
        _call(q, k, v, g, beta, cu, slot_in, slot_out, -1, state, o, N, shape=shape, ckpt=ckpt, cp_base=cp_base)
    return o, state, ckpt, cp_base


def backward(q, k, v, g, beta, cu_seqlens, ckpt, cp_base, do, dht=None, *, need_dh0=False, grid=None):
    """Gradients of :func:`forward_train` (deterministic, no atomics; the roundings are identities). Returns
    ``(dq, dk, dv, dg, dbeta, dh0)``; ``dbeta`` has beta's dtype, ``dh0`` is None unless requested."""
    T, H = q.shape[:2]
    cu = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    N = cu.numel() - 1
    dev = q.device
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    grid = grid or max(1, min(N * H, sms))
    dq, dk, dv = (torch.empty(T, H, HEAD_DIM, dtype=torch.bfloat16, device=dev) for _ in range(3))
    dg = torch.empty(T, H, HEAD_DIM, dtype=torch.float32, device=dev)
    dbeta = torch.empty(T, H, dtype=torch.float32, device=dev)
    dh0 = torch.empty(N, H, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=dev) if need_dh0 else None
    scratch = torch.empty(grid * 64 * HEAD_DIM * HEAD_DIM, dtype=torch.float32, device=dev)
    mode = _BETA_MODE[beta.dtype]
    if T == 0 or N == 0:
        return dq.zero_(), dk.zero_(), dv.zero_(), dg.zero_(), dbeta.to(beta.dtype).zero_(), dh0
    extension().bwd(q, k, v, g, beta, mode, cu, ckpt, cp_base, do.contiguous(), dht, dq, dk, dv, dg, dbeta, dh0, scratch, grid)
    return dq, dk, dv, dg, dbeta.to(beta.dtype), dh0


class _CoreFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, cu, h0):
        o, ht, ckpt, base = forward_train(q, k, v, g, beta, cu, h0)
        ctx.save_for_backward(q, k, v, g, beta, cu, ckpt, base)
        ctx.has_h0 = h0 is not None
        return o, ht

    @staticmethod
    def backward(ctx, do, dht):
        q, k, v, g, beta, cu, ckpt, base = ctx.saved_tensors
        if do is None:
            do = torch.zeros_like(q)
        dq, dk, dv, dg, db, dh0 = backward(q, k, v, g, beta, cu, ckpt, base, do, dht, need_dh0=ctx.has_h0)
        return dq, dk, dv, dg, db, None, dh0


def core(q, k, v, g, beta, cu_seqlens, initial_state=None):
    """Autograd form of :func:`forward` (``(o, final_state)``): backward is :func:`backward`."""
    return _CoreFn.apply(q, k, v, g, beta, cu_seqlens.to(torch.int32), initial_state)


def prefill(q, k, v, g, beta, cu_seqlens, state, slot_in, slot_out, *, null_slot=-1, shape=None):
    """Engine prefill: sequence n continues slot ``slot_in[n]`` (negative: a fresh +0 state) and leaves its state
    in ``slot_out[n]`` (negative: not stored). ``state`` is the fp32 ``[slots, H, 128, 128]`` pool."""
    _check_count(cu_seqlens.numel() - 1)
    _check_qkv(q, k, v)
    T, H = q.shape[:2]
    _check_device(q, g, beta)
    _check_state(state, H, q.device)
    N = cu_seqlens.numel() - 1
    si, so = (_slots(n, x, N, q.device, state, null_slot) for n, x in (("slot_in", slot_in), ("slot_out", slot_out)))
    cu = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    o = torch.empty(T, H, HEAD_DIM, dtype=torch.bfloat16, device=q.device)
    if N:
        _call(q, k, v, g, beta, cu, si, so, null_slot, state, o, N, shape=shape)
    return o


def prefill_fused(q, k, v, f, b, dt_bias, A_log, lower_bound, cu_seqlens, state, slot_in, slot_out, *, null_slot=-1,
                  shape=None):
    """:func:`prefill` with the gate and beta stages evaluated in-kernel from the raw bf16 projections."""
    _check_count(cu_seqlens.numel() - 1)
    _check_qkv(q, k, v)
    T, H = q.shape[:2]
    if f.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or f.shape != (T, H * HEAD_DIM) or b.shape != (T, H):
        raise ValueError("prefill_fused expects raw bf16 f [T, H*128] and b [T, H]")
    dt_bias, A_log = _gate_params(dt_bias, A_log, H)
    _check_device(q, f, b, dt_bias, A_log)
    _check_state(state, H, q.device)
    N = cu_seqlens.numel() - 1
    si, so = (_slots(n, x, N, q.device, state, null_slot) for n, x in (("slot_in", slot_in), ("slot_out", slot_out)))
    cu = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    o = torch.empty(T, H, HEAD_DIM, dtype=torch.bfloat16, device=q.device)
    if N:
        _call(q, k, v, f, b, cu, si, so, null_slot, state, o, N,
              fused=True, dt_bias=dt_bias, A_log=A_log, lb=lower_bound, beta_mode=2, shape=shape)
    return o


def decode_step(q, k, v, g, beta, state, slots, *, null_slot=-1, out=None, shape=None):
    """One token per sequence, in place in ``state[slots]``. Lanes whose slot is ``null_slot`` (vLLM's padded
    lanes) produce zeros and touch no state. CUDA-graph capturable (no host sync)."""
    _check_count(q.shape[0], "decode lanes")
    _check_qkv(q, k, v)
    N, H = q.shape[:2]
    _check_device(q, g, beta)
    _check_state(state, H, q.device)
    sl = _slots("slots", slots, N, q.device, state, null_slot)
    o = _check_out(out, N, H, q.device)
    if N:
        _call(q, k, v, g, beta, None, sl, sl, null_slot, state, o, N, decode=True, shape=shape)
    return o


def decode_step_fused(q, k, v, f, b, dt_bias, A_log, lower_bound, state, slots, *, null_slot=-1, out=None, shape=None):
    """:func:`decode_step` with the forget gate and beta evaluated in-kernel from the raw bf16 projections
    ``f [N, H*128]`` and ``b [N, H]``: bit for bit ``decode_step(..., forget_gate(f, ...), sigmoid_beta(b), ...)``."""
    _check_count(q.shape[0], "decode lanes")
    _check_qkv(q, k, v)
    N, H = q.shape[:2]
    if f.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or f.shape != (N, H * HEAD_DIM) or b.shape != (N, H):
        raise ValueError("decode_step_fused expects raw bf16 f [N, H*128] and b [N, H]")
    dt_bias, A_log = _gate_params(dt_bias, A_log, H)
    _check_device(q, f, b, dt_bias, A_log)
    _check_state(state, H, q.device)
    sl = _slots("slots", slots, N, q.device, state, null_slot)
    o = _check_out(out, N, H, q.device)
    if N:
        _call(q, k, v, f, b, None, sl, sl, null_slot, state, o, N, fused=True, dt_bias=dt_bias, A_log=A_log,
              lb=lower_bound, beta_mode=2, decode=True, shape=shape)
    return o


def _gate_params(dt_bias, A_log, H: int):
    """fp32 contiguous ``dt_bias [H*128]`` and ``A_log [H]`` (the kernels read fp32; an fp32 tensor is returned as is)."""
    if not (isinstance(dt_bias, torch.Tensor) and isinstance(A_log, torch.Tensor)) or dt_bias.numel() != H * HEAD_DIM \
            or A_log.numel() != H or not (dt_bias.is_floating_point() and A_log.is_floating_point()):
        raise ValueError(f"kda glm_rec_sm100 expects floating-point dt_bias [{H * HEAD_DIM}] and A_log [{H}]")
    return dt_bias.float().contiguous(), A_log.float().contiguous()


def forget_gate(f, dt_bias, A_log, lower_bound):
    """fp32 log decay ``lb * sigmoid(exp(A_log_h) * (f + dt_bias))`` of the raw bf16 forget projection f
    ``[T, H*128]`` (the recipe's gate definition; same bits as glm_sm100's gate op and the fused kernels)."""
    out = torch.empty(f.shape, dtype=torch.float32, device=f.device)
    extension().gate(f, dt_bias.float().contiguous(), A_log.float().contiguous(), float(lower_bound), out)
    return out


def sigmoid_beta(b):
    """bf16 ``sigmoid(b)`` of the raw bf16 beta projection."""
    out = torch.empty_like(b)
    extension().beta(b.contiguous(), out)
    return out


__all__ = ["forward_train", "backward", "core", "checkpoint_layout", "forward", "prefill", "prefill_fused", "decode_step", "decode_step_fused", "forget_gate", "sigmoid_beta"]
