"""Autograd bindings of the SM100 GLM KDA core and the layer's element-wise stages, and the native
composition of ``isoexec.nn.GLMKimiDeltaAttention(backend="sm100")``.

Native stages: Q/K/V causal SiLU convolutions (``gdn.conv_sm100``, whose equation is GLM's: width-4 causal
taps oldest-to-current in fp32, SiLU, one bf16 rounding), the bounded forget gate, sigmoid beta, the KDA
recurrence and the sigmoid-gated RMS output (this package). The projections stay the module's own linear
layers (GEMMs are the ``mm`` family's contract). Torch here is allocation, views and the attention-mask
multiply the eager composition also performs.
"""

from __future__ import annotations

import torch

from isoexec.ops.kda.glm_sm100 import ops
from isoexec.ops.kda.glm_sm100._extension import extension
from isoexec.ops.kda.glm_sm100.spec import HEAD_DIM, IMPL_ID

IMPL = IMPL_ID


class _Core(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, h0, cu):
        o, ht = ops.forward(q, k, v, g, beta, cu, h0, output_final_state=True)
        ctx.save_for_backward(q, k, v, g, beta, h0, cu)
        ctx.has_h0 = h0 is not None
        return o, ht

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, do, dht):
        q, k, v, g, beta, h0, cu = ctx.saved_tensors
        dq, dk, dv, dg, db, dh0 = ops.backward(q, k, v, g, beta, cu, h0 if ctx.has_h0 else None, do, dht)
        return dq, dk, dv, dg, db, (dh0 if ctx.has_h0 else None), None


def kda_core(q, k, v, log_decay, beta, initial_state=None, *, cu_seqlens=None, return_state=False):
    """The KDA recurrence with gradients. ``[B, T, H, 128]`` inputs (every batch row one sequence), or
    packed ``[T, H, 128]`` with int32 ``cu_seqlens``. Returns ``o`` (and the fp32 final state)."""
    batched = q.ndim == 4
    if batched:
        B, T, H, K = q.shape
        if q.shape != k.shape or log_decay.shape != k.shape or v.shape[:3] != q.shape[:3] or beta.shape != q.shape[:3]:
            raise ValueError("KDA expects matching [B,T,H,K] Q, K, and per-channel decay")
        if T == 0:
            raise ValueError("KDA expects nonempty time")
        if cu_seqlens is not None:
            raise ValueError("cu_seqlens is for packed [T, H, K] inputs")
        cu = torch.arange(B + 1, device=q.device, dtype=torch.int32) * T
        flat = [t.reshape(B * T, *t.shape[2:]) for t in (q, k, v, log_decay, beta)]
    else:
        if cu_seqlens is None:
            raise ValueError("packed [T, H, K] KDA inputs need cu_seqlens")
        cu = cu_seqlens.to(device=q.device, dtype=torch.int32)
        flat = [q, k, v, log_decay, beta]
    if initial_state is not None:
        initial_state = initial_state.float()
    o, ht = _Core.apply(*flat, initial_state, cu)
    if batched:
        o = o.view(B, T, H, v.shape[-1])
    return (o, ht) if return_state else o


class _Gate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, dt_bias, A_log, lb):
        HK = dt_bias.numel()
        f2 = f.contiguous()
        # The kernels read fp32 parameters (as gdn_cpr_sm100's and glm_rec_sm100's gate do, a bf16 / fp16 parameter
        # is widened exactly; an fp32 one is the same tensor, no copy). Gradients return in the parameter's dtype.
        dt32, A32 = dt_bias.detach().float().contiguous(), A_log.detach().float().contiguous()
        out = torch.empty(f2.shape, dtype=torch.float32, device=f.device)
        extension().gate_fwd(f2, dt32, A32, float(lb), out)
        ctx.save_for_backward(f2, dt32, A32)
        ctx.lb, ctx.HK, ctx.dtypes = float(lb), HK, (dt_bias.dtype, A_log.dtype)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dld):
        f, dt_bias, A_log = ctx.saved_tensors
        HK, H = ctx.HK, A_log.numel()
        T = f.numel() // HK
        nseg = max(1, -(-T // extension().seg()))
        f32 = dict(dtype=torch.float32, device=f.device)
        pdt, pA = torch.empty(nseg, HK, dtype=torch.float32, device=f.device), torch.empty(nseg, HK, dtype=torch.float32, device=f.device)
        colA, ddt, dA = torch.empty(HK, dtype=torch.float32, device=f.device), torch.empty(HK, dtype=torch.float32, device=f.device), torch.empty(H, dtype=torch.float32, device=f.device)
        df = torch.empty_like(f)
        extension().gate_bwd(dld.float().contiguous(), f, df, dt_bias, A_log, ctx.lb, pdt, pA, colA, ddt, dA)
        return df, ddt.to(ctx.dtypes[0]), dA.to(ctx.dtypes[1]), None


class _Beta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b):
        b2 = b.contiguous()
        y = torch.empty_like(b2)
        extension().beta_fwd(b2, y)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        (y,) = ctx.saved_tensors
        db = torch.empty_like(y)
        extension().beta_bwd(dy.contiguous(), y, db)
        return db


class _GatedNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z, w, eps):
        x2, z2 = x.contiguous(), z.contiguous()
        y = torch.empty_like(x2)
        rstd = torch.empty(x2.numel() // HEAD_DIM, dtype=torch.float32, device=x.device)
        extension().gnorm_fwd(x2, z2, w.contiguous(), float(eps), y, rstd)
        ctx.save_for_backward(x2, z2, w, rstd)
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        x, z, w, rstd = ctx.saved_tensors
        rows = x.numel() // HEAD_DIM
        nseg = max(1, -(-rows // extension().seg()))
        pw = torch.empty(nseg, HEAD_DIM, dtype=torch.float32, device=x.device)
        dx, dz, dw = torch.empty_like(x), torch.empty_like(z), torch.empty_like(w)
        extension().gnorm_bwd(dy.to(torch.bfloat16).contiguous(), x, z, w.contiguous(), rstd, dx, dz, pw, dw)
        return dx, dz, dw, None


def forget_gate(f, dt_bias, A_log, lower_bound):
    """``lower_bound * sigmoid(exp(A_log)[h] * (f + dt_bias))`` in fp32, f ``[..., H * 128]``."""
    HK = f.shape[-1] if f.ndim else 0
    if f.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"glm_sm100 forget gate expects a bf16 or fp32 projection, got {f.dtype}")
    if not f.is_cuda or dt_bias.device != f.device or A_log.device != f.device:
        raise ValueError("glm_sm100 forget gate requires f, dt_bias and A_log on one CUDA device")
    if HK == 0 or HK % HEAD_DIM or dt_bias.numel() != HK or A_log.numel() != HK // HEAD_DIM:
        raise ValueError(f"glm_sm100 forget gate expects f [..., H * 128], dt_bias [H * 128] and A_log [H]; got "
                         f"{tuple(f.shape)}, {tuple(dt_bias.shape)}, {tuple(A_log.shape)}")
    if not (dt_bias.is_floating_point() and A_log.is_floating_point()):
        raise ValueError("glm_sm100 forget gate expects floating-point dt_bias and A_log")
    return _Gate.apply(f, dt_bias, A_log, lower_bound)


def sigmoid_beta(b):
    """bf16 ``sigmoid(b)``, the published bf16 boundary."""
    if b.dtype != torch.bfloat16:
        raise ValueError("glm_sm100 sigmoid beta expects the bf16 projection output")
    return _Beta.apply(b)


def gated_rms_norm(core, gate, weight, eps):
    """``rn(core * rsqrt(mean(core^2) + eps) * weight * sigmoid(gate))`` per 128-wide head row."""
    if core.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16 or core.numel() != gate.numel():
        raise ValueError("glm_sm100 gated norm expects bf16 core and gate of equal size")
    if core.numel() % HEAD_DIM or weight.numel() != HEAD_DIM or weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("glm_sm100 gated norm expects 128-wide rows and a bf16 or fp32 weight [128]")
    if not core.is_cuda or gate.device != core.device or weight.device != core.device:
        raise ValueError("glm_sm100 gated norm requires core, gate and weight on one CUDA device")
    return _GatedNorm.apply(core, gate, weight, eps)


def glm_kda_attention(module, x, attention_mask=None):
    """The native composition of ``GLMKimiDeltaAttention.forward`` (module carries the parameters)."""
    from isoexec.ops.gdn.conv_sm100 import ops as conv_ops

    cfg = module.cfg
    if x.ndim != 3:
        raise ValueError("GLM KDA expects [batch,time,hidden]")
    if x.dtype != torch.bfloat16 or cfg.linear_head_dim != HEAD_DIM:
        raise ValueError("GLM KDA backend='sm100' requires bf16 activations and linear_head_dim == 128")
    if attention_mask is not None:
        x = (x * attention_mask[..., None]).to(x.dtype)
    B, T = x.shape[:2]
    H, K = cfg.linear_num_heads, cfg.linear_head_dim
    cu = torch.arange(B + 1, device=x.device, dtype=torch.int32) * T
    qkv = []
    for name in ("q", "k", "v"):
        projected = getattr(module, name + "_proj")(x).reshape(B * T, H * K)
        weight = getattr(module, name + "_conv1d").weight
        qkv.append(conv_ops.conv(projected, weight, None, cu_seqlens=cu, activation="silu").view(B, T, H, K))
    forget = module.f_b_proj(module.f_a_proj(x))
    log_decay = forget_gate(forget, module.dt_bias, module.A_log, cfg.linear_gate_lower_bound).view(B, T, H, K)
    beta = sigmoid_beta(module.b_proj(x))
    core = kda_core(*qkv, log_decay, beta)
    gate = module.g_b_proj(module.g_a_proj(x))
    normalized = gated_rms_norm(core, gate.view(B, T, H, K), module.o_norm.weight, cfg.rms_norm_eps)
    return module.o_proj(normalized.flatten(-2))


__all__ = ["IMPL", "forget_gate", "gated_rms_norm", "glm_kda_attention", "kda_core", "sigmoid_beta"]
