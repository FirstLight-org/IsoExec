"""Plain-torch semantic references for short convolution and the gated delta rule.
The core includes L2 normalization and gating, matching fused ownership; these references do not reproduce kernel rounding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isoexec.core.bindings import BoundOp

if TYPE_CHECKING:  # the plan is DUCK-TYPED at runtime -- ops/reference.py imports this module, never the reverse
    from isoexec.ops.reference import ReferencePlan


def ref_conv_short(x: torch.Tensor, w: torch.Tensor, cu_seqlens) -> torch.Tensor:
    """Causal depthwise conv over each packed sequence + silu; [T, C], w [C, 1, K]."""
    C, _, K = w.shape
    out = torch.empty_like(x)
    cu = cu_seqlens.tolist()
    for i in range(len(cu) - 1):
        a, b = cu[i], cu[i + 1]
        xs = x[a:b].to(torch.float32).t()[None]  # [1, C, L]
        y = F.conv1d(xs, w.to(torch.float32), padding=K - 1, groups=C)[..., : b - a]
        out[a:b] = F.silu(y)[0].t().to(x.dtype)
    return out


def ref_gdn_l2norm(q, k):
    def n(x):
        xf = x.to(torch.float32)
        return (xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) + 1e-6)).to(x.dtype)

    return n(q), n(k)


def ref_gdn_core(q, k, v, alpha, beta, A_log, dt_bias, cu_seqlens,
                 *, cu_seqlens_cpu=None) -> torch.Tensor:
    """Recurrent gated delta rule in fp32 (l2norm + gating in-kernel, like the fused impl):
    S_t = S_{t-1} * exp(g_t); delta = beta_t * (v_t - k_t S_{t-1}exp(g_t)); S_t += k_t^T delta; o_t = q_t S_t."""
    T, hk, dk = q.shape
    hv, dv = v.shape[1], v.shape[2]
    rep = hv // hk
    qn, kn = ref_gdn_l2norm(q.to(torch.float32), k.to(torch.float32))
    # The native fused core applies the default attention scale after L2 normalization.
    qn = qn.repeat_interleave(rep, dim=1) * (dk ** -0.5)
    kn = kn.repeat_interleave(rep, dim=1)
    g = -torch.exp(A_log.to(torch.float32)) * F.softplus(alpha.to(torch.float32) + dt_bias.to(torch.float32))  # [T, hv]
    b = torch.sigmoid(beta.to(torch.float32))
    vf = v.to(torch.float32)
    out = torch.empty(T, hv, dv, dtype=torch.float32, device=v.device)
    cu = (cu_seqlens if cu_seqlens_cpu is None else cu_seqlens_cpu).tolist()
    for i in range(len(cu) - 1):
        S = torch.zeros(hv, dk, dv, dtype=torch.float32, device=v.device)
        for t in range(cu[i], cu[i + 1]):
            S = S * torch.exp(g[t])[:, None, None]
            kv_mem = torch.einsum("hk,hkv->hv", kn[t], S)
            delta = b[t][:, None] * (vf[t] - kv_mem)
            S = S + kn[t][:, :, None] * delta[:, None, :]
            out[t] = torch.einsum("hk,hkv->hv", qn[t], S)
    return out.to(v.dtype)


def ref_gdn_core_stateful(q, k, v, alpha, beta, A_log, dt_bias, *, cu_seqlens, lengths, h0=None,
                          want_last: bool = False):
    """:func:`ref_gdn_core` with a state hand-off (the reference twin of ``gdn_native_stateful``).

    ``h0`` (``[hv, dv, dk]`` fp32, the native pool-row layout) starts the first sequence; with
    ``want_last`` the state after the last sequence's last token comes back in the same layout.
    Differentiable through plain torch autograd. Each token's update is the same op sequence on the
    same shapes whether or not earlier tokens were in this call, so a hand-off is exact here too.
    """
    del cu_seqlens
    T, hk, dk = q.shape
    hv, dv = v.shape[1], v.shape[2]
    rep = hv // hk
    vf = v.to(torch.float32)
    decay = -torch.exp(A_log.to(torch.float32))

    # Every op below runs on one token's tensors, whose shapes never depend on T: CPU kernels pick
    # vector/remainder paths (and transcendental approximations) by size, so a [T, ...] op would make
    # a token's bits depend on how many tokens share the call.
    def normed(x):
        xf = x.to(torch.float32)
        return (xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) + 1e-6)).repeat_interleave(rep, dim=0)

    rows = []
    S = None
    start = 0
    for n, length in enumerate(lengths):
        if n == 0 and h0 is not None:
            S = h0.transpose(-1, -2)
        else:
            S = torch.zeros(hv, dk, dv, dtype=torch.float32, device=v.device)
        for t in range(start, start + length):
            qn, kn = normed(q[t]) * (dk ** -0.5), normed(k[t])
            g = decay * F.softplus(alpha[t].to(torch.float32) + dt_bias.to(torch.float32))
            S = S * torch.exp(g)[:, None, None]
            kv_mem = torch.einsum("hk,hkv->hv", kn, S)
            delta = torch.sigmoid(beta[t].to(torch.float32))[:, None] * (vf[t] - kv_mem)
            S = S + kn[:, :, None] * delta[:, None, :]
            rows.append(torch.einsum("hk,hkv->hv", qn, S))
        start += length
    out = torch.stack(rows).to(v.dtype)
    return out, (S.transpose(-1, -2).contiguous() if want_last else None)


def bindings(plan: "ReferencePlan") -> dict[str, BoundOp]:
    """This family's contribution to ``ops/reference.py::reference_bindings``.

    ``gdn.core`` OWNS ``gdn.l2norm`` and ``gdn.gating`` because :func:`ref_gdn_core` computes both
    in-kernel, like the production fused impl -- so a table built from this module skips exactly the
    steps the production fused impl skips.
    """
    del plan
    return {
        "conv.short": BoundOp("conv.short", "ref.conv_silu", None, "ops.reference", fn=ref_conv_short),
        "gdn.core": BoundOp(
            "gdn.core",
            "ref.gdn_recurrent",
            None,
            "ops.reference",
            fn=ref_gdn_core,
            owns={"gdn.l2norm": "ref.gdn_recurrent", "gdn.gating": "ref.gdn_recurrent"},
        ),
    }


__all__ = [
    "bindings",
    "ref_conv_short",
    "ref_gdn_core",
    "ref_gdn_core_stateful",
    "ref_gdn_l2norm",
]
