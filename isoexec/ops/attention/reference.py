"""Plain-torch attention references for RoPE, GQA, unabsorbed MLA and output gates.
These semantic oracles use thd layouts and do not reproduce kernel rounding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isoexec.core.bindings import BoundOp

if TYPE_CHECKING:  # the plan is DUCK-TYPED at runtime -- ops/reference.py imports this module, never the reverse
    from isoexec.ops.reference import ReferencePlan


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def ref_rope_apply(q, k, freqs=None, *, style: str, rotary_dim: int, seq=None, rotary=None, input_pairs=None):
    """Partial rotary with fp32 angles and math, rounding once per tensor.

    The binding owns reference angle preparation and checkpoint lane conversion.
    Explicit angle tensors remain useful to callers of this semantic oracle.
    """
    if freqs is None:
        from isoexec.core.rope import ensure_rope_freqs

        if seq is None or rotary is None:
            raise ValueError("ref_rope_apply: seq and rotary are required without explicit freqs")
        freqs = ensure_rope_freqs(seq, rotary)
    if input_pairs is not None and input_pairs != style:
        if input_pairs != "interleaved" or style != "neox":
            raise ValueError(f"ref_rope_apply: unsupported lane conversion {input_pairs!r} to {style!r}")
        def gather(x):
            rotated, suffix = x[..., :rotary_dim], x[..., rotary_dim:]
            halves = rotated.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2)
            return torch.cat((halves, suffix), dim=-1)

        q, k = gather(q), gather(k)
    cos, sin = torch.cos(freqs), torch.sin(freqs)

    def one(x):
        xf = x.to(torch.float32)
        xr, xp = xf[..., :rotary_dim], xf[..., rotary_dim:]
        if style == "neox":
            c = torch.cat((cos, cos), dim=-1)[:, None, :]
            s = torch.cat((sin, sin), dim=-1)[:, None, :]
            rot = xr * c + _rotate_half(xr) * s
        elif style == "interleaved":
            c = cos.repeat_interleave(2, dim=-1)[:, None, :]
            s = sin.repeat_interleave(2, dim=-1)[:, None, :]
            rot = xr * c + _rotate_interleaved(xr) * s
        else:
            raise ValueError(style)
        return torch.cat((rot, xp), dim=-1).to(x.dtype)

    return one(q), one(k)


def ref_attention_core(q, k, v, cu_seqlens, max_seqlen: int, scale: float) -> torch.Tensor:
    """Causal GQA per packed sequence, fp32 softmax; returns [T, hq, dv] in q's dtype."""
    del max_seqlen
    T, hq, _d = q.shape
    hk = k.shape[1]
    rep = hq // hk
    out = torch.empty(T, hq, v.shape[-1], dtype=q.dtype, device=q.device)
    cu = cu_seqlens.tolist()
    for i in range(len(cu) - 1):
        a, b = cu[i], cu[i + 1]
        qs = q[a:b].to(torch.float32).transpose(0, 1)  # [hq, L, d]
        ks = k[a:b].to(torch.float32).repeat_interleave(rep, dim=1).transpose(0, 1)
        vs = v[a:b].to(torch.float32).repeat_interleave(rep, dim=1).transpose(0, 1)
        L = b - a
        s = (qs @ ks.transpose(-1, -2)) * scale
        mask = torch.ones(L, L, dtype=torch.bool, device=q.device).tril()
        s = s.masked_fill(~mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        out[a:b] = (p @ vs).transpose(0, 1).to(q.dtype)
    return out


def ref_attention_core_qk(q, k, v, cu_q, cu_k, max_q: int, max_k: int, scale: float) -> torch.Tensor:
    """Causal GQA with separate query/key boundaries, queries right-aligned to their keys (the
    chunked-prefill / context-parallel form). Computed one query row at a time, so a row's arithmetic
    depends only on the keys it sees, never on how the queries were split."""
    del max_q, max_k
    hq, hk = q.shape[1], k.shape[1]
    rep = hq // hk
    out = torch.empty(q.shape[0], hq, v.shape[-1], dtype=q.dtype, device=q.device)
    qc, kc = cu_q.tolist(), cu_k.tolist()
    for i in range(len(qc) - 1):
        qa, qb, ka, kb = qc[i], qc[i + 1], kc[i], kc[i + 1]
        shift = (kb - ka) - (qb - qa)
        ks = k[ka:kb].to(torch.float32).repeat_interleave(rep, dim=1).transpose(0, 1)  # [hq, Lk, d]
        vs = v[ka:kb].to(torch.float32).repeat_interleave(rep, dim=1).transpose(0, 1)
        for row in range(qb - qa):
            seen = shift + row + 1
            s = torch.einsum("hd,hkd->hk", q[qa + row].to(torch.float32), ks[:, :seen]) * scale
            p = torch.softmax(s, dim=-1)
            out[qa + row] = torch.einsum("hk,hkd->hd", p, vs[:, :seen]).to(q.dtype)
    return out


def ref_attention_gate(o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (o.to(torch.float32) * torch.sigmoid(gate.to(torch.float32))).to(o.dtype)


def ref_attention_gate_swish(o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply ``o * silu(gate)`` in fp32, then round once to the output dtype."""
    return (o.to(torch.float32) * torch.nn.functional.silu(gate.to(torch.float32))).to(o.dtype)


GATE_BINDINGS = {
    "sigmoid": BoundOp(
        "attention.gate",
        "ref.sigmoid_gate",
        None,
        "ops.reference",
        inputs={"gate": "sigmoid"},
        fn=ref_attention_gate,
    ),
    "swish": BoundOp(
        "attention.gate",
        "ref.swish_gate",
        None,
        "ops.reference",
        inputs={"gate": "swish"},
        fn=ref_attention_gate_swish,
    ),
}


def ref_mla_core(q_nope, q_pe, c, k_pe, uk, uv, seq, scale):
    """Unabsorbed torch algebra reference, not the contracted absorbed forward."""
    heads = q_nope.shape[1]
    latent = c.unsqueeze(0).expand(heads, -1, -1)
    k_nope = torch.bmm(latent, uk.transpose(1, 2)).transpose(0, 1)
    values = torch.bmm(latent, uv.transpose(1, 2)).transpose(0, 1)
    q = torch.cat((q_nope, q_pe), dim=-1)
    k = torch.cat((k_nope, k_pe.unsqueeze(1).expand(-1, heads, -1)), dim=-1)
    return ref_attention_core(q, k, values, seq.cu_seqlens, seq.max_q, scale)


def bindings(plan: ReferencePlan) -> dict[str, BoundOp]:
    """This family's contribution to ``ops/reference.py::reference_bindings``."""
    selectors = plan.selectors.get("attention.gate", {})
    unknown = selectors.keys() - {"gate"}
    if unknown:
        raise ValueError(f"attention.gate: unsupported reference selectors {sorted(unknown)}")
    gate = selectors.get("gate", "sigmoid")
    if gate not in GATE_BINDINGS:
        raise ValueError(f"attention.gate: no reference binding for gate {gate!r} (on offer: {tuple(GATE_BINDINGS)})")
    return {
        "rope.apply": BoundOp("rope.apply", "ref.rope", None, "ops.reference", fn=ref_rope_apply),
        "attention.core.gqa": BoundOp(
            "attention.core.gqa", "ref.attention", None, "ops.reference", fn=ref_attention_core
        ),
        "attention.core.mla": BoundOp(
            "attention.core.mla", "ref.mla_unabsorbed", None, "ops.reference", fn=ref_mla_core
        ),
        "attention.gate": GATE_BINDINGS[gate],
    }


__all__ = [
    "GATE_BINDINGS",
    "bindings",
    "ref_attention_core",
    "ref_attention_core_qk",
    "ref_attention_gate",
    "ref_attention_gate_swish",
    "ref_mla_core",
    "ref_rope_apply",
]
