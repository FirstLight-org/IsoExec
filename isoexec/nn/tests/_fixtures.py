"""Shared tiny attention geometry for component CPU tests."""

from __future__ import annotations

import torch

from isoexec.nn import Attention
from isoexec.schema import Attn, Axis, Placement

#: A handle sentinel. These tests never run a collective; a placement only has to carry a group.
GROUP = object()

#: The tiny Qwen3.5 text shape (``models/qwen3_5/tests/tiny.py::TINY_TEXT_COMMON``), which is
#: 4 query heads over 2 kv heads at head_dim 64 with a quarter-rotary and a sigmoid output gate.
HIDDEN = 256
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 64
ROPE_THETA = 1e7
ROTARY_FRACTION = 0.25
EPS = 1e-6
NS = "lm.layers.0"
SITE = f"{NS}.mixer"


def spec(
    *,
    heads: int = HEADS,
    kv_heads: int = KV_HEADS,
    qk_norm: str | None = "rms_zero_centered",
    out_gate: str | None = "sigmoid",
    qkv_bias: bool = False,
    mrope: bool = False,
) -> Attn:
    return Attn(
        num_q_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=HEAD_DIM,
        rope_theta=ROPE_THETA,
        partial_rotary=ROTARY_FRACTION,
        qk_norm=qk_norm,
        qkv_bias=qkv_bias,
        out_gate=out_gate,
        mrope_section=(3, 3, 2) if mrope else None,
        mrope_interleaved=mrope,
    )


def component(row: Attn, *, device: str = "meta") -> Attention:
    """``nn.Attention`` with the same configuration, spelled as the constructor spells it."""
    return Attention(
        hidden=HIDDEN,
        heads=row.num_q_heads,
        kv_heads=row.num_kv_heads,
        head_dim=row.head_dim,
        rope_theta=row.rope_theta,
        rotary_fraction=row.partial_rotary,
        rope_style=row.rope_style,
        mrope_section=row.mrope_section,
        mrope_interleaved=row.mrope_interleaved,
        qkv_bias=row.qkv_bias,
        qk_norm=row.qk_norm,
        eps=EPS,
        out_gate=row.out_gate,
        device=device,
    )


def placement(tp: int = 1, rank: int = 0, *, side: str = "trainer") -> Placement:
    """A ``Placement`` at the TP degree under test; every other axis is 1."""
    one = Axis(1, 0, None)
    tp_axis = Axis(tp, rank, GROUP if tp > 1 else None)
    return Placement(tp=tp_axis, ep=one, dp=one, pp=one, etp=tp_axis if tp > 1 else one, cp=one, side=side, leaves=8)


def mirror_weights(
    left: torch.nn.Module, right: torch.nn.Module, *, seed: int, scales=None, default: float = 1.0
) -> None:
    """Fill both modules with identical values matched by parameter name. Scaling controls the
    exponent range used by underflow tests.
    """
    gen = torch.Generator().manual_seed(seed)
    scales = scales or {}
    for (a, p), (b, q) in zip(left.named_parameters(), right.named_parameters(), strict=True):
        assert a == b, f"parameter order differs: {a!r} vs {b!r}"
        assert tuple(p.shape) == tuple(q.shape), f"{a}: {tuple(p.shape)} vs {tuple(q.shape)}"
        w = torch.randn(p.shape, generator=gen, dtype=torch.float32) * scales.get(a, default)
        with torch.no_grad():
            p.copy_(w.to(p.dtype))
            q.copy_(p)


def bits(t: torch.Tensor) -> torch.Tensor:
    """View tensor bit patterns as integers, preserving signed-zero and NaN payload differences."""
    view = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[t.dtype]
    return t.contiguous().view(view)
