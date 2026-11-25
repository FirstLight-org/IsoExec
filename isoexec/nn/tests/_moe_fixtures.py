"""Shared tiny mixture, embedding and head fixtures for component CPU tests."""

from __future__ import annotations

import torch

from isoexec.nn.embed import Embedding
from isoexec.nn.head import LMHead
from isoexec.nn.moe import MoE
from isoexec.schema import Attn, Axis, Mlp, ModelConfig, Placement

#: A handle sentinel. These tests never run a collective; a placement only has to carry a group.
GROUP = object()

#: A Qwen3.5-shaped mixture, shrunk: 8 experts, top 2, a shared expert a quarter of the routed width.
HIDDEN = 64
NUM_EXPERTS = 8
TOP_K = 2
INTERMEDIATE = 32
SHARED_INTERMEDIATE = 16
VOCAB = 96
PADDED_VOCAB = 128
EPS = 1e-6
NS = "lm.layers.0"
SITE = f"{NS}.mlp"

#: The attention row the one-row config needs; nothing in these proofs runs it.
MIXER = Attn(num_q_heads=4, num_kv_heads=2, head_dim=16, rope_theta=1e4)


def spec(
    *,
    num_experts: int = NUM_EXPERTS,
    top_k: int = TOP_K,
    intermediate: int = INTERMEDIATE,
    routing: str = "softmax_renorm",
    shared_intermediate: int = SHARED_INTERMEDIATE,
    shared_expert_gate: bool = True,
    routed_scale: float = 1.0,
) -> Mlp:
    return Mlp(
        intermediate_size=intermediate,
        num_experts=num_experts,
        top_k=top_k,
        shared_intermediate=shared_intermediate,
        routing=routing,
        routed_scale=routed_scale,
        shared_expert_gate=shared_expert_gate,
    )


def config(mlp: Mlp, *, tied_head: bool = False) -> ModelConfig:
    """A one-row model around ``mlp`` -- what the schema vocabulary folds."""
    return ModelConfig(
        model_id="tiny_moe",
        hidden=HIDDEN,
        vocab=VOCAB,
        padded_vocab=PADDED_VOCAB,
        rms_eps=EPS,
        tied_head=tied_head,
        layers=((MIXER, mlp),),
    )


def component(mlp: Mlp, *, device: str = "meta") -> MoE:
    """``nn.MoE`` with the same configuration, spelled as the constructor spells it."""
    return MoE(
        hidden=HIDDEN,
        num_experts=mlp.num_experts,
        top_k=mlp.top_k,
        intermediate=mlp.intermediate_size,
        routing=mlp.routing,
        shared_intermediate=mlp.shared_intermediate,
        shared_expert_gate=mlp.shared_expert_gate,
        routed_scale=mlp.routed_scale,
        device=device,
    )


def embedding(*, device: str = "meta") -> Embedding:
    return Embedding(vocab=PADDED_VOCAB, hidden=HIDDEN, device=device)


def head(*, tied_to=None, device: str = "meta") -> LMHead:
    return LMHead(vocab=VOCAB, hidden=HIDDEN, padded_vocab=PADDED_VOCAB, tied_to=tied_to, device=device)


def placement(
    tp: int = 1,
    rank: int = 0,
    *,
    ep: int = 1,
    ep_rank: int = 0,
    side: str = "trainer",
    sp: bool = False,
    leaves: int = 8,
) -> Placement:
    """Build the requested placement using sentinel groups for record inspection; these fixtures
    run no collectives.
    """
    one = Axis(1, 0, None)
    tp_axis = Axis(tp, rank, GROUP if tp > 1 else None)
    return Placement(
        tp=tp_axis,
        ep=Axis(ep, ep_rank, GROUP if ep > 1 else None),
        dp=one,
        pp=one,
        etp=(tp_axis if tp > 1 else one) if ep == 1 else one,
        cp=one,
        side=side,
        leaves=leaves,
        sp=sp,
    )


def mirror_weights(left, right, *, seed: int, scales=None, default: float = 1.0) -> None:
    """Fill both modules with identical values matched by parameter name. Scaling controls the
    exponent range used by underflow tests.
    """
    gen = torch.Generator().manual_seed(seed)
    scales = scales or {}
    lhs = dict(left.named_parameters()) | {n: b for n, b in left.named_buffers() if b is not None}
    rhs = dict(right.named_parameters()) | {n: b for n, b in right.named_buffers() if b is not None}
    assert set(lhs) == set(rhs), f"tensor names differ: {sorted(set(lhs) ^ set(rhs))}"
    # one fixed name order, taken from the left module's own walk and extended with whatever it holds
    # as a buffer, so the generator draws the same sequence for both sides whatever a tier registers.
    order = [n for n, _ in left.named_parameters()]
    order += [n for n in lhs if n not in set(order)]
    for name in order:
        p, q = lhs[name], rhs[name]
        assert tuple(p.shape) == tuple(q.shape), f"{name}: {tuple(p.shape)} vs {tuple(q.shape)}"
        scale = next((s for key, s in scales.items() if name.endswith(key)), default)
        w = torch.randn(p.shape, generator=gen, dtype=torch.float32) * scale
        with torch.no_grad():
            p.copy_(w.to(p.dtype))
            q.copy_(p)


def bits(t: torch.Tensor) -> torch.Tensor:
    """View tensor bit patterns as integers, preserving signed-zero and NaN payload differences."""
    view = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[t.dtype]
    return t.contiguous().view(view)
