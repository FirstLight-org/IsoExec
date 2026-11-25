"""Measure which registered eager implementations match their plain-torch references bitwise.
Both equalities and expected inequalities are asserted on integer views, including signed zero."""

from __future__ import annotations

import pytest
import torch

#: Small, ragged, and not a power of two in either dimension: a schedule that depended on the tiling
#: would show it here rather than on a round shape.
ROWS, WIDTH, VOCAB, EXPERTS = 48, 132, 500, 7

#: bf16 is the fleet's activation dtype, and it is where a fp32-upcast-then-round schedule and a
#: native-dtype one part company. A test written in fp32 would find every pair below equal.
DTYPE = torch.bfloat16


def _bits(t: torch.Tensor) -> torch.Tensor:
    """The integer view of a tensor's bit pattern. See the module docstring for why."""
    view = {torch.bfloat16: torch.int16, torch.float16: torch.int16,
            torch.float32: torch.int32, torch.float64: torch.int64}
    return t.contiguous().view(view[t.dtype]) if t.dtype in view else t.contiguous()


def _same_bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.dtype == right.dtype and left.shape == right.shape and bool(torch.equal(_bits(left), _bits(right)))


def _differing(left: torch.Tensor, right: torch.Tensor) -> int:
    return int((_bits(left) != _bits(right)).sum())


@pytest.fixture
def inputs():
    """One seeded set of activations, shared by every case, so the cases compare like with like."""
    gen = torch.Generator().manual_seed(0x15E0)
    return {
        "x": (torch.randn(ROWS, WIDTH, generator=gen) * 2).to(DTYPE),
        "y": (torch.randn(ROWS, WIDTH, generator=gen) * 2).to(DTYPE),
        "embed_w": torch.randn(VOCAB, WIDTH, generator=gen).to(DTYPE),
        "ids": torch.randint(0, VOCAB, (ROWS,), generator=gen),
        "router_w": torch.randn(EXPERTS, WIDTH, generator=gen).to(DTYPE),
    }


# -- the gaps that are CLOSED: the registered impl is bitwise the reference -------------------------


def test_embed_lookup_eager_is_bitwise_the_reference_embedding(inputs):
    """``embed.lookup:eager`` == ``ref.embedding``, so the composition proofs cover the real impl."""
    from isoexec.ops.embed.eager import embed_lookup
    from isoexec.ops.embed.reference import ref_embed_lookup

    got, want = embed_lookup(inputs["ids"], inputs["embed_w"]), ref_embed_lookup(inputs["ids"], inputs["embed_w"])
    assert _same_bits(got, want), f"{_differing(got, want)} elements differ"


def test_residual_add_eager_is_bitwise_the_reference_add(inputs):
    """``residual.add.attn:eager`` and ``residual.add.mlp:eager`` are ONE callable, and it == ``ref.add``.

    Both op ids resolve ``ops/residual/eager.py::residual_add_megatron``, so one comparison settles
    two rows of the readiness matrix; the parametrization names both so the matrix can be read off
    the test ids.
    """
    from isoexec.ops.residual.eager import residual_add_megatron
    from isoexec.ops.residual.reference import ref_residual_add

    got, want = residual_add_megatron(inputs["x"], inputs["y"]), ref_residual_add(inputs["x"], inputs["y"])
    assert _same_bits(got, want), f"{_differing(got, want)} elements differ"


def test_route_score_eager_is_bitwise_the_reference_router_gemm(inputs):
    """``moe.route_score:deterministic_fp32_gemm`` == ``ref.router_gemm_fp32``.

    Two different spellings of the same contraction (``torch.mm`` on upcast operands against
    ``@`` on them), and the fp32 router is the one place in the mixture where a last-bit difference
    reorders a top-k rather than perturbing a sum -- so it is worth knowing they agree exactly.
    """
    from isoexec.ops.moe.eager import route_score_fp32
    from isoexec.ops.moe.reference import ref_route_score

    got = route_score_fp32(inputs["x"], inputs["router_w"])
    want = ref_route_score(inputs["x"], inputs["router_w"])
    assert got.dtype == torch.float32
    assert _same_bits(got, want), f"{_differing(got, want)} elements differ"


# -- the gaps that are OPEN, asserted so they cannot close by accident ------------------------------


def test_mlp_act_eager_is_NOT_the_reference_and_the_gap_is_real(inputs):
    """Witness the eager activation-dtype SiLU/product rounds differing from the fp32 single-round reference."""
    from isoexec.ops.mlp.eager import silu_mul_megatron
    from isoexec.ops.mlp.reference import ref_silu_mul

    got, want = silu_mul_megatron(inputs["x"], inputs["y"]), ref_silu_mul(inputs["x"], inputs["y"])
    assert got.dtype == want.dtype == DTYPE
    assert not _same_bits(got, want)
    # Not one straggler element: the schedules genuinely differ over a measurable fraction.
    assert _differing(got, want) > got.numel() // 100


def test_the_swish_reference_matches_eager_swish_and_differs_from_sigmoid(inputs):
    """Explicit swish selection covers the production eager schedule on these operands."""
    from isoexec.ops.attention.attention_gate import attention_gate_swish
    from isoexec.ops.attention.reference import bindings, ref_attention_gate_swish
    from isoexec.ops.reference import ReferencePlan

    plan = ReferencePlan(selectors={"attention.gate": {"gate": "swish"}})
    selected = bindings(plan)["attention.gate"]
    assert selected.fn is ref_attention_gate_swish
    swish = selected.fn(inputs["x"], inputs["y"])
    eager = attention_gate_swish(inputs["x"], inputs["y"])
    sigmoid = bindings(ReferencePlan())["attention.gate"].fn(inputs["x"], inputs["y"])
    assert _same_bits(swish, eager)
    assert not _same_bits(swish, sigmoid)
    assert _differing(swish, sigmoid) > (swish.numel() * 9) // 10
