"""Portable GQA and MLA facts supplied to host projection."""

from dataclasses import dataclass, replace

import pytest

from isoexec.ops.attention.facts import (
    GQAOpFacts,
    MLAOpFacts,
    infer_gqa_op_facts,
    infer_mla_op_facts,
)
from isoexec.state.validation import ContractError


@dataclass(frozen=True, kw_only=True)
class GQAMetadata:
    num_q_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 4
    scale: float = 0.37


@dataclass(frozen=True, kw_only=True)
class MLAMetadata:
    num_q_heads: int = 16
    kv_lora_rank: int = 32
    qk_rope_head_dim: int = 8
    scale: float = 0.29


def test_attention_op_facts_preserve_the_final_authoring_scale():
    # GQA head_dim=4 would produce 0.5 if inference recomputed the usual default;
    # 0.37 proves the contract records the already-decided authoring value.
    assert infer_gqa_op_facts(GQAMetadata()) == GQAOpFacts(8, 2, 4, 4, 0.37)
    assert infer_mla_op_facts(MLAMetadata()) == MLAOpFacts(16, 32, 8, 0.29)


@pytest.mark.parametrize(
    "infer,metadata",
    [
        (infer_gqa_op_facts, replace(GQAMetadata(), scale=float("nan"))),
        (infer_gqa_op_facts, replace(GQAMetadata(), scale=1)),
        (infer_mla_op_facts, replace(MLAMetadata(), scale=float("inf"))),
        (infer_mla_op_facts, replace(MLAMetadata(), scale=None)),
    ],
)
def test_attention_op_facts_require_an_explicit_finite_float_scale(infer, metadata):
    with pytest.raises(ContractError, match="scale.*finite float"):
        infer(metadata)
