"""Portable GQA and MLA state requirements, without host or kernel execution."""

from dataclasses import replace

import pytest

from isoexec.ops.attention.state import infer_attention_state_specs, infer_mla_state_specs
from isoexec.schema import MLA, Attn, col, replicated
from isoexec.state.contract import StateSpec
from isoexec.state.validation import ContractError

ATTN = Attn(num_q_heads=8, num_kv_heads=4, head_dim=32, rope_theta=10000.0)
MLA_SPEC = MLA(
    num_q_heads=8,
    q_lora_rank=16,
    kv_lora_rank=32,
    qk_nope_head_dim=8,
    qk_rope_head_dim=8,
    v_head_dim=16,
    rope_theta=10000.0,
)


@pytest.mark.parametrize("num_kv_heads,head_dim,shape", [(8, 32, (8, 32)), (2, 64, (2, 64)), (1, 128, (1, 128))])
def test_gqa_derives_separate_k_and_v(num_kv_heads, head_dim, shape):
    specs = infer_attention_state_specs(replace(ATTN, num_kv_heads=num_kv_heads, head_dim=head_dim))

    assert specs == (
        StateSpec("k", shape, "bf16", col(dim=0), "per_token", 1, "full_history"),
        StateSpec("v", shape, "bf16", col(dim=0), "per_token", 1, "full_history"),
    )


@pytest.mark.parametrize("kv_lora_rank,qk_rope_head_dim,shape", [(32, 8, (40,)), (512, 64, (576,))])
def test_mla_derives_replicated_latent_kv(kv_lora_rank, qk_rope_head_dim, shape):
    specs = infer_mla_state_specs(replace(MLA_SPEC, kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope_head_dim))

    assert specs == (StateSpec("kv_latent", shape, "bf16", replicated(), "per_token", 1, "full_history"),)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_q_heads", 0),
        ("num_q_heads", True),
        ("num_kv_heads", 0),
        ("num_kv_heads", 2.0),
        ("head_dim", -1),
        ("head_dim", "32"),
    ],
)
def test_gqa_rejects_invalid_heads_or_dimension(field, value):
    with pytest.raises(ContractError, match=field):
        infer_attention_state_specs(replace(ATTN, **{field: value}))


def test_gqa_rejects_query_heads_that_do_not_group_over_kv_heads():
    with pytest.raises(ContractError, match="num_q_heads.*divisible by num_kv_heads"):
        infer_attention_state_specs(replace(ATTN, num_q_heads=6))


@pytest.mark.parametrize(
    "field,value",
    [
        ("kv_lora_rank", -1),
        ("kv_lora_rank", 0),
        ("kv_lora_rank", True),
        ("kv_lora_rank", "32"),
        ("qk_rope_head_dim", -2),
        ("qk_rope_head_dim", 0),
        ("qk_rope_head_dim", True),
        ("qk_rope_head_dim", 8.0),
    ],
)
def test_mla_validates_each_dimension_before_adding_them(field, value):
    with pytest.raises(ContractError, match=field):
        infer_mla_state_specs(replace(MLA_SPEC, **{field: value}))


def test_mla_rejects_odd_rotary_width():
    with pytest.raises(ContractError, match="qk_rope_head_dim.*even"):
        infer_mla_state_specs(replace(MLA_SPEC, qk_rope_head_dim=3))


@pytest.mark.parametrize(
    "infer_state_specs,metadata,expected_type",
    [(infer_attention_state_specs, MLA_SPEC, "Attn"), (infer_mla_state_specs, ATTN, "MLA")],
)
def test_inference_rejects_wrong_metadata_type(infer_state_specs, metadata, expected_type):
    with pytest.raises(ContractError, match=f"requires {expected_type} metadata"):
        infer_state_specs(metadata)


def test_gqa_state_specs_do_not_depend_on_query_head_count_or_projection_options():
    changed = replace(ATTN, num_q_heads=16, rope_theta=100000.0, qk_norm=None, qkv_bias=True, out_gate=None)

    assert infer_attention_state_specs(changed) == infer_attention_state_specs(ATTN)


def test_mla_state_specs_do_not_depend_on_query_or_output_projection_dimensions():
    changed = replace(MLA_SPEC, num_q_heads=16, q_lora_rank=32, qk_nope_head_dim=16, v_head_dim=32, rope_theta=100000.0)

    assert infer_mla_state_specs(changed) == infer_mla_state_specs(MLA_SPEC)
