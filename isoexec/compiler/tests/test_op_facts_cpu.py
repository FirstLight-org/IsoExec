"""Collection of immutable op facts in a ModelContract."""

from dataclasses import FrozenInstanceError

import pytest
from torch import nn

from isoexec.compiler.contract.inference import OP_FACTS_BY_OP, OpFactsRule
from isoexec.compiler.contract.sema import build_model_contract
from isoexec.compiler.source import ModelSource
from isoexec.models.qwen3_5_moe.tests.tiny import TINY_MOE_HF
from isoexec.nn import GDN, Attention, LatentAttention
from isoexec.ops.attention.facts import GQAOpFacts, MLAOpFacts, validate_gqa_op_facts
from isoexec.ops.attention.state import infer_attention_state_specs, infer_mla_state_specs
from isoexec.ops.gdn.facts import GDNOpFacts
from isoexec.ops.gdn.state import infer_gdn_state_specs
from isoexec.schema import MLA, Attn, replicated
from isoexec.schema.op_metadata import GQAMetadata, MLAMetadata
from isoexec.state.contract import StateRef, StateSpec
from isoexec.state.validation import ContractError

GQA_OP_ID = "attention.core.gqa"
MLA_OP_ID = "attention.core.mla"
GDN_OP_ID = "gdn.core"


def gqa_metadata(*, num_q_heads=8, scale=0.37):
    return GQAMetadata(
        num_q_heads=num_q_heads,
        num_kv_heads=2,
        head_dim=4,
        rope_theta=10000.0,
        scale=scale,
    )


def mla_metadata():
    return MLAMetadata(
        num_q_heads=16,
        q_lora_rank=16,
        kv_lora_rank=32,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        rope_theta=10000.0,
        scale=0.29,
    )


class StatefulOp(nn.Module):
    def __init__(self, op_id, spec):
        super().__init__()
        self.stateful_op_id = op_id
        self.spec = spec

    def forward(self, *args, **kwargs):
        raise AssertionError("contract building must not execute forward")


def build(model, *, op_facts_registry=OP_FACTS_BY_OP):
    return build_model_contract(
        model,
        op_ids={GQA_OP_ID, MLA_OP_ID, GDN_OP_ID},
        inference_registry={
            GQA_OP_ID: infer_attention_state_specs,
            MLA_OP_ID: infer_mla_state_specs,
            GDN_OP_ID: infer_gdn_state_specs,
        },
        op_facts_registry=op_facts_registry,
    )


def test_contract_keeps_facts_for_each_attention_invocation():
    model = nn.ModuleDict(
        {
            "gqa_a": StatefulOp(GQA_OP_ID, gqa_metadata()),
            "gqa_b": StatefulOp(GQA_OP_ID, gqa_metadata(num_q_heads=16, scale=0.41)),
            "mla": StatefulOp(MLA_OP_ID, mla_metadata()),
        }
    )

    contract = build(model)

    assert contract.op_facts == {
        "gqa_a": GQAOpFacts(8, 2, 4, 4, 0.37),
        "gqa_b": GQAOpFacts(16, 2, 4, 4, 0.41),
        "mla": MLAOpFacts(16, 32, 8, 0.29),
    }
    assert {decl.state_id: tuple(spec.shape for spec in decl.specs) for decl in contract.state_decls} == {
        "gqa_a": ((2, 4), (2, 4)),
        "gqa_b": ((2, 4), (2, 4)),
        "mla": ((40,),),
    }


def test_production_attention_components_author_the_same_scale_execution_and_contract_use():
    gqa = Attention(hidden=16, heads=4, kv_heads=2, head_dim=4, rope_theta=10000.0)
    mla = LatentAttention(
        hidden=16,
        heads=4,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        rope_theta=10000.0,
    )

    contract = build(nn.ModuleDict({"gqa": gqa, "mla": mla}))

    assert contract.op_facts == {
        "gqa": GQAOpFacts(4, 2, 4, 4, gqa.scale),
        "mla": MLAOpFacts(4, 8, 4, mla.scale),
    }


def test_production_gdn_authors_one_aggregate_stateful_op():
    gdn = GDN(
        hidden=16,
        key_heads=2,
        value_heads=4,
        key_head_dim=4,
        value_head_dim=8,
        conv_kernel=4,
    )

    contract = build(gdn)

    assert contract.state_refs == (StateRef("", GDN_OP_ID, "<root>"),)
    assert contract.op_facts == {"": GDNOpFacts(2, 4, 4, 8, 4)}
    assert {spec.name: spec.shape for spec in contract.state_decls[0].specs} == {
        "conv_q": (2, 4, 3),
        "conv_k": (2, 4, 3),
        "conv_v": (4, 8, 3),
        "ssm": (4, 4, 8),
    }


def test_production_qwen_root_declares_one_contract_per_mixer():
    model = ModelSource("isoexec.models.qwen3_5_moe").build_meta(TINY_MOE_HF)

    contract = build(model)

    assert [(ref.module_path, ref.op_id) for ref in contract.state_refs] == [
        ("layers.0.mixer", GDN_OP_ID),
        ("layers.1.mixer", GDN_OP_ID),
        ("layers.2.mixer", GDN_OP_ID),
        ("layers.3.mixer", GQA_OP_ID),
    ]
    assert {decl.state_id: tuple(spec.name for spec in decl.specs) for decl in contract.state_decls} == {
        "layers.0.mixer": ("conv_q", "conv_k", "conv_v", "ssm"),
        "layers.1.mixer": ("conv_q", "conv_k", "conv_v", "ssm"),
        "layers.2.mixer": ("conv_q", "conv_k", "conv_v", "ssm"),
        "layers.3.mixer": ("k", "v"),
    }


def test_scale_changes_facts_without_changing_state_specs():
    first = build(StatefulOp(GQA_OP_ID, gqa_metadata(scale=0.37)))
    second = build(StatefulOp(GQA_OP_ID, gqa_metadata(scale=0.41)))

    assert first.state_decls == second.state_decls
    assert first.op_facts[""].scale == 0.37
    assert second.op_facts[""].scale == 0.41


def test_contract_snapshots_metadata_in_immutable_facts():
    module = StatefulOp(GQA_OP_ID, gqa_metadata())
    contract = build(module)

    module.spec = gqa_metadata(scale=0.41)
    assert contract.op_facts[""].scale == 0.37

    with pytest.raises(TypeError):
        contract.op_facts[""] = GQAOpFacts(8, 2, 4, 4, 0.41)
    with pytest.raises(FrozenInstanceError):
        contract.op_facts[""].scale = 0.41


def test_registry_keys_select_which_stateful_ops_produce_facts():
    other_op_id = "test.stateful"
    other = StatefulOp(other_op_id, object())
    gqa = StatefulOp(GQA_OP_ID, gqa_metadata())
    state_specs = (StateSpec("history", (2,), "bf16", replicated(), "per_request"),)

    contract = build_model_contract(
        nn.ModuleDict({"gqa": gqa, "other": other}),
        op_ids={GQA_OP_ID, other_op_id},
        inference_registry={GQA_OP_ID: infer_attention_state_specs, other_op_id: lambda _spec: state_specs},
        op_facts_registry=OP_FACTS_BY_OP,
    )

    assert {ref.module_path for ref in contract.state_refs} == {"gqa", "other"}
    assert contract.op_facts == {"gqa": GQAOpFacts(8, 2, 4, 4, 0.37)}


def test_invalid_facts_report_the_module_path_and_op_id():
    registry = {GQA_OP_ID: OpFactsRule(lambda _spec: object(), validate_gqa_op_facts)}

    with pytest.raises(ContractError, match=f"module_path 'attention'.*op_id '{GQA_OP_ID}'.*must be GQAOpFacts"):
        build(nn.ModuleDict({"attention": StatefulOp(GQA_OP_ID, gqa_metadata())}), op_facts_registry=registry)


def test_omitting_the_facts_registry_preserves_upstream_state_metadata():
    model = nn.ModuleDict(
        {
            "gqa": StatefulOp(
                GQA_OP_ID,
                Attn(num_q_heads=8, num_kv_heads=2, head_dim=4, rope_theta=10000.0),
            ),
            "mla": StatefulOp(
                MLA_OP_ID,
                MLA(
                    num_q_heads=16,
                    q_lora_rank=16,
                    kv_lora_rank=32,
                    qk_nope_head_dim=8,
                    qk_rope_head_dim=8,
                    v_head_dim=16,
                    rope_theta=10000.0,
                ),
            ),
        }
    )

    contract = build(model, op_facts_registry=None)

    assert contract.op_facts == {}
    assert {ref.op_id for ref in contract.state_refs} == {GQA_OP_ID, MLA_OP_ID}
