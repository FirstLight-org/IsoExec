from unittest.mock import Mock

import pytest
from torch import nn

from isoexec import schema
from isoexec.core.registry import ImplSpec, Registry
from isoexec.compiler.contract.contract import ModelContract
from isoexec.compiler.contract.sema import build_model_contract
from isoexec.ops.attention.state import infer_attention_state_specs
from isoexec.ops.gdn.conv_state import infer_conv_state_specs
from isoexec.ops.gdn.state import infer_gdn_state_specs
from isoexec.state.contract import StateDecl, StateRef, StateSpec
from isoexec.state.inference import STATE_SPECS_BY_OP
from isoexec.state.tests._fixtures import (
    GDN,
    REGISTERED_OP_IDS,
    Attention,
    LatentAttention,
    Model,
    attention_spec,
    gdn_spec,
    mla_spec,
)
from isoexec.state.validation import ContractError


def test_nn_module_model_definition_derives_gqa_state_at_attention_path():
    model = Model(Attention(attention_spec()))
    contract = build_model_contract(model, op_ids=REGISTERED_OP_IDS, inference_registry=STATE_SPECS_BY_OP)

    assert len(contract.state_decls) == len(contract.state_refs) == 1
    (decl,) = contract.state_decls
    (ref,) = contract.state_refs
    assert (decl.state_id, ref.state_id, ref.module_path, ref.op_id) == (
        "layers.0.attention",
        "layers.0.attention",
        "layers.0.attention",
        "attention.core.gqa",
    )
    assert {spec.name for spec in decl.specs} == {"k", "v"}
    for spec in decl.specs:
        assert (spec.shape, spec.dtype, spec.granularity, spec.tokens_per_entry, spec.retention) == (
            (4, 32),
            "bf16",
            "per_token",
            1,
            "full_history",
        )
        assert spec.shard == schema.col(dim=0)

    # ``Attention.mm`` is a registered stateless child. The GQA call is
    # represented by the containing Attention path plus StateRef.op_id.
    assert [ref.module_path for ref in contract.state_refs] == ["layers.0.attention"]
    assert "layers.0.attention.mm" in dict(model.named_modules())


def test_build_does_not_execute_or_trace_forward():
    class NeverExecute(Model):
        def forward(self, hidden):
            raise AssertionError("Sema must not execute forward")

    contract = build_model_contract(
        NeverExecute(Attention()), op_ids=REGISTERED_OP_IDS, inference_registry=STATE_SPECS_BY_OP
    )
    assert contract.state_refs[0].module_path == "layers.0.attention"


def test_gqa_and_mla_build_one_contract_through_the_inference_registry():
    model = nn.ModuleDict({"gqa": Attention(), "mla": LatentAttention(), "norm": nn.Identity()})
    model["norm"].op_id = "norms.rms.final"

    contract = build_model_contract(model, op_ids=REGISTERED_OP_IDS, inference_registry=STATE_SPECS_BY_OP)

    assert contract == ModelContract(
        (
            StateDecl(
                "gqa",
                (
                    StateSpec("k", (4, 32), "bf16", schema.col(dim=0), "per_token", 1, "full_history"),
                    StateSpec("v", (4, 32), "bf16", schema.col(dim=0), "per_token", 1, "full_history"),
                ),
            ),
            StateDecl(
                "mla", (StateSpec("kv_latent", (40,), "bf16", schema.replicated(), "per_token", 1, "full_history"),)
            ),
        ),
        (StateRef("gqa", "attention.core.gqa", "gqa"), StateRef("mla", "attention.core.mla", "mla")),
    )


def test_gdn_builds_separate_convolution_and_recurrence_state():
    contract = build_model_contract(
        GDN(gdn_spec(conv_kernel=5)), op_ids=REGISTERED_OP_IDS, inference_registry=STATE_SPECS_BY_OP
    )
    assert contract == ModelContract(
        (
            StateDecl(
                "conv",
                (
                    StateSpec("conv_q", (4, 16, 4), "bf16", schema.col(dim=0), "per_request"),
                    StateSpec("conv_k", (4, 16, 4), "bf16", schema.col(dim=0), "per_request"),
                    StateSpec("conv_v", (8, 32, 4), "bf16", schema.col(dim=0), "per_request"),
                ),
            ),
            StateDecl("core", (StateSpec("ssm", (8, 16, 32), "fp32", schema.col(dim=0), "per_request"),)),
        ),
        (StateRef("conv", "conv.short", "conv"), StateRef("core", "gdn.core", "core")),
    )


def test_each_module_path_derives_exactly_once():
    infer = Mock(wraps=infer_attention_state_specs)
    model = Model(Attention(), Attention())

    contract = build_model_contract(model, op_ids=REGISTERED_OP_IDS, inference_registry={"attention.core.gqa": infer})
    assert len(contract.state_decls) == 2
    assert infer.call_count == 2


@pytest.mark.parametrize(
    "op_id,message",
    [("unknown", "unknown op_id"), ("mm", "missing state inference"), ("norms.rms.final", "missing state inference")],
)
def test_unknown_op_or_missing_inference_refuses(op_id, message):
    module = Attention()
    module.stateful_op_id = op_id
    with pytest.raises(ContractError, match=message):
        build_model_contract(module, op_ids=REGISTERED_OP_IDS, inference_registry=STATE_SPECS_BY_OP)


@pytest.mark.parametrize("op_id", [(), ("attention.core.gqa",), ["attention.core.gqa"], "", None])
def test_stateful_op_id_must_be_one_non_empty_string(op_id):
    module = Attention()
    module.stateful_op_id = op_id
    with pytest.raises(ContractError, match="stateful_op_id"):
        build_model_contract(module, op_ids=REGISTERED_OP_IDS, inference_registry=STATE_SPECS_BY_OP)


@pytest.mark.parametrize(
    "module,message",
    [
        (Attention(gdn_spec()), "requires Attn"),
        (Attention(mla_spec()), "requires Attn"),
        (LatentAttention(attention_spec()), "requires MLA"),
        (GDN(attention_spec()), "requires GDN"),
        (Attention(attention_spec(num_kv_heads=0)), "num_kv_heads"),
        (GDN(gdn_spec(num_value_heads=True)), "num_value_heads"),
        (GDN(gdn_spec(conv_kernel=1)), "conv_kernel"),
        (GDN(gdn_spec(num_key_heads=3)), "num_value_heads.*divisible by num_key_heads"),
    ],
)
def test_invalid_model_metadata_refuses(module, message):
    with pytest.raises(ContractError, match=message):
        build_model_contract(module, op_ids=REGISTERED_OP_IDS, inference_registry=STATE_SPECS_BY_OP)


@pytest.mark.parametrize("op_id", ["conv.short", "gdn.core"])
def test_gdn_requires_inference_for_both_stateful_children(op_id):
    inference_registry = {key: infer for key, infer in STATE_SPECS_BY_OP.items() if key != op_id}
    with pytest.raises(ContractError, match=f"missing state inference for op_id '{op_id}'"):
        build_model_contract(GDN(), op_ids=REGISTERED_OP_IDS, inference_registry=inference_registry)


def test_missing_model_metadata_refuses():
    module = Attention()
    del module.spec
    with pytest.raises(ContractError, match="missing module.spec"):
        build_model_contract(module, op_ids=REGISTERED_OP_IDS, inference_registry=STATE_SPECS_BY_OP)


@pytest.mark.parametrize("derived", [(), [], None])
def test_derivation_must_return_a_non_empty_tuple(derived):
    with pytest.raises(ContractError, match="non-empty tuple"):
        build_model_contract(
            Attention(), op_ids=REGISTERED_OP_IDS, inference_registry={"attention.core.gqa": lambda _spec: derived}
        )


@pytest.mark.parametrize("nested", [False, True])
def test_same_stateful_module_at_multiple_module_paths_refuses(nested):
    attention = Attention()
    module = nn.ModuleDict({"attention": attention}) if nested else attention
    model = nn.ModuleDict({"a": module, "b": module})
    infer = Mock(wraps=infer_attention_state_specs)
    with pytest.raises(ContractError, match="multiple module paths"):
        build_model_contract(model, op_ids=REGISTERED_OP_IDS, inference_registry={"attention.core.gqa": infer})
    infer.assert_called_once_with(attention.spec)


@pytest.mark.parametrize("shadow,message", [(False, "cannot be resolved"), (True, "different module")])
def test_module_path_must_resolve_to_the_declaring_instance_before_inference(shadow, message):
    model = nn.Module()
    if shadow:
        model.block = nn.ModuleDict({"attention": nn.Identity()})
    setattr(model, "block.attention", Attention())
    infer = Mock(wraps=infer_attention_state_specs)

    with pytest.raises(ContractError, match=f"module_path 'block.attention'.*{message}"):
        build_model_contract(model, op_ids=REGISTERED_OP_IDS, inference_registry={"attention.core.gqa": infer})

    infer.assert_not_called()


@pytest.mark.parametrize("module", [nn.Identity(), Attention()])
def test_explicit_state_id_is_unsupported(module):
    module.state_id = "state"
    with pytest.raises(ContractError, match="explicit state_id is unsupported"):
        build_model_contract(module, op_ids=REGISTERED_OP_IDS, inference_registry=STATE_SPECS_BY_OP)


def test_stateless_modules_are_ignored():
    assert build_model_contract(
        nn.Sequential(nn.Identity(), nn.Linear(2, 2)), op_ids=REGISTERED_OP_IDS, inference_registry={}
    ) == ModelContract((), ())


@pytest.mark.parametrize("op_id", ["mm", "norms.rms.final", "attention.core.gqa"])
def test_plain_op_id_does_not_declare_state(op_id):
    module = nn.Identity()
    module.op_id = op_id
    module.spec = object()
    infer = Mock()

    contract = build_model_contract(module, op_ids=REGISTERED_OP_IDS, inference_registry={"attention.core.gqa": infer})

    assert contract == ModelContract((), ())
    infer.assert_not_called()


def test_equal_geometry_and_module_types_keep_independent_state_ids():
    contract = build_model_contract(
        Model(Attention(), Attention()), op_ids=REGISTERED_OP_IDS, inference_registry=STATE_SPECS_BY_OP
    )

    assert {decl.state_id for decl in contract.state_decls} == {
        "layers.0.attention",
        "layers.1.attention",
    }
    assert {ref.module_path: ref.state_id for ref in contract.state_refs} == {
        "layers.0.attention": "layers.0.attention",
        "layers.1.attention": "layers.1.attention",
    }


def test_module_traversal_order_is_not_a_binding_key():
    def build(names):
        return nn.ModuleDict({name: Attention() for name in names})

    assert build_model_contract(
        build(("a", "b")), op_ids=REGISTERED_OP_IDS, inference_registry=STATE_SPECS_BY_OP
    ) == build_model_contract(build(("b", "a")), op_ids=REGISTERED_OP_IDS, inference_registry=STATE_SPECS_BY_OP)


def test_stateful_op_id_selects_inference_when_plain_op_id_is_present():
    module = Attention()
    module.op_id = "mm"

    contract = build_model_contract(module, op_ids=REGISTERED_OP_IDS, inference_registry=STATE_SPECS_BY_OP)

    assert contract.state_refs == (StateRef("", "attention.core.gqa", "<root>"),)
    assert {spec.name for spec in contract.state_decls[0].specs} == {"k", "v"}


def test_build_requires_nn_module():
    with pytest.raises(ContractError, match="requires an nn.Module"):
        build_model_contract(object(), op_ids=REGISTERED_OP_IDS, inference_registry=STATE_SPECS_BY_OP)


def test_state_contract_does_not_replace_legacy_schema():
    assert schema.StateSpec is not StateSpec
    assert "ModelContract" not in schema.__all__


def test_injected_inference_accepts_an_op_with_opaque_metadata():
    module = nn.Module()
    module.stateful_op_id = "test.stateful"
    module.spec = object()
    specs = (StateSpec("history", (2, 3), "bf16", schema.replicated(), "per_request"),)
    infer = Mock(return_value=specs)
    op_registry = Registry()
    op_registry.add(ImplSpec(op=module.stateful_op_id, impl_id="test", version=1, supported_archs=frozenset({"cpu"})))

    contract = build_model_contract(
        module, op_ids=op_registry.ops.keys(), inference_registry={module.stateful_op_id: infer}
    )

    assert contract == ModelContract(
        (StateDecl("<root>", specs),),
        (StateRef("", "test.stateful", "<root>"),),
    )
    infer.assert_called_once_with(module.spec)


def test_inference_mapping_does_not_register_an_op():
    module = Attention()
    module.stateful_op_id = "test.stateful"
    infer = Mock(wraps=infer_attention_state_specs)

    with pytest.raises(ContractError, match="unknown op_id 'test.stateful'"):
        build_model_contract(module, op_ids=REGISTERED_OP_IDS, inference_registry={module.stateful_op_id: infer})

    infer.assert_not_called()


def test_missing_inference_does_not_fall_back_to_default_mapping():
    with pytest.raises(ContractError, match="missing state inference for op_id 'attention.core.gqa'"):
        build_model_contract(Attention(), op_ids=REGISTERED_OP_IDS, inference_registry={})


@pytest.mark.parametrize("infer", [None, (), "infer_attention_state_specs"])
def test_non_callable_inference_refuses(infer):
    with pytest.raises(ContractError, match="inference for op_id 'attention.core.gqa' must be callable"):
        build_model_contract(Attention(), op_ids=REGISTERED_OP_IDS, inference_registry={"attention.core.gqa": infer})


def test_one_parent_with_multiple_stateful_children_and_stateless_work():
    block = nn.ModuleDict(
        {
            "projection": nn.Linear(2, 2, device="meta"),
            "attention": Attention(),
            "gdn": GDN(),
            "output": nn.Identity(),
        }
    )
    block.op_id = "mm"
    block["projection"].op_id = "mm"
    model = nn.ModuleDict({"block": block, "norm": nn.Identity()})
    model["norm"].op_id = "norms.rms.final"
    model["projection_alias"] = block["projection"]
    attention_infer = Mock(wraps=infer_attention_state_specs)
    conv_infer = Mock(wraps=infer_conv_state_specs)
    gdn_infer = Mock(wraps=infer_gdn_state_specs)

    contract = build_model_contract(
        model,
        op_ids=REGISTERED_OP_IDS,
        inference_registry={"attention.core.gqa": attention_infer, "conv.short": conv_infer, "gdn.core": gdn_infer},
    )

    assert contract.state_refs == (
        StateRef("block.attention", "attention.core.gqa", "block.attention"),
        StateRef("block.gdn.conv", "conv.short", "block.gdn.conv"),
        StateRef("block.gdn.core", "gdn.core", "block.gdn.core"),
    )
    assert {decl.state_id: {spec.name for spec in decl.specs} for decl in contract.state_decls} == {
        "block.attention": {"k", "v"},
        "block.gdn.conv": {"conv_q", "conv_k", "conv_v"},
        "block.gdn.core": {"ssm"},
    }
    attention_infer.assert_called_once_with(block["attention"].spec)
    conv_infer.assert_called_once_with(block["gdn"].spec)
    gdn_infer.assert_called_once_with(block["gdn"].spec)
    assert not hasattr(block["attention"], "state_id")
    assert not hasattr(block["gdn"].conv, "state_id")
    assert not hasattr(block["gdn"].core, "state_id")


def test_rebuilding_uses_current_metadata_without_mutating_the_previous_contract():
    module = Attention()
    first = build_model_contract(module, op_ids=REGISTERED_OP_IDS, inference_registry=STATE_SPECS_BY_OP)
    module.spec = attention_spec(head_dim=64)
    second = build_model_contract(module, op_ids=REGISTERED_OP_IDS, inference_registry=STATE_SPECS_BY_OP)

    assert first.state_refs == second.state_refs
    assert {spec.shape for spec in first.state_decls[0].specs} == {(4, 32)}
    assert {spec.shape for spec in second.state_decls[0].specs} == {(4, 64)}


def test_inference_failure_propagates_without_fallback():
    error = RuntimeError("inference failed")
    infer = Mock(side_effect=error)
    with pytest.raises(RuntimeError) as caught:
        build_model_contract(Attention(), op_ids=REGISTERED_OP_IDS, inference_registry={"attention.core.gqa": infer})
    assert caught.value is error
