"""Bind through a direct StateRef or a lowering-provided authored/native path mapping."""

import pytest
from torch import nn

from isoexec.core.bindings import BoundOp
from isoexec.compiler.contract.binding import (
    StatefulOpModulePathMapping,
    bind_stateful_op,
    validate_stateful_op_module_path_mappings,
)
from isoexec.compiler.contract.contract import ModelContract
from isoexec.nn import GDN, Attention, LatentAttention
from isoexec.ops.table import OpTable
from isoexec.schema import Axis, Placement, replicated
from isoexec.state.contract import StateDecl, StateRef, StateSpec
from isoexec.state.validation import ContractError

OP_ID = "attention.core.gqa"
MLA_OP_ID = "attention.core.mla"
STATE_SPECS = (StateSpec("state", (1,), "bf16", replicated(), "per_request"),)


class StatefulModule(nn.Module):
    def __init__(self, op_id=OP_ID):
        super().__init__()
        self.stateful_op_id = op_id
        self.bound = None

    @property
    def spec(self):
        raise AssertionError("binding must not inspect module.spec")

    def bind_stateful_op(self, op_id, fn):
        if op_id != self.stateful_op_id:
            raise ContractError("op_id does not belong to this module")
        if self.bound is not None:
            raise ContractError("stateful op is already bound")
        self.bound = (op_id, fn)


def _contract():
    return ModelContract(
        state_decls=(
            StateDecl("layers.0.mixer", STATE_SPECS),
            StateDecl("layers.1.mixer", STATE_SPECS),
        ),
        state_refs=(
            StateRef("layers.0.mixer", OP_ID, "layers.0.mixer"),
            StateRef("layers.1.mixer", MLA_OP_ID, "layers.1.mixer"),
        ),
    )


MAPPINGS = (
    StatefulOpModulePathMapping("layers.0.mixer", "native.attention", OP_ID),
    StatefulOpModulePathMapping("layers.1.mixer", "native.latent_attention", MLA_OP_ID),
)


def _engine_placement():
    one = Axis(1, 0, None)
    return Placement(tp=one, ep=one, dp=one, pp=one, etp=one, cp=one, side="engine", leaves=8)


@pytest.mark.parametrize("module_path", ["", "layer.attention"])
def test_binding_resolves_root_or_nested_module_without_reading_spec(module_path):
    target = StatefulModule()
    if module_path:
        model = nn.Module()
        model.layer = nn.Module()
        model.layer.attention = target
    else:
        model = target
    fn = lambda: None

    bind_stateful_op(model, StateRef(module_path, OP_ID, module_path or "<root>"), fn)

    assert target.bound == (OP_ID, fn)


def test_binding_rejects_a_missing_or_different_target():
    model = nn.Module()
    ref = StateRef("attention", OP_ID, "attention")

    with pytest.raises(ContractError, match="module_path 'attention'.*no longer resolves"):
        bind_stateful_op(model, ref, lambda: None)

    model.attention = StatefulModule("attention.core.mla")
    with pytest.raises(ContractError, match="op_id 'attention.core.gqa'.*declares stateful_op_id 'attention.core.mla'"):
        bind_stateful_op(model, ref, lambda: None)


def test_binding_requires_the_explicit_module_method_and_a_callable():
    model = nn.Module()
    model.attention = nn.Module()
    model.attention.stateful_op_id = OP_ID
    ref = StateRef("attention", OP_ID, "attention")

    with pytest.raises(ContractError, match="must provide callable bind_stateful_op"):
        bind_stateful_op(model, ref, lambda: None)

    target = StatefulModule()
    with pytest.raises(ContractError, match="stateful op binding must be callable"):
        bind_stateful_op(target, StateRef("", OP_ID, "<root>"), object())
    assert target.bound is None


def test_module_refusal_prevents_repeated_binding():
    target = StatefulModule()
    ref = StateRef("", OP_ID, "<root>")
    first = lambda: None

    bind_stateful_op(target, ref, first)

    with pytest.raises(ContractError, match="module_path ''.*stateful op is already bound"):
        bind_stateful_op(target, ref, lambda: None)
    assert target.bound == (OP_ID, first)


def test_lowering_mappings_bind_different_native_paths_without_inference():
    native_model = nn.Module()
    native_model.native = nn.Module()
    native_model.native.attention = StatefulModule(OP_ID)
    native_model.native.latent_attention = StatefulModule(MLA_OP_ID)
    functions = {"layers.0.mixer": lambda: None, "layers.1.mixer": lambda: None}
    contract = _contract()
    validate_stateful_op_module_path_mappings(contract, MAPPINGS, native_model=native_model)
    mapping_by_path = {mapping.authored_module_path: mapping for mapping in MAPPINGS}
    for ref in contract.state_refs:
        bind_stateful_op(
            native_model,
            ref,
            functions[ref.module_path],
            mapping=mapping_by_path[ref.module_path],
        )

    assert native_model.native.attention.bound == (OP_ID, functions["layers.0.mixer"])
    assert native_model.native.latent_attention.bound == (MLA_OP_ID, functions["layers.1.mixer"])


def test_production_stateful_components_retain_the_complete_bound_op():
    modules = (
        Attention(hidden=16, heads=4, kv_heads=2, head_dim=4, rope_theta=10000.0),
        LatentAttention(
            hidden=16,
            heads=4,
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            rope_theta=10000.0,
        ),
        GDN(
            hidden=16,
            key_heads=2,
            value_heads=4,
            key_head_dim=4,
            value_head_dim=8,
            conv_kernel=4,
        ),
    )
    for module in modules:
        module.apply_placement(_engine_placement(), site="mixer", op_table=OpTable(mode="reference"))
        owns = {module.stateful_op_id: "paged"}
        if module.stateful_op_id == "gdn.core":
            owns.update({op: "paged" for op in ("gdn.l2norm", "gdn.gating", "conv.short")})
        binding = BoundOp(
            module.stateful_op_id,
            "paged",
            7,
            "test.runtime",
            {"scale": module.scale} if hasattr(module, "scale") else {},
            lambda *args: args,
            owns=owns,
        )

        bind_stateful_op(module, StateRef("", module.stateful_op_id, "<root>"), binding)

        assert module.slots.resolved()["core"] is binding
