"""State-only integration across op families and registry inputs."""

from types import MappingProxyType
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from isoexec.core.registry import ImplSpec
from isoexec.core.registry_build import build_registry
from isoexec.compiler.contract.contract import ModelContract
from isoexec.compiler.contract.sema import build_model_contract
from isoexec.schema import col, replicated
from isoexec.state.contract import StateDecl, StateRef, StateSpec
from isoexec.state.inference import STATE_SPECS_BY_OP
from isoexec.state.tests._fixtures import GDN, REGISTERED_OP_IDS, Attention, LatentAttention, attention_spec


class Model(nn.Module):
    """Metadata fixture with all supported stateful ops and unrelated parameters."""

    def __init__(self, *, device):
        super().__init__()
        self.layers = nn.ModuleList(
            (
                nn.ModuleDict({"attention": Attention(), "gdn": GDN()}),
                nn.ModuleDict(
                    {
                        "attention": Attention(attention_spec(num_q_heads=16, num_kv_heads=8, head_dim=64)),
                        "mla": LatentAttention(),
                    }
                ),
            )
        )
        self.projection = nn.Linear(8, 8, device=device)
        self.projection.op_id = "mm"

    def forward(self, *args, **kwargs):
        raise AssertionError("contract building must not execute or trace forward")


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_all_stateful_ops_build_the_same_contract_without_weight_values(device):
    model = Model(device=device)
    if device == "cpu":
        with torch.no_grad():
            for parameter in model.projection.parameters():
                parameter.fill_(float("nan"))

    contract = build_model_contract(model, op_ids=REGISTERED_OP_IDS, inference_registry=STATE_SPECS_BY_OP)

    assert {decl.state_id: decl.specs for decl in contract.state_decls} == {
        "layers.0.attention": (
            StateSpec("k", (4, 32), "bf16", col(dim=0), "per_token", 1, "full_history"),
            StateSpec("v", (4, 32), "bf16", col(dim=0), "per_token", 1, "full_history"),
        ),
        "layers.0.gdn.conv": (
            StateSpec("conv_q", (4, 16, 3), "bf16", col(dim=0), "per_request"),
            StateSpec("conv_k", (4, 16, 3), "bf16", col(dim=0), "per_request"),
            StateSpec("conv_v", (8, 32, 3), "bf16", col(dim=0), "per_request"),
        ),
        "layers.0.gdn.core": (StateSpec("ssm", (8, 16, 32), "fp32", col(dim=0), "per_request"),),
        "layers.1.attention": (
            StateSpec("k", (8, 64), "bf16", col(dim=0), "per_token", 1, "full_history"),
            StateSpec("v", (8, 64), "bf16", col(dim=0), "per_token", 1, "full_history"),
        ),
        "layers.1.mla": (StateSpec("kv_latent", (40,), "bf16", replicated(), "per_token", 1, "full_history"),),
    }
    assert contract.state_refs == (
        StateRef("layers.0.attention", "attention.core.gqa", "layers.0.attention"),
        StateRef("layers.0.gdn.conv", "conv.short", "layers.0.gdn.conv"),
        StateRef("layers.0.gdn.core", "gdn.core", "layers.0.gdn.core"),
        StateRef("layers.1.attention", "attention.core.gqa", "layers.1.attention"),
        StateRef("layers.1.mla", "attention.core.mla", "layers.1.mla"),
    )


def test_registered_op_extends_a_mixed_model_through_a_read_only_mapping():
    model = Model(device="meta")
    op_registry = build_registry()
    original = build_model_contract(model, op_ids=op_registry.ops.keys(), inference_registry=STATE_SPECS_BY_OP)

    model.extra = nn.Module()
    model.extra.stateful_op_id = "test.stateful"
    model.extra.spec = object()
    op_registry.add(
        ImplSpec(op=model.extra.stateful_op_id, impl_id="test", version=1, supported_archs=frozenset({"cpu"}))
    )
    specs = (StateSpec("history", (2, 3), "bf16", replicated(), "per_request"),)
    infer_state_specs = Mock(return_value=specs)
    inference_registry = MappingProxyType({**STATE_SPECS_BY_OP, model.extra.stateful_op_id: infer_state_specs})

    contract = build_model_contract(model, op_ids=op_registry.ops.keys(), inference_registry=inference_registry)

    decls = {decl.state_id: decl for decl in contract.state_decls}
    refs = {ref.module_path: ref for ref in contract.state_refs}
    assert decls["extra"] == StateDecl("extra", specs)
    assert refs["extra"] == StateRef("extra", "test.stateful", "extra")
    assert (
        ModelContract(
            tuple(decl for decl in contract.state_decls if decl.state_id != "extra"),
            tuple(ref for ref in contract.state_refs if ref.module_path != "extra"),
        )
        == original
    )
    infer_state_specs.assert_called_once_with(model.extra.spec)
    assert infer_state_specs.call_args.args[0] is model.extra.spec
