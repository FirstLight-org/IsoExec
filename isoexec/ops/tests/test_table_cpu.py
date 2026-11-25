"""Per-build selection and component binding safety."""

from dataclasses import FrozenInstanceError

import pytest

from isoexec.core.bindings import BoundOp
from isoexec.ops import _dispatch
from isoexec.ops.table import OpResolutionError, OpTable


def test_production_does_not_fall_back_to_reference(monkeypatch):
    monkeypatch.setattr(_dispatch, "ARCH", "cpu")
    with pytest.raises(OpResolutionError, match="no production implementation"):
        OpTable().bind("mm", site="lm.head")
    assert OpTable(mode="reference").bind("mm", site="lm.head").provider == "ops.reference"


@pytest.mark.parametrize(
    ("op", "facts"),
    (
        ("rope.apply", {"style": "neox", "rotary_dim": 64, "theta": 10000}),
        ("norms.qk", {"convention": "rms_zero_centered"}),
    ),
)
def test_family_resolvers_refuse_missing_rope_provider(monkeypatch, op, facts):
    from isoexec.ops._resolve import ResolutionError, resolve

    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    selected = _dispatch.selected
    monkeypatch.setattr(_dispatch, "selected", lambda family: None if family == "rope" else selected(family))
    with pytest.raises(ResolutionError, match="reference"):
        resolve(op, site="lm.attn", **facts)
    assert OpTable(mode="reference").bind(op, site="lm.attn", facts=facts).provider == "ops.reference"


@pytest.mark.parametrize("missing", ("qk_norm_rope", "QK_IMPL"))
def test_qk_norm_refuses_incomplete_fused_provider(monkeypatch, missing):
    from isoexec.ops.attention.rope_sm90 import integration
    from isoexec.ops.norms.resolve import resolve
    from isoexec.ops._resolve import ResolutionError

    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    monkeypatch.delattr(integration, missing)
    with pytest.raises(ResolutionError, match="no fused q/k norm"):
        resolve("norms.qk", site="lm.attn", convention="rms_zero_centered")


def test_build_context_is_immutable_and_refuses_architecture_drift(monkeypatch):
    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    table = OpTable()
    with pytest.raises(FrozenInstanceError):
        table.mode = "reference"
    monkeypatch.setattr(_dispatch, "ARCH", "cpu")
    with pytest.raises(OpResolutionError, match="created for 'sm90'"):
        table.bind("mm", site="lm.head")


def test_selection_facts_cannot_disagree_with_recorded_values():
    table = OpTable(mode="reference")
    with pytest.raises(OpResolutionError, match="disagrees with its recorded input"):
        table.bind("mm", site="lm.head", facts={"width": 16}, inputs={"width": 32})
    with pytest.raises(OpResolutionError, match="conflicting norm conventions"):
        table.bind("norms.rms.final", site="lm.norm", convention="rms", inputs={"convention": "rms_zero_centered"})


def test_binding_ownership_is_copied_from_explicit_metadata():
    owns = {"other": "fused"}
    binding = BoundOp("op", "impl", 1, "test", fn=lambda x: x, owns=owns)
    owns["other"] = "changed"
    assert binding.owns == {"other": "fused"}
    assert binding.ops == ("op", "other")
    with pytest.raises(ValueError, match="ownership disagrees"):
        BoundOp("op", "impl", 1, "test", fn=lambda x: x, owns={"op": "different"})


def test_registry_does_not_silently_omit_missing_family(monkeypatch):
    from isoexec.core import registry_build
    from isoexec.core.registry import RegistryError

    monkeypatch.setattr(registry_build, "_FAMILIES", ("missing_family",))
    with pytest.raises(RegistryError, match="missing_family"):
        registry_build.build_registry()


def test_registry_propagates_internal_import_failures(monkeypatch):
    from isoexec.core import registry_build

    def broken_import(name):
        raise ModuleNotFoundError("missing implementation dependency", name="missing_dependency")

    monkeypatch.setattr(registry_build.importlib, "import_module", broken_import)
    with pytest.raises(ModuleNotFoundError, match="missing implementation dependency"):
        registry_build.build_registry()


def test_reference_gate_selection_matches_independent_swish_math_and_gradients():
    import torch

    table = OpTable(mode="reference")
    binding = table.bind("attention.gate", site="lm.attn", facts={"gate": "swish"}, inputs={"gate": "swish"})
    assert binding.impl == "ref.swish_gate" and binding.inputs == {"gate": "swish"}
    output = torch.tensor([-2.0, 1.0, 3.0, -1.0, 0.5, 4.0], requires_grad=True)
    gate = torch.tensor([-5.0, -1.0, 0.0, 0.4, 3.0, 9.0], requires_grad=True)
    actual = binding.fn(output, gate)
    o64, g64 = output.detach().double(), gate.detach().double()
    sigmoid64 = 1 / (1 + torch.exp(-g64))
    expected = o64 * g64 * sigmoid64
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=2e-7)
    grad_output, grad_gate = torch.autograd.grad(actual.sum(), (output, gate))
    torch.testing.assert_close(grad_output.double(), g64 * sigmoid64, rtol=2e-6, atol=2e-7)
    expected_grad_gate = o64 * (sigmoid64 + g64 * sigmoid64 * (1 - sigmoid64))
    torch.testing.assert_close(grad_gate.double(), expected_grad_gate, rtol=2e-6, atol=2e-7)
    sigmoid = table.bind("attention.gate", site="lm.other", facts={"gate": "sigmoid"})
    assert sigmoid.impl == "ref.sigmoid_gate" and sigmoid.inputs == {"gate": "sigmoid"}
    assert not torch.equal(sigmoid.fn(output, gate), actual)


def test_reference_gate_refuses_unknown_variant_and_selector():
    table = OpTable(mode="reference")
    with pytest.raises(OpResolutionError, match="no reference binding for gate 'relu'"):
        table.bind("attention.gate", site="lm.attn", facts={"gate": "relu"})
    with pytest.raises(OpResolutionError, match="unsupported reference selectors"):
        table.bind("attention.gate", site="lm.attn", facts={"gait": "swish"})


def test_reference_norm_convention_in_facts_is_selected_and_recorded():
    binding = OpTable(mode="reference").bind(
        "norms.rms.final", site="lm.norm", facts={"convention": "rms", "width": 8}
    )
    assert binding.impl == "ref.rms"
    assert binding.inputs["convention"] == "rms"
