"""One non-Qwen fixture exercises dataflow preservation and author diagnostics."""

import json
from collections.abc import Callable
from types import MethodType

import pytest
import torch
from torch import nn

import isoexec.nn as inn
from isoexec.compiler.capture.capture import capture
from isoexec.compiler.capture.graph import PipelineError
from isoexec.ops.table import OpTable
from isoexec.runtimes.parallel import trivial_placement
from isoexec.schema import make_sequence_context


class Block(nn.Module):
    def __init__(self, device="meta"):
        super().__init__()
        self.norm = inn.RMSNorm(hidden=16, device=device)
        self.ffn = inn.MLP(hidden=16, intermediate=32, device=device)
        self.mix: nn.Module | None = None
        self.repeat = self.skip = False

    def forward(self, x, seq):
        if self.mix is not None and not self.skip:
            x = x + self.mix(self.norm(x), seq)
            if self.repeat:
                x = x + self.mix(x, seq)
        return x + self.ffn(self.norm(x), seq)


class Model(nn.Module):
    pipeline_layers: Callable[[], object]

    def __init__(self, device="meta"):
        super().__init__()
        self.lexicon = inn.Embedding(vocab=32, hidden=16, device=device)
        self.units = nn.ModuleList([Block(device), Block(device)])
        self.finish = inn.RMSNorm(hidden=16, role="final", device=device)
        self.readout = inn.LMHead(vocab=32, hidden=16, tied_to=self.lexicon.weight, device=device)
        self.scale = 0.5
        self.parallel = self.dynamic = self.suffix = self.inplace = self.no_grad = False
        self.seq_write = self.slice_inplace = self.module_write = False

    def forward(self, input_ids, seq, *, factor=1.0):
        if self.module_write:
            self.scale += 1
        if self.seq_write:
            seq.positions = seq.positions + 1
        if self.no_grad:
            with torch.no_grad():
                x = self.lexicon(input_ids)
        else:
            x = self.lexicon(input_ids)
        if self.dynamic and x.sum() > 0:
            x = x + 1
        if self.inplace:
            x += 1
        if self.slice_inplace:
            x[:, :1] += 1
        if self.parallel:
            x = self.units[0](x, seq) + self.units[1](x, seq)
        else:
            for unit in self.units:
                x = unit(x, seq)
        logits = self.readout(self.finish(x) * self.scale * factor, replicated_tokens=False)
        return logits + 1 if self.suffix else logits


def example_inputs(tokens=4):
    ids = torch.arange(tokens, dtype=torch.int64) + 1
    seq = make_sequence_context(
        torch.arange(tokens), torch.tensor([0, tokens], dtype=torch.int32), max_q=tokens, input_ids=ids,
    )
    return ids, seq


@pytest.mark.parametrize("inplace", [False, True], ids=["plain", "whole_tensor_iadd"])
def test_capture_preserves_forward_gradients_ties_and_op_table(inplace):
    model = Model(device="cpu")
    model.inplace = inplace
    ids, seq = example_inputs()
    result = capture(model, (ids, seq))
    assert result.layer_source == "inferred"
    assert [layer.path for layer in result.layers] == ["units.0", "units.1"]
    multiply = next(
        node for node in result.program.graph.nodes
        if node.target is torch.ops.aten.mul.Tensor and node.args[1] == 0.5
    )
    assert isinstance(multiply.args[0], torch.fx.Node) and multiply.args[1] == 0.5
    assert all(multiply.name not in layer.nodes for layer in result.layers)
    head = next(node for node in result.program.graph.nodes if node.name == result.head)
    assert head.kwargs == {"replicated_tokens": False}
    assert result.inventory.ties[0].aliases == ("readout.weight",)
    assert result.program.get_submodule("lexicon").weight is model.lexicon.weight
    assert result.program.get_submodule("readout").weight is model.lexicon.weight

    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.04)
    table = OpTable(mode="reference")
    inn.place(model, trivial_placement(side="trainer"), op_table=table)
    assert result.program.get_submodule("units.0.ffn") is model.units[0].ffn
    assert result.program.get_submodule("units.0.ffn")._op_table is table
    def run(forward):
        model.zero_grad(set_to_none=True)
        logits = forward(ids, seq)
        logits.square().mean().backward()
        assert torch.isfinite(logits).all()
        grads = {}
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            grads[name] = parameter.grad.clone()
        return logits.detach(), grads

    expected, expected_grads = run(model)
    actual, actual_grads = run(result.program)
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    assert expected_grads.keys() == actual_grads.keys()
    for name in expected_grads:
        assert torch.equal(expected_grads[name].view(torch.uint8), actual_grads[name].view(torch.uint8)), name
    # Negative control: dropping the authored factor must change the answer.
    model.scale = 1.0
    assert not torch.equal(model(ids, seq), actual)


def test_instance_forward_override_is_captured_by_the_private_export_clone():
    model = Model(device="cpu")

    def forward(this, input_ids, seq, *, factor=1.0):
        return Model.forward(this, input_ids, seq, factor=factor * 2)

    model.forward = MethodType(forward, model)
    inputs = example_inputs()
    captured = capture(model, inputs)
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.04)
    inn.place(model, trivial_placement(side="trainer"), op_table=OpTable(mode="reference"))
    def run(forward):
        model.zero_grad(set_to_none=True)
        output = forward(*inputs)
        output.float().square().mean().backward()
        gradients = {
            name: parameter.grad.clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return output.detach(), gradients

    expected, expected_gradients = run(model)
    actual, actual_gradients = run(captured.program)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert actual_gradients.keys() == expected_gradients.keys()
    for name in expected_gradients:
        assert torch.equal(actual_gradients[name].view(torch.uint8), expected_gradients[name].view(torch.uint8)), name


def test_module_hooks_remain_an_explicit_preflight_refusal():
    model = Model()
    handle = model.units[0].register_forward_hook(lambda _module, _args, output: output)
    try:
        with pytest.raises(PipelineError, match="Module hooks are not captured"):
            capture(model, example_inputs())
    finally:
        handle.remove()


def test_nested_layers_require_annotation_and_explicit_dag_is_preserved():
    class Group(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = nn.ModuleList([Block(), Block()])

        def forward(self, x, seq):
            for block in self.inner:
                x = block(x, seq)
            return x

    model = Model()
    model.units = nn.ModuleList([Group(), Group()])
    with pytest.raises(PipelineError) as caught:
        capture(model, example_inputs())
    assert caught.value.code == "PP_LAYERS_REQUIRED"
    candidates = caught.value.context["candidates"]
    assert isinstance(candidates, dict)
    assert set(candidates) == {"units", "units.0.inner", "units.1.inner"}
    assert "pipeline_layers" in caught.value.action
    model.pipeline_layers = lambda: model.units
    assert [layer.path for layer in capture(model, example_inputs()).layers] == ["units.0", "units.1"]

    model = Model()
    model.parallel = True
    with pytest.raises(PipelineError, match="PP_LAYERS_REQUIRED"):
        capture(model, example_inputs())
    model.pipeline_layers = lambda: list(model.units)
    result = capture(model, example_inputs())
    assert result.layer_source == "explicit" and len(result.layers) == 2


@pytest.mark.parametrize("entries", [
    lambda m: ["units.0"],
    lambda m: [m.units[0], Block()],
    lambda m: [m.units[0], m.units[0]],
    lambda m: [m.units[1], m.units[0]],
])
def test_invalid_explicit_layers_never_fall_back_to_inference(entries):
    model = Model()
    model.pipeline_layers = lambda: entries(model)
    with pytest.raises(PipelineError) as caught:
        capture(model, example_inputs())
    assert caught.value.code == "PP_LAYERS_INVALID"
    assert caught.value.phase == "layer_mapping"
    assert "pipeline_layers()[" in caught.value.location


def test_state_contract_links_real_attention_and_gdn_calls_and_rejects_reuse():
    model = Model()
    first, second = model.units
    assert isinstance(first, Block) and isinstance(second, Block)
    first.mix = inn.Attention(hidden=16, heads=4, kv_heads=2, head_dim=4, rope_theta=10000)
    second.mix = inn.GDN(hidden=16, key_heads=2, value_heads=4, key_head_dim=4, value_head_dim=4, conv_kernel=4)
    result = capture(model, example_inputs())
    assert {ref.op_id for ref in result.contract.state_refs} == {"attention.core.gqa", "gdn.core"}
    assert set(result.state_nodes) == {"units.0.mix", "units.1.mix"}
    assert set(result.contract.op_facts) == set(result.state_nodes)
    # An existing composite component remains opaque and owns its child's state call.
    composite = Model()
    composite.units[0] = inn.DecoderBlock(
        hidden=16, mixer=first.mix, mlp=inn.MLP(hidden=16, intermediate=32),
    )
    composite_result = capture(composite, example_inputs())
    assert set(composite_result.state_nodes) == {"units.0.mixer"}
    state_owner = composite_result.state_nodes["units.0.mixer"]
    assert next(
        node for node in composite_result.program.graph.nodes if node.name == state_owner
    ).target == "units.0"
    first.repeat = True
    with pytest.raises(PipelineError) as caught:
        capture(model, example_inputs())
    assert caught.value.code == "PP_STATEFUL_INVOCATION"
    state_nodes = caught.value.context["nodes"]
    assert isinstance(state_nodes, list) and len(state_nodes) == 2
    first.repeat, first.skip = False, True
    with pytest.raises(PipelineError, match="PP_STATEFUL_INVOCATION"):
        capture(model, example_inputs())
    second.mix = first.mix
    with pytest.raises(PipelineError) as caught:
        capture(model, example_inputs())
    assert caught.value.context["paths"] == ["units.0.mix", "units.1.mix"]


@pytest.mark.parametrize("flag", ["dynamic", "suffix", "no_grad", "seq_write", "slice_inplace", "module_write"])
def test_unsupported_forward_reports_capture_location_and_action(flag):
    model = Model()
    setattr(model, flag, True)
    with pytest.raises(PipelineError) as caught:
        capture(model, example_inputs())
    error = caught.value
    assert json.loads(json.dumps(error.to_dict()))["code"] == error.code
    assert error.code == "PP_CAPTURE_UNSUPPORTED"
    assert error.model.endswith(".Model") and error.location and error.action
    assert error.required_information == "none"
    if flag == "dynamic":
        assert "data-dependent" in error.reason
        assert error.__cause__ is not None and ".py:" in error.location
    elif flag == "suffix":
        assert "one terminal isoexec.nn.LMHead" in error.reason
    elif flag == "no_grad":
        assert "context" in error.reason
    elif flag == "seq_write":
        assert "Writing SequenceContext.positions" in error.reason
    elif flag == "module_write":
        assert "mutated module state" in error.reason
    else:
        assert "aten.copy" in error.reason and "backward" in error.reason


def test_layer_aliases_and_annotation_mutation_are_refused():
    model = Model()
    model.alias = model.units[0]
    model.pipeline_layers = lambda: model.units
    with pytest.raises(PipelineError, match="one registered non-root path"):
        capture(model, example_inputs())
    del model.alias

    def mutating_annotation():
        model.scale = 2.0
        return model.units

    model.pipeline_layers = mutating_annotation
    with pytest.raises(PipelineError, match="annotation changed"):
        capture(model, example_inputs())


class AliasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.units = nn.ModuleList([Block("cpu"), Block("cpu")])
        self.finish = inn.RMSNorm(hidden=16, role="final", device="cpu")
        self.readout = inn.LMHead(vocab=32, hidden=16, device="cpu")
        self.mutate_alias = True

    def forward(self, hidden, seq):
        base = self.units[0](hidden, seq)
        view = base.view(base.shape[0], 2, 8)
        if self.mutate_alias:
            view += 0.25
        # Both reads must observe the mutation of their common storage.
        hidden = base + view.reshape_as(base)
        hidden = self.units[1](hidden, seq)
        return self.readout(self.finish(hidden))


def test_view_iadd_preserves_alias_reads_forward_and_all_gradients():
    model = AliasModel()
    generator = torch.Generator().manual_seed(71)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.04)
    hidden = torch.randn(4, 16, generator=generator, dtype=torch.bfloat16).requires_grad_()
    _, seq = example_inputs()
    result = capture(model, (hidden, seq))
    table = OpTable(mode="reference")
    inn.place(model, trivial_placement(side="trainer"), op_table=table)
    assert result.program.get_submodule("units.0.ffn") is model.units[0].ffn
    assert result.program.get_submodule("units.1.norm") is model.units[1].norm
    assert result.program.get_submodule("units.0.ffn")._op_table is table

    def run(forward):
        model.zero_grad(set_to_none=True)
        inputs = hidden.detach().clone().requires_grad_()
        output = forward(inputs, seq)
        output.square().mean().backward()
        assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
        assert torch.count_nonzero(inputs.grad) > 0
        gradients = {}
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            gradients[name] = parameter.grad.clone()
        assert torch.equal(inputs.detach(), hidden)
        return output.detach(), inputs.grad.clone(), gradients

    expected, expected_input_grad, expected_grads = run(model)
    actual, actual_input_grad, actual_grads = run(result.program)
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    assert torch.equal(expected_input_grad.view(torch.uint8), actual_input_grad.view(torch.uint8))
    assert expected_grads.keys() == actual_grads.keys()
    for name in expected_grads:
        assert torch.equal(expected_grads[name].view(torch.uint8), actual_grads[name].view(torch.uint8)), name
    model.mutate_alias = False
    assert not torch.equal(model(hidden, seq), actual)


def test_export_shape_guard_rejects_inputs_before_unplaced_components_execute():
    model = Model()
    result = capture(model, example_inputs())
    ids, seq = example_inputs(tokens=5)
    seq.max_q = 4  # Keep the static scalar identical so this tests the tensor-shape guard.
    with pytest.raises(ValueError, match="captured shape/value contract.*Guard failed"):
        result.program(ids, seq)
    from isoexec.compiler.text import text_inputs

    examples = text_inputs(model, device='cpu', max_tokens=8)
    dynamic = capture(model, examples.args, examples.kwargs, dynamic_shapes=examples.dynamic_shapes)
    too_long = text_inputs(model, device='cpu', tokens=9, max_tokens=16)
    with pytest.raises(ValueError, match=r"captured_ranges=.*8.*sufficient max_tokens bound"):
        dynamic.program(*too_long.args, **too_long.kwargs)


@pytest.mark.parametrize("duplicate_at_capture", [False, True])
def test_export_guard_preserves_tensor_identity_specialization(duplicate_at_capture):
    ids, seq = example_inputs()
    if not duplicate_at_capture:
        seq.input_ids = ids.clone()
    result = capture(Model(), (ids, seq))
    # Flip the captured identity relationship while keeping shape, dtype and values valid.
    runtime_ids = ids + 1 if duplicate_at_capture else seq.input_ids
    with pytest.raises(ValueError, match="identity/alias specialization"):
        result.program(runtime_ids, seq)


def test_root_keyword_default_remains_a_checked_scalar_specialization():
    ids, seq = example_inputs()
    result = capture(Model(), (ids, seq))
    with pytest.raises(ValueError, match="specialized scalar"):
        result.program(ids, seq, factor=2.0)


def test_exact_registered_derived_component_preserves_eager_gradients_and_op_table(monkeypatch):
    from isoexec.compiler.capture import component_capture as registration

    class ScaledMLP(inn.MLP):
        def __init__(self, *, scale=0.5, device="cpu"):
            super().__init__(hidden=16, intermediate=32, device=device)
            self.output_scale = scale

        def facts(self):
            return {**super().facts(), "output_scale": self.output_scale}

        def forward(self, x, seq=None, aux=None):
            return super().forward(x, seq, aux=aux) * self.output_scale

    monkeypatch.setattr(registration, "_REGISTERED", dict(registration._REGISTERED))
    model = Model(device="cpu")
    derived = ScaledMLP()
    model.units[0].ffn = derived
    inputs = example_inputs()
    with pytest.raises(PipelineError, match="register_capture_component"):
        capture(model, inputs)
    assert registration.register_capture_component(ScaledMLP) is ScaledMLP
    graph = capture(model, inputs)
    assert graph.program.get_submodule("units.0.ffn") is derived
    table = OpTable(mode="reference")
    inn.place(model, trivial_placement(), op_table=table)
    assert derived._op_table is table and derived.facts()["output_scale"] == 0.5
    generator = torch.Generator().manual_seed(214)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.04)
    calls = []
    handle = derived.register_forward_hook(lambda *_args: calls.append(1))

    def run(forward):
        model.zero_grad(set_to_none=True)
        output = forward(*inputs)
        output.square().mean().backward()
        assert torch.isfinite(output).all()
        gradients = {}
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            gradients[name] = parameter.grad.clone()
        return output.detach(), gradients

    try:
        eager, eager_gradients = run(model)
        captured, captured_gradients = run(graph.program)
        assert calls == [1, 1]
        assert torch.equal(eager.view(torch.uint8), captured.view(torch.uint8))
        for name in eager_gradients:
            assert torch.equal(eager_gradients[name].view(torch.uint8), captured_gradients[name].view(torch.uint8))
    finally:
        handle.remove()

    class UnregisteredChild(ScaledMLP):
        pass

    unknown = Model()
    unknown.units[0].ffn = UnregisteredChild(device="meta")
    with pytest.raises(PipelineError, match="register_capture_component"):
        capture(unknown, inputs)

    class HiddenStateMLP(ScaledMLP):
        def __init__(self):
            super().__init__(device="meta")
            self.inner = inn.Attention(hidden=16, heads=4, kv_heads=1, head_dim=4, rope_theta=10000.0)

    registration.register_capture_component(HiddenStateMLP)
    unknown.units[0].ffn = HiddenStateMLP()
    with pytest.raises(PipelineError, match="contains another stateful invocation"):
        capture(unknown, inputs)
