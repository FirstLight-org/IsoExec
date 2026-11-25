"""Internal sequence inputs preserve component calls, guards and stage gradients."""

import gc
import importlib
import inspect
import weakref
from dataclasses import replace
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.utils import _pytree as pytree

import isoexec.nn as inn
from isoexec.compiler.capture.capture import capture
from isoexec.compiler.capture.component_capture import register_capture_component
from isoexec.compiler.capture.export_boundary import Argument, CaptureSession, ExportRoot
from isoexec.compiler.capture.graph import PipelineError
from isoexec.compiler.tests.test_capture_cpu import Block, Model
from isoexec.compiler.tests.test_output_contract_cpu import _equal, _fill, _gradients
from isoexec.compiler.text import text_inputs, text_roots
from isoexec.ops.table import OpTable
from isoexec.runtimes.parallel import trivial_placement


class ExplicitBlock(Block):
    def __init__(self, device="cpu"):
        super().__init__(device)
        self.mix = inn.Attention(hidden=16, heads=4, kv_heads=2, head_dim=4,
                                 rope_theta=10000, device=device)


class ImplicitBlock(ExplicitBlock):
    def forward(self, x):
        assert self.mix is not None
        x = x + self.mix(self.norm(x))
        return x + self.ffn(self.norm(x))


class ExplicitModel(Model):
    def __init__(self, device="cpu"):
        super().__init__(device)
        self.units = nn.ModuleList(ExplicitBlock(device) for _ in range(2))


class ImplicitModel(ExplicitModel):
    def __init__(self, device="cpu"):
        super().__init__(device)
        self.units = nn.ModuleList(ImplicitBlock(device) for _ in range(2))

    def forward(self, input_ids):
        x = self.lexicon(input_ids)
        for unit in self.units:
            x = unit(x)
        return self.readout(self.finish(x) * self.scale, replicated_tokens=False)


def test_internal_seq_preserves_authored_calling_convention_and_defaults():
    class CallingConvention(ImplicitModel):
        def forward(self, input_ids, /, *scales, factor=1.0, **offsets):
            assert "seq" not in offsets
            x = self.lexicon(input_ids)
            for unit in self.units:
                x = unit(x)
            return self.readout(self.finish(x) * (factor + sum(scales) + sum(offsets.values())))

    model = CallingConvention()
    _fill(model)
    examples = text_inputs(ImplicitModel(), device="cpu")
    captured = capture(model, (*examples.args, 0.25, 0.5), {"offset": 0.125}, seq=examples.seq)
    signature = captured.program.meta["isoexec_input_signature"]
    assert signature.parameters["seq"].kind == inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["factor"].default == 1.0
    inn.place(model, trivial_placement(), op_table=OpTable(mode="reference"))
    output = captured.program(examples.args[0], 0.25, 0.5, seq=examples.seq, offset=0.125)
    assert output.shape == (4, 32)
    with pytest.raises(ValueError, match="specialized scalar"):
        captured.program(examples.args[0], 0.25, 0.5, seq=examples.seq, offset=0.125, factor=2.0)


def test_internal_seq_refuses_authored_root_collision():
    model = ExplicitModel()
    examples = text_inputs(model, device="cpu")
    with pytest.raises(PipelineError, match="conflicts with an authored seq"):
        capture(model, examples.args, examples.kwargs, seq=examples.args[1])


def test_internal_seq_fills_defaults_before_positional_only_component_argument(monkeypatch):
    registration = importlib.import_module("isoexec.compiler.capture.component_capture")
    monkeypatch.setattr(registration, "_REGISTERED", dict(registration._REGISTERED))

    class PositionalMLP(inn.MLP):
        def forward(self, x, scale=0.5, seq=None, /):
            assert seq is not None
            return super().forward(x * scale, seq)

    register_capture_component(PositionalMLP, sequence_context=True)
    model = ImplicitModel()
    for unit in model.units:
        unit.ffn = PositionalMLP(hidden=16, intermediate=32, device="cpu")
    _fill(model)
    examples = text_inputs(model, device="cpu", max_tokens=8)
    captured = capture(model, examples.args, seq=examples.seq, dynamic_shapes=examples.dynamic_shapes)
    calls = [node for node in captured.program.graph.nodes
             if node.op == "call_module" and str(node.target).endswith(".ffn")]
    assert len(calls) == 2
    assert all(len(node.args) == 3 and node.args[1] == 0.5 and not node.kwargs
               and isinstance(node.args[2], torch.fx.Node) and node.args[2].target == "seq" for node in calls)
    inn.place(model, trivial_placement(), op_table=OpTable(mode="reference"))
    sequence = examples.seq
    assert sequence is not None

    def explicit():
        seq = text_roots(examples.args[0], sequence)["seq"]
        hidden = model.lexicon(examples.args[0])
        for unit in model.units:
            assert isinstance(unit, ExplicitBlock) and unit.mix is not None
            hidden = hidden + unit.mix(unit.norm(hidden), seq)
            hidden = hidden + unit.ffn(unit.norm(hidden), 0.5, seq)
        return model.readout(model.finish(hidden) * model.scale, replicated_tokens=False)

    expected, expected_grads = _gradients(model, explicit)
    actual, actual_grads = _gradients(model, lambda: captured.program(**text_roots(examples.args[0], sequence)))
    _equal(actual, expected)
    assert actual_grads.keys() == expected_grads.keys()
    for name in expected_grads:
        _equal(actual_grads[name], expected_grads[name])


def test_trace_sequence_is_current_and_cleared_after_failure():
    class Failing(nn.Module):
        def forward(self, input_ids):
            raise RuntimeError("authored failure")

    examples = text_inputs(ImplicitModel(), device="cpu")
    assert examples.seq is not None
    session = CaptureSession(Failing(), examples.args, seq=examples.seq)
    first = session.begin(session.example_values)
    other = replace(examples.seq, positions=examples.seq.positions + 7)
    second = session.begin((examples.args[0], other))
    assert session.sequence is second[1] and second[1] is not first[1]
    assert session.sequence is not None
    assert session.sequence.positions is other.positions
    assert session.encode(second[1], []) == Argument("root", 1)
    root = ExportRoot(Failing(), session)
    with pytest.raises(RuntimeError, match="authored failure"):
        root(*session.export_values)
    assert session.sequence is None


def test_normalized_internal_seq_releases_session_and_examples(monkeypatch):
    sessions = []

    def session(*args: Any, **kwargs: Any):
        result = CaptureSession(*args, **kwargs)
        sessions.append(weakref.ref(result))
        return result

    monkeypatch.setattr(importlib.import_module("isoexec.compiler.capture.capture"), "CaptureSession", session)

    def build():
        model = ImplicitModel("meta")
        examples = text_inputs(model, device="cpu")
        result = capture(model, examples.args, seq=examples.seq, dynamic_shapes=examples.dynamic_shapes)
        tensors = [weakref.ref(value) for value in pytree.tree_leaves((examples.args, examples.seq))
                   if isinstance(value, torch.Tensor)]
        return result, tensors

    captured, tensors = build()
    gc.collect()
    assert captured.program.graph.nodes
    assert sessions and sessions[0]() is None
    assert tensors and all(reference() is None for reference in tensors)


@pytest.mark.parametrize("supply", [False, True])
@pytest.mark.parametrize("explicit_none", [False, True])
def test_custom_component_requires_registration_and_preserves_explicit_none(supply, explicit_none, monkeypatch):
    registration = importlib.import_module("isoexec.compiler.capture.component_capture")
    monkeypatch.setattr(registration, "_REGISTERED", dict(registration._REGISTERED))

    class CustomMLP(inn.MLP):
        def forward(self, x, seq=None, aux=None):
            return super().forward(x, seq, aux)

    register_capture_component(CustomMLP, sequence_context=supply)

    class CustomBlock(ImplicitBlock):
        def __init__(self):
            super().__init__()
            self.ffn = CustomMLP(hidden=16, intermediate=32, device="cpu")

        def forward(self, x):
            assert self.mix is not None
            x = x + self.mix(self.norm(x))
            return x + (self.ffn(self.norm(x), None) if explicit_none else self.ffn(self.norm(x)))

    class CustomModel(ImplicitModel):
        def __init__(self):
            super().__init__()
            self.units[0] = CustomBlock()

    model = CustomModel()
    examples = text_inputs(model, device="cpu")
    captured = capture(model, examples.args, seq=examples.seq, dynamic_shapes=examples.dynamic_shapes)
    call = next(node for node in captured.program.graph.nodes
                if node.op == "call_module" and node.target == "units.0.ffn")
    if explicit_none:
        assert call.args[1] is None
    elif supply:
        assert isinstance(call.args[1], torch.fx.Node) and call.args[1].target == "seq"
    else:
        assert len(call.args) == 1 and not call.kwargs


def test_sequence_registration_refuses_invalid_or_changed_contracts(monkeypatch):
    registration = importlib.import_module("isoexec.compiler.capture.component_capture")
    monkeypatch.setattr(registration, "_REGISTERED", dict(registration._REGISTERED))

    class WithSequence(inn.MLP):
        pass

    class WithoutSequence(inn.MLP):
        def forward(self, x):
            return super().forward(x)

    with pytest.raises(TypeError, match="sequence_context must be a bool"):
        register_capture_component(WithSequence, sequence_context=cast(Any, 1))
    with pytest.raises(ValueError, match="requires an explicit seq parameter"):
        register_capture_component(WithoutSequence, sequence_context=True)
    assert WithSequence not in registration._REGISTERED and WithoutSequence not in registration._REGISTERED
    register_capture_component(WithSequence, sequence_context=True)
    with pytest.raises(ValueError, match="sequence contract changed"):
        register_capture_component(WithSequence, sequence_context=False)
    assert registration.supplies_sequence_context(WithSequence(hidden=16, intermediate=32))
