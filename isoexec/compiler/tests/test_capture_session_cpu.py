"""Capture input binding, trace resets and temporary-object lifetimes."""

import gc
import importlib
import inspect
import weakref
from typing import Any

import pytest
import torch
from torch.utils import _pytree as pytree

import isoexec.nn as inn
from isoexec.compiler.capture.capture import capture
from isoexec.compiler.capture.export_boundary import CaptureSession
from isoexec.compiler.tests.test_capture_cpu import Model, example_inputs
from isoexec.compiler.tests.test_implicit_sequence_cpu import ExplicitModel, ImplicitModel
from isoexec.compiler.tests.test_output_contract_cpu import _equal, _fill, _gradients
from isoexec.compiler.text import text_inputs, text_roots
from isoexec.ops.table import OpTable
from isoexec.runtimes.parallel import trivial_placement


def test_begin_resets_calls_without_replacing_input_description():
    model = Model()
    session = CaptureSession(model, example_inputs())
    description = {name: getattr(session, name) for name in
                   ("signature", "names", "example_values", "leaf_paths", "tree_spec")}
    first = session.begin(session.example_values)
    session.enter("units.0", type(model.units[0]), model.units[0])
    assert session.calls and session.active and session.counts and session.module_states

    second = session.begin(session.example_values)
    assert not session.calls and not session.active and not session.counts and not session.module_states
    assert first[1] is not second[1]
    assert session.roots == {id(second[1]): 1}
    for name, value in description.items():
        assert getattr(session, name) is value


class CallingConventionModel(ExplicitModel):
    def forward(self, input_ids, /, *scales, seq, factor=1.0, **offsets):
        return super().forward(input_ids, seq, factor=factor + sum(scales) + sum(offsets.values()))


class ImplicitCallingConventionModel(ImplicitModel):
    def forward(self, input_ids, /, *scales, factor=1.0, **offsets):
        assert "seq" not in offsets
        x = self.lexicon(input_ids)
        for unit in self.units:
            x = unit(x)
        return self.readout(self.finish(x) * self.scale * (factor + sum(scales) + sum(offsets.values())),
                            replicated_tokens=False)


@pytest.mark.parametrize("implicit_sequence", [False, True], ids=["explicit_seq", "implicit_seq"])
@pytest.mark.parametrize("explicit_factor", [False, True], ids=["default", "keyword"])
def test_capture_preserves_positional_keyword_and_variadic_arguments(explicit_factor: bool, implicit_sequence: bool):
    model = ImplicitCallingConventionModel() if implicit_sequence else CallingConventionModel()
    reference = CallingConventionModel()
    _fill(model)
    reference.load_state_dict(model.state_dict())
    examples = text_inputs(ImplicitModel(), device="cpu")
    ids, seq = examples.args[0], examples.seq
    assert seq is not None
    args = (ids, 0.25, 0.5)
    kwargs: dict[str, Any] = {"offset": 0.125}
    if explicit_factor:
        kwargs["factor"] = 2.0
    graph = capture(model, args, kwargs if implicit_sequence else {**kwargs, "seq": seq},
                    seq=seq if implicit_sequence else None)
    signature = graph.program.meta["isoexec_input_signature"]
    assert signature.parameters["seq"].kind == inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["factor"].default == 1.0
    for owner in (model, reference):
        inn.place(owner, trivial_placement(), op_table=OpTable(mode="reference"))
    expected, expected_gradients = _gradients(reference, lambda: reference(
        *args, seq=text_roots(ids, seq)["seq"], **kwargs))
    actual, actual_gradients = _gradients(model, lambda: graph.program(
        *args, seq=text_roots(ids, seq)["seq"], **kwargs))
    assert actual.shape == (4, 32)
    _equal(actual, expected)
    assert actual_gradients.keys() == expected_gradients.keys()
    for name in expected_gradients:
        _equal(actual_gradients[name], expected_gradients[name])
    with pytest.raises(ValueError, match="specialized scalar"):
        graph.program(*args, seq=seq, **(kwargs | {"factor": 3.0}))


@pytest.mark.parametrize("implicit_sequence", [False, True], ids=["explicit_seq", "implicit_seq"])
def test_normalized_program_releases_session_and_example_tensors(implicit_sequence: bool, monkeypatch: pytest.MonkeyPatch):
    sessions: list[weakref.ReferenceType[CaptureSession]] = []

    def session(*args: Any, **kwargs: Any) -> CaptureSession:
        created = CaptureSession(*args, **kwargs)
        sessions.append(weakref.ref(created))
        return created

    monkeypatch.setattr(importlib.import_module("isoexec.compiler.capture.capture"), "CaptureSession", session)

    def build():
        model = ImplicitModel("meta") if implicit_sequence else Model()
        examples = text_inputs(model, device="cpu")
        graph = capture(model, examples.args, examples.kwargs, seq=examples.seq, dynamic_shapes=examples.dynamic_shapes)
        tensors = [weakref.ref(value) for value in pytree.tree_leaves((examples.args, examples.kwargs, examples.seq))
                   if isinstance(value, torch.Tensor)]
        return graph, tensors

    graph, tensors = build()
    gc.collect()
    assert graph.program.graph.nodes
    assert len(sessions) == 1 and sessions[0]() is None
    assert tensors and all(tensor() is None for tensor in tensors)
