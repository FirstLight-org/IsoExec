"""Target graph changes cannot mutate a stage definition's graph or state."""

import pytest
import torch
from torch import fx, nn

import isoexec.nn as inn
from isoexec.compiler.realization.graph import clone_graph_module, drop_auxiliary_outputs
from isoexec.compiler.capture.capture import capture
from isoexec.compiler.partition import define_stage
from isoexec.compiler.tests.test_capture_cpu import Model, example_inputs
from isoexec.ops.table import OpTable
from isoexec.runtimes.parallel import trivial_placement


def test_target_graph_and_module_changes_leave_source_and_other_targets_unchanged():
    model = Model().eval()
    captured = capture(model, example_inputs())
    source = define_stage(captured, ((0, 2),), pp_rank=0).module
    source_graph = str(source.graph)
    source_gate = source.get_parameter("units.0.ffn.gate")
    source_eps = source.get_submodule("units.0.norm").eps
    first, second = clone_graph_module(source), clone_graph_module(source)

    assert tuple(dict(first.named_parameters(remove_duplicate=False))) == tuple(
        dict(source.named_parameters(remove_duplicate=False))
    )
    for target in (first, second):
        assert target.get_parameter("lexicon.weight") is target.get_parameter("readout.weight")
        assert target.get_parameter("lexicon.weight") is not source.get_parameter("lexicon.weight")
        assert target.training is False
        assert all(parameter.is_meta for parameter in target.parameters())
    assert first.get_parameter("lexicon.weight") is not second.get_parameter("lexicon.weight")

    # Exercise mutations needed by subsequent lowering, not just unequal IDs.
    first.get_submodule("units.0.ffn").gate = nn.Parameter(torch.empty(8, 16, device="meta"))
    first.get_submodule("units.0.norm").eps = 0.1
    output = next(node for node in first.graph.nodes if node.op == "output")
    with first.graph.inserting_before(output):
        negated = first.graph.call_function(torch.neg, (output.args[0],))
    output.args = (negated,)
    first.recompile()
    first.graph.lint()

    assert str(source.graph) == source_graph == str(second.graph)
    assert source.get_parameter("units.0.ffn.gate") is source_gate
    assert tuple(source_gate.shape) == tuple(second.get_parameter("units.0.ffn.gate").shape) == (32, 16)
    assert source.get_submodule("units.0.norm").eps == second.get_submodule("units.0.norm").eps == source_eps
    for original, copied in zip(source.graph.nodes, second.graph.nodes, strict=True):
        assert original is not copied and original.meta is not copied.meta
        if "val" in original.meta:
            assert original.meta["val"] is copied.meta["val"]


def test_clone_preserves_untied_parameters_and_requires_grad():
    model = Model()
    model.readout = inn.LMHead(vocab=32, hidden=16)
    model.lexicon.weight.requires_grad_(False)
    captured = capture(model, example_inputs())
    source = define_stage(captured, ((0, 2),), pp_rank=0).module
    target = clone_graph_module(source)

    assert target.get_parameter("lexicon.weight") is not target.get_parameter("readout.weight")
    assert target.get_parameter("lexicon.weight").requires_grad is False
    assert target.get_parameter("readout.weight").requires_grad is True
    for name, original in source.named_parameters():
        copied = target.get_parameter(name)
        assert copied is not original
        assert copied.shape == original.shape and copied.dtype == original.dtype


def test_clone_preserves_buffer_values_aliases_and_per_name_persistence():
    root = nn.Module()
    shared = torch.tensor([1.0, 2.0], dtype=torch.float32)
    root.register_buffer("persistent", shared)
    root.register_buffer("scratch", shared, persistent=False)
    graph = fx.Graph()
    value = graph.placeholder("value")
    persistent, scratch = graph.get_attr("persistent"), graph.get_attr("scratch")
    added = graph.call_function(torch.add, (value, persistent))
    result = graph.call_function(torch.add, (added, scratch))
    graph.output(result)
    source = fx.GraphModule(root, graph)
    source._non_persistent_buffers_set.add("scratch")
    target = clone_graph_module(source)

    assert target.persistent is target.scratch
    assert target.persistent is not source.persistent
    assert target.persistent.dtype == torch.float32
    assert tuple(target.state_dict()) == tuple(source.state_dict()) == ("persistent",)
    assert torch.equal(target(torch.tensor([3.0, 4.0])), torch.tensor([5.0, 8.0]))
    target.scratch.add_(10)
    assert torch.equal(source.scratch, torch.tensor([1.0, 2.0]))


# TODO(jiangjiawei1103): Consider removing this test if we handle pre-hooks differently.
def test_clone_preserves_captured_input_defaults_and_scalar_guards():
    authored = Model(device="cpu")
    with torch.no_grad():
        for parameter in authored.parameters():
            parameter.fill_(0.02)
    ids, seq = example_inputs()
    source = capture(authored, (ids, seq)).program
    target = clone_graph_module(source)
    inn.place(target, trivial_placement(side="trainer"), op_table=OpTable(mode="reference"))

    assert torch.equal(target(ids, seq), target(ids, seq, factor=1.0))
    with pytest.raises(ValueError, match="specialized scalar"):
        target(ids, seq, factor=2.0)


def test_dropping_auxiliary_outputs_keeps_the_primary_result():
    graph = fx.Graph()
    x = graph.placeholder("x")
    graph.output((graph.call_function(torch.neg, (x,)), {"scores": graph.call_function(torch.exp, (x,))}))
    module = fx.GraphModule(nn.Module(), graph)

    drop_auxiliary_outputs(module)

    assert torch.equal(module(torch.ones(2)), -torch.ones(2))
    assert not any(node.target is torch.exp for node in module.graph.nodes)
