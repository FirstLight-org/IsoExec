"""Allocation preserves target declarations without loading or executing them."""

import pytest
import torch
from torch import fx, nn

import isoexec.nn as inn
from isoexec.core.bindings import BoundOp
from isoexec.compiler.realization.graph import clone_graph_module
from isoexec.compiler.realization.storage import allocate_tensors


def _must_not_execute(*args, **kwargs):
    raise AssertionError("Allocation executed an operation")


@pytest.mark.parametrize("tied", [False, True])
def test_allocation_preserves_parameter_properties(tied):
    root = nn.Module()
    root.embed = inn.Embedding(vocab=8, hidden=4)
    root.head = inn.LMHead(vocab=8, hidden=4, tied_to=root.embed.weight if tied else None)
    root.projection = nn.Linear(8, 4, bias=False, device="meta", dtype=torch.float32)
    root.embed.weight.requires_grad_(False)
    graph = fx.Graph()
    ids = graph.placeholder("ids")
    features = graph.placeholder("features")
    hidden = graph.call_module("embed", (ids,))
    logits = graph.call_module("head", (hidden,))
    projected = graph.call_module("projection", (features,))
    graph.output((logits, projected))
    source = fx.GraphModule(root, graph).eval()
    target = clone_graph_module(source)
    target.projection.weight = nn.Parameter(torch.empty(4, 4, dtype=torch.float32, device="meta"), requires_grad=False)
    target.head.slots.head = BoundOp("logprobs.head", "test.head", None, "test", fn=_must_not_execute)
    bound = target.head.slots.resolved()["head"]
    declarations = {name: (parameter.shape, parameter.dtype, parameter.requires_grad)
                    for name, parameter in target.named_parameters(remove_duplicate=False)}

    allocate_tensors(target, device=torch.device("cpu"))

    assert str(target.graph) == str(source.graph)
    assert target.training is target.embed.training is target.head.training is False
    assert (target.embed.weight is target.head.weight) is tied
    assert target.head.slots.resolved()["head"] is bound
    assert not source.head.slots.resolved()
    assert source.projection.weight.shape == (4, 8) and source.projection.weight.requires_grad
    for name, parameter in target.named_parameters(remove_duplicate=False):
        assert parameter.device == torch.device("cpu")
        assert (parameter.shape, parameter.dtype, parameter.requires_grad) == declarations[name]
        assert source.get_parameter(name).is_meta


def test_allocation_preserves_buffer_values_sharing_and_per_name_persistence():
    root = nn.Module()
    shared = torch.tensor([1.0, 2.0], dtype=torch.float32)
    root.register_buffer("constant", shared)
    root.register_buffer("scratch", shared, persistent=False)
    root.register_buffer("pending", torch.empty(2, dtype=torch.float32, device="meta"))
    graph = fx.Graph()
    value = graph.placeholder("value")
    for name in ("constant", "scratch", "pending"):
        value = graph.call_function(torch.add, (value, graph.get_attr(name)))
    graph.output(value)
    source = fx.GraphModule(root, graph)
    source.register_buffer("scratch", source.scratch, persistent=False)
    target = clone_graph_module(source)
    target.register_parameter("absent_parameter", None)
    target.register_buffer("absent_buffer", None)

    allocate_tensors(target, device=torch.device("cpu"))

    assert target.constant is target.scratch and target.constant is not source.constant
    assert torch.equal(target.constant, shared) and target.constant.dtype == torch.float32
    assert target.pending.shape == (2,) and target.pending.dtype == torch.float32
    assert all(buffer.device == torch.device("cpu") for buffer in target.buffers())
    assert tuple(target.state_dict()) == ("constant", "pending")
    assert target.absent_parameter is target.absent_buffer is None
    target.scratch.add_(10)
    assert torch.equal(source.constant, shared) and source.pending.is_meta


def test_allocation_refuses_parameters_that_already_have_storage():
    root = nn.Module()
    root.linear = nn.Linear(4, 4, bias=False)
    graph = fx.Graph()
    hidden = graph.placeholder("hidden")
    graph.output(graph.call_module("linear", (hidden,)))
    target = fx.GraphModule(root, graph)
    parameter = target.linear.weight
    before = parameter.detach().clone()

    with pytest.raises(ValueError, match="requires meta parameters"):
        allocate_tensors(target, device=torch.device("cpu"))

    assert target.linear.weight is parameter and torch.equal(parameter, before)
