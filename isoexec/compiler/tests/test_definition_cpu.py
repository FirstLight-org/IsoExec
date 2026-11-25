"""Validate stage definition contracts."""

import operator
from dataclasses import replace

import pytest
import torch
from torch import fx, nn

import isoexec.nn as inn
from isoexec.compiler.contract.contract import ModelContract
from isoexec.compiler.capture.capture import capture
from isoexec.compiler.capture.graph import ExecutionGraph, LogicalLayer, PipelineError
from isoexec.compiler.partition import define_stage
from isoexec.compiler.tests.test_capture_cpu import Block, Model as StateModel
from isoexec.compiler.tests.test_capture_cpu import example_inputs
from isoexec.ops.table import OpTable
from isoexec.runtimes.parallel import trivial_placement
from isoexec.schema import replicated


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lexicon = inn.Embedding(vocab=32, hidden=16, device="cpu")
        self.blocks = nn.ModuleList([Block("cpu") for _ in range(4)])
        self.register_buffer("offset", torch.full((16,), 0.02, dtype=torch.bfloat16))
        self.finish = inn.RMSNorm(hidden=16, role="final", device="cpu")
        self.head = inn.LMHead(vocab=31, padded_vocab=32, hidden=16,
                                 tied_to=self.lexicon.weight, device="cpu")

    def synced_buffers(self):
        return {"offset": replicated()}

    def forward(self, input_ids, seq, residual, *, factor=0.5):
        x = self.lexicon(input_ids) + self.offset
        carried = x + residual
        view = carried.view_as(carried)
        for index, block in enumerate(self.blocks):
            x = block(x, seq)
            if index == 0:
                carried += x
            if index == 2:
                x = x + view + residual
        return self.head(self.finish(x + self.offset) * factor, replicated_tokens=False)


def _forward(definitions, **roots):
    values = dict(roots)
    auxiliary = {}
    for definition in definitions:
        module = definition.module
        arguments = {str(node.target): values[node.name] for node in module.graph.nodes if node.op == "placeholder"}
        result = module(**arguments)
        outputs = next(node for node in module.graph.nodes if node.op == "output").args[0]
        if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], dict):
            outputs, _ = outputs
            result, local_aux = result
            assert auxiliary.keys().isdisjoint(local_aux)
            auxiliary.update(local_aux)
        if isinstance(outputs, fx.Node):
            return (result, auxiliary) if auxiliary else result
        assert len(set(outputs)) == len(outputs)
        values.update((node.name, value) for node, value in zip(outputs, result, strict=True))
    raise AssertionError("The final graph must return the terminal value")


@pytest.mark.parametrize("assignments", [
    ((0, 4),),
    ((0, 2), (2, 4)),
    ((0, 1), (1, 2), (2, 3), (3, 4)),
], ids=["pp1", "pp2", "pp4"])
@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
def test_definition_preserves_authored_values_gradients_and_registered_identities(assignments, tied):
    model = Model().eval()
    if not tied:
        model.head = inn.LMHead(vocab=31, padded_vocab=32, hidden=16, device="cpu")
    ids, seq = example_inputs()
    residual = torch.full((ids.numel(), 16), 0.02, dtype=torch.bfloat16, requires_grad=True)
    captured = capture(model, (ids, seq, residual))
    # A supplied nonpersistent buffer must not become persistent during extraction.
    captured.program.register_buffer("offset", model.offset, persistent=False)
    definitions = tuple(define_stage(captured, assignments, pp_rank=rank) for rank in range(len(assignments)))
    first, last = definitions[0], definitions[-1]
    if len(assignments) == 1:
        assert first.module is captured.program
        assert first.pp_shared_parameter is None
    else:
        assert first.pp_shared_parameter == ("lexicon.weight" if tied else None)
        assert last.pp_shared_parameter == ("head.weight" if tied else None)
        assert first.module.get_submodule("blocks.0.ffn") is model.blocks[0].ffn
        last_start = assignments[-1][0]
        assert last.module.get_submodule(f"blocks.{last_start}.ffn") is model.blocks[last_start].ffn
        assert not any(name.startswith("blocks.2.") for name, _ in first.module.named_parameters())
        assert not any(name.startswith("blocks.0.") for name, _ in last.module.named_parameters())
    assert first.module.get_parameter("lexicon.weight") is model.lexicon.weight
    assert last.module.get_parameter("head.weight") is model.head.weight
    assert all(definition.module.training is False for definition in definitions)
    for definition in (first, last):
        assert definition.module.offset is model.offset
        assert "offset" not in definition.module.state_dict()

    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.04)
    inn.place(model, trivial_placement(side="trainer"), op_table=OpTable(mode="reference"))

    def run(forward):
        model.zero_grad(set_to_none=True)
        residual.grad = None
        output = forward(input_ids=ids, seq=seq, residual=residual, factor=0.5)
        output.square().mean().backward()
        gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
        gradients["residual"] = residual.grad.clone()
        return output.detach(), gradients

    expected, expected_gradients = run(model)
    actual, actual_gradients = run(lambda **roots: _forward(definitions, **roots))
    assert torch.equal(actual, expected)
    assert actual_gradients.keys() == expected_gradients.keys()
    for name, gradient in expected_gradients.items():
        assert torch.isfinite(actual_gradients[name]).all(), name
        assert torch.equal(actual_gradients[name], gradient), name


class _Block(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def tensor_axes(self):
        return {"weight": replicated()}

    def forward(self, x):
        return self.weight * x


class _AuxModel(nn.Module):
    def __init__(self, reuse_aux=False):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(2), _Block(5)])
        self.reuse_aux = reuse_aux

    def forward(self, x):
        hidden = self.blocks[0](x)
        first_aux = hidden.square()
        logits = self.blocks[1](hidden)
        if self.reuse_aux:
            logits = logits + first_aux
        return logits, {"first": first_aux, "second": logits.square()}


def _aux_graph(model):
    gm = fx.symbolic_trace(model)
    layers = tuple(LogicalLayer(index, f"blocks.{index}", index, (node.name,)) for index, node in enumerate(
        node for node in gm.graph.nodes if node.op == "call_function" and node.target is operator.mul
    ))
    head = layers[-1].nodes[0]
    return ExecutionGraph(gm, (), layers, "explicit", head, inn.inventory(model), ModelContract((), ()), {},
                          tuple((name, module.training) for name, module in model.named_modules()), (True, False, ()))


@pytest.mark.parametrize("reuse_aux", [False, True], ids=["local_aux_only", "aux_also_used_after_cut"])
def test_stage_local_aux_stays_out_of_pp_values_but_contributes_to_backward(reuse_aux):
    model = _AuxModel(reuse_aux)
    captured = _aux_graph(model)
    first = define_stage(captured, ((0, 1), (1, 2)), pp_rank=0).module
    second = define_stage(captured, ((0, 1), (1, 2)), pp_rank=1).module
    reference_input = torch.tensor(3.0, requires_grad=True)
    logits, auxiliary = model(reference_input)
    (logits + auxiliary["first"] + auxiliary["second"]).backward()
    expected = [parameter.grad.clone() for parameter in model.parameters()]
    expected_input = reference_input.grad.clone()
    assert [float(gradient) for gradient in expected] == ([6819, 798] if reuse_aux else [951, 366])
    assert float(expected_input) == (4546 if reuse_aux else 634)
    model.zero_grad(set_to_none=True)

    x = torch.tensor(3.0, requires_grad=True)
    outgoing, first_aux = first(x)
    assert tuple(first_aux) == ("first",)
    assert [float(value.detach()) for value in outgoing] == ([6, 36] if reuse_aux else [6])
    output_nodes, auxiliary_nodes = next(node for node in first.graph.nodes if node.op == "output").args[0]
    assert tuple(node.name for node in second.graph.nodes if node.op == "placeholder") == tuple(
        node.name for node in output_nodes
    )
    assert (auxiliary_nodes["first"] in output_nodes) is reuse_aux
    if reuse_aux:
        assert outgoing[1] is first_aux["first"]
    # Emulate only the PP value/gradient boundary. This is not a native PP test.
    received = tuple(value.detach().requires_grad_() for value in outgoing)
    actual_logits, second_aux = second(*received)
    assert tuple(second_aux) == ("second",)
    (actual_logits + second_aux["second"]).backward()
    assert [float(value.grad) for value in received] == ([665, 133] if reuse_aux else [305])
    torch.autograd.backward((*outgoing, first_aux["first"]), (
        *(value.grad for value in received), torch.ones_like(first_aux["first"]),
    ))
    for parameter, gradient in zip(model.parameters(), expected, strict=True):
        assert torch.equal(parameter.grad, gradient)
    assert torch.equal(x.grad, expected_input)


def test_definition_refuses_invalid_assignments_and_rank():
    captured = _aux_graph(_AuxModel())
    for assignments in ((), ((0, 1),), ((0, 1), (0, 2)), ((0, 0), (0, 2)), ((0, 3),), ((0, 2.0),)):
        with pytest.raises(PipelineError) as caught:
            define_stage(captured, assignments, pp_rank=0)
        assert caught.value.code == "PP_ASSIGNMENT_INVALID"
    for rank in (-1, 2, True, 0.5):
        with pytest.raises(PipelineError):
            define_stage(captured, ((0, 1), (1, 2)), pp_rank=rank)


def test_definition_refuses_cross_stage_parameters_other_than_the_embedding_tie():
    model = StateModel()
    model.units[1].ffn.gate = model.units[0].ffn.gate
    captured = capture(model, example_inputs())
    with pytest.raises(PipelineError, match="only the first/last embedding"):
        define_stage(captured, ((0, 1), (1, 2)), pp_rank=0)


def test_definition_retains_meta_state_modules():
    model = StateModel()
    model.units[0].mix = inn.Attention(hidden=16, heads=4, kv_heads=2, head_dim=4, rope_theta=10000)
    model.units[1].mix = inn.GDN(hidden=16, key_heads=2, value_heads=4, key_head_dim=4, value_head_dim=4, conv_kernel=4)
    captured = capture(model, example_inputs())
    for rank in range(2):
        module = define_stage(captured, ((0, 1), (1, 2)), pp_rank=rank).module
        assert module.get_submodule(f"units.{rank}.mix") is model.units[rank].mix
        assert all(parameter.is_meta for parameter in module.parameters())
        assert f"units.{1 - rank}.mix" not in dict(module.named_modules())


@pytest.mark.parametrize("assignments", [((0, 2),), ((0, 1), (1, 2))], ids=["same_stage", "cross_stage"])
def test_definition_refuses_shared_state(assignments):
    model = StateModel()
    model.units[0].mix = inn.Attention(hidden=16, heads=4, kv_heads=2, head_dim=4, rope_theta=10000)
    model.units[1].mix = inn.GDN(hidden=16, key_heads=2, value_heads=4, key_head_dim=4, value_head_dim=4, conv_kernel=4)
    captured = capture(model, example_inputs())
    first, second = captured.contract.state_refs
    shared = replace(captured, contract=replace(captured.contract, state_refs=(
        first, replace(second, state_id=first.state_id),
    )))
    with pytest.raises(PipelineError, match="state sharing is unsupported") as caught:
        define_stage(shared, assignments, pp_rank=0)
    assert caught.value.code == "PP_BOUNDARY_UNSUPPORTED"
    assert caught.value.location == second.module_path
    assert first.state_id in caught.value.reason
