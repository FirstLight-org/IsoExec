"""Native vLLM PP stages of an MoE model cross only the hidden tensor."""

import operator

import pytest
import torch

from isoexec.compiler.capture.capture import capture
from isoexec.compiler.partition import define_stage
from isoexec.compiler.realization.graph import clone_graph_module
from isoexec.compiler.realization.pp import prepare_pipeline_graph
from isoexec.compiler.text import text_inputs
from isoexec.models.qwen2_moe.model import Qwen2MoeForCausalLM
from isoexec.models.qwen2_moe.tests.test_qwen1_5_moe_a2_7b_cpu import SOURCE, configuration
from isoexec.nn import MoE
from isoexec.runtimes.vllm._definition import _split_body_and_head


@pytest.fixture
def definitions():
    author = Qwen2MoeForCausalLM(SOURCE.build_config(configuration())).eval()
    inputs = text_inputs(author, device="cpu", max_tokens=8, profile="trainer")
    captured = capture(author, inputs.args, inputs.kwargs, dynamic_shapes=inputs.dynamic_shapes,
                       root_layouts=inputs.root_layouts, seq=inputs.seq)
    return [define_stage(captured, ((0, 1), (1, 2)), pp_rank=rank) for rank in range(2)]


@pytest.mark.parametrize("rank", [0, 1])
def test_engine_pp_stage_drops_router_scores_and_crosses_one_tensor(definitions, rank):
    source = definitions[rank].module
    code = str(source.graph)
    names = {name for name, _ in source.named_parameters(remove_duplicate=False)}
    body, head = _split_body_and_head(clone_graph_module(source))
    assert (head is None) == (rank == 0)
    if head is None:
        # Weight sync and loading address every registered parameter by name.
        assert {name for name, _ in body.named_parameters(remove_duplicate=False)} == names
    prepare_pipeline_graph(body)
    output = next(node for node in body.graph.nodes if node.op == "output")
    assert isinstance(output.args[0], torch.fx.Node)
    moe_nodes = [node for node in body.graph.nodes if node.op == "call_module"
                 and isinstance(body.get_submodule(str(node.target)), MoE)]
    assert len(moe_nodes) == 1
    assert all(node.kwargs.get("return_aux") is False for node in moe_nodes)
    assert not any(node.target is operator.getitem and node.args[0] in moe_nodes for node in body.graph.nodes)
    assert str(source.graph) == code


def test_stage_zero_source_returns_router_scores(definitions):
    # The definition itself keeps the trainer's auxiliary outputs; only the engine copy drops them.
    output = next(node for node in definitions[0].module.graph.nodes if node.op == "output")
    assert isinstance(output.args[0], tuple) and isinstance(output.args[0][1], dict) and output.args[0][1]
