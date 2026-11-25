"""Native PP graph inputs preserve captured values and registered state."""

import re

import pytest
import torch
import transformers
from torch import fx, nn

from isoexec.compiler.realization.graph import clone_graph_module
from isoexec.compiler.realization.pp import prepare_pipeline_graph, uses_mrope
from isoexec.compiler.realization.storage import allocate_tensors
from isoexec.compiler.realization.tests.test_weights_cpu import qwen_config
from isoexec.compiler.realization.weights import ignored_patterns, load_weights
from isoexec.compiler.capture.capture import capture
from isoexec.compiler.partition import define_stage
from isoexec.compiler.source import ModelSource
from isoexec.compiler.text import text_inputs
from isoexec.models.qwen3_5.model import Qwen3_5ForConditionalGeneration


@pytest.fixture
def stages():
    author = Qwen3_5ForConditionalGeneration(qwen_config()).eval()
    inputs = text_inputs(author, device="cpu", max_tokens=8, profile="trainer")
    captured = capture(author, inputs.args, inputs.kwargs, dynamic_shapes=inputs.dynamic_shapes,
                       root_layouts=inputs.root_layouts, seq=inputs.seq)
    return author, [define_stage(captured, ((0, 1), (1, 2)), pp_rank=rank) for rank in range(2)]


def test_pipeline_graph_derives_token_count_and_preserves_source_and_state(stages):
    author, definitions = stages
    assert author.embed.weight is author.head.weight
    assert [definition.pp_shared_parameter for definition in definitions] == ["embed.weight", "head.weight"]
    for rank, definition in enumerate(definitions):
        source = definition.module
        code = str(source.graph)
        target = clone_graph_module(source)
        parameters = dict(target.named_parameters())
        prepare_pipeline_graph(target)
        assert [node.name for node in target.graph.nodes if node.op == "placeholder"] == [
            "input_ids", "seq", "hidden_states",
        ]
        output = next(node for node in target.graph.nodes if node.op == "output")
        assert isinstance(output.args[0], fx.Node)
        if rank == 1:
            assert any(node.target == torch.ops.aten.sym_size.int and node.args[0].name == "hidden_states"
                       for node in target.graph.nodes if node.op == "call_function")
        assert uses_mrope(target)
        assert str(source.graph) == code
        assert all(target.get_parameter(name) is parameter for name, parameter in parameters.items())
        assert all(parameter.is_meta for parameter in target.parameters())
        assert all(name.startswith(f"layers.{rank}.") or name in ("embed.weight", "norm.weight", "head.weight")
                   for name in parameters)


def test_native_pp_refuses_multiple_carried_tensors():
    graph = fx.Graph()
    left, right = graph.placeholder("left"), graph.placeholder("right")
    left.meta["val"] = right.meta["val"] = torch.empty(3, 4, device="meta")
    graph.output((left, right))
    target = fx.GraphModule(nn.Module(), graph)
    with pytest.raises(ValueError, match="one hidden tensor"):
        prepare_pipeline_graph(target)


@pytest.mark.parametrize("rank", [0, 1])
def test_stage_loading_skips_other_stages_and_resolves_the_tied_head(stages, rank):
    author, definitions = stages
    config = qwen_config()
    with torch.device("meta"):
        reference = transformers.Qwen3_5ForConditionalGeneration(config)
    ignored = ignored_patterns(config)
    weights = {name: torch.full(tensor.shape, float(index))
               for index, (name, tensor) in enumerate(reference.state_dict().items())
               if not any(re.search(pattern, name) for pattern in ignored)}
    definition = definitions[rank]
    names = set(dict(author.named_parameters(remove_duplicate=False)))
    local = set(dict(definition.module.named_parameters(remove_duplicate=False)))
    # Each global name is registered here, aliased to a local tied Parameter, or owned by the other stage.
    assert local | definition.remote_parameters | set(definition.parameter_aliases) == names
    assert not local & definition.remote_parameters and not local & set(definition.parameter_aliases)
    assert definition.parameter_aliases == ({"head.weight": "embed.weight"} if rank == 0
                                            else {"embed.weight": "head.weight"})
    target = clone_graph_module(definition.module)
    allocate_tensors(target, device=torch.device("cpu"))

    loaded = load_weights(target, weights.items(),
                          source=ModelSource("isoexec.models.qwen3_5"), hf_config=config, remote=definition.remote_parameters,
                          aliases=definition.parameter_aliases)

    assert set(dict(target.named_parameters())) <= loaded
    shared = target.get_parameter(definition.pp_shared_parameter)
    assert torch.equal(shared, weights["model.language_model.embed_tokens.weight"].to(shared.dtype))
    with pytest.raises(ValueError, match="not registered"):
        load_weights(target, weights.items(), source=ModelSource("isoexec.models.qwen3_5"), hf_config=config)


FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def test_single_tensor_cuts_keep_the_plan_they_had_before_named_carries(stages):
    """A single-tensor cut never takes the named-carry form, and its prepared stage code is byte for byte the
    code the realization produced before named carries existed (fixtures written by that version)."""
    from isoexec.compiler.realization.pp import crosses_several_tensors
    _, definitions = stages
    for rank, definition in enumerate(definitions):
        target = clone_graph_module(definition.module)
        assert not crosses_several_tensors(target)
        assert prepare_pipeline_graph(target) is None
        assert target.code == (FIXTURES / f"pp_single_tensor_qwen_stage{rank}.txt").read_text()


def test_the_ced_cut_crosses_named_token_leading_carries_in_one_order():
    """DeepSeek V4.1 at the CED cut: five tensors cross (the last encoder block's carry x, pre, out, post,
    comb). Stage 0 sends them token-leading in the cut's order under ``carry_names``; stage 1 takes the same
    keys with the same per-token shapes and dtypes and restores the captured layout (pure moves)."""
    import json
    from pathlib import Path

    from isoexec.compiler.realization.pp import carry_names, crosses_several_tensors
    package = Path(__file__).parents[3] / "models" / "deepseek_v4_1_flash" / "tests" / "fixtures"
    raw = json.loads((package / "debug_config.json").read_text())
    model = ModelSource("isoexec.models.deepseek_v4_1_flash").build_meta(raw)
    b, S, H = model.cfg.boundary, model.cfg.hc_mult, model.cfg.hidden_size
    inputs = text_inputs(model, device="cpu", max_tokens=16, profile="trainer")
    captured = capture(model, inputs.args, inputs.kwargs, dynamic_shapes=inputs.dynamic_shapes,
                       root_layouts=inputs.root_layouts, seq=inputs.seq)
    n = len(captured.layers)
    first = clone_graph_module(define_stage(captured, ((0, b), (b, n)), pp_rank=0).module)
    last = clone_graph_module(define_stage(captured, ((0, b), (b, n)), pp_rank=1).module)
    assert crosses_several_tensors(first) and crosses_several_tensors(last)
    in0, out0 = prepare_pipeline_graph(first, named_carries=True, final=False)
    in1, out1 = prepare_pipeline_graph(last, named_carries=True, final=True)
    assert in0 == () and out1 == ()
    assert [c.key for c in out0] == list(carry_names(5)) == [c.key for c in in1]
    assert [(c.tail, c.dtype, c.lead) for c in out0] == [(c.tail, c.dtype, c.lead) for c in in1] == [
        ((S, H), torch.bfloat16, 1), ((S,), torch.float32, 1), ((H,), torch.bfloat16, 1),
        ((S,), torch.float32, 1), ((S, S), torch.float32, 1)]
    assert [node.name for node in last.graph.nodes if node.op == "placeholder"] == ["input_ids", "seq",
                                                                                    *carry_names(5)]
    out = next(node for node in first.graph.nodes if node.op == "output").args[0]
    assert len(out) == 5 and all(node.target == torch.ops.aten.squeeze.dim for node in out)
    # wire bytes per token: the raw carry, a byte move (native sends the settled streams + pre_mix)
    assert sum(torch.empty(c.tail, dtype=c.dtype).nbytes for c in out0) == 2 * (S * H + H) + 4 * (S + S + S * S)
