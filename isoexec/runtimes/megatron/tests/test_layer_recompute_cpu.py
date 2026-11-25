"""Full authored layer regions preserve norms, residuals, auxiliary outputs and state."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from isoexec import nn as inn
from isoexec.compiler.capture.capture import capture
from isoexec.compiler.tests.test_capture_cpu import Model, example_inputs
from isoexec.ops.table import OpTable
from isoexec.runtimes.megatron._recompute import checkpoint_layers
from isoexec.runtimes.megatron.tests.test_recompute_cpu import cpu_checkpoint_rng  # noqa: F401 -- pytest fixture
from isoexec.runtimes.parallel import trivial_placement


class MoEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplace = False
        self.norm = inn.RMSNorm(hidden=16, device="cpu")
        self.mlp = inn.MoE(hidden=16, num_experts=4, top_k=2, intermediate=8,
                          shared_intermediate=8, device="cpu")

    def forward(self, x, seq):
        hidden, scores = self.mlp(self.norm(x), seq, return_aux=True)
        if self.inplace:
            x += hidden
            return x, scores
        return x + hidden, scores


class MoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = inn.Embedding(vocab=32, hidden=16, device="cpu")
        self.layers = nn.ModuleList([MoEBlock(), MoEBlock()])
        self.head = inn.LMHead(vocab=32, hidden=16, tied_to=self.embed.weight, device="cpu")

    def forward(self, input_ids, seq):
        hidden = self.embed(input_ids)
        aux = {}
        for index, layer in enumerate(self.layers):
            hidden, scores = layer(hidden, seq)
            aux[f"layers.{index}.mlp.router.scores_full"] = scores
        return self.head(hidden), aux


@pytest.mark.usefixtures("cpu_checkpoint_rng")
@pytest.mark.parametrize("kind", ["dense", "moe_unused_aux", "moe_aux", "moe_inplace"])
@pytest.mark.parametrize("span", [1, 2, 3])
def test_full_layer_regions_exact_gradients_accumulation_updates_and_storage(kind, span):
    torch.manual_seed(29)
    author = Model(device="cpu") if kind == "dense" else MoEModel()
    if kind == "moe_inplace":
        for layer in author.layers:
            layer.inplace = True
    captured = capture(author, example_inputs())
    inn.place(author, trivial_placement(side="trainer"), op_table=OpTable(mode="reference"))
    with torch.no_grad():
        for parameter in author.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.03)
    graph = captured.program
    parameters = dict(graph.named_parameters())
    state_names = tuple(graph.state_dict())
    pointers = {name: parameter.data_ptr() for name, parameter in parameters.items()}
    regions = checkpoint_layers(graph, span=span)
    assert sum(len(region.layer_paths) for region in regions) == 2
    assert len(regions) == (2 if span == 1 else 1)
    assert dict(graph.named_parameters()).keys() == parameters.keys()
    assert tuple(graph.state_dict()) == state_names
    for name, parameter in graph.named_parameters():
        assert parameter is parameters[name] and parameter.data_ptr() == pointers[name]
    for region in regions:
        assert any(isinstance(module, inn.RMSNorm) for module in region.program.modules())
        assert any(node.target is torch.ops.aten.add.Tensor for node in region.program.graph.nodes)
        assert all(region.program.get_parameter(name) is graph.get_parameter(name)
                   for name, _ in region.program.named_parameters())
    norm = next(module for module in regions[0].program.modules() if isinstance(module, inn.RMSNorm))
    norm_modes = []
    hook = norm.register_forward_pre_hook(lambda *unused: norm_modes.append(torch.is_grad_enabled()))
    for _ in range(2):
        observations = []
        for enabled in (False, True):
            for region in regions:
                region.enabled = enabled
            graph.zero_grad(set_to_none=True)
            norm_modes.clear()
            saved = []
            outputs = []
            for microbatch in range(2):
                ids, seq = example_inputs()
                ids = ids + microbatch
                seq.input_ids = ids
                with torch.autograd.graph.saved_tensors_hooks(
                    lambda value, saved=saved: (saved.append(value.numel()), value)[1], lambda value: value,
                ):
                    result = graph(ids, seq)
                logits, aux = (result, {}) if kind == "dense" else result
                loss = logits.float().square().sum()
                if kind == "moe_aux":
                    loss = loss + sum(value.float().square().sum() for value in aux.values()) * 0.01
                loss.backward()
                outputs.append((logits.detach(), {key: value.detach() for key, value in aux.items()}))
            gradients = {name: parameter.grad.clone() for name, parameter in graph.named_parameters()}
            observations.append((outputs, gradients, sum(saved)))
            if enabled:
                assert norm_modes.count(False) == norm_modes.count(True) == 2
        assert observations[1][2] < observations[0][2]
        for left, right in zip(observations[0][0], observations[1][0], strict=True):
            assert torch.equal(left[0], right[0])
            assert all(torch.equal(value, right[1][key]) for key, value in left[1].items())
        assert all(torch.equal(value, observations[1][1][name]) for name, value in observations[0][1].items())
        with torch.no_grad():
            for parameter in graph.parameters():
                parameter.add_(parameter.grad, alpha=-0.01)
    hook.remove()


def test_mutable_region_refused():
    graph = torch.fx.Graph()
    hidden = graph.placeholder("hidden")
    result = graph.call_function(torch.ops.aten.add_.Tensor, (hidden, 1))
    result.meta["isoexec_logical_layer"] = (0, "layer")
    graph.output(result)
    module = torch.fx.GraphModule(nn.Module(), graph)
    with pytest.raises(ValueError, match="functional layer operations"):
        checkpoint_layers(module)


def test_aux_binding_discovers_moe_inside_checkpoint_regions():
    from isoexec.integrations.skyrl.aux_loss import _bind_native, _native_aux_loss

    author = MoEModel()
    graph = capture(author, example_inputs()).program
    regions = checkpoint_layers(graph)
    stage = SimpleNamespace(
        graph_module=graph, _activation_checkpoint_regions=regions,
        pg_collection=SimpleNamespace(pp=SimpleNamespace(size=lambda: 1)),
        config=SimpleNamespace(num_moe_experts=4, calculate_per_token_loss=False),
        placement=trivial_placement(side="trainer"),
        aux_outputs=tuple(f"layers.{i}.mlp.router.scores_full" for i in range(2)),
    )
    helper = _bind_native(stage, 0.01)
    scores = {name: torch.randn(4, 4).softmax(-1).requires_grad_() for name in stage.aux_outputs}
    loss, metrics = helper(scores, torch.ones(4), torch.tensor([0, 4], dtype=torch.int32))
    expected_helper = _native_aux_loss(
        [(name, 4, 2) for name in stage.aux_outputs], stage.placement, 0.01,
    )
    expected_scores = {name: value.detach().clone().requires_grad_() for name, value in scores.items()}
    expected, _ = expected_helper(expected_scores, torch.ones(4), torch.tensor([0, 4], dtype=torch.int32))
    assert torch.equal(loss, expected)
    loss.backward()
    expected.backward()
    assert all(torch.equal(value.grad, expected_scores[name].grad) for name, value in scores.items())
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in scores.values())
    assert sum(key.startswith("moe_aux/") for key in metrics) == 2


def test_region_observes_parameter_data_rebinding_and_default_state_dict_load():
    torch.manual_seed(73)
    author = Model(device="cpu")
    graph = capture(author, example_inputs()).program
    inn.place(author, trivial_placement(side="trainer"), op_table=OpTable(mode="reference"))
    with torch.no_grad():
        for parameter in graph.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.03)
    regions = checkpoint_layers(graph)
    original = {name: tensor.clone() for name, tensor in graph.state_dict().items()}
    parameters = dict(graph.named_parameters())
    with torch.no_grad():
        baseline = graph(*example_inputs())
        # DDP and SkyRL residency retain Parameter objects while rebinding data.
        for parameter in graph.parameters():
            parameter.data = torch.randn_like(parameter) * 0.06
        rebound = graph(*example_inputs())
        assert not torch.equal(rebound, baseline)
        for region in regions:
            assert all(parameter is graph.get_parameter(name)
                       for name, parameter in region.program.named_parameters())
        graph.load_state_dict(original)  # production default: assign=False
        restored = graph(*example_inputs())
    assert torch.equal(restored, baseline)
    assert all(parameter is parameters[name] for name, parameter in graph.named_parameters())


@pytest.mark.usefixtures("cpu_checkpoint_rng")
@pytest.mark.parametrize("external", ["functional", "inplace", "module"])
def test_region_outputs_copied_only_for_possible_in_place_users(external):
    """A region output is returned without a copy unless an external expression may write it in place
    (an in-place schema op, or anything not provably functional such as a module call)."""
    graph = torch.fx.Graph()
    hidden = graph.placeholder("hidden")
    inner = graph.call_function(torch.ops.aten.mul.Tensor, (hidden, 2.0))
    inner.meta["isoexec_logical_layer"] = (0, "layer")
    if external == "functional":
        out = graph.call_function(torch.ops.aten.add.Tensor, (inner, 1.0))
    elif external == "inplace":
        out = graph.call_function(torch.ops.aten.add_.Tensor, (inner, 1.0))
    else:
        out = graph.call_module("act", (inner,))
    graph.output(out)
    root = nn.Module()
    root.act = nn.ReLU()
    module = torch.fx.GraphModule(root, graph)
    (region,) = checkpoint_layers(module)
    assert region.mutated_outputs == ((external != "functional"),)
    x = torch.randn(4, requires_grad=True)
    y = module(x)
    y.sum().backward()
    assert torch.equal(x.grad, torch.full((4,), 2.0)) or external == "module"
