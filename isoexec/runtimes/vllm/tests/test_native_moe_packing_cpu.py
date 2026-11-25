"""Native MoE expert storage is packed before capture and stays current across updates."""
import pytest
import torch

from isoexec.core.bindings import BoundOp
from isoexec.compiler.capture.capture import capture
from isoexec.compiler.partition import define_stage
from isoexec.compiler.text import text_inputs
from isoexec.models.qwen2_moe.tests.test_qwen1_5_moe_a2_7b_cpu import SOURCE, checkpoint_state, configuration
from isoexec.nn import _seams
from isoexec.nn.attention import Attention
from isoexec.runtimes.vllm.model import VllmModel
from isoexec.runtimes.vllm.tests import test_definition_cpu

production_bindings = test_definition_cpu.production_bindings
tp_group = test_definition_cpu.tp_group
vllm_config = test_definition_cpu.vllm_config


@pytest.fixture
def model(vllm_config, monkeypatch, request):
    monkeypatch.setenv("ISOEXEC_MOE_PACKED_EXPERTS", "1" if getattr(request, "param", True) else "0")
    raw = configuration()
    author = SOURCE.build_meta(raw).eval()
    examples = text_inputs(author, device='cpu', max_tokens=16, profile='engine')
    captured = capture(author, examples.args, examples.kwargs, dynamic_shapes=examples.dynamic_shapes,
                       root_layouts=examples.root_layouts, seq=examples.seq)
    definition = define_stage(captured, ((0, 2),), pp_rank=0)
    return VllmModel(definition, source=SOURCE, hf_config=SOURCE.build_config(raw), vllm_config=vllm_config)


def _views(model):
    result = {}
    with torch.no_grad():
        for name, module in model.body.named_modules():
            if hasattr(module, 'expert_weight_views'):
                assert module.expert_views is not None
                result[name + '.gate_up'], result[name + '.down'] = module.expert_weight_views()
            if hasattr(module, "packed_proj_roles"):
                result[name + ".proj"] = module.packed_proj()
                assert result[name + ".proj"] is not None
    return result


def _assert_values(model):
    with torch.no_grad():
        for module in model.body.modules():
            if hasattr(module, 'expert_weight_views'):
                gate_up, down = module.expert_weight_views()
                experts = module.held_experts()
                assert torch.equal(gate_up, torch.stack([torch.cat((e.gate, e.up)) for e in experts]))
                assert torch.equal(down, torch.stack([e.down for e in experts]))
            if hasattr(module, "packed_proj_roles"):
                assert torch.equal(module.packed_proj(), torch.cat(module.packed_proj_members(), dim=0))


def _load_hf(model):
    weights = {name: torch.arange(parameter.numel()).reshape(parameter.shape).float().remainder(113).bfloat16()
               for name, parameter in checkpoint_state(configuration()).items()}
    model.load_weights(weights.items())


def _update(model, value):
    model.start_weight_update()
    updates = [(name, torch.full_like(parameter, value)) for name, parameter in model.named_parameters()]
    model.load_weights(updates)
    model.finish_weight_update()


def test_packed_storage_is_ready_before_first_forward_and_updates_without_cat_stack(model, monkeypatch):
    pointers = {name: value.data_ptr() for name, value in _views(model).items()}
    parameters = dict(model.named_parameters())
    _load_hf(model)
    _assert_values(model)
    _update(model, 0.5)
    _assert_values(model)
    assert pointers == {name: value.data_ptr() for name, value in _views(model).items()}
    assert all(model.get_parameter(name) is parameter for name, parameter in parameters.items())
    assert all(torch.equal(value, torch.full_like(value, 0.5)) for value in _views(model).values())

    def forbidden(*args, **kwargs):
        raise AssertionError('steady packed read must not concatenate or stack weights')

    with monkeypatch.context() as patch:
        patch.setattr(torch, 'cat', forbidden)
        patch.setattr(torch, 'stack', forbidden)
        assert pointers == {name: value.data_ptr() for name, value in _views(model).items()}


def test_weight_update_refuses_moved_captured_pack_buffer(model):
    _load_hf(model)
    layer = model.body.get_submodule('layers.0.mlp')
    layer._buffers['_isoexec_packed_gate_up'] = layer._buffers['_isoexec_packed_gate_up'].clone()
    with pytest.raises(RuntimeError, match='GRAPH-UNSAFE'):
        _update(model, 0.75)
    assert model._weight_update_failed
    assert model._loaded_weights is not None


@pytest.mark.parametrize("model", [False], indirect=True)
def test_existing_opt_out_keeps_expert_views_unpacked(model):
    layers = [module for module in model.body.modules() if hasattr(module, "expert_weight_views")]
    assert len(layers) == 2
    assert all(layer.expert_views() is None for layer in layers)
    assert all("_isoexec_packed_gate_up" not in layer._buffers for layer in layers)


def test_attention_projection_packing_requires_bound_no_movement_seam():
    attention = Attention(hidden=16, heads=2, kv_heads=1, head_dim=8, rope_theta=10000, out_gate=None)
    assert not attention.placed
    assert attention.packed_proj_roles() == ()
    for impl, expected in ((_seams.NO_MOVEMENT, ("q", "k", "v")), ("test.gather", ())):
        attention.slots.kv_gather = BoundOp("collectives.all_gather", impl, None, "test", fn=lambda x: x)
        assert attention.packed_proj_roles() == expected
