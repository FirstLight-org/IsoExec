"""HF checkpoint coverage, packed expert conversion, and native rank slicing."""
import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen3_5MoeConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

from isoexec.compiler.capture.capture import capture
from isoexec.compiler.realization.weights import _conversion_mapping, _find_mapping, ignored_patterns
from isoexec.compiler.source import ModelSource
from isoexec.compiler.text import text_inputs
from isoexec.nn import GDN, MoE
from isoexec.runtimes.parallel import trivial_placement

SOURCE = ModelSource("isoexec.models.qwen3_5_moe")


def configuration(tiny=True):
    raw = json.loads((Path(__file__).parent / "fixtures/qwen3_5_35b_a3b.json").read_text())
    if tiny:
        raw['text_config'].update(hidden_size=16, vocab_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_num_key_heads=4, linear_num_value_heads=8,
            linear_key_head_dim=4, linear_value_head_dim=4,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=8,
            shared_expert_intermediate_size=8, layer_types=['linear_attention', 'full_attention'],
            rope_parameters={'rope_type':'default', 'rope_theta':10000000,
                'partial_rotary_factor':0.25, 'mrope_section':[1,0,0], 'mrope_interleaved':True})
    return raw


def hf_shapes(raw):
    config = Qwen3_5MoeConfig.from_dict(raw)
    with torch.device('meta'):
        hf = Qwen3_5MoeTextModel(config.get_text_config())
    shapes = {'model.language_model.' + k: tuple(v.shape) for k,v in hf.state_dict().items()}
    shapes['lm_head.weight'] = (config.text_config.vocab_size, config.text_config.hidden_size)
    return shapes


def test_released_geometry_and_hf_key_coverage():
    raw = configuration(False)
    model = SOURCE.build_meta(raw)
    assert len(model.layers) == 40
    assert sum(isinstance(layer.mixer, GDN) for layer in model.layers) == 30
    assert all(isinstance(layer.mlp, MoE) and layer.mlp.num_experts == 256 and
               layer.mlp.top_k == 8 and layer.mlp.weight_after_fc2 for layer in model.layers)
    assert all(p.is_meta for p in model.parameters())
    config = SOURCE.build_config(raw)
    renamings, converters = _conversion_mapping(SOURCE, config)
    logical_names = set(dict(model.named_parameters()))
    mapped = {name for key in hf_shapes(raw) for name in _find_mapping(key, renamings, converters)[2]}
    assert mapped == logical_names
    assert ignored_patterns(config) == model._keys_to_ignore_on_load_unexpected


@pytest.mark.parametrize("key,value", [("mlp_only_layers", [0]), ("hidden_act", "relu")])
def test_unsupported_geometry_refused(key, value):
    raw = configuration()
    raw['text_config'][key] = value
    with pytest.raises(ValueError):
        SOURCE.build_meta(raw)


def test_moe_forward_matches_independent_transformers_block():
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

    from isoexec.models.qwen3_5_moe.model import Qwen3_5MoeForConditionalGeneration
    from isoexec.ops.table import OpTable
    from isoexec.schema import make_sequence_context

    config = Qwen3_5MoeConfig.from_dict(configuration())
    torch.manual_seed(29)
    reference = Qwen3_5MoeSparseMoeBlock(config.text_config).bfloat16()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)
    actual = Qwen3_5MoeForConditionalGeneration(config, device='cpu').layers[0].mlp
    actual.apply_placement(trivial_placement(), site='layers.0.mlp', op_table=OpTable(mode='reference'))
    with torch.no_grad():
        actual.router.weight.copy_(reference.gate.weight)
        for i, expert in enumerate(actual.experts):
            gate, up = reference.experts.gate_up_proj[i].chunk(2, dim=0)
            expert.gate.copy_(gate)
            expert.up.copy_(up)
            expert.down.copy_(reference.experts.down_proj[i])
        for role in ('gate','up','down'):
            getattr(actual.shared, role).copy_(getattr(reference.shared_expert, role + '_proj').weight)
        actual.shared.expert_gate.copy_(reference.shared_expert_gate.weight)
        x = torch.randn(7,16).bfloat16()
        cu = torch.tensor([0,7], dtype=torch.int32)
        seq = make_sequence_context(torch.arange(7), cu, max_q=7)
        output = actual(x, seq)
        expected = reference(x.unsqueeze(0)).squeeze(0)
    torch.testing.assert_close(output.float(), expected.float(), rtol=0.03, atol=0.001)


@pytest.mark.parametrize("tp_size,tp_rank,ep", [(1, 0, False), (2, 0, False), (2, 1, False), (2, 0, True)])
def test_native_runtime_loader_handles_fused_experts_and_remote_ownership(tp_size, tp_rank, ep, monkeypatch):
    from isoexec.compiler.realization.graph import clone_graph_module
    from isoexec.compiler.realization.storage import allocate_tensors
    from isoexec.compiler.realization.tp import shard_parameter_shapes, slice_weight
    from isoexec.compiler.realization.weights import ignored_patterns, load_weights
    from isoexec.compiler.partition import define_stage

    raw = configuration()
    config = SOURCE.build_config(raw)
    author = SOURCE.build_meta(raw).eval()
    assert ignored_patterns(config) == author._keys_to_ignore_on_load_unexpected
    examples = text_inputs(author, device='cpu', max_tokens=16, profile='engine')
    captured = capture(author, examples.args, examples.kwargs, dynamic_shapes=examples.dynamic_shapes,
                       root_layouts=examples.root_layouts, seq=examples.seq)
    for rank in range(2):
        stage = define_stage(captured, ((0, 1), (1, 2)), pp_rank=rank)
        assert stage.module.get_submodule(f"layers.{rank}.mlp") is not None
    target = clone_graph_module(define_stage(captured, ((0, 2),), pp_rank=0).module)
    remote = set()
    if ep:
        remote = {name for name, _ in target.named_parameters()
                  if '.mlp.experts.' in name and int(name.split('.mlp.experts.')[1].split('.')[0]) >= 2}
        for index in range(2):
            experts = target.get_submodule(f'layers.{index}.mlp.experts')
            for expert in (2, 3):
                del experts._modules[str(expert)]
    etp = 1 if ep else tp_size
    shard_parameter_shapes(target, tp_size=tp_size, side='trainer', etp_size=etp)
    allocate_tensors(target, device=torch.device('cpu'))
    shapes = hf_shapes(raw)
    weights = {name: torch.arange(torch.tensor(shape).prod().item()).reshape(shape).float().remainder(113).bfloat16()
               for name, shape in shapes.items()}
    weights.update({'mtp.fc.weight': torch.ones(1), 'model.visual.patch.weight': torch.ones(1)})
    original = {name: value.clone() for name, value in weights.items()}
    loaded = load_weights(target, weights.items(), source=SOURCE, hf_config=config, tp_size=tp_size, tp_rank=tp_rank,
                          side='trainer', etp_size=etp, remote=remote)
    assert loaded == dict(target.named_parameters()).keys()
    assert not loaded & remote
    for name, value in weights.items():
        assert torch.equal(value, original[name])
    for expert in range(2 if ep else 4):
        for role in ('gate', 'up', 'down'):
            name = f'layers.0.mlp.experts.{expert}.{role}'
            full = weights['model.language_model.layers.0.mlp.experts.' +
                           ('down_proj' if role == 'down' else 'gate_up_proj')][expert]
            if role != 'down':
                full = full.chunk(2, dim=0)[role == 'up']
            expected = slice_weight(target, name, full, tp_size=tp_size, tp_rank=tp_rank,
                                    side='trainer', etp_size=etp)
            assert torch.equal(target.get_parameter(name), expected)
    qkv = weights['model.language_model.layers.0.linear_attn.in_proj_qkv.weight']
    for role, start, width in zip(('q', 'k', 'v'), (0, 16, 32), (16, 16, 32)):
        name = f'layers.0.mixer.{role}'
        expected = slice_weight(target, name, qkv[start:start + width], tp_size=tp_size,
                                tp_rank=tp_rank, side='trainer', etp_size=etp)
        assert torch.equal(target.get_parameter(name), expected)
    if ep:
        fused_key = 'model.language_model.layers.0.mlp.experts.gate_up_proj'
        fully_remote = remote | {f'layers.0.mlp.experts.{expert}.{role}'
                                 for expert in range(4) for role in ('gate', 'up')}
        assert load_weights(target, [(fused_key, torch.ones(1))], source=SOURCE, hf_config=config,
                            remote=fully_remote) == set()
        # Conversion must still produce all declared outputs, including remote outputs.
        from isoexec.models.qwen3_5_moe.weight import _ExpertProjection
        convert = _ExpertProjection.convert

        def incomplete(self, *args, **kwargs):
            result = convert(self, *args, **kwargs)
            result.pop(next(reversed(result)))
            return result

        with monkeypatch.context() as patch:
            patch.setattr(_ExpertProjection, 'convert', incomplete)
            with pytest.raises(ValueError, match='converted logical names do not match'):
                load_weights(target, [(fused_key, weights[fused_key])], source=SOURCE, hf_config=config,
                             tp_size=tp_size, tp_rank=tp_rank, side='trainer', etp_size=etp, remote=remote)
        # A genuinely missing local target remains an error, despite remote siblings.
        del target.get_submodule('layers.0.mlp.experts.0')._parameters['gate']
        fused = ('model.language_model.layers.0.mlp.experts.gate_up_proj',
                 weights['model.language_model.layers.0.mlp.experts.gate_up_proj'])
        with pytest.raises(ValueError, match='mapped logical name is not registered'):
            load_weights(target, [fused], source=SOURCE, hf_config=config, tp_size=tp_size, tp_rank=tp_rank,
                         side='trainer', etp_size=etp, remote=remote)
