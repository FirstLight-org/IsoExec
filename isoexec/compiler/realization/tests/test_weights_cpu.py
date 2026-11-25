from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig, Qwen3_5Config
from transformers.core_model_loading import ConversionOps, WeightRenaming

from isoexec.compiler.realization import weights as weight_loading
from isoexec.compiler.realization.graph import clone_graph_module
from isoexec.compiler.realization.storage import allocate_tensors
from isoexec.compiler.realization.weights import ignored_patterns, load_weights
from isoexec.compiler.capture.capture import capture
from isoexec.compiler.partition import define_stage
from isoexec.compiler.source import ModelSource
from isoexec.compiler.text import text_inputs
from isoexec.models.qwen3_5.model import Qwen3_5ForConditionalGeneration

SOURCE = ModelSource("isoexec.models.qwen3_5")


def qwen_config(*, tied: bool = True) -> Qwen3_5Config:
    return Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        tie_word_embeddings=tied,
        text_config={
            "model_type": "qwen3_5_text", "hidden_size": 16, "vocab_size": 128,
            "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
            "intermediate_size": 32, "rms_norm_eps": 1e-6,
            "linear_num_key_heads": 4, "linear_num_value_heads": 4,
            "linear_key_head_dim": 4, "linear_value_head_dim": 4, "linear_conv_kernel_dim": 4,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000,
                                "partial_rotary_factor": 1.0, "mrope_section": [2, 0, 0],
                                "mrope_interleaved": False},
        },
    )


@pytest.fixture
def qwen_stage(request):
    model = Qwen3_5ForConditionalGeneration(qwen_config()).eval()
    examples = text_inputs(model, device="cpu", max_tokens=8, profile=getattr(request, "param", "trainer"))
    captured = capture(model, examples.args, examples.kwargs, dynamic_shapes=examples.dynamic_shapes,
                       root_layouts=examples.root_layouts, seq=examples.seq)
    return define_stage(captured, ((0, 2),), pp_rank=0)


@pytest.mark.parametrize("tied", [False, True])
def test_qwen_mapping_loads_stage_weights_and_preserves_registered_objects(tied):
    config = qwen_config(tied=tied)
    model = Qwen3_5ForConditionalGeneration(config).eval()
    assert ignored_patterns(config) == model._keys_to_ignore_on_load_unexpected
    examples = text_inputs(model, device="cpu", max_tokens=8)
    captured = capture(model, examples.args, examples.kwargs, dynamic_shapes=examples.dynamic_shapes,
                       root_layouts=examples.root_layouts, seq=examples.seq)
    definition = define_stage(captured, ((0, 2),), pp_rank=0)
    target = clone_graph_module(definition.module)
    allocate_tensors(target, device=torch.device("cpu"))
    parameters = dict(target.named_parameters(remove_duplicate=False))
    pointers = {name: tensor.data_ptr() for name, tensor in parameters.items()}

    prefix = "model.language_model."
    embedding = torch.arange(128 * 16, dtype=torch.float32).reshape(128, 16)
    head = torch.full((128, 16), -0.375)
    qkv = torch.arange(48 * 16, dtype=torch.float32).reshape(48, 16)
    q_gate = torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16)
    weights = {
        prefix + "embed_tokens.weight": embedding,
        "lm_head.weight": head,
        prefix + "layers.0.linear_attn.in_proj_qkv.weight": qkv,
        prefix + "layers.1.self_attn.q_proj.weight": q_gate,
        prefix + "layers.0.linear_attn.norm.weight": torch.tensor([1.00390625, 1.0, 1.5, 2.0]),
        prefix + "layers.0.mlp.gate_proj.weight": torch.full((32, 16), 0.25),
        "model.visual.patch_embed.proj.weight": torch.ones(1),
        "mtp.fc.weight": torch.ones(1),
    }
    before = {key: value.clone() for key, value in weights.items()}
    loaded = load_weights(target, weights.items(), source=SOURCE, hf_config=config)

    expected = {"embed.weight": embedding, "head.weight": embedding if tied else head,
                "layers.0.mixer.out_norm": torch.tensor([0.00390625, 0.0, 0.5, 1.0]),
                "layers.0.mlp.gate": torch.full((32, 16), 0.25)}
    for role, offset in (("q", 0), ("k", 16), ("v", 32)):
        expected[f"layers.0.mixer.{role}"] = qkv[offset:offset + 16]
    for role, part in (("q", 0), ("gate", 1)):
        expected[f"layers.1.mixer.{role}"] = q_gate.reshape(4, 2, 4, 16)[:, part].reshape(16, 16)
    assert loaded == expected.keys() - ({"head.weight"} if tied else set())
    for name, value in expected.items():
        assert torch.equal(target.get_parameter(name), value.bfloat16()), name
    assert (target.get_parameter("embed.weight") is target.get_parameter("head.weight")) is tied
    for name, tensor in parameters.items():
        assert target.get_parameter(name) is tensor and tensor.data_ptr() == pointers[name]
    assert all(tensor.is_meta for tensor in definition.module.parameters())
    assert all(torch.equal(value, before[key]) for key, value in weights.items())

    with pytest.raises(RuntimeError, match="split_with_sizes"):
        load_weights(target, [(prefix + "layers.0.linear_attn.in_proj_qkv.weight", torch.zeros(49, 16))],
                     source=SOURCE, hf_config=config)
    assert torch.equal(target.get_parameter("layers.0.mixer.q"), qkv[:16].bfloat16())
    with pytest.raises(ValueError, match="not registered"):
        load_weights(target, [(prefix + "unexpected.weight", torch.ones(1))], source=SOURCE, hf_config=config)


@pytest.fixture
def allocated(monkeypatch):
    model = nn.Module()
    model.weight = nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16), requires_grad=False)
    model.alias = model.weight
    model.register_buffer("offset", torch.zeros(2, dtype=torch.float32))

    def checkpoint_conversion_mapping(config):
        return [WeightRenaming("^hf.weight$", "weight"), WeightRenaming("^hf.alias$", "alias"),
                WeightRenaming("^hf.offset$", "offset"), WeightRenaming("^hf.missing$", "missing")]

    mapping_module = SimpleNamespace(checkpoint_conversion_mapping=checkpoint_conversion_mapping)
    monkeypatch.setattr(ModelSource, "weight_module", lambda self: mapping_module)
    monkeypatch.setattr(weight_loading, "ignored_patterns", lambda config: ())
    return model, PretrainedConfig(), mapping_module


def test_load_weights_updates_existing_parameters_and_buffers_and_allows_later_updates(allocated):
    model, config, _ = allocated
    weight, offset = model.weight, model.offset
    loaded = load_weights(model, [
        ("hf.weight", torch.ones(2, 2)), ("hf.alias", torch.ones(2, 2)),
        ("hf.offset", torch.tensor([0.25, 0.5], dtype=torch.float64)),
    ], source=SOURCE, hf_config=config)
    assert loaded == {"weight", "alias", "offset"}
    assert model.weight is model.alias is weight and not weight.requires_grad
    assert model.offset is offset and offset.dtype == torch.float32
    assert torch.equal(offset, torch.tensor([0.25, 0.5]))

    assert load_weights(model, [("hf.weight", torch.full((2, 2), 2.0))], source=SOURCE, hf_config=config) == {"weight"}
    assert model.weight is weight and torch.equal(weight, torch.full((2, 2), 2.0, dtype=torch.bfloat16))
    assert load_weights(model, (), source=SOURCE, hf_config=config) == set()


@pytest.mark.parametrize("hf_name,tensor,error", [
    ("unknown.bias", torch.zeros(2, 2), "Unknown HF weight name"),
    ("hf.missing", torch.zeros(2, 2), "logical name is not registered"),
    ("hf.weight", torch.zeros(1, 2), "loaded shape"),
    ("hf.weight", torch.zeros(2, 2, dtype=torch.int8), "floating tensor"),
    ("hf.weight", torch.zeros(2, 2).to(torch.float8_e4m3fn), "floating tensor"),
    ("hf.weight", torch.empty(2, 2, device="meta"), "allocated"),
    ("hf.alias", -torch.zeros(2, 2), "conflicting values"),
])
def test_load_weights_rejects_invalid_entries_before_overwriting_registered_values(allocated, hf_name, tensor, error):
    model, config, _ = allocated
    weight = model.weight
    with pytest.raises(ValueError, match=error):
        load_weights(model, [("hf.weight", torch.zeros(2, 2)), (hf_name, tensor)], source=SOURCE, hf_config=config)
    assert model.weight is model.alias is weight
    assert torch.equal(model.weight.view(torch.uint8), torch.zeros(2, 2, dtype=torch.bfloat16).view(torch.uint8))


def test_load_weights_rejects_a_registered_tensor_without_storage(allocated):
    model, config, _ = allocated
    model.weight = nn.Parameter(torch.empty(2, 2, device="meta"))
    with pytest.raises(ValueError, match="allocated.*registered tensor"):
        load_weights(model, [("hf.weight", torch.zeros(2, 2))], source=SOURCE, hf_config=config)


def test_load_weights_accepts_strided_inputs_and_records_both_shared_names(allocated):
    model, config, _ = allocated
    src = torch.arange(8, dtype=torch.float32).reshape(2, 4)[:, ::2]
    pointer = model.weight.data_ptr()
    loaded = load_weights(model, [("hf.weight", src), ("hf.alias", src)], source=SOURCE, hf_config=config)
    assert loaded == {"weight", "alias"}
    assert torch.equal(model.weight, src.bfloat16())
    assert model.weight.data_ptr() == pointer and not src.is_contiguous()


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.int8])
def test_load_weights_copies_encoded_checkpoint_tensors_byte_for_byte(allocated, dtype):
    model, config, mapping = allocated
    model.weight = nn.Parameter(torch.zeros(2, 2, dtype=dtype), requires_grad=False)
    model.alias = model.weight
    codes = torch.tensor([[1, 2], [127, 200]], dtype=torch.uint8).view(dtype)
    loaded = load_weights(model, [("hf.weight", codes), ("hf.alias", codes)], source=SOURCE, hf_config=config)
    assert loaded == {"weight", "alias"}
    assert torch.equal(model.weight.view(torch.uint8), codes.view(torch.uint8))
    # An encoding is never converted: a float source or another encoding is refused.
    for wrong in (torch.ones(2, 2), torch.zeros(2, 2, dtype=torch.uint8 if dtype != torch.uint8 else torch.int8)):
        with pytest.raises(ValueError, match="encoded tensor"):
            load_weights(model, [("hf.weight", wrong)], source=SOURCE, hf_config=config)
    assert torch.equal(model.weight.view(torch.uint8), codes.view(torch.uint8))


class _Scaled(ConversionOps):
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        codes, scale = (input_dict[pattern][0] for pattern in source_patterns)
        return {target_patterns[0]: codes.float() * scale.float()}


def test_many_to_one_converter_waits_for_every_source_and_package_ignores(monkeypatch):
    from transformers.core_model_loading import WeightConverter

    model = nn.Module()
    model.weight = nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16), requires_grad=False)

    def checkpoint_conversion_mapping(config):
        return [WeightConverter(source_patterns=[r"^proj\.weight$", r"^proj\.scale$"], target_patterns="weight",
                                operations=[_Scaled()])]

    mapping = SimpleNamespace(checkpoint_conversion_mapping=checkpoint_conversion_mapping,
                              ignored_checkpoint_patterns=lambda config: (r"^tower\.",))
    monkeypatch.setattr(ModelSource, "weight_module", lambda self: mapping)
    codes, scale = torch.full((2, 2), 3.0), torch.full((2, 2), 0.5)
    for order in ((("proj.scale", scale), ("proj.weight", codes)), (("proj.weight", codes), ("proj.scale", scale))):
        model.weight.data.zero_()
        loaded = load_weights(model, [*order, ("tower.x", torch.ones(1))], source=SOURCE, hf_config=PretrainedConfig())
        assert loaded == {"weight"} and torch.equal(model.weight, torch.full((2, 2), 1.5, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="sources incomplete"):
        load_weights(model, [("proj.weight", codes)], source=SOURCE, hf_config=PretrainedConfig())
    with pytest.raises(ValueError, match="Unknown HF weight name"):
        load_weights(model, [("other.x", torch.ones(1))], source=SOURCE, hf_config=PretrainedConfig())
