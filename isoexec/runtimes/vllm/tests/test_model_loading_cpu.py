import pytest
import torch

from isoexec.compiler.realization.tests import test_weights_cpu
from isoexec.compiler.source import ModelSource
from isoexec.runtimes.vllm.model import VllmModel
from isoexec.runtimes.vllm.tests import test_definition_cpu

qwen_stage = test_weights_cpu.qwen_stage
hf_config = test_definition_cpu.hf_config
production_bindings = test_definition_cpu.production_bindings
tp_group = test_definition_cpu.tp_group
vllm_config = test_definition_cpu.vllm_config


@pytest.fixture
def model(qwen_stage, hf_config, vllm_config):
    return VllmModel(qwen_stage,
                     source=ModelSource("isoexec.models.qwen3_5"), hf_config=hf_config,
                     vllm_config=vllm_config)


def test_load_weights_streams_hf_transforms_into_body_without_replacing_state(model, qwen_stage):
    parameters = dict(model.named_parameters(remove_duplicate=False))
    qkv = torch.arange(48 * 16, dtype=torch.float32).reshape(48, 16)
    norm = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
    qkv_before, norm_before = qkv.clone(), norm.clone()

    def weights():
        yield "model.language_model.layers.0.linear_attn.in_proj_qkv.weight", qkv
        assert torch.equal(model.get_parameter("body.layers.0.mixer.q"), qkv[:16].bfloat16())
        yield "model.language_model.layers.0.linear_attn.norm.weight", norm

    assert model.load_weights(weights()) == {
        "body.layers.0.mixer.q", "body.layers.0.mixer.k", "body.layers.0.mixer.v", "body.layers.0.mixer.out_norm",
    }
    for role, start in (("q", 0), ("k", 16), ("v", 32)):
        assert torch.equal(model.get_parameter(f"body.layers.0.mixer.{role}"), qkv[start:start + 16].bfloat16())
    expected_norm = torch.tensor([-0.5, 0.0, 0.5, 1.0], dtype=torch.bfloat16)
    assert torch.equal(model.get_parameter("body.layers.0.mixer.out_norm"), expected_norm)
    for name, parameter in parameters.items():
        assert model.get_parameter(name) is parameter
    assert all(parameter.is_meta for parameter in qwen_stage.module.parameters())
    assert torch.equal(qkv, qkv_before) and torch.equal(norm, norm_before)
    assert model.load_weights(()) == set()


def test_load_weights_preserves_ties_across_later_updates(model):
    weight = model.head.weight
    hf_weight_name = "model.language_model.embed_tokens.weight"
    incoming = torch.full(weight.shape, 0.25, dtype=torch.float32)
    expected_names = {"body.embed.weight"}

    assert model.load_weights([(hf_weight_name, incoming)]) == expected_names

    assert model.body.embed.weight is model.head.weight is weight
    assert torch.equal(weight, incoming.bfloat16())
    updated = torch.full(weight.shape, 0.75, dtype=torch.float32)
    assert model.load_weights([(hf_weight_name, updated)]) == expected_names
    assert model.body.embed.weight is model.head.weight is weight
    assert torch.equal(weight, updated.bfloat16())


def test_load_weights_refuses_conflicting_tied_entries_without_rolling_back_prior_writes(model):
    weight = model.head.weight
    first = torch.full(weight.shape, 0.25, dtype=torch.float32)
    conflicting = torch.full(weight.shape, 0.5, dtype=torch.float32)

    with pytest.raises(ValueError, match="conflicting values"):
        model.load_weights([
            ("model.language_model.embed_tokens.weight", first),
            ("model.language_model.embed_tokens.weight", conflicting),
        ])

    assert model.head.weight is model.body.embed.weight is weight
    assert torch.equal(weight, first.bfloat16())


@pytest.mark.parametrize("hf_weight_name,shape,error", [
    ("unknown.weight", (128, 16), "Unknown HF weight name"),
    ("model.language_model.embed_tokens.weight", (1, 16), "loaded shape"),
])
def test_load_weights_refuses_unknown_names_and_bad_shapes(model, hf_weight_name, shape, error):
    weight = model.head.weight
    model.load_weights([("model.language_model.embed_tokens.weight", torch.full(weight.shape, 0.25))])

    with pytest.raises(ValueError, match=error):
        model.load_weights([(hf_weight_name, torch.zeros(shape))])

    assert model.head.weight is model.body.embed.weight is weight
    assert torch.equal(weight, torch.full_like(weight, 0.25))


def test_load_weights_ignores_only_author_declared_prefixes(model):
    assert model.load_weights([
        ("model.visual.weight", torch.ones(1)),
        ("mtp.weight", torch.ones(1)),
        ("lm_head.weight", torch.ones(1)),
    ]) == set()
