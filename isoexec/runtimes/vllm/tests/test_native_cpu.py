"""Native kwargs construct and load the new module through its registered class."""

import copy
import json
import os
import sys
from importlib import import_module
from types import ModuleType, SimpleNamespace

import pytest
import torch

from isoexec.compiler.realization.tests import test_weights_cpu
from isoexec.compiler.source import ModelSource
from isoexec.runtimes.vllm import registration, scoring_temperature
from isoexec.runtimes.vllm.model import VllmModel
from isoexec.runtimes.vllm.tests import _gdn_fixtures, test_definition_cpu

qwen_config = pytest.fixture(test_weights_cpu.qwen_config)
production_bindings = test_definition_cpu.production_bindings
tp_group = test_definition_cpu.tp_group
vllm_config = test_definition_cpu.vllm_config


@pytest.fixture
def model_source():
    return ModelSource("isoexec.models.qwen3_5")


class HasInnerState:
    has_inner_state = True


class IsHybrid:
    is_hybrid = True

    @classmethod
    def get_mamba_state_copy_funcs(cls, mamba_types):
        return {kind: cls.get_mamba_state_copy_func() for kind in mamba_types}


class SupportsMRoPE:
    supports_mrope = True


class SupportsPP:
    supports_pp = True


@pytest.fixture
def native_registry(monkeypatch, vllm_config):
    vllm_config.structured_outputs_config = SimpleNamespace(backend="auto")
    monkeypatch.setattr(scoring_temperature, "_OWNER", None)
    monkeypatch.setattr(scoring_temperature, "_TEMPERATURE", 1.0)
    interfaces = ModuleType("vllm.model_executor.models.interfaces")
    interfaces.HasInnerState, interfaces.IsHybrid, interfaces.SupportsMRoPE = HasInnerState, IsHybrid, SupportsMRoPE
    interfaces.SupportsPP = SupportsPP
    registry = ModuleType("vllm.model_executor.models.registry")
    models = {}
    registry.ModelRegistry = SimpleNamespace(register_model=lambda name, cls: models.update({name: cls}))
    configs = ModuleType("vllm.model_executor.models.config")
    configs.MODELS_CONFIG_MAP = {}
    configs.HybridAttentionMambaModelConfig = object()
    for module in (interfaces, registry, configs):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(registration, "register_plugin", registration.register_native_model_class)
    native = import_module("isoexec.runtimes.vllm.factory")
    vllm_config.model_config.model = "configured/checkpoint"

    monkeypatch.setattr(native, "install_arch_impls", dict)
    return models, configs


@pytest.mark.parametrize("exact_file", [False, True])
def test_native_registration_kwargs_queries_construction_and_loading(
    native_registry, qwen_config, model_source, vllm_config, tmp_path, exact_file,
):
    models, configs = native_registry
    raw = json.loads(qwen_config.to_json_string())
    original = copy.deepcopy(raw)
    options = {"max_num_batched_tokens": 8, "additional_config": {"keep": True}, "hf_overrides": {"keep": True},
               "served_model_name": None}
    options_before = copy.deepcopy(options)

    checkpoint = tmp_path / "selected.safetensors" if exact_file else tmp_path
    if exact_file:
        checkpoint.touch()
    kwargs = registration.register_native(model_source, raw, checkpoint=str(checkpoint), engine_kwargs=options,
                                          score_temperature=0.7)

    assert kwargs["model"] == str(tmp_path) and kwargs["load_format"] == "auto"
    assert kwargs["enforce_eager"] and kwargs["dtype"] == "bfloat16"
    assert kwargs["compilation_config"]["mode"] == 0 and kwargs["compilation_config"]["cudagraph_mode"] == "NONE"
    assert kwargs["attention_backend"] == "CUSTOM" and kwargs["mamba_ssm_cache_dtype"] == "float32"
    assert kwargs["hf_overrides"] == {"keep": True, "architectures": [registration.NATIVE_ARCH]}
    assert kwargs["additional_config"]["keep"]
    if exact_file:
        assert kwargs["served_model_name"] == [str(checkpoint)]
    assert options == options_before and raw == original
    assert configs.MODELS_CONFIG_MAP[registration.NATIVE_ARCH] is configs.HybridAttentionMambaModelConfig
    module_name, class_name = models[registration.NATIVE_ARCH].split(":")
    factory = getattr(import_module(module_name), class_name)
    vllm_config.additional_config = kwargs["additional_config"]
    vllm_config.device_config = SimpleNamespace(device=torch.device("cpu"))
    vllm_config.scheduler_config.max_num_batched_tokens = 8
    vllm_config.model_config.hf_text_config = qwen_config.get_text_config()

    assert factory.has_inner_state and factory.is_hybrid and factory.supports_mrope
    assert factory.supports_pp
    assert factory.get_mamba_state_dtype_from_config(vllm_config) == (torch.bfloat16, torch.float32)
    assert factory.get_mamba_state_shape_from_config(vllm_config) == ((3, 48), (4, 4, 4))
    assert factory.get_mamba_state_copy_funcs({_gdn_fixtures.MambaAttentionBackendEnum.GDN_ATTN}) == {
        _gdn_fixtures.MambaAttentionBackendEnum.GDN_ATTN: (_gdn_fixtures.conv_copy, _gdn_fixtures.ssm_copy),
    }

    # Native worker setup can leave export's legacy cuDNN flag inconsistent
    # with the per-operator IEEE settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    model = factory(vllm_config=vllm_config, prefix="native")

    assert isinstance(model, VllmModel)
    assert torch.equal(scoring_temperature.temperature_tensor(), torch.tensor(0.7, dtype=torch.float32))
    assert torch.backends.cudnn.allow_tf32 is False
    if exact_file:
        assert model.allow_patterns_overrides == [checkpoint.name]
    assert model.body.embed.weight is model.head.weight
    assert model.body.get_submodule("layers.0.mixer.core").prefix == "native.body.layers.0.mixer.core"
    weight = model.body.embed.weight
    assert model.load_weights([("model.language_model.embed_tokens.weight", torch.full(weight.shape, 0.25))]) == {
        "body.embed.weight",
    }
    assert model.body.embed.weight is model.head.weight is weight
    assert torch.equal(weight, torch.full_like(weight, 0.25))
    assert raw == original


def test_native_registration_selects_the_dense_class_without_gdn(native_registry, tmp_path):
    models, configs = native_registry
    fixture = "isoexec/models/qwen2_moe/tests/fixtures/qwen1_5_moe_a2_7b.json"
    raw = {**json.loads(open(fixture).read()), "num_hidden_layers": 1, "num_experts": 4}

    kwargs = registration.register_native(ModelSource("isoexec.models.qwen2_moe"), raw, checkpoint=str(tmp_path))

    assert kwargs["hf_overrides"] == {"architectures": [registration.NATIVE_DENSE_ARCH]}
    module_name, class_name = models[registration.NATIVE_DENSE_ARCH].split(":")
    factory = getattr(import_module(module_name), class_name)
    assert factory.supports_pp and factory.supports_mrope
    assert not getattr(factory, "is_hybrid", False) and not getattr(factory, "has_inner_state", False)
    assert registration.NATIVE_DENSE_ARCH not in configs.MODELS_CONFIG_MAP


@pytest.mark.parametrize("field,value,reason", [
    ("structured_outputs_config.backend", "xgrammar", "grammar bitmask"),
    ("cache_config.cache_dtype", "fp8", "cache_dtype='fp8'"),
])
def test_native_construction_rejects_unsupported_scoring_and_cache_settings(
    native_registry, model_source, qwen_config, vllm_config, field, value, reason,
):
    native = import_module("isoexec.runtimes.vllm.factory")
    vllm_config.additional_config = {"isoexec": {
        "model_source": model_source.to_dict(),
        "raw_model_config": json.loads(qwen_config.to_json_string()),
    }}
    vllm_config.device_config = SimpleNamespace(device=torch.device("cpu"))
    vllm_config.scheduler_config.max_num_batched_tokens = 8
    vllm_config.model_config.hf_text_config = qwen_config.get_text_config()
    config, name = field.split(".")
    setattr(getattr(vllm_config, config), name, value)

    with pytest.raises(ValueError, match=reason):
        native.VllmModel(vllm_config=vllm_config)


def test_native_pp_stages_follow_the_vllm_layer_partition(native_registry, model_source, qwen_config, vllm_config,
                                                          monkeypatch):
    native = import_module("isoexec.runtimes.vllm.factory")
    vllm_config.additional_config = {"isoexec": {
        "model_source": model_source.to_dict(),
        "raw_model_config": json.loads(qwen_config.to_json_string()),
    }}
    vllm_config.device_config = SimpleNamespace(device=torch.device("cpu"))
    vllm_config.scheduler_config.max_num_batched_tokens = 8
    vllm_config.model_config.hf_text_config = qwen_config.get_text_config()
    vllm_config.parallel_config.pipeline_parallel_size = 2
    monkeypatch.setattr(sys.modules["vllm.distributed"], "get_pp_group",
                        lambda: SimpleNamespace(device_group=torch.distributed.ProcessGroup(0, 2)))
    # A VLLM_PP_LAYER_PARTITION of "2,0" reaches the native stage definition, which still refuses the empty stage.
    monkeypatch.setattr(sys.modules["vllm.distributed.utils"], "get_pp_indices",
                        lambda layers, rank, size: ((0, 2), (2, 2))[rank])

    with pytest.raises(ValueError, match="Stage 1 must contain at least one logical layer"):
        native.VllmModel(vllm_config=vllm_config)


def test_native_registration_captures_decode_graphs_without_compilation(native_registry, qwen_config, model_source):
    raw = json.loads(qwen_config.to_json_string())

    kwargs = registration.register_native(model_source, raw, checkpoint="checkpoint", cudagraph_mode="FULL_DECODE_ONLY",
                                          engine_kwargs={"max_num_seqs": 1024, "compilation_config": {"mode": 3}})

    assert not kwargs["enforce_eager"]
    assert kwargs["compilation_config"] == {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                            "cudagraph_num_of_warmups": 1, "max_cudagraph_capture_size": 1024}
    with pytest.raises(ValueError, match="expected NONE, FULL_DECODE_ONLY or FULL_AND_PIECEWISE"):
        registration.register_native(model_source, raw, checkpoint="checkpoint", cudagraph_mode="PIECEWISE")


def test_native_registration_chunked_prefill_follows_the_gdn_composition(native_registry, qwen_config, model_source,
                                                                         monkeypatch):
    """engine_kwargs' chunked prefill stands under the compositions that resume it exactly (native recurrent
    from vLLM's own state blocks, CPR from its private pool); any other forces it off unless the caller opts
    in explicitly."""
    raw = json.loads(qwen_config.to_json_string())
    asked = {"enable_chunked_prefill": True}

    monkeypatch.setenv("ISOEXEC_GDN_CHUNKED_PREFILL", "")  # register_native exports it; restore it afterwards
    monkeypatch.setenv("ISOEXEC_GDN_KERNEL", "cpr")
    monkeypatch.setenv("ISOEXEC_GDN_NATIVE_KERNELS", "0")  # cpr with eager kernels: no resume path
    assert registration.register_native(model_source, raw, checkpoint="checkpoint",
                                        engine_kwargs=asked)["enable_chunked_prefill"] is False
    monkeypatch.setenv("ISOEXEC_GDN_NATIVE_KERNELS", "1")
    monkeypatch.setenv("ISOEXEC_GDN_NATIVE_STATE", "0")
    assert registration.register_native(model_source, raw, checkpoint="checkpoint",
                                        engine_kwargs=asked)["enable_chunked_prefill"] is True
    assert os.environ["ISOEXEC_GDN_CHUNKED_PREFILL"] == "1"
    monkeypatch.setenv("ISOEXEC_GDN_NATIVE_KERNELS", "0")
    assert registration.register_native(model_source, raw, checkpoint="checkpoint", engine_kwargs=asked,
                                        enable_chunked_prefill=True)["enable_chunked_prefill"] is True
    for name, value in (("ISOEXEC_GDN_KERNEL", "recurrent"), ("ISOEXEC_GDN_NATIVE_KERNELS", "1"),
                        ("ISOEXEC_GDN_NATIVE_STATE", "1")):
        monkeypatch.setenv(name, value)
    assert registration.register_native(model_source, raw, checkpoint="checkpoint",
                                        engine_kwargs=asked)["enable_chunked_prefill"] is True


def test_native_registration_piecewise_selects_breakable_graphs(native_registry, qwen_config, model_source,
                                                                 monkeypatch):
    raw = json.loads(qwen_config.to_json_string())
    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)

    kwargs = registration.register_native(model_source, raw, checkpoint="checkpoint",
                                          cudagraph_mode="FULL_AND_PIECEWISE")

    assert not kwargs["enforce_eager"]
    assert kwargs["compilation_config"]["mode"] == 0
    assert kwargs["compilation_config"]["cudagraph_mode"] == "FULL_AND_PIECEWISE"
    assert os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "1"
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    with pytest.raises(ValueError, match="requires VLLM_USE_BREAKABLE_CUDAGRAPH=1"):
        registration.register_native(model_source, raw, checkpoint="checkpoint", cudagraph_mode="FULL_AND_PIECEWISE")


def test_native_registration_applies_engine_settings_the_model_declares(native_registry, qwen_config, model_source,
                                                                         monkeypatch):
    raw = json.loads(qwen_config.to_json_string())
    pinned = registration.register_native(model_source, raw, checkpoint="checkpoint",
                                          engine_kwargs={"block_size": 64})
    assert pinned["attention_backend"] == "CUSTOM" and pinned["enable_chunked_prefill"] is False
    assert pinned["block_size"] == 64 and "kv_cache_dtype" not in pinned

    meta = model_source.build_meta(raw)
    declared = {"enable_chunked_prefill": True, "attention_backend": None, "kv_cache_dtype": "fp8_ds_mla",
                "block_size": 128}
    monkeypatch.setattr(type(meta), "engine_settings", lambda self: declared, raising=False)
    monkeypatch.setattr(ModelSource, "build_meta", lambda self, config: meta)
    kwargs = registration.register_native(model_source, raw, checkpoint="checkpoint",
                                          engine_kwargs={"attention_backend": "FLASH_ATTN", "block_size": 64})
    assert kwargs["enable_chunked_prefill"] is True and kwargs["kv_cache_dtype"] == "fp8_ds_mla"
    assert kwargs["block_size"] == 128 and "attention_backend" not in kwargs
    # The engine contract itself is not model-selectable.
    monkeypatch.setattr(type(meta), "engine_settings", lambda self: {"enable_prefix_caching": True})
    with pytest.raises(ValueError, match="not model-selectable"):
        registration.register_native(model_source, raw, checkpoint="checkpoint")


def test_native_registration_keeps_vllm_from_quantizing_a_self_decoding_checkpoint(native_registry, qwen_config,
                                                                                    model_source):
    raw = json.loads(qwen_config.to_json_string())
    assert "quantization_config" not in registration.register_native(
        model_source, raw, checkpoint="checkpoint")["hf_overrides"]
    encoded = {**raw, "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32]}}
    kwargs = registration.register_native(model_source, encoded, checkpoint="checkpoint")
    assert kwargs["hf_overrides"]["quantization_config"] is None
    # the package still reads the stored encoding declaration
    assert kwargs["additional_config"]["isoexec"]["raw_model_config"]["quantization_config"]["quant_method"] == "fp8"
