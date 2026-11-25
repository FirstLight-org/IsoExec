"""Native provider contracts with CPU allocation and DDP/optimizer collaborators."""

import importlib
import json
import math
import re
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import transformers
from safetensors.torch import load_file, save_file
from torch import nn

from isoexec.integrations.skyrl.config import SkyRLConfig
from isoexec.integrations.skyrl import config as integration_config
from isoexec.integrations.skyrl.tests.test_config_cpu import config as skyrl_config
from isoexec.compiler.realization.tests.test_weights_cpu import qwen_config
from isoexec.compiler.realization.weights import ignored_patterns
from isoexec.compiler.source import ModelSource
from isoexec.ops.table import OpTable


@pytest.fixture
def framework(monkeypatch):
    from megatron.core import distributed, optimizer
    from megatron.core.process_groups_config import ProcessGroupCollection

    from isoexec.ops.collectives import pik_tp_invariant
    from isoexec.runtimes import trainer_installs
    from isoexec.runtimes.megatron import _binding, model

    # SkyRL's real configuration helpers (get_config_as_dict, init_megatron_optim_config) build the
    # DDP and optimizer configurations; only the device-bound collaborators below are CPU stand-ins.
    integration = importlib.import_module("isoexec.integrations.skyrl.megatron")
    installs, transports = [], []

    def install(path, *, source, raw_config, deterministic_attention_backward):
        installs.append((path, source, raw_config, deterministic_attention_backward))
        return {}

    def select_transport(side):
        transports.append(side)
        return "nccl"

    monkeypatch.setattr(trainer_installs, "install_trainer_kernels", install)
    monkeypatch.setattr(pik_tp_invariant, "select_trainer_transport", select_transport)
    group = torch.distributed.ProcessGroup(0, 1)
    groups = SimpleNamespace(tp=group, pp=group)
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", lambda: groups)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    allocate = model.allocate_tensors
    monkeypatch.setattr(model, "allocate_tensors", lambda module, **kwargs: allocate(module, device=torch.device("cpu")))
    reference = OpTable(mode="reference")
    monkeypatch.setattr(_binding, "OpTable", lambda: reference)
    monkeypatch.setattr(_binding, "bind_row_parallel",
                        lambda **kwargs: reference.bind("collectives.row_parallel", site=kwargs["site"]))
    factory = importlib.import_module("isoexec.runtimes.megatron.factory")
    inputs = factory.text_inputs
    bounds = []

    def cpu_inputs(author, **kwargs):
        bounds.append(kwargs["max_tokens"])
        return inputs(author, **{**kwargs, "device": "cpu"})

    monkeypatch.setattr(factory, "text_inputs", cpu_inputs)

    class DDP(nn.Module):
        def __init__(self, config, ddp_config, module, *, pg_collection):
            super().__init__()
            self.config, self.ddp_config, self.module = config, ddp_config, module
            self.pg_collection = pg_collection

        def no_sync(self):
            raise AssertionError("CPU provider test executed DDP synchronization")

    def optimizer_for(config, model_chunks, **kwargs):
        return SimpleNamespace(config=config, model_chunks=model_chunks, pg_collection=kwargs["pg_collection"],
                               param_groups=[{"params": list(model_chunks[0].parameters())}])

    monkeypatch.setattr(distributed, "DistributedDataParallel", DDP)
    monkeypatch.setattr(optimizer, "get_megatron_optimizer", optimizer_for)
    return SimpleNamespace(integration=integration, model=model, groups=groups, bounds=bounds,
                           installs=installs, transports=transports)


def test_tp2_provider_keeps_current_parameters_in_ddp_and_optimizer(framework, worker, monkeypatch):
    from isoexec.ops import _dispatch
    from isoexec.runtimes.megatron import _binding

    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    production, reference = OpTable(), OpTable(mode="reference")

    def bind(op, **kwargs):
        return (production if op == "logprobs.log_softmax" else reference).bind(op, **kwargs)

    monkeypatch.setattr(_binding, "OpTable", lambda: SimpleNamespace(bind=bind))
    framework.groups.tp = torch.distributed.ProcessGroup(0, 2)
    worker.cfg.policy.megatron_config.tensor_model_parallel_size = 2
    modules = framework.integration.build_module(
        worker, wrap_with_ddp=True, ddp_config=None, lora_config=None, bf16=True,
    )
    model = modules[0].module
    graph = model.graph_module
    assert graph.embed.weight.shape == (64, 16) and graph.head.weight is graph.embed.weight
    assert graph.get_parameter("layers.0.mlp.down").shape == (16, 16)
    assert graph.get_submodule("layers.1.mixer").kv_heads_local == 1
    from isoexec.compiler.capture.export_normalize import _checked_output

    guard = next(node for node in graph.graph.nodes if node.target is _checked_output and node.args[2] == "head")
    assert guard.args[3][-1] == guard.args[4][-2] == 64
    assert all(not parameter.is_meta for parameter in model.parameters())
    assert graph.embed.weight.tensor_model_parallel
    assert not graph.norm.weight.tensor_model_parallel
    assert modules[0].pg_collection is framework.groups
    assert {id(parameter) for group in worker._isoexec_optimizer.param_groups for parameter in group["params"]} == {
        id(parameter) for parameter in model.parameters()
    }


@pytest.mark.parametrize("rank", [0, 1])
def test_pp2_provider_loads_local_layers_and_shared_hf_aliases(framework, worker, rank):
    framework.groups.pp = torch.distributed.ProcessGroup(rank, 2)
    worker.cfg.policy.megatron_config.pipeline_model_parallel_size = 2
    modules = framework.integration.build_module(
        worker, wrap_with_ddp=True, ddp_config=None, lora_config=None, bf16=True,
    )
    model = modules[0].module
    graph = model.graph_module
    assert model.pre_process == (rank == 0) and model.post_process == (rank == 1)
    assert model._mrope
    assert set(graph.layers._modules) == {str(rank)}
    shared = model.shared_embedding_or_output_weight()
    assert shared.shared_embedding
    assert bool(getattr(shared, "shared", False)) == (rank == 1)
    assert torch.equal(shared, torch.full_like(shared, 0.25))
    assert all(not parameter.is_meta for parameter in model.parameters())
    assert {id(parameter) for group in worker._isoexec_optimizer.param_groups for parameter in group["params"]} == {
        id(parameter) for parameter in model.parameters()
    }


@pytest.fixture(params=["original", "renamed_without_config"])
def worker(tmp_path, framework, monkeypatch, request):
    source = ModelSource("isoexec.models.qwen3_5")
    if request.param != "original":
        package = tmp_path / ("custom_weight_author" if request.param == "custom_weights" else "another_author")
        package.mkdir()
        (package / "model.py").write_text(
            "from isoexec.models.qwen3_5.model import Qwen3_5ForConditionalGeneration as Base\n"
            "class Qwen3_5ForConditionalGeneration(Base):\n"
            "    def __init__(self, config):\n"
            "        super().__init__(config)\n"
            "        del self.config\n"
        )
        if request.param == "custom_weights":
            (package / "weight.py").write_text(
                "from transformers.core_model_loading import WeightRenaming\n"
                "from isoexec.models.qwen3_5.weight import checkpoint_conversion_mapping as base_mapping\n"
                "def checkpoint_conversion_mapping(config):\n"
                "    return [WeightRenaming(r'^custom\\.', ''), *base_mapping(config)]\n"
            )
        else:
            (package / "weight.py").write_text(
                "from isoexec.models.qwen3_5.weight import checkpoint_conversion_mapping\n"
            )
        monkeypatch.syspath_prepend(str(tmp_path))
        source = ModelSource(package.name)
    raw = json.loads(qwen_config().to_json_string())
    author = source.build_meta(raw)
    # HF's own model names the checkpoint tensors that conversion consumes.
    config = source.build_config(raw)
    with torch.device("meta"):
        reference = transformers.Qwen3_5ForConditionalGeneration(config)
    ignored = ignored_patterns(config)
    weights = {name: torch.full(tensor.shape, 0.25) for name, tensor in reference.state_dict().items()
               if not any(re.search(pattern, name) for pattern in ignored)}
    if request.param == "custom_weights":
        weights = {f"custom.{name}": tensor for name, tensor in weights.items()}
    save_file(weights, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(raw))
    from skyrl.train.config.config import TrainerConfig

    cfg = TrainerConfig()
    cfg.gradient_checkpointing = False
    for role, deterministic in (("policy", True), ("ref", False)):
        section = getattr(cfg, role)
        section.model.path = str(tmp_path)
        mc = section.megatron_config
        mc.tensor_model_parallel_size = mc.pipeline_model_parallel_size = mc.context_parallel_size = 1
        mc.expert_model_parallel_size = mc.expert_tensor_parallel_size = 1
        mc.transformer_config_kwargs["deterministic_mode"] = deterministic
    cfg.policy.optimizer_config.lr = 3e-5
    cfg._isoexec_config = SkyRLConfig(trainer_capture_max_tokens=16, model_descriptor={"model_package": source.package})
    worker = SimpleNamespace(cfg=cfg, _isoexec_section_name="policy")
    monkeypatch.setattr(framework.integration.model_folder, "source_config", lambda *args: (source, raw))
    worker.source, worker.raw = source, raw
    worker.parameter_dtypes = {name: parameter.dtype for name, parameter in author.named_parameters(remove_duplicate=False)}
    yield worker
    if request.param != "original":
        for name in (f"{source.package}.model", f"{source.package}.weight", source.package):
            sys.modules.pop(name, None)


@pytest.mark.parametrize("role,training", [("policy", True), ("policy", False), ("ref", False)])
def test_policy_and_reference_use_loaded_native_parameters(framework, worker, role, training):
    worker._isoexec_section_name = role
    original = deepcopy(worker.raw)
    modules = framework.integration.build_module(
        worker, wrap_with_ddp=training,
        ddp_config=getattr(worker.cfg, role).megatron_config.ddp_config,
        lora_config=None, bf16=True,
    )
    wrapped = modules[0]
    model = wrapped.module if training else wrapped

    assert isinstance(model, framework.model.MegatronModel)
    assert model.source == worker.source
    assert model.training is model.graph_module.training is training
    assert model.hf_config.to_dict() == worker.source.build_config(worker.raw).to_dict()
    assert model.pg_collection is framework.groups
    assert model.config.deterministic_mode is (role == "policy")
    assert model.config.hidden_size == 16 and model.config.num_layers == 2
    assert model.config.bf16 and model.config.pipeline_dtype == torch.bfloat16
    assert not model.config.gradient_accumulation_fusion and model.config.variable_seq_lengths
    assert model.graph_module.embed.weight is model.graph_module.head.weight
    assert {name: parameter.dtype for name, parameter in model.graph_module.named_parameters(remove_duplicate=False)} == (
        worker.parameter_dtypes
    )
    assert torch.equal(model.graph_module.embed.weight, torch.full_like(model.graph_module.embed.weight, 0.25))
    assert framework.bounds == [16] and worker.raw == original
    # The census inputs are the same source and config the model was built from.
    assert [(path, source, deterministic) for path, source, _, deterministic in framework.installs] == [
        (getattr(worker.cfg, role).model.path, worker.source, role == "policy")
    ]
    assert worker.source.build_config(framework.installs[0][2]).to_dict() == model.hf_config.to_dict()
    assert framework.transports == ["TRAINER"]
    if training:
        optimizer = worker._isoexec_optimizer
        assert optimizer.model_chunks == modules and optimizer.pg_collection is framework.groups
        assert optimizer.config.lr == 3e-5
        assert wrapped.ddp_config.grad_reduce_in_fp32 and not wrapped.ddp_config.overlap_grad_reduce
        assert all(actual is expected for actual, expected in
                   zip(optimizer.param_groups[0]["params"], model.parameters(), strict=True))
        assert model.config.no_sync_func == wrapped.no_sync
        assert model.config.finalize_model_grads_func.keywords == {"pg_collection": framework.groups}
    else:
        assert worker._isoexec_optimizer is None


def test_initial_loading_requires_every_registered_parameter(framework, worker):
    checkpoint = worker.cfg.policy.model.path + "/model.safetensors"
    weights = load_file(checkpoint)
    del weights["model.language_model.layers.0.mlp.gate_proj.weight"]
    save_file(weights, checkpoint)

    with pytest.raises(ValueError, match=r"missing model parameters.*graph_module.layers.0.mlp.gate"):
        framework.integration.build_module(worker, wrap_with_ddp=True, ddp_config=None, lora_config=None, bf16=True)


@pytest.mark.parametrize("bf16,recompute,message", [
    (False, False, "trainer.bf16=True"),
])
def test_native_construction_rejects_unsupported_settings(framework, worker, bf16, recompute, message):
    worker.cfg.gradient_checkpointing = recompute

    with pytest.raises(ValueError, match=re.escape(message)):
        framework.integration.build_module(worker, wrap_with_ddp=True, ddp_config=None, lora_config=None, bf16=bf16)
    assert not framework.installs


@pytest.mark.parametrize("span", [1, 2, 3])
def test_native_construction_enables_full_layer_checkpointing(framework, worker, span):
    worker.cfg.gradient_checkpointing = True
    worker.cfg.policy.megatron_config.transformer_config_kwargs["recompute_num_layers"] = span
    wrapped, = framework.integration.build_module(
        worker, wrap_with_ddp=True, ddp_config=None, lora_config=None, bf16=True,
    )
    model = wrapped.module
    assert model.config.recompute_granularity == "full"
    assert model.config.recompute_num_layers == span
    assert sum(len(paths) for paths in model.activation_checkpoint_layers) == 2
    assert len(model.activation_checkpoint_layers) == (2 if span == 1 else 1)
    model.set_activation_checkpointing(False)
    assert all(not region.enabled for region in model._activation_checkpoint_regions)
    model.set_activation_checkpointing(True)
    assert all(region.enabled for region in model._activation_checkpoint_regions)


def test_native_checkpointing_rejects_invalid_spans_before_kernel_install(framework, worker):
    worker.cfg.gradient_checkpointing = True
    worker.cfg.policy.megatron_config.transformer_config_kwargs["recompute_num_layers"] = 0
    with pytest.raises(ValueError, match="recompute_num_layers must be positive"):
        framework.integration.build_module(
            worker, wrap_with_ddp=True, ddp_config=None, lora_config=None, bf16=True,
        )
    assert not framework.installs


def test_precision_aware_optimizer_installs_mixed_dtype_grad_kernels(framework, worker, monkeypatch):
    from isoexec.runtimes.megatron import grad_kernels

    installed = []
    monkeypatch.setattr(grad_kernels, "install_mixed_dtype_grad_kernels", lambda: installed.append(True))
    worker.cfg.policy.megatron_config.optimizer_config_kwargs["use_precision_aware_optimizer"] = True
    # HybridDeviceOptimizer admits precision-aware Adam on CPU without the
    # unavailable Transformer Engine FusedAdam extension on this test host.
    worker.cfg.policy.megatron_config.optimizer_config_kwargs["optimizer_cpu_offload"] = True

    framework.integration.build_module(worker, wrap_with_ddp=True, ddp_config=None, lora_config=None, bf16=True)

    assert worker._isoexec_optimizer.config.use_precision_aware_optimizer
    assert installed == [True]


def test_wrapper_scores_with_the_registered_head(framework, worker):
    from isoexec.integrations.skyrl import scoring

    modules = framework.integration.build_module(
        worker, wrap_with_ddp=False, ddp_config=None, lora_config=None, bf16=True,
    )
    wrapper = SimpleNamespace(actor_module=modules, remove_microbatch_padding=True)

    assert scoring._find_isoexec_stage(modules) is modules[0]
    assert scoring.lm_head_log_probs(wrapper) == modules[0].graph_module.head.log_probs
    wrapper.remove_microbatch_padding = False
    with pytest.raises(ValueError, match="remove_microbatch_padding=true"):
        scoring.lm_head_log_probs(wrapper)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_model_tempers_its_logits_with_the_engines_fp32_tensor_divide(framework, worker, dtype):
    from isoexec.ops.logprobs.temperature import temperature_divide

    model, = framework.integration.build_module(
        worker, wrap_with_ddp=False, ddp_config=None, lora_config=None, bf16=True,
    )
    logits = torch.randn(5, 64, generator=torch.Generator().manual_seed(0)).to(dtype)
    assert model.score_temperature == 1.0 and model._tempered(logits) is logits
    model.score_temperature = 0.7
    tempered = model._tempered(logits)
    expected = logits.float() / torch.tensor(0.7, dtype=torch.float32)
    assert tempered.dtype == torch.float32
    assert torch.equal(tempered.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(tempered.view(torch.int32), temperature_divide(logits.float(), 0.7).view(torch.int32))


def test_rollout_uses_native_registration_and_preserves_transport(worker, monkeypatch):
    from isoexec.runtimes.vllm import registration
    monkeypatch.setattr("isoexec.integrations.skyrl.weight_transfer.require_plugins", lambda: None)

    cfg = skyrl_config()
    cfg.trainer.algorithm.temperature = 0.7
    cfg.trainer.policy.model.path = worker.cfg.policy.model.path
    cfg.generator.inference_engine.tensor_parallel_size = 1
    # Without an explicit cudagraph_mode, an eager engine resolves to NONE.
    cfg.generator.inference_engine.enforce_eager = True
    monkeypatch.setattr(integration_config, "prepare_checkpoints", lambda *args: None)
    monkeypatch.setattr(integration_config, "source_config", lambda *args: (worker.source, worker.raw))
    monkeypatch.setattr(integration_config.dependencies, "check", lambda **kwargs: [])
    monkeypatch.setattr("isoexec.integrations.skyrl.trainer_profile.trainer_capture_bound", lambda cfg: 16)
    monkeypatch.setattr(registration, "register_plugin", lambda: None)
    monkeypatch.setenv("VLLM_PLUGINS", "")
    transport = {"backend": "skyrl_ipc"}
    args = SimpleNamespace(model=worker.cfg.policy.model.path, tokenizer=None,
                           additional_config={"keep": True}, weight_transfer_config=transport,
                           worker_extension_cls="skyrl.WorkerExtension")
    original = deepcopy(worker.raw)

    integration_config.resolve(cfg)
    integration_config.configure_isoexec_vllm_engine(cfg, args)

    assert args.hf_overrides == {"architectures": [registration.NATIVE_ARCH]}
    assert args.load_format == "auto" and args.dtype == "bfloat16" and args.enforce_eager
    assert args.model == args.tokenizer == worker.cfg.policy.model.path
    assert args.weight_transfer_config.backend == "isoexec_ipc"
    assert transport == {"backend": "skyrl_ipc"} and args.worker_extension_cls == "skyrl.WorkerExtension"
    assert args.additional_config["keep"]
    assert args.additional_config["isoexec"]["model_source"] == worker.source.to_dict()
    assert args.additional_config["isoexec"]["raw_model_config"] == worker.raw == original
    assert args.additional_config["isoexec"]["score_temperature"] == 0.7
    assert cfg.trainer._isoexec_config.cudagraph_mode == "NONE"
    assert not cfg.trainer._isoexec_config.enable_chunked_prefill
