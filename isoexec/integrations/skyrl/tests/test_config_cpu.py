"""The IsoExec configuration boundary on SkyRL's real configuration, parser and vLLM argument builder."""

import json
from typing import Any, cast

import pytest
import torch
from safetensors.torch import save_file

from skyrl.train.config import SkyRLTrainConfig

from isoexec.integrations.skyrl import config as integration
from isoexec.integrations.skyrl.config import resolve

PACKAGE = "isoexec.models.qwen3_5"


@pytest.fixture(autouse=True)
def native_driver(monkeypatch):
    # Native dependency admission runs inside the GPU image.
    monkeypatch.setattr("isoexec.integrations.skyrl.config.dependencies.check", lambda **_kwargs: [])
    # register_native installs the engine's process-wide vLLM hooks; this process is not an engine.
    monkeypatch.setattr("isoexec.runtimes.vllm.registration.register_plugin", lambda: None)


@pytest.fixture
def checkpoint(tmp_path):
    """A local artifact the driver resolves: config.json beside one safetensors file."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    save_file({"probe": torch.tensor([1.0])}, str(tmp_path / "model.safetensors"))
    return str(tmp_path)


@pytest.fixture
def qwen_checkpoint(tmp_path):
    """A local artifact with a complete Qwen3.5 HF configuration, as the engine registration reads it."""
    from isoexec.compiler.realization.tests.test_weights_cpu import qwen_config

    (tmp_path / "config.json").write_text(qwen_config().to_json_string())
    save_file({"probe": torch.tensor([1.0])}, str(tmp_path / "model.safetensors"))
    return str(tmp_path)


def overrides(checkpoint: str, **extra: Any) -> list[str]:
    """SkyRL command-line overrides of a resolvable IsoExec run; ``extra`` replaces or adds keys."""
    values = {
        "isoexec": {"model_package": PACKAGE, "trainer_capture_max_tokens": 4096},
        "trainer.strategy": "megatron", "trainer.remove_microbatch_padding": True,
        "trainer.ckpt_interval": -1, "trainer.hf_save_interval": -1, "trainer.resume_mode": None,
        "trainer.policy.model.path": checkpoint, "trainer.ref.model.path": checkpoint,
        "generator.inference_engine.tensor_parallel_size": 4,
        "generator.inference_engine.enable_prefix_caching": False,
        "generator.inference_engine.max_num_batched_tokens": 4096,
        **extra,
    }
    return [f"{key}={json.dumps(value, separators=(',', ':'))}" for key, value in values.items()]


def real(checkpoint: str, **extra: Any):
    return SkyRLTrainConfig.from_cli_overrides(overrides(checkpoint, **extra))


def config(checkpoint: str = "tiny", **extra: Any):
    """A real SkyRL configuration of an IsoExec run (policy only) for tests that set their own fields."""
    return real(checkpoint, **{"trainer.algorithm.use_kl_loss": False, "trainer.algorithm.use_kl_in_reward": False,
                               "isoexec": {"model_package": PACKAGE}, **extra})


def resolved(cfg):
    result = resolve(cfg)
    assert result is not None, "These cases explicitly enable the integration"
    return result


def test_validate_config_resolves_the_run_and_hands_skyrl_the_actor_environment(checkpoint):
    """``bootstrap.validate_config`` is what SkyRL's ``validate_cfg`` calls in the driver; SkyRL adds
    ``trainer.extra_runtime_env_vars`` to the Ray runtime environment of every actor."""
    from isoexec.integrations.skyrl.bootstrap import validate_config

    cfg = real(checkpoint, **{"trainer.extra_runtime_env_vars": {"USER_FLAG": "1"}})
    options = validate_config(cfg)
    assert cfg.trainer._isoexec_config is options
    extra = cfg.trainer.extra_runtime_env_vars
    assert extra["USER_FLAG"] == "1" and extra["VLLM_PLUGINS"] == "isoexec,isoexec_skyrl"
    assert all(isinstance(value, str) for value in extra.values()), "Ray refuses non-string environment values"


def test_defaults_resolve_to_model_runner_v2_graphs(checkpoint):
    cfg = real(checkpoint)
    result = resolved(cfg)
    assert result.engine_tp == 4 and result.cudagraph_mode == "FULL_AND_PIECEWISE"
    assert result.logprob_bound == 0 and result.trainer_capture_max_tokens == 4096
    assert result.full_distribution_comparison is False
    assert cfg.trainer._isoexec_config is result and result.model_descriptor is cfg.isoexec
    cfg.generator.inference_engine.enforce_eager = True
    assert resolved(cfg).cudagraph_mode == "NONE"


@pytest.mark.parametrize("mode", ["NONE", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"])
def test_every_v2_graph_mode_is_admitted(checkpoint, mode):
    cfg = real(checkpoint, **{"generator.inference_engine.engine_init_kwargs": {
        "compilation_config": {"mode": 0, "cudagraph_mode": mode}}})
    assert resolved(cfg).cudagraph_mode == mode


@pytest.mark.parametrize("compilation,match", [
    ({"mode": 0, "cudagraph_mode": "PIECEWISE"}, "cudagraph_mode in"),
    ({"mode": 0, "cudagraph_mode": "FULL"}, "cudagraph_mode in"),
    ({"mode": 3}, "compilation_config.mode=0"),
])
def test_compiled_or_unbreakable_graphs_are_refused(checkpoint, compilation, match):
    cfg = real(checkpoint, **{"generator.inference_engine.engine_init_kwargs": {"compilation_config": compilation}})
    with pytest.raises(ValueError, match=match):
        resolved(cfg)


def test_chunked_prefill_is_requested_through_the_engine_kwargs_only(checkpoint):
    cfg = real(checkpoint)
    assert cfg.generator.inference_engine.enable_chunked_prefill  # SkyRL's default, not plumbed to the engine
    assert resolved(cfg).enable_chunked_prefill is False
    cfg.generator.inference_engine.engine_init_kwargs = {"enable_chunked_prefill": True}
    assert resolved(cfg).enable_chunked_prefill is True


def test_resolved_descriptor_reaches_native_actor_config(checkpoint, monkeypatch):
    import ray
    import ray.cloudpickle
    from types import SimpleNamespace as NS

    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup

    cfg = real(checkpoint)
    options = resolved(cfg)
    assert "_resolved_source_config" in options.model_descriptor
    serializer = cast(Any, ray.cloudpickle)
    received = []

    class Actor:
        def options(self, **kwargs):
            return self

        def remote(self, **kwargs):
            received.append(serializer.loads(serializer.dumps(kwargs)))
            return NS(get_master_addr_port=NS(remote=lambda: ("localhost", 23456)),
                      init_worker_process_group=NS(remote=lambda: None),
                      get_mesh_rank=NS(remote=lambda: NS(dp_size=2)))

    monkeypatch.setattr(ray, "get", lambda value: value)
    group = PPORayActorGroup(cfg.trainer, 2, 1, cast(Any, Actor()))
    assert len(group._actor_handlers) == len(received) == 2
    for rank, arguments in enumerate(received):
        assert arguments["rank"] == rank
        assert "isoexec" not in arguments
        assert arguments["cfg"]._isoexec_config.model_descriptor == cfg.isoexec


def test_trainer_pipeline_and_context_parallel_degrees(checkpoint):
    """PP>1 resolves here (the mesh and the Megatron host own its admission); CP resolves at 1 or 2."""
    cfg = real(checkpoint)
    mc = cfg.trainer.policy.megatron_config
    mc.pipeline_model_parallel_size = 2
    assert resolved(cfg).engine_tp == 4
    for bad in (0, -1, True, "2", None):
        mc.pipeline_model_parallel_size = bad
        with pytest.raises(ValueError, match="pipeline_model_parallel_size"):
            resolved(cfg)
    mc.pipeline_model_parallel_size = 2
    mc.context_parallel_size = 2
    assert resolved(cfg).engine_tp == 4
    for bad in (0, 3, 4, True, "2", None):
        mc.context_parallel_size = bad
        with pytest.raises(ValueError, match="context_parallel_size in \\(1, 2\\)"):
            resolved(cfg)


@pytest.mark.parametrize("key,value,match", [
    ("trainer.strategy", "fsdp", "trainer.strategy"),
    ("trainer.policy.megatron_config.freeze_dsa_indexer", True, "freeze_dsa_indexer=False"),
    ("trainer.algorithm.enable_sample_support_replay", True, "enable_sample_support_replay=False"),
    ("trainer.fused_lm_head_logprob", True, "fused_lm_head_logprob=False"),
    ("generator.inference_engine.weight_sync_backend", "delta", "weight_sync_backend='nccl'"),
    ("generator.inference_engine.engine_init_kwargs", {"logprobs_mode": "processed_logprobs"}, "logprobs_mode unset"),
    ("generator.inference_engine.enable_pd", True, "enable_pd"),
    ("generator.inference_engine.external_proxy_url", "http://localhost:8000", "external inference servers"),
    ("generator.inference_engine.external_server_urls", ["http://localhost:8001"], "external inference servers"),
])
def test_settings_the_isoexec_model_cannot_honor_refuse_before_workers(checkpoint, key, value, match):
    cfg = real(checkpoint)
    section, _, name = key.rpartition(".")
    target = cfg
    for part in section.split("."):
        target = getattr(target, part)
    setattr(target, name, value)
    with pytest.raises(ValueError, match=match):
        resolved(cfg)


def test_full_vocabulary_comparison_resolves_for_skyrl(checkpoint):
    cfg = real(checkpoint)
    cfg.isoexec["full_logprob_comparison"] = True
    assert resolved(cfg).full_distribution_comparison is True


def test_a_weight_file_is_not_a_checkpoint(checkpoint):
    cfg = real(checkpoint)
    cfg.trainer.policy.model.path = f"{checkpoint}/model.safetensors"
    with pytest.raises(ValueError, match="not a weight file"):
        resolved(cfg)


def test_missing_native_source_fails_before_build(checkpoint):
    cfg = real(checkpoint)
    cfg.isoexec = {}
    with pytest.raises(ValueError, match="isoexec.model_package"):
        resolved(cfg)


def test_null_config_skips_integration_before_dependencies(monkeypatch):
    monkeypatch.setattr("isoexec.integrations.skyrl.config.dependencies.check",
                        lambda **kwargs: pytest.fail("disabled integration must not check its dependencies"))
    assert resolve(SkyRLTrainConfig()) is None


@pytest.mark.parametrize("selected", [{}, {"model_package": ""}, {"model_package": None},
                                     {"model_package": True}, {"source": {"package": PACKAGE}},
                                     {"model": PACKAGE}, False, PACKAGE])
def test_invalid_config_refuses_before_dependencies(monkeypatch, selected):
    monkeypatch.setattr("isoexec.integrations.skyrl.config.dependencies.check",
                        lambda **kwargs: pytest.fail("invalid configuration must fail before dependency admission"))
    cfg = SkyRLTrainConfig()
    cfg.isoexec = selected
    with pytest.raises(ValueError, match="isoexec"):
        resolve(cfg)


@pytest.mark.parametrize("colocated", [False, True])
def test_engine_expert_parallelism_resolves_only_across_the_engines_data_parallel_ranks(checkpoint, colocated):
    cfg = real(checkpoint, **{"trainer.placement.colocate_all": colocated})
    engine = cfg.generator.inference_engine
    engine.tensor_parallel_size, engine.data_parallel_size, engine.expert_parallel_size = 1, 2, 2
    assert resolved(cfg).engine_ep == 2
    engine.tensor_parallel_size, engine.expert_parallel_size = 2, 4
    assert resolved(cfg).engine_ep == 4
    for tp, dp, ep in ((1, 1, 2), (1, 2, 4), (2, 2, 2)):
        engine.tensor_parallel_size, engine.data_parallel_size, engine.expert_parallel_size = tp, dp, ep
        with pytest.raises(ValueError, match="only across the engine's data-parallel ranks"):
            resolved(cfg)
    engine.tensor_parallel_size, engine.data_parallel_size, engine.expert_parallel_size = 1, 2, 0
    with pytest.raises(ValueError, match="positive integer"):
        resolved(cfg)


def test_worker_environment_selects_model_runner_v2_and_breakable_graphs(checkpoint, monkeypatch, tmp_path):
    import os

    from isoexec.runtimes import trainer_installs as host

    assert host.TRAINER_ENV
    monkeypatch.setenv("ISOEXEC_TEST_FLAG", "job")
    monkeypatch.setenv("NCCL_ALGO", "allreduce:tree")
    monkeypatch.setenv("NCCL_MAX_NCHANNELS", "1")
    private_mm = str(tmp_path / "private-mm")
    monkeypatch.setenv("ISOEXEC_MM_SM90_BUILD_DIR", private_mm)
    cfg = real(checkpoint)
    options = resolved(cfg)
    before = dict(os.environ)
    env = integration._worker_environment(cfg, options)
    assert dict(os.environ) == before
    assert cfg.trainer.extra_runtime_env_vars == env
    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "1" and env["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "1"
    assert env["VLLM_PLUGINS"] == "isoexec,isoexec_skyrl" and env["VLLM_ISOEXEC_REGISTERED"] == "1"
    assert env["VLLM_BATCH_INVARIANT"] == "1" and env["VLLM_USE_AOT_COMPILE"] == "0"
    assert env["ISOEXEC_MODEL_PATH"] == checkpoint and env["ISOEXEC_TEST_FLAG"] == "job"
    assert env["ISOEXEC_MM_SM90_BUILD_DIR"] == private_mm
    assert not {"NCCL_ALGO", "NCCL_MAX_NCHANNELS", "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"} & env.keys()
    eager = real(checkpoint, **{"generator.inference_engine.enforce_eager": True})
    resolved(eager)
    assert "VLLM_USE_BREAKABLE_CUDAGRAPH" not in eager.trainer.extra_runtime_env_vars


@pytest.mark.parametrize("colocated,receiver", [(True, "isoexec_ipc"), (False, "isoexec_nccl")])
def test_engine_configuration_registers_the_native_model_on_skyrls_own_server_arguments(qwen_checkpoint, colocated, receiver):
    checkpoint = qwen_checkpoint
    pytest.importorskip("vllm")
    from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args

    cfg = real(checkpoint, **{"trainer.placement.colocate_all": colocated,
                              "generator.inference_engine.tensor_parallel_size": 1,
                              "generator.inference_engine.engine_init_kwargs": {"max_model_len": 2048}})
    options = resolved(cfg)
    args = build_vllm_cli_args(cfg)
    assert args.logprobs_mode == "processed_logprobs" and args.weight_transfer_config.backend.startswith("skyrl_")
    integration.configure_isoexec_vllm_engine(cfg, args)
    assert args.model == args.tokenizer == checkpoint
    assert args.logprobs_mode == "raw_logprobs" and args.enforce_eager is False
    assert args.compilation_config["mode"] == 0
    assert args.compilation_config["cudagraph_mode"] == options.cudagraph_mode == "FULL_AND_PIECEWISE"
    assert args.hf_overrides["architectures"] in (["VllmModel"], ["DenseVllmModel"])
    assert args.additional_config["isoexec"]["score_temperature"] == options.score_temperature
    assert args.weight_transfer_config.backend == receiver
    # Weight updates are verified only under the full comparison, which this run leaves off.
    assert args.additional_config["isoexec"]["verify_weight_updates"] is False
    # SkyRL's own settings survive: the placement-derived ones and engine_init_kwargs.
    assert args.max_model_len == 2048 and args.tensor_parallel_size == 1
    assert args.enable_sleep_mode is colocated


def test_engine_configuration_refuses_a_model_other_than_the_resolved_policy(qwen_checkpoint):
    checkpoint = qwen_checkpoint
    pytest.importorskip("vllm")
    from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args

    cfg = real(checkpoint)
    resolved(cfg)
    args = build_vllm_cli_args(cfg)
    args.model = "another/model"
    with pytest.raises(ValueError, match="resolved policy checkpoint"):
        integration.configure_isoexec_vllm_engine(cfg, args)
