"""The IsoExec rollout server reads verl's converted model config, not the raw Hydra DictConfig."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


def test_server_configures_environment_after_verl_converts_configs(monkeypatch):
    from omegaconf import OmegaConf
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    from isoexec.integrations.verl import rollout

    calls = []

    def base_init(self, config, model_config, *args, **kwargs):
        # verl's own __init__ converts the Hydra DictConfig into the dataclass carrying local_path.
        calls.append("base")
        self.config = SimpleNamespace(calculate_log_probs=True, enable_chunked_prefill=False,
                                      enable_prefix_caching=False, enforce_eager=False,
                                      logprobs_mode="raw_logprobs", dtype="bfloat16", topk_log_probs=0)
        self.model_config = SimpleNamespace(local_path="/models/converted")

    monkeypatch.setattr(vLLMHttpServer, "__init__", base_init)
    monkeypatch.setattr(rollout, "configure_process_environment", lambda path: calls.append(("env", path)))
    monkeypatch.setattr("isoexec.runtimes.vllm.registration.register_plugin", lambda: calls.append("plugin"))
    raw = OmegaConf.create({"path": "Qwen/x"})
    OmegaConf.set_struct(raw, True)  # the struct DictConfig verl hands the actor; it has no local_path
    rollout.IsoExecvLLMHttpServer(OmegaConf.create({}), raw)
    assert calls == ["base", ("env", "/models/converted"), "plugin"]


def test_server_refuses_settings_that_would_change_rows():
    from isoexec.integrations.verl.rollout import validate_rollout_config

    good = dict(calculate_log_probs=True, enable_chunked_prefill=False, enable_prefix_caching=False,
                enforce_eager=False, logprobs_mode="raw_logprobs", dtype="bfloat16", topk_log_probs=0)
    validate_rollout_config(SimpleNamespace(**good))
    with pytest.raises(ValueError, match="logprobs_mode"):
        validate_rollout_config(SimpleNamespace(**{**good, "logprobs_mode": "processed_logprobs"}))


def test_server_admits_chunked_prefill_and_prefix_caching_only_where_the_gdn_state_resumes(monkeypatch):
    from isoexec.integrations.verl.rollout import validate_rollout_config

    on = dict(calculate_log_probs=True, enable_chunked_prefill=True, enable_prefix_caching=True,
              enforce_eager=False, logprobs_mode="raw_logprobs", dtype="bfloat16", topk_log_probs=0)
    for key, value in {"ISOEXEC_GDN_KERNEL": "recurrent", "ISOEXEC_GDN_NATIVE_KERNELS": "1",
                       "ISOEXEC_GDN_NATIVE_STATE": "1"}.items():
        monkeypatch.setenv(key, value)
    validate_rollout_config(SimpleNamespace(**on))
    with pytest.raises(ValueError, match="speculative"):
        validate_rollout_config(SimpleNamespace(**on, mtp=SimpleNamespace(enable=True, enable_rollout=True)))
    monkeypatch.setenv("ISOEXEC_GDN_NATIVE_STATE", "0")
    with pytest.raises(ValueError, match="enable_chunked_prefill=False"):
        validate_rollout_config(SimpleNamespace(**on))
    validate_rollout_config(SimpleNamespace(**{**on, "enable_chunked_prefill": False, "enable_prefix_caching": False}))
    # CPR on its private pool resumes both; the server exports the engine flags they need.
    for key, value in {"ISOEXEC_GDN_KERNEL": "cpr", "ISOEXEC_GDN_NATIVE_KERNELS": "1", "ISOEXEC_GDN_NATIVE_STATE": "0",
                       "ISOEXEC_GDN_CHUNKED_PREFILL": "", "ISOEXEC_GDN_CPR_APC": ""}.items():
        monkeypatch.setenv(key, value)
    validate_rollout_config(SimpleNamespace(**on))
    assert os.environ["ISOEXEC_GDN_CPR_APC"] == "shm" and os.environ["ISOEXEC_GDN_CHUNKED_PREFILL"] == "1"
    monkeypatch.setenv("ISOEXEC_GDN_CPR_APC", "0")
    with pytest.raises(ValueError, match="ISOEXEC_GDN_CPR_APC"):
        validate_rollout_config(SimpleNamespace(**on))


def test_registration_sees_the_rollouts_chunked_prefill_and_prefix_caching():
    """verl hands vLLM the rollout config's values with engine_kwargs applied last, so register_native must see
    them in engine_kwargs (else its decision for an unset value would override the rollout's)."""
    from isoexec.integrations.verl.rollout import carry_state_features

    kwargs = {"enable_prefix_caching": False}
    carry_state_features(SimpleNamespace(enable_chunked_prefill=True, enable_prefix_caching=True), kwargs)
    assert kwargs == {"enable_chunked_prefill": True, "enable_prefix_caching": False}


def test_cudagraph_mode_is_the_callers_or_full_and_piecewise():
    import json

    from isoexec.integrations.verl.rollout import CUDAGRAPH_MODE, cudagraph_mode

    assert CUDAGRAPH_MODE == "FULL_AND_PIECEWISE"
    assert cudagraph_mode({}) == CUDAGRAPH_MODE
    assert cudagraph_mode({"compilation_config": {"cudagraph_capture_sizes": [1, 2]}}) == CUDAGRAPH_MODE
    assert cudagraph_mode({"compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}}) == "FULL_DECODE_ONLY"
    assert cudagraph_mode({"compilation_config": json.dumps({"cudagraph_mode": "NONE"})}) == "NONE"
