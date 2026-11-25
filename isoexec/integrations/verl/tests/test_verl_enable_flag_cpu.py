"""``isoexec.enable=true`` expands verl's real composed config into the IsoExec settings (needs verl on the path)."""

from __future__ import annotations

from pathlib import Path

import pytest


def _compose(*overrides):
    hydra = pytest.importorskip("hydra")
    import verl

    config_dir = str(Path(verl.__file__).parent / "trainer" / "config")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        return hydra.compose(config_name="ppo_trainer", overrides=list(overrides))


@pytest.fixture(autouse=True)
def _restore_exported_flags(monkeypatch):
    """``apply`` exports the engine flags an enabled feature needs; restore them after every test."""
    import os

    for key in ("ISOEXEC_GDN_CHUNKED_PREFILL", "ISOEXEC_GDN_CPR_APC"):
        monkeypatch.setenv(key, os.environ.get(key, ""))


def test_flag_defaults_off_and_expands_every_component(monkeypatch):
    from omegaconf import OmegaConf

    from isoexec.integrations.verl import config as isoexec_config

    # async_scheduling follows GDN state ownership (test_async_scheduling_follows_gdn_state_ownership);
    # pin the default, private-pool state here so the expectation does not depend on the caller's env.
    monkeypatch.delenv("ISOEXEC_GDN_KERNEL", raising=False)
    monkeypatch.delenv("ISOEXEC_GDN_NATIVE_STATE", raising=False)
    base = _compose("model_engine=megatron")
    assert base.isoexec.enable is False
    enabled = _compose("model_engine=megatron", "isoexec.enable=true",
                       "actor_rollout_ref.rollout.tensor_model_parallel_size=2")
    changes = isoexec_config.apply(enabled)
    assert changes
    for key, value in isoexec_config.REQUIRED.items():
        assert OmegaConf.select(enabled, key) == value, key
    assert enabled.actor_rollout_ref.model.external_lib == isoexec_config.REGISTER
    assert enabled.actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling is False
    # Caller-owned settings are untouched, and a second application changes nothing.
    assert enabled.actor_rollout_ref.rollout.tensor_model_parallel_size == 2
    assert isoexec_config.apply(enabled) == []


def test_async_scheduling_follows_gdn_state_ownership(monkeypatch):
    from isoexec.integrations.verl import config as isoexec_config

    monkeypatch.setenv("ISOEXEC_GDN_KERNEL", "recurrent")
    monkeypatch.setenv("ISOEXEC_GDN_NATIVE_STATE", "1")
    native = _compose("model_engine=megatron", "isoexec.enable=true")
    isoexec_config.apply(native)
    assert native.actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling is True
    monkeypatch.setenv("ISOEXEC_GDN_NATIVE_STATE", "0")
    pooled = _compose("model_engine=megatron", "isoexec.enable=true")
    isoexec_config.apply(pooled)
    assert pooled.actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling is False
    # CPR keeps a private pool driven by the scheduler's own host output on Model Runner V2: async is safe.
    monkeypatch.setenv("ISOEXEC_GDN_KERNEL", "cpr")
    cpr = _compose("model_engine=megatron", "isoexec.enable=true")
    isoexec_config.apply(cpr)
    assert cpr.actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling is True


def test_flag_keeps_other_external_libraries():
    from isoexec.integrations.verl import config as isoexec_config

    enabled = _compose("model_engine=megatron", "isoexec.enable=true", "actor_rollout_ref.model.external_lib=my.lib")
    isoexec_config.apply(enabled)
    assert list(enabled.actor_rollout_ref.model.external_lib) == ["my.lib", isoexec_config.REGISTER]


_NATIVE_GDN = {"ISOEXEC_GDN_KERNEL": "recurrent", "ISOEXEC_GDN_NATIVE_KERNELS": "1", "ISOEXEC_GDN_NATIVE_STATE": "1",
               "ISOEXEC_GDN_CPR_MIN_PAGES": "0"}
_CPR_GDN = {"ISOEXEC_GDN_KERNEL": "cpr", "ISOEXEC_GDN_NATIVE_KERNELS": "1", "ISOEXEC_GDN_NATIVE_STATE": "0",
            "ISOEXEC_GDN_CPR_MIN_PAGES": "1", "ISOEXEC_GDN_CHUNKED_PREFILL": "1"}
_KEYS = ("actor_rollout_ref.rollout.enable_chunked_prefill", "actor_rollout_ref.rollout.enable_prefix_caching")


@pytest.mark.parametrize("overrides, want", [
    ((), (True, True)),  # verl's defaults
    (("actor_rollout_ref.rollout.enable_prefix_caching=false",), (True, False)),
    (("actor_rollout_ref.rollout.enable_chunked_prefill=false", "actor_rollout_ref.rollout.enable_prefix_caching=false"),
     (False, False)),
], ids=["defaults", "no-prefix-cache", "both-off"])
def test_native_gdn_composition_keeps_the_callers_chunked_prefill_and_prefix_caching(monkeypatch, overrides, want):
    from omegaconf import OmegaConf

    from isoexec.integrations.verl import config as isoexec_config

    for key, value in _NATIVE_GDN.items():
        monkeypatch.setenv(key, value)
    enabled = _compose("model_engine=megatron", "isoexec.enable=true", *overrides)
    changes = isoexec_config.apply(enabled)
    assert tuple(OmegaConf.select(enabled, key) for key in _KEYS) == want
    assert not any(key in change for change in changes for key in _KEYS)


@pytest.mark.parametrize("overrides, want, flags", [
    ((), (True, True), {"ISOEXEC_GDN_CHUNKED_PREFILL": "1", "ISOEXEC_GDN_CPR_APC": "shm"}),  # verl's defaults
    (("actor_rollout_ref.rollout.enable_prefix_caching=false",), (True, False), {"ISOEXEC_GDN_CHUNKED_PREFILL": "1"}),
    (("actor_rollout_ref.rollout.enable_chunked_prefill=false", "actor_rollout_ref.rollout.enable_prefix_caching=false"),
     (False, False), {}),
], ids=["defaults", "no-prefix-cache", "both-off"])
def test_cpr_keeps_the_callers_features_and_forwards_their_engine_flags(monkeypatch, overrides, want, flags):
    import os

    from omegaconf import OmegaConf

    from isoexec.integrations.verl import config as isoexec_config

    for key, value in {**_CPR_GDN, "ISOEXEC_GDN_CHUNKED_PREFILL": "", "ISOEXEC_GDN_CPR_APC": ""}.items():
        monkeypatch.setenv(key, value)
    enabled = _compose("model_engine=megatron", "isoexec.enable=true", *overrides)
    changes = isoexec_config.apply(enabled)
    assert tuple(OmegaConf.select(enabled, key) for key in _KEYS) == want
    assert not any(key in change for change in changes for key in _KEYS)
    env_vars = OmegaConf.select(enabled, "ray_kwargs.ray_init.runtime_env.env_vars", default=None) or {}
    assert {k: v for k, v in dict(env_vars).items() if k in ("ISOEXEC_GDN_CHUNKED_PREFILL", "ISOEXEC_GDN_CPR_APC")} \
        == flags
    assert all(os.environ[k] == v for k, v in flags.items())
    assert isoexec_config.apply(enabled) == []


def test_cpr_refuses_a_feature_whose_flag_the_caller_turned_off(monkeypatch):
    from isoexec.integrations.verl import config as isoexec_config

    for key, value in {**_CPR_GDN, "ISOEXEC_GDN_CPR_APC": "0"}.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="ISOEXEC_GDN_CPR_APC"):
        isoexec_config.apply(_compose("model_engine=megatron", "isoexec.enable=true"))


@pytest.mark.parametrize("env, overrides, reason", [
    ({**_NATIVE_GDN, "ISOEXEC_GDN_NATIVE_STATE": "0"}, (), "GDN composition"),
    ({**_CPR_GDN, "ISOEXEC_GDN_NATIVE_KERNELS": "0"}, (), "GDN composition"),
    (_NATIVE_GDN, ("+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config={method: mtp}",), "speculative"),
    (_CPR_GDN, ("+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config={method: mtp}",), "speculative"),
], ids=["private-pool", "cpr-eager-kernels", "speculative", "cpr-speculative"])
def test_other_compositions_force_both_off_with_a_reason(monkeypatch, env, overrides, reason):
    from omegaconf import OmegaConf

    from isoexec.integrations.verl import config as isoexec_config

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    enabled = _compose("model_engine=megatron", "isoexec.enable=true", *overrides)
    changes = isoexec_config.apply(enabled)
    assert all(OmegaConf.select(enabled, key) is False for key in _KEYS)
    for key in _KEYS:
        assert any(change.startswith(f"{key}: True -> False") and reason in change for change in changes), changes
    assert isoexec_config.apply(enabled) == []
