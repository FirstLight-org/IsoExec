"""Chunked prefill and prefix caching follow the GDN state composition, decided in one place.

The engine gate (``gdn_engine_patch.assert_engine_args_compatible``), ``register_native`` and the
verl integration all read ``resumable_engine_args_for`` / ``admitted_engine_args``: the native
recurrent composition (recurrent + NATIVE_KERNELS + NATIVE_STATE) resumes from vLLM's own state
blocks and CPR on its private pool from its carried rows and boundary checkpoints, so both features
are admitted there (CPR with the engine flags ``state_feature_env`` names, which the integrations turn
on); the private recurrent pool admits only its chunked-prefill resume, and speculative decoding nothing.
"""

from __future__ import annotations

import os

import pytest

from isoexec.runtimes.vllm import gdn_engine_patch as patch

FEATURES = {"enable_chunked_prefill", "enable_prefix_caching"}

NATIVE = {"ISOEXEC_GDN_KERNEL": "recurrent", "ISOEXEC_GDN_NATIVE_KERNELS": "1", "ISOEXEC_GDN_NATIVE_STATE": "1",
          "ISOEXEC_GDN_CPR_MIN_PAGES": "0", "ISOEXEC_GDN_CHUNKED_PREFILL": "1"}
CPR = {"ISOEXEC_GDN_KERNEL": "cpr", "ISOEXEC_GDN_NATIVE_KERNELS": "1", "ISOEXEC_GDN_NATIVE_STATE": "0",
       "ISOEXEC_GDN_CPR_MIN_PAGES": "1", "ISOEXEC_GDN_CHUNKED_PREFILL": "1", "ISOEXEC_GDN_CPR_APC": ""}
CPR_APC = {**CPR, "ISOEXEC_GDN_CPR_APC": "shm"}
# Native kernels on a private pool: the pool cannot see a prefix vLLM cached, so prefix caching is out.
POOLED_NATIVE_KERNELS = {**NATIVE, "ISOEXEC_GDN_NATIVE_STATE": "0"}
RECURRENT_POOL_NO_RESUME = {**POOLED_NATIVE_KERNELS, "ISOEXEC_GDN_NATIVE_KERNELS": "0", "ISOEXEC_GDN_CHUNKED_PREFILL": "0"}


def _env(monkeypatch, values):
    # setenv on every flag the functions under test may export, so teardown restores them
    monkeypatch.setenv("ISOEXEC_GDN_CPR_APC", os.environ.get("ISOEXEC_GDN_CPR_APC", ""))
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    # The provider check is the engine's own business; these tests pin the shared decision.
    monkeypatch.setattr("isoexec.ops.gdn.refuse_incompatible_flags", lambda: None)


@pytest.mark.parametrize(
    "env, resumable, admitted",
    [
        (NATIVE, FEATURES, FEATURES),
        (CPR, FEATURES, {"enable_chunked_prefill"}),
        (CPR_APC, FEATURES, FEATURES),
        (POOLED_NATIVE_KERNELS, set(), {"enable_chunked_prefill"}),
        (RECURRENT_POOL_NO_RESUME, set(), set()),
    ],
    ids=["native", "cpr", "cpr-apc", "pooled-native-kernels", "pooled-no-resume"],
)
def test_integrations_enable_a_subset_of_what_the_gate_admits(monkeypatch, env, resumable, admitted):
    _env(monkeypatch, env)
    assert patch.resumable_engine_args() == resumable
    assert patch.admitted_engine_args() == admitted
    allowed, reason = patch.resumable_engine_args_for(speculative=False)
    assert allowed == resumable and bool(reason) == (not resumable)
    # Once the integration exported the flags its features need, the gate admits every one of them.
    patch.enable_state_features(sorted(resumable))
    assert patch.resumable_engine_args() <= patch.admitted_engine_args()


def test_cpr_features_name_their_engine_flags(monkeypatch):
    _env(monkeypatch, {**CPR, "ISOEXEC_GDN_CHUNKED_PREFILL": ""})
    assert patch.state_feature_env([]) == {}
    assert patch.state_feature_env(["enable_chunked_prefill"]) == {"ISOEXEC_GDN_CHUNKED_PREFILL": "1"}
    want = {"ISOEXEC_GDN_CHUNKED_PREFILL": "1", "ISOEXEC_GDN_CPR_APC": "shm"}
    assert patch.state_feature_env(["enable_prefix_caching"]) == want
    assert patch.enable_state_features(["enable_prefix_caching"]) == want
    assert all(os.environ[k] == v for k, v in want.items())
    assert patch.enable_state_features(["enable_prefix_caching"]) == {}  # already on: nothing to set
    _env(monkeypatch, NATIVE)
    assert patch.state_feature_env(sorted(FEATURES)) == {}  # vLLM's own blocks: no flag needed


def test_a_flag_turned_off_explicitly_refuses_the_feature(monkeypatch):
    _env(monkeypatch, {**CPR, "ISOEXEC_GDN_CPR_APC": "0"})
    with pytest.raises(ValueError, match="ISOEXEC_GDN_CPR_APC=shm.*ISOEXEC_GDN_CPR_APC=0"):
        patch.enable_state_features(["enable_prefix_caching"])
    _env(monkeypatch, {**CPR, "ISOEXEC_GDN_CHUNKED_PREFILL": "0"})
    with pytest.raises(ValueError, match="enable_chunked_prefill"):
        patch.enable_state_features(["enable_chunked_prefill"])


def test_speculative_decoding_turns_both_features_off(monkeypatch):
    _env(monkeypatch, NATIVE)
    allowed, reason = patch.resumable_engine_args_for(speculative=True)
    assert allowed == frozenset() and "speculative" in reason
    with pytest.raises(ValueError, match="speculative_config"):
        patch.assert_engine_args_compatible({"speculative_config": object()}, capturing=True)


def test_gate_admits_both_under_the_native_composition(monkeypatch):
    _env(monkeypatch, NATIVE)
    patch.assert_engine_args_compatible(
        {"enable_chunked_prefill": True, "enable_prefix_caching": True, "mamba_cache_mode": "align"}, capturing=True)


def test_gate_admits_both_under_cpr_with_the_boundary_store(monkeypatch):
    _env(monkeypatch, CPR_APC)
    monkeypatch.setattr(patch, "install_cpr_lazy_driver", lambda: True)
    patch.assert_engine_args_compatible(
        {"enable_chunked_prefill": True, "enable_prefix_caching": True, "mamba_cache_mode": "align"}, capturing=True)


@pytest.mark.parametrize("env", [CPR, POOLED_NATIVE_KERNELS], ids=["cpr-without-store", "pooled-native-kernels"])
def test_gate_refuses_prefix_caching_on_a_private_pool(monkeypatch, env):
    _env(monkeypatch, env)
    monkeypatch.setattr(patch, "install_cpr_lazy_driver", lambda: True)
    with pytest.raises(ValueError, match="enable_prefix_caching"):
        patch.assert_engine_args_compatible(
            {"enable_chunked_prefill": True, "enable_prefix_caching": True, "mamba_cache_mode": "align"},
            capturing=True)


def test_gate_refuses_prefix_caching_outside_align_mode(monkeypatch):
    _env(monkeypatch, NATIVE)
    with pytest.raises(ValueError, match="mamba_cache_mode='none'"):
        patch.assert_engine_args_compatible(
            {"enable_chunked_prefill": True, "enable_prefix_caching": True, "mamba_cache_mode": "none"},
            capturing=True)


@pytest.mark.parametrize(
    "env, given, expected",
    [
        (NATIVE, {}, {}),  # unset: vLLM's defaults (on) stand
        (NATIVE, {"enable_prefix_caching": True, "enable_chunked_prefill": True}, {}),
        (NATIVE, {"enable_prefix_caching": False}, {}),  # the caller's off stands too
        (NATIVE, {"speculative_config": {"method": "mtp"}},
         {"enable_prefix_caching": False, "enable_chunked_prefill": False}),
        (NATIVE, {"enable_prefix_caching": True, "enable_chunked_prefill": False}, {"enable_chunked_prefill": True}),
        (CPR, {}, {"enable_prefix_caching": False, "enable_chunked_prefill": False}),  # unset: off under cpr
        (CPR, {"enable_prefix_caching": False, "enable_chunked_prefill": True}, {}),
        (CPR, {"enable_prefix_caching": True, "enable_chunked_prefill": True}, {}),
        (CPR, {"enable_prefix_caching": True, "enable_chunked_prefill": False}, {"enable_chunked_prefill": True}),
        (CPR, {"enable_prefix_caching": True, "speculative_config": {"method": "mtp"}},
         {"enable_prefix_caching": False, "enable_chunked_prefill": False}),
        (RECURRENT_POOL_NO_RESUME, {"enable_prefix_caching": True, "enable_chunked_prefill": True},
         {"enable_prefix_caching": False, "enable_chunked_prefill": False}),
    ],
    ids=["native-unset", "native-on", "native-off", "native-spec", "native-prefix-needs-chunked", "cpr-unset",
         "cpr-chunked", "cpr-both", "cpr-prefix-needs-chunked", "cpr-spec", "pooled-off"],
)
def test_register_native_keeps_admitted_values_and_forces_the_rest_off(monkeypatch, capsys, env, given, expected):
    from isoexec.runtimes.vllm.registration import gdn_state_engine_args

    _env(monkeypatch, env)
    assert gdn_state_engine_args(given) == expected
    if any(given.get(name) and expected.get(name) is False for name in expected):
        assert "-> False" in capsys.readouterr().out
    final = {**given, **expected}
    on = {name for name in FEATURES if final.get(name)}
    if env is CPR:  # the engine flags each feature that stays on needs are exported for vLLM's processes
        assert (os.environ.get("ISOEXEC_GDN_CPR_APC") == "shm") == ("enable_prefix_caching" in on)
