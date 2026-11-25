"""Expandable segments and NCCL graph buffer registration are never on together in the engine."""

import pytest

from isoexec.checks import Refusal
from isoexec.runtimes.vllm import vllm_plugin
from isoexec.runtimes.vllm.vllm_plugin import NCCL_GRAPH_REGISTER, resolve_nccl_graph_registration


def test_default_expandable_segments_turn_graph_registration_off():
    env = {}
    receipt = resolve_nccl_graph_registration(env)
    assert env == {NCCL_GRAPH_REGISTER: "0"}
    assert receipt == {"expandable_segments": True, NCCL_GRAPH_REGISTER: "0"}


@pytest.mark.parametrize("name", ["PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"])
def test_user_fixed_segments_keep_nccls_default(name):
    env = {name: "expandable_segments:False,max_split_size_mb:512"}
    receipt = resolve_nccl_graph_registration(env)
    assert NCCL_GRAPH_REGISTER not in env
    assert receipt == {"expandable_segments": False, NCCL_GRAPH_REGISTER: "nccl-default"}


def test_user_fixed_segments_with_registration_on_is_admitted():
    env = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512", NCCL_GRAPH_REGISTER: "1"}
    assert resolve_nccl_graph_registration(env) == {"expandable_segments": False, NCCL_GRAPH_REGISTER: "1"}


@pytest.mark.parametrize("conf", [None, "expandable_segments:True", "garbage_collection_threshold:0.8,expandable_segments:1"])
@pytest.mark.parametrize("value", ["1", " 2", "yes"])
def test_both_on_explicitly_is_a_named_refusal(conf, value):
    env = {NCCL_GRAPH_REGISTER: value}
    if conf is not None:
        env["PYTORCH_CUDA_ALLOC_CONF"] = conf
    with pytest.raises(Refusal, match="nccl.graph_register_with_expandable_segments") as info:
        resolve_nccl_graph_registration(env)
    assert "expandable_segments:False" in str(info.value) and f"{NCCL_GRAPH_REGISTER}=0" in str(info.value)


def test_user_expandable_segments_with_registration_off_is_kept():
    env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", NCCL_GRAPH_REGISTER: "0"}
    assert resolve_nccl_graph_registration(env) == {"expandable_segments": True, NCCL_GRAPH_REGISTER: "0"}


def test_plugin_sets_the_default_in_the_process_environment_and_refuses_both_on(monkeypatch):
    from isoexec.ops.moe import param_storage
    from isoexec.runtimes.vllm import registration, vllm_patches

    monkeypatch.setenv("VLLM_ISOEXEC_REGISTERED", "1")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # skip the live allocator write
    monkeypatch.delenv(NCCL_GRAPH_REGISTER, raising=False)
    monkeypatch.delenv("NCCL_NVLS_ENABLE", raising=False)
    monkeypatch.delenv("ISOEXEC_PIK_OWNER_AR", raising=False)
    monkeypatch.setattr(param_storage, "set_buffer_allocator", lambda alloc: None)
    monkeypatch.setattr(vllm_patches, "neutralize_vllm_nccl_channel_pin", lambda: None)
    monkeypatch.setattr(vllm_patches, "patch_vllm_pp_kv_cache_tensors", lambda: None)
    monkeypatch.setattr(registration, "register_plugin", lambda: None)
    monkeypatch.setattr(vllm_plugin, "_installed", False)
    vllm_plugin.register()
    import os
    assert os.environ[NCCL_GRAPH_REGISTER] == "0"
    assert vllm_plugin.NCCL_REGISTRATION_RECEIPT == {"expandable_segments": True, NCCL_GRAPH_REGISTER: "0"}

    monkeypatch.setenv(NCCL_GRAPH_REGISTER, "1")
    monkeypatch.setattr(vllm_plugin, "_installed", False)
    with pytest.raises(Refusal, match="nccl.graph_register_with_expandable_segments"):
        vllm_plugin.register()
    assert vllm_plugin._installed is False
