"""The vLLM plugin must bind its sleepable allocator through the MoE storage seam."""


def test_plugin_binds_sleepable_allocator_through_moe_storage_seam(monkeypatch):
    from isoexec.ops.moe import param_storage
    from isoexec.runtimes.vllm import registration, vllm_patches, vllm_plugin

    installed = []
    monkeypatch.setenv("VLLM_ISOEXEC_REGISTERED", "1")
    monkeypatch.setattr(vllm_plugin, "_installed", False)
    # Patch the provider so undo does not leave a cached attribute on the lazy facade.
    monkeypatch.setattr(param_storage, "set_buffer_allocator", installed.append)
    monkeypatch.setattr(vllm_patches, "neutralize_vllm_nccl_channel_pin", lambda: None)
    monkeypatch.setattr(registration, "register_plugin", lambda: None)
    vllm_plugin.register()
    assert installed == [vllm_plugin.alloc_sleepable]
