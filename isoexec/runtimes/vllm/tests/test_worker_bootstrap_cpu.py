"""Explicit registration must bootstrap spawned workers through ``VLLM_ISOEXEC_REGISTERED``."""
import pytest
from isoexec.runtimes.vllm import registration, vllm_plugin


def test_explicit_registration_exports_bootstrap_even_when_already_installed(monkeypatch):
    monkeypatch.delenv('VLLM_ISOEXEC_REGISTERED', raising=False)
    monkeypatch.setattr(registration, '_PLUGIN_INSTALLED', True)
    registration.register_plugin()
    import os
    assert os.environ['VLLM_ISOEXEC_REGISTERED'] == '1'


@pytest.mark.parametrize('job,explicit,expected', [('0', '1', True), ('1', '0', False), ('1', '1', True)])
def test_worker_plugin_requires_the_explicit_marker(monkeypatch, job, explicit, expected):
    """``ISOEXEC=1`` alone (a trainer process, say) does not install the engine stack."""
    from isoexec.ops.moe import param_storage
    from isoexec.runtimes.vllm import vllm_patches
    monkeypatch.setenv('ISOEXEC', job)
    monkeypatch.setenv('VLLM_ISOEXEC_REGISTERED', explicit)
    monkeypatch.setattr(vllm_plugin, '_installed', False)
    calls = []
    monkeypatch.setattr(registration, 'register_plugin', lambda: calls.append('registered'))
    monkeypatch.setattr(param_storage, 'set_buffer_allocator', lambda fn: calls.append('allocator'))
    monkeypatch.setattr(vllm_patches, 'neutralize_vllm_nccl_channel_pin', lambda: None)
    vllm_plugin.register()
    assert calls == (['allocator', 'registered'] if expected else [])


def test_worker_plugin_retries_failed_installation_without_publishing_readiness(monkeypatch):
    from isoexec.ops.moe import param_storage
    from isoexec.runtimes.vllm import vllm_patches

    monkeypatch.setenv('ISOEXEC', '0')
    monkeypatch.setenv('VLLM_ISOEXEC_REGISTERED', '1')
    monkeypatch.setattr(vllm_plugin, '_installed', False)
    monkeypatch.setattr(param_storage, 'set_buffer_allocator', lambda fn: None)
    monkeypatch.setattr(vllm_patches, 'neutralize_vllm_nccl_channel_pin', lambda: None)
    attempts = []

    def install():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError('worker installation failed')

    monkeypatch.setattr(registration, 'register_plugin', install)
    with pytest.raises(RuntimeError, match='worker installation failed'):
        vllm_plugin.register()
    assert not vllm_plugin._installed
    vllm_plugin.register()
    assert vllm_plugin._installed and attempts == [1, 1]
    vllm_plugin.register()
    assert attempts == [1, 1]


def test_fresh_worker_plugin_uses_resolved_environment_without_driver_registration(monkeypatch):
    from isoexec.ops.moe import param_storage
    from isoexec.runtimes.environment import resolved_environment
    from isoexec.runtimes.vllm import vllm_patches

    monkeypatch.setenv('ISOEXEC', '0')
    monkeypatch.delenv('VLLM_ISOEXEC_REGISTERED', raising=False)
    environment = resolved_environment('mounted/checkpoint')
    assert environment['VLLM_PLUGINS'] == 'isoexec'
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv('ISOEXEC', '0')
    calls = []
    monkeypatch.setattr(vllm_plugin, '_installed', False)
    monkeypatch.setattr(registration, 'register_plugin', lambda: calls.append('registered'))
    monkeypatch.setattr(param_storage, 'set_buffer_allocator', lambda fn: calls.append('allocator'))
    monkeypatch.setattr(vllm_patches, 'neutralize_vllm_nccl_channel_pin', lambda: None)
    vllm_plugin.register()
    assert calls == ['allocator', 'registered'] and vllm_plugin._installed


@pytest.mark.parametrize('explicit,expected', [(None, True), ('0', False), ('1', True)])
def test_engine_owner_all_reduce_is_on_unless_set_and_read_at_first_use(monkeypatch, explicit, expected):
    """Engine row-parallel sites default to pik's owner all-reduce; an explicit setting wins. The owner
    module resolves its flag at first use, so a value set after it was imported (here: after the plugin
    registered) still applies."""
    from isoexec.ops.collectives.pik import owner
    from isoexec.ops.moe import param_storage
    from isoexec.runtimes.vllm import vllm_patches

    monkeypatch.setenv('ISOEXEC', '0')
    monkeypatch.setenv('VLLM_ISOEXEC_REGISTERED', '1')
    monkeypatch.setenv('ISOEXEC_PIK_OWNER_AR', '')  # restored after the test, whatever register() writes
    if explicit is None:
        monkeypatch.delenv('ISOEXEC_PIK_OWNER_AR')
    else:
        monkeypatch.setenv('ISOEXEC_PIK_OWNER_AR', explicit)
    monkeypatch.setattr(owner, '_ON', None)  # imported, not yet used
    monkeypatch.setattr(vllm_plugin, '_installed', False)
    monkeypatch.setattr(registration, 'register_plugin', lambda: None)
    monkeypatch.setattr(param_storage, 'set_buffer_allocator', lambda fn: None)
    monkeypatch.setattr(vllm_patches, 'neutralize_vllm_nccl_channel_pin', lambda: None)
    vllm_plugin.register()
    assert owner.enabled() is expected


def test_owner_flag_is_resolved_at_first_use_not_at_import(monkeypatch):
    """The trainer side reads the same lazy flag: a launcher that sets it after the import still decides."""
    from isoexec.ops.collectives.pik import owner

    monkeypatch.setattr(owner, '_ON', None)
    monkeypatch.setenv('ISOEXEC_PIK_OWNER_AR', '0')
    assert owner.enabled() is False and owner.lamport_enabled() is False
    monkeypatch.setattr(owner, '_ON', None)
    monkeypatch.delenv('ISOEXEC_PIK_OWNER_AR')
    assert owner.enabled() is True
