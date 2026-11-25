"""Reject stale native workspace caches before installing numerical overrides."""

import os

import pytest
import torch


def test_workspace_defaults_reach_fresh_actors_and_direct_host(monkeypatch):
    from isoexec.runtimes.trainer_installs import apply_trainer_env
    from isoexec.runtimes.environment import resolved_environment

    expected = {"CUBLAS_WORKSPACE_CONFIG": ":16:8", "CUBLASLT_WORKSPACE_SIZE": "1"}
    for key in expected:
        monkeypatch.delenv(key, raising=False)
    before = dict(os.environ)
    actor_env = resolved_environment("unused")
    assert {key: actor_env.get(key) for key in expected} == expected
    assert dict(os.environ) == before
    # Isolate the other defaults applied by this process-wide host entry point.
    with monkeypatch.context() as scoped:
        scoped.setattr(os, "environ", dict(os.environ))
        apply_trainer_env(None)
        assert {key: os.environ.get(key) for key in expected} == expected


@pytest.mark.parametrize("classic,lt", [(32 << 20, 1024), (131072, 1 << 20)])
def test_late_environment_assignment_cannot_hide_cached_workspace_sizes(monkeypatch, classic, lt):
    from isoexec.ops.mm import batch_invariant_install as installer

    monkeypatch.setattr(installer, "_library", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.backends.cuda, "cublas_workspace_size", lambda: classic)
    monkeypatch.setattr(torch.backends.cuda, "cublaslt_workspace_size", lambda: lt)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setenv("CUBLASLT_WORKSPACE_SIZE", "1")

    def must_not_install(*args, **kwargs):
        pytest.fail("numerical override installation reached with stale workspace cache")

    monkeypatch.setattr(installer.gemm, "configure_fp16_block_size_n", must_not_install)
    with pytest.raises(RuntimeError, match="workspace.*restart"):
        installer.install()
    assert installer._library is None
