"""Provider failures and obsolete aliases cannot masquerade as production bindings."""

import pytest


def test_mm_identity_propagates_provider_failure(monkeypatch):
    from isoexec.ops.mm import trainer

    def broken():
        raise ImportError("provider dependency missing")

    monkeypatch.setattr(trainer, "_mm", broken)
    with pytest.raises(ImportError, match="provider dependency missing"):
        trainer.mm_impl_id()


def test_mlp_refuses_silu_alias():
    from isoexec.ops._resolve import ResolutionError
    from isoexec.ops.mlp.resolve import resolve

    with pytest.raises(ResolutionError, match="silu"):
        resolve("mlp.act", site="test.mlp", activation="silu")


def test_gdn_facade_does_not_export_retired_arithmetic():
    from isoexec.ops import gdn

    assert "gdn_native_cpr" not in gdn.__all__
    with pytest.raises(AttributeError, match="no public export"):
        getattr(gdn, "gdn_native_cpr")


def test_convolution_refuses_missing_selected_kernel(monkeypatch):
    from isoexec.ops.gdn import trainer

    monkeypatch.setattr(trainer, "_conv_kernel", lambda arch: None)
    with pytest.raises(RuntimeError, match="no selected production convolution kernel"):
        trainer._refuse_uninstalled_conv(lambda *args: None)
