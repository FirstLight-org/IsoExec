"""The CPR provider uses the real op table and the shared engine ownership objects."""

import importlib

import pytest

from isoexec.ops import _dispatch, _resolve, gdn


@pytest.fixture
def cpr_provider(monkeypatch):
    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    monkeypatch.setitem(_dispatch.FAMILIES["gdn"].archs, "sm90", "gdn.gdn_cpr_sm90")
    _resolve._registry.cache_clear()
    yield
    _resolve._registry.cache_clear()


def test_cpr_provider_resolves_trainer_and_engine_ownership(cpr_provider):
    from isoexec.ops.table import OpTable
    from isoexec.runtimes.engine_ops import gdn_core_resolution

    table = OpTable()
    trainer = table.bind("gdn.core", site="layer.mixer.core")
    engine = gdn_core_resolution("layer.mixer", lambda: None, table=table, provider="test.engine")
    assert trainer.impl == engine.impl == gdn.IMPL == "cuda_cpr_v1"
    assert trainer.version == engine.version == 1
    assert trainer.owns == {op: "cuda_cpr_v1" for op in ("gdn.core", "gdn.l2norm", "gdn.gating")}
    assert engine.owns == {**trainer.owns, "conv.short": "sm90_cuda_v1"}
    assert trainer.provider == "ops.gdn.gdn_cpr_sm90"


def test_every_advertised_family_export_exists_with_cpr(cpr_provider):
    for name in gdn.__all__:
        assert getattr(gdn, name) is not None, name
    assert "Plan" not in gdn.__all__ and "choose_plan" not in gdn.__all__


@pytest.mark.parametrize("name", [
    "CPR_LAYERS", "RecurrentGDN", "_HOST_BOOKKEEPING_STATS", "fla_chunk_size", "cpr_mode",
    "chunked_prefill_enabled", "gdn_native_conv", "gdn_cpr_state", "gdn_recurrent_state",
    "flush_slot_maps", "layer_batched_resync", "causal_conv1d_metadata",
])
def test_cpr_provider_shares_lifecycle_and_live_adapters(cpr_provider, name):
    module, symbol = gdn._SHARED_EXPORTS[name]
    owner = importlib.import_module(f"isoexec.ops.gdn.shared.{module}")
    assert getattr(gdn, name) is (owner if symbol is None else getattr(owner, symbol))


def test_shared_conv_lookup_observes_installation(cpr_provider, monkeypatch):
    from isoexec.ops.gdn.shared import gdn_ops

    first, second = object(), object()
    monkeypatch.setattr(gdn_ops, "gdn_native_conv", first)
    assert gdn.gdn_native_conv is first
    monkeypatch.setattr(gdn_ops, "gdn_native_conv", second)
    assert gdn.gdn_native_conv is second


def test_shared_helpers_do_not_bypass_arch_refusal(monkeypatch):
    """A shared helper is reached THROUGH the arch table, so an arch the row cannot back still refuses.

    Both arch packages now declare (`gdn_cpr_sm90` on sm90, `gdn_cpr_sm100` on sm100), so the refusal is
    staged two ways: an accelerator arch the row does not name at all (sm120), and the sm100 row pointed
    at a package whose `supported_archs` says sm90 (as
    `test_dispatch_cpu.test_a_row_without_the_declarations_to_back_it_still_refuses` stages it).
    """
    monkeypatch.setattr(_dispatch, "ARCH", "sm120")
    with pytest.raises(_dispatch.ArchSelectionError, match="sm120"):
        gdn.__getattr__("CPR_LAYERS")
    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    monkeypatch.setitem(_dispatch.FAMILIES["gdn"].archs, "sm100", "gdn.gdn_cpr_sm90")
    with pytest.raises(_dispatch.ArchSelectionError, match="supported_archs"):
        gdn.__getattr__("CPR_LAYERS")


