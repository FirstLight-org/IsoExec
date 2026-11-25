"""Extraction preserves shared state identity and live convolution installation."""

from isoexec.ops import _dispatch, gdn
from isoexec.ops.gdn.conv_sm90 import integration as conv_integration
from isoexec.ops.gdn.conv_sm90.vendored import causal_conv1d as conv
from isoexec.ops.gdn.gdn_cpr_sm90 import engine_state, fla_backward
from isoexec.ops.gdn.shared import gdn_cpr_sleep, gdn_cpr_state, gdn_fla_backward, gdn_ops, gdn_recurrent_state


def test_provider_and_facade_use_the_same_state_and_sleep_owners(monkeypatch):
    monkeypatch.setattr(_dispatch, "ARCH", "cpu")
    assert gdn.CPR_LAYERS is engine_state.CPR_LAYERS is gdn_cpr_state.CPR_LAYERS
    assert gdn.RecurrentGDN is engine_state.RecurrentGDN is gdn_recurrent_state.RecurrentGDN
    assert gdn._HOST_BOOKKEEPING_STATS is gdn_cpr_state._HOST_BOOKKEEPING_STATS
    assert engine_state.alloc_cpr_arena is gdn_cpr_sleep.alloc_cpr_arena
    assert engine_state.register_layer is gdn_cpr_sleep.register_layer
    assert fla_backward.get_fla_chunk_backward is gdn_fla_backward.get_fla_chunk_backward
    marker = object()
    monkeypatch.setattr(gdn_cpr_state, "CPR_APC_STORE", marker)
    assert gdn.gdn_cpr_state.CPR_APC_STORE is marker


def test_convolution_installation_reaches_provider_and_facade_and_restores(monkeypatch):
    monkeypatch.setattr(_dispatch, "ARCH", "cpu")
    # Isolate the installation registry while preserving any installation made by
    # another test or the host before this test started.
    monkeypatch.setattr(conv_integration, "_INSTALLED", {})
    original = (conv.causal_conv1d_fn, conv.causal_conv1d_update, gdn_ops.gdn_native_conv)
    assert engine_state.conv is conv
    try:
        assert conv_integration.install()
        assert engine_state.conv.causal_conv1d_fn is conv_integration.causal_conv1d_fn
        assert engine_state.conv.causal_conv1d_update is conv_integration.causal_conv1d_update
        assert gdn.gdn_native_conv is gdn_ops.gdn_native_conv is conv_integration.gdn_native_conv
        assert conv_integration.install()  # repeated installation must preserve originals
    finally:
        conv_integration.uninstall()
    assert (conv.causal_conv1d_fn, conv.causal_conv1d_update, gdn.gdn_native_conv) == original
