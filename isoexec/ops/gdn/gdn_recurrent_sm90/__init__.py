"""SM90 Gated DeltaNet core, native recurrent composition with vLLM-owned state.

The family surface provides ``gdn_core`` (trainer), ``build_state_pool`` (engine state over vLLM's
mamba pages) and ``refuse_incompatible_flags``. ``needs_native_cache`` tells the state layer to defer
building the state until vLLM binds the layer's cache.
"""

__all__ = ["build_state_pool", "gdn_core", "needs_native_cache", "refuse_incompatible_flags"]

#: The engine state lives in the layer's bound vLLM mamba cache (``GDNStateLayer.bind_kv_cache``).
needs_native_cache = True


def gdn_core(*args, **kwargs):
    from isoexec.ops.gdn.gdn_recurrent_sm90.integration import gdn_recurrent_sm90_core

    return gdn_recurrent_sm90_core(*args, **kwargs)


def build_state_pool(**kwargs):
    from isoexec.ops.gdn.gdn_recurrent_sm90.engine_state import build_recurrent_sm90_gdn

    return build_recurrent_sm90_gdn(**kwargs)


def refuse_incompatible_flags():
    from isoexec.ops.gdn.gdn_recurrent_sm90.engine_state import refuse_incompatible_flags as check

    return check()
