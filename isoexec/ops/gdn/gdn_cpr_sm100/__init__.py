"""SM100 Gated DeltaNet core uses one chunk identity."""

from isoexec.ops.gdn.gdn_cpr_sm100.ops import CprState, decode, prefill

__all__ = ["CprState", "build_state_pool", "decode", "gdn_core", "prefill", "refuse_incompatible_flags"]


def gdn_core(*args, **kwargs):
    """Apply this backend's trainer core (forward + FLA chunked backward)."""
    from isoexec.ops.gdn.gdn_cpr_sm100.integration import gdn_cpr_sm100_core

    return gdn_cpr_sm100_core(*args, **kwargs)


def build_state_pool(**kwargs):
    """Construct this backend's engine state (a record pool behind the shared slot map)."""
    from isoexec.ops.gdn.gdn_cpr_sm100.engine_state import build_cpr_sm100_gdn

    return build_cpr_sm100_gdn(**kwargs)


def refuse_incompatible_flags():
    """Check engine flags against this backend's state contract."""
    from isoexec.ops.gdn.gdn_cpr_sm100.engine_state import refuse_incompatible_flags as check_flags

    return check_flags()
