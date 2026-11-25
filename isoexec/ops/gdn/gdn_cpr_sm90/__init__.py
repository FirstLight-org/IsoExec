"""SM90 Gated DeltaNet core, chunk-parallel with recurrent decode: one chunk-form identity for decode,
(chunked) prefill and trainer forward, with the boundary resync inside the decode launch.

The family surface provides: ``prefill``, ``decode``, ``gdn_core`` (trainer),
``build_state_pool`` (the engine state machine) and ``refuse_incompatible_flags``.
"""

from isoexec.ops.gdn.gdn_cpr_sm90.ops import CprState, decode, prefill

__all__ = ["CprState", "build_state_pool", "decode", "gdn_core", "prefill", "refuse_incompatible_flags"]


def gdn_core(*args, **kwargs):
    """Apply this backend's trainer core (forward + FLA chunked backward)."""
    from isoexec.ops.gdn.gdn_cpr_sm90.integration import gdn_cpr_sm90_core

    return gdn_cpr_sm90_core(*args, **kwargs)


def build_state_pool(**kwargs):
    """Construct this backend's engine state (a record pool behind the shared slot map)."""
    from isoexec.ops.gdn.gdn_cpr_sm90.engine_state import build_cpr_sm90_gdn

    return build_cpr_sm90_gdn(**kwargs)


def refuse_incompatible_flags():
    """Check engine flags against this backend's state contract."""
    from isoexec.ops.gdn.gdn_cpr_sm90.engine_state import refuse_incompatible_flags as check_flags

    return check_flags()
