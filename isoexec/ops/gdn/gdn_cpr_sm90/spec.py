"""Import-light declarations for the SM90 chunk-parallel GDN core and its owned stages.
Decode, prefill and trainer forward share the recipe in csrc/recipe.cuh.
Convolution is declared separately by conv_sm90.

Engine state record: format 3 (``ops.RECORD_FORMAT``, csrc/state.cuh ``Rec::FORMAT``), 137,984 bytes per
(slot, value head). The record's format is engine-private state, not arithmetic: every format holds the same
fp32 boundary state bits, so the impl id and version below do not change with it.
"""

from isoexec.core.registry import ImplSpec

#: The engine state record format this package's kernels read and write (see ``ops.RECORD_FORMAT``).
STATE_RECORD_FORMAT = 3

_SM90 = frozenset({"sm90"})
_FUSED_OPS = ("gdn.core", "gdn.l2norm", "gdn.gating")

IMPLS: tuple[ImplSpec, ...] = tuple(
    ImplSpec(op=op, impl_id="cuda_cpr_v1", version=1, supported_archs=_SM90)
    for op in _FUSED_OPS
)
