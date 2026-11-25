"""Import-light declarations for the SM100 native recurrent GDN core and the stages it owns.

The recipe is gdn_recurrent_sm90's: trainer and engine run one kernel, vLLM's
``fused_sigmoid_gating_delta_rule_update`` vendored in ``ops/gdn/shared/vendored_fla`` (l2norm and gating
in-kernel), and the engine keeps its fp32 state in vLLM's own mamba pages. The Triton source is shared;
the proof is per arch (``tests/``). Convolution is declared separately by conv_sm100.
"""

from isoexec.core.registry import ImplSpec

#: This package's architecture. ``ops/_dispatch.FAMILIES`` names it as the sm100 row's ``recurrent``
#: alternative.
ARCH = "sm100"

_SM100 = frozenset({ARCH})
_FUSED_OPS = ("gdn.core", "gdn.l2norm", "gdn.gating")

IMPLS: tuple[ImplSpec, ...] = tuple(
    ImplSpec(op=op, impl_id="fla_native_recurrent_v1", version=1, supported_archs=_SM100)
    for op in _FUSED_OPS
)
