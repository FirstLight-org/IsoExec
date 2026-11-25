"""Import-light declarations for the SM90 native recurrent GDN core and the stages it owns.

Trainer and engine run one kernel, vLLM's ``fused_sigmoid_gating_delta_rule_update`` vendored in
``ops/gdn/shared/vendored_fla`` (l2norm and gating in-kernel). The engine keeps its fp32 state in vLLM's
own mamba cache pages. Convolution is declared separately by conv_sm90.
"""

from isoexec.core.registry import ImplSpec

_SM90 = frozenset({"sm90"})
_FUSED_OPS = ("gdn.core", "gdn.l2norm", "gdn.gating")

IMPLS: tuple[ImplSpec, ...] = tuple(
    ImplSpec(op=op, impl_id="fla_native_recurrent_v1", version=1, supported_archs=_SM90)
    for op in _FUSED_OPS
)
