"""SM100 declaration for DeepSeek-V4.1 shifted mHC (``residual.deepseek_v41.mhc``). Import-light."""

from __future__ import annotations

from isoexec.core.registry import ANY, ImplSpec, OneOf
from isoexec.ops.residual.mhc_sm100 import spec as _mhc

ARCH = "sm100"
OP_ID = "residual.deepseek_v41.mhc"
OP_ID_BWD = "residual.deepseek_v41.mhc.backward"
OP_ID_COLLAPSE = "residual.deepseek_v41.mhc.collapse"
OP_ID_COLLAPSE_BWD = "residual.deepseek_v41.mhc.collapse.backward"
OP_ID_EXPAND = "residual.deepseek_v41.mhc.expand"
OP_ID_EXPAND_BWD = "residual.deepseek_v41.mhc.expand.backward"
IMPL_ID = "sm100_deepseek_v41_mhc_v1"
IMPL_ID_BWD = "sm100_deepseek_v41_mhc_bwd_v1"
IMPL_ID_COLLAPSE = "sm100_deepseek_v41_mhc_collapse_v1"
IMPL_ID_COLLAPSE_BWD = "sm100_deepseek_v41_mhc_collapse_bwd_v1"
IMPL_ID_EXPAND = "sm100_deepseek_v41_mhc_expand_v1"
IMPL_ID_EXPAND_BWD = "sm100_deepseek_v41_mhc_expand_bwd_v1"
RECIPE_VERSION = 1

HC_MULT = 4
HIDDEN = 5120
#: Published epsilons (the checkpoint config.json): rms 1e-20 (a pure floor, fp32), Sinkhorn /
#: pre-mix 1e-6; 20 Sinkhorn iterations. They are runtime facts of the call, recorded in the placement.
#: The reduction program is mhc_sm100's (32 h-slices of 160, exact-product mma chains, balanced trees) with
#: an fp32 ``fn`` consumed as three exact bf16 pieces (hi, mid, lo) per step.
NUM_SLICES = _mhc.NUM_SLICES
FN_PIECES = 3
SEG_ROWS = _mhc.SEG_ROWS
#: Boundaries:
#:   params : mix = (x . fn) * rstd ; pre = sigmoid(mix*s0 + b) + eps ; post = 2 sigmoid(...) ; comb = Sinkhorn(softmax+eps)
#:   collapse : bf16( ((p0*x0 + p1*x1) + p2*x2) + p3*x3 )  (fp32 products/sums; pre is the SHIFTED one supplied by the caller)
#:   expand   : bf16( post[i]*x[h] + ((c[0][i]*r0 + c[1][i]*r1) + c[2][i]*r2) + c[3][i]*r3 ), all fp32, one rounding
COLLAPSE = "fp32_left_to_right_over_streams_one_bf16_rounding"
EXPAND = "fp32_post_x_plus_sequential_comb_residual_one_bf16_rounding"

_ARCHS = frozenset({ARCH})
_INPUTS = {"hidden": OneOf(HIDDEN), "hc_mult": OneOf(HC_MULT), "rms_eps": ANY, "sinkhorn_eps": ANY, "iterations": ANY}

IMPLS: tuple[ImplSpec, ...] = tuple(
    ImplSpec(op=op, impl_id=impl, version=RECIPE_VERSION, supported_archs=_ARCHS, inputs=dict(_INPUTS))
    for op, impl in ((OP_ID, IMPL_ID), (OP_ID_BWD, IMPL_ID_BWD), (OP_ID_COLLAPSE, IMPL_ID_COLLAPSE),
                     (OP_ID_COLLAPSE_BWD, IMPL_ID_COLLAPSE_BWD), (OP_ID_EXPAND, IMPL_ID_EXPAND),
                     (OP_ID_EXPAND_BWD, IMPL_ID_EXPAND_BWD))
)

__all__ = ["ARCH", "HC_MULT", "HIDDEN", "IMPLS", "OP_ID", "RECIPE_VERSION"]
