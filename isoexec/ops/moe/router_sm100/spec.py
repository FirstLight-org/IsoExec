"""SM100 DeepSeek-class router declarations. The MoE family registers these additional
recipes alongside its architecture-selected BF16 expert package. Import-light: no torch, no build."""

from __future__ import annotations

ARCH = "sm100"

#: The two routing names this package serves. Each is one numerical identity.
GROUPED = "sigmoid_noaux_bias_select_grouped"
SQRTSOFTPLUS = "sqrtsoftplus_bias_select"

#: impl ids -- the id IS the recipe (logits program + selection program + table contract).
#: v2: the schedule is the schedule policy's (``isoexec/ops/schedule_policy.py``): the logits are ONE chain over
#: the hidden axis (v1: 4 segments and a tree) and the renormaliser is the ascending-expert-id chain from +0
#: (v1: a half fold in selection-rank order). Both move the weights' low bits, so the ids moved.
IMPL_GROUPED = "sm100_route_grouped_v2"
IMPL_SQRTSOFTPLUS = "sm100_route_sqrtsoftplus_v2"
#: the backward recipe: scalar weight backward, sparse dX chain, dW segment
#: tree with SEG_T = 1024. v2: it recomputes the forward's v2 renormaliser.
IMPL_BACKWARD = "sm100_route_bwd_v2"

#: The logits program: S fixed hidden segments, each one tcgen05 chain, a balanced tree. S is not a table: it is
#: rule 3 of the schedule policy on the reducer's block (16) with no published shortest segment, i.e. 1 for every H.
LOGIT_REDUCER_BLOCK = 16
LOGIT_REDUCER_SHORTEST = None


def logit_segments(hidden: int) -> int:
    """The logits program's segment count for a hidden size (``ops.declared_segments`` restated, no torch)."""
    from isoexec.ops import schedule_policy

    return schedule_policy.reduce(hidden, b=LOGIT_REDUCER_BLOCK, min_segment=LOGIT_REDUCER_SHORTEST).G


LOGIT_MERGE = "balanced_tree_lower_left"
LOGIT_CHAIN = "tcgen05.mma.cta_group::1.kind::f16 bf16xbf16->fp32, K=16/instr, h ascending"

#: The table contract both consumers read (moe_sm100 dispatch, moe_fp8_sm100 SPEC §3.3).
TABLE_SLOT_ORDER = "ascending_global_expert_id"
TABLE_DTYPES = ("int32", "fp32")
#: Rule 2 of the schedule policy: one segment per selected expert, a chain in ascending expert id from +0.
RENORM_FOLD = "ascending_expert_id_chain_from_zero_plus_1e-20"
SEG_T = 1024
#: SEG_T by expert count (deployment-fixed): E <= 128 -> 512, else 1024.
SEG_T_BY_E = {64: 512, 256: 1024, 288: 1024, 384: 1024, 512: 1024}


__all__ = ["ARCH", "GROUPED", "IMPLS", "IMPL_BACKWARD", "IMPL_GROUPED", "IMPL_SQRTSOFTPLUS",
           "LOGIT_REDUCER_BLOCK", "LOGIT_REDUCER_SHORTEST", "SEG_T", "SQRTSOFTPLUS", "TABLE_SLOT_ORDER",
           "logit_segments"]


from isoexec.core.registry import ANY, ImplSpec, OneOf  # noqa: E402

_INPUTS = {"routing": OneOf(GROUPED, SQRTSOFTPLUS), "top_k": ANY,
           "n_group": ANY, "topk_group": ANY, "num_shared": ANY, "routed_scale": ANY}
IMPLS = tuple(
    ImplSpec(op="moe.route", impl_id=impl, version=2, supported_archs=frozenset({ARCH}),
             inputs=dict(_INPUTS, routing=OneOf(routing)))
    for routing, impl in ((GROUPED, IMPL_GROUPED), (SQRTSOFTPLUS, IMPL_SQRTSOFTPLUS))
) + (ImplSpec(op="moe.route.backward", impl_id=IMPL_BACKWARD, version=2,
              supported_archs=frozenset({ARCH}), inputs=_INPUTS),)
