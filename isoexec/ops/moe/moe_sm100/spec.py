"""SM100 MoE experts, routing and combination."""

from __future__ import annotations

from isoexec.core.registry import ANY, ImplSpec, OneOf

#: This package's architecture. ``ops/_dispatch.FAMILIES`` names it on the sm100 row.
ARCH = "sm100"

_SM100 = frozenset({"sm100"})

#: The FC1 GLU epilogue's activation; the arithmetic is ``csrc/numerics.cuh::glu_epilogue``,
#: which spells ``x / (1 + expf(-x))`` with each step rounded to the operand dtype. ``mlp.act``
#: owns the reference function this names.
FC1_GLU_ACT = "silu"

#: Combine walks ascending GLOBAL expert id, then route slot: an EP or ETP split may not
#: reorder a token's sum. moe_sm90's ``SM90_COMBINE_ORDER`` states the same thing.
#: Schedule policy rule 2 (``isoexec/ops/schedule_policy.py`` ``selected()``): a sum over data-selected
#: coordinates is one segment per coordinate, a chain in ascending expert id. The combine's chain starts
#: from the token's first term (its lowest selected expert), taken as is: a channel whose k terms are all
#: -0 sums to -0. The same rule orders the selected-set sum of ``route_mechanics`` (v2; v1 folded it by a
#: power-of-two half fold in selection order); ``SM100_SELECTED_SUM`` names that sum, not the combine.
SM100_COMBINE_ORDER = "ascending_expert_then_slot"
SM100_SELECTED_SUM = "ascending_expert_chain_from_zero"

#: The combine IS moe_sm90's program bit for bit: the same order and the same start. moe_sm90's folds
#: (``fused_combine``, the owner exchange's ``_unified_owner_core``) take the first term as is and add the
#: others in ascending expert id, as ``csrc/route.cu`` ``combine_kernel`` does. ``ops/moe/sm100.py`` binds the
#: owner exchange on that ground (``tests/test_combine_seed_cpu.py``). The routing weights are another
#: matter: the selected-set sum above is not sm90's half fold, so the two archs' weights still differ.
SM100_COMBINE_MATCHES_SM90 = True

#: Dispatch rows are exact integer ranks over the flat ``(token, slot)`` index, so
#: within-expert placement is canonical -- stronger than sm90, where it is unspecified and the
#: result is made independent of it by the inverse map instead.
SM100_DISPATCH_ORDER = "canonical_token_major"

#: Every K segment is a multiple of one 128-byte TMA swizzle atom. The served geometries
#: (I/8 for I in {512, 1536}) satisfy this and sm90's multiple-of-8 bound alike, so the
#: segments are the same segments.
SM100_SEGMENT_MULTIPLE = 64

#: The one score-program difference from sm90, stated where a reader of the declarations will see
#: it: sm90's ``moe.route_score:deterministic_fp32_gemm`` is the installed batch-invariant Triton
#: mm, whose fold is not this ascending chain, so the two architectures' router logits differ and
#: the pre-update score gate is a within-arch gate on sm100. README.md "Remaining gaps" item 7.
SM100_SCORE_MATCHES_SM90 = False

#: ``KORDER``: this package declares TWO members of it (the forward-only ``sm100_gather_korder_v1`` and
#: the after-fc2 ``sm100_after_fc2_combine_v1``, one native combine kernel), so the class has an on-arch
#: counterparty and carries the field. ``runtimes/parallel`` admits EP > 1 only against a combine whose
#: RECORD names the class, and ``tests/test_ep_korder_gpu.py`` is the proof the record points at.
#: What the field claims is the equality of THESE two members at every EP / ETP layout. moe_sm90's own
#: two members carry the same class name and fold by the same program (``SM100_COMBINE_MATCHES_SM90``),
#: but ``ops/moe/spec.py`` is lazy (``family_impls`` builds the family from the package the LIVE arch
#: selects, and ``ops/_dispatch._assert_proven`` refuses a package that declares an impl of another arch),
#: so a registry holds one arch's members and the field never compares the two packages' members.
#: ``core/registry.TWIN_PROOFS[KORDER]`` lists the batteries of both archs.
#:
#: ``SORTED_TOPK`` was claimed with moe_sm90 through ``core/registry.TWIN_PROOFS`` (never on a
#: ``twin_class`` field: with the sm90 sibling undeclarable here, the field would read as a one-member
#: class, which ``core/tests/test_op_gate_alignment_cpu.py`` refuses); ``sm100_fused_topk_v2`` withdrew
#: that claim (its record below). The reason a cross-arch class cannot be spelled on these records stays
#: as stated.
WHY_NO_TWIN_FIELD = "cross-arch class; the sibling cannot be declared in an arch-selected package"

KORDER = "moe.combine.korder_fp32"
SORTED_TOPK = "moe.route.sorted_topk"
#: The Blackwell expert MLP's class. Arch-free by design: the arch is which member a binder
#: resolves, the name says what the members compute.
SEGMENT_TREE_GLU = "moe.expert_mlp.segment_tree_glu"

#: The fc2 leaf-tree plan the caller supplies, plus the activation the expert MLP calls.
_EXPERT_INPUTS = {"leaves": OneOf(1, 2, 4, 8), "leaf_dtype": OneOf("bf16", "fp16"), "act": ANY}

IMPLS: tuple[ImplSpec, ...] = (
    ImplSpec(op="moe.route_score", impl_id="sm100_router_gemm_fp32_v1", version=1, supported_archs=_SM100),
    ImplSpec(op="moe.route_mechanics", impl_id="sm100_softmax_full_v1", version=1, supported_archs=_SM100),
    ImplSpec(op="moe.expert_mlp", impl_id="sm100_tcgen05_after_fc2_v1", version=1,
             supported_archs=_SM100, inputs=dict(_EXPERT_INPUTS)),
    # The after-fc2 recipe's fold: the same native combine kernel ``sm100_gather_korder_v1`` launches,
    # on rows already weighted after fc2, under an exact row-gather adjoint. At EP > 1
    # ``ops/moe/expert_parallel.expert_dispatch`` places every returned term at its token's
    # canonical slot and calls this fold over them, so the bits are EP1's;
    # ``tests/test_ep_korder_gpu.py`` proves it at EP 1/2/4 (and ETP 4) on the 35B layer.
    ImplSpec(op="moe.combine", impl_id="sm100_after_fc2_combine_v1", version=1, supported_archs=_SM100,
             twin_class=KORDER),
    # v2: gate | up on ``mm_sm100``, the N = 1 gate on the vendored Triton GEMM, both named rather than
    # reached through ``matmul_persistent`` (whose wrappers differ between a trainer and an engine).
    ImplSpec(op="moe.shared_expert", impl_id="sm100_after_fc2_shared_v2", version=2, supported_archs=_SM100),
    # An ascending fp32 chain over the hidden axis, one accumulator per (token, expert). NOT a
    # widening of sm90's `deterministic_fp32_gemm`, which is the installed batch-invariant mm
    # and folds in a different order. It reads operand-dtype router weights; ``nn.MoE`` holds an
    # fp32 router, so ``ops/moe/sm100.py`` binds ``sm100_router_gemm_fp32_v1`` for every routing.
    ImplSpec(
        op="moe.route_score",
        impl_id="sm100_fp32_chain_v1",
        version=1,
        supported_archs=_SM100,
    ),
    # Sorted top-k with the (score desc, index asc) tie rule: the SELECTION is ``SORTED_TOPK``'s. v2: the
    # weights divide by the selected-set sum taken as the schedule policy's chain (ascending expert id
    # from +0), not by sm90's half fold, so this impl is NOT a member of the cross-arch ``SORTED_TOPK``
    # class any more and ``core/registry.TWIN_PROOFS[SORTED_TOPK]`` no longer names this package.
    ImplSpec(
        op="moe.route_mechanics",
        impl_id="sm100_fused_topk_v2",
        version=2,
        supported_archs=_SM100,
    ),
    # Sigmoid scores with a no-aux expert bias: a different selection function, not a different
    # implementation of the same one, so it carries no twin class.
    ImplSpec(
        op="moe.route_mechanics",
        impl_id="sm100_sigmoid_noaux_bias_select_v2",
        version=2,
        supported_archs=_SM100,
    ),
    ImplSpec(
        op="moe.expert_mlp",
        impl_id="sm100_tcgen05_v1",
        version=1,
        supported_archs=_SM100,
        inputs=dict(_EXPERT_INPUTS),
        twin_class=SEGMENT_TREE_GLU,
    ),
    # The same numerical program with the GLU epilogue taken OUT of the grouped GEMM: a raw FC1
    # projection (``ops.fc1``) followed by the standalone ``activation`` kernel, which calls the
    # same ``csrc/numerics.cuh::glu_epilogue``. tests/moe_relations_gpu.py proves
    # ``fc1_glu == activation(fc1)`` bitwise for every served geometry, which is exactly the
    # equivalence this class names, so the two are members of it and a host running either may
    # digest the other's table.
    ImplSpec(
        op="moe.expert_mlp",
        impl_id="sm100_tcgen05_unfused_act_v1",
        version=1,
        supported_archs=_SM100,
        inputs=dict(_EXPERT_INPUTS),
        twin_class=SEGMENT_TREE_GLU,
    ),
    # ``KORDER``'s program: the k slots folded in canonical slot order by one fp32 chain with one
    # rounding, which is bit-for-bit what moe_sm90's ``pik_leaf_tree`` computes from the same rows
    # (the fold has no tensor core in it, so tcgen05 changes nothing here) -- including the EP
    # home-rank fold, which is the same call on rows shipped home. The chain starts from the first slot,
    # taken as is (schedule policy rule 2 orders it and states no seed). ``TWIN_PROOFS[KORDER]`` names
    # the batteries that check it; ``tests/test_ep_korder_gpu.py`` proves this member and
    # ``sm100_after_fc2_combine_v1`` bitwise equal on the same rows and EP-invariant.
    ImplSpec(
        op="moe.combine",
        impl_id="sm100_gather_korder_v1",
        version=1,
        supported_archs=_SM100,
        twin_class=KORDER,
    ),
    # v2: ``ops.shared_expert``'s GEMMs on ``mm_sm100`` and the vendored Triton GEMM, not ``torch.mm``.
    ImplSpec(
        op="moe.shared_expert",
        impl_id="sm100_eager_v2",
        version=2,
        supported_archs=_SM100,
    ),
)

__all__ = (
    "ARCH",
    "FC1_GLU_ACT",
    "IMPLS",
    "KORDER",
    "SEGMENT_TREE_GLU",
    "SM100_COMBINE_MATCHES_SM90",
    "SM100_COMBINE_ORDER",
    "SM100_DISPATCH_ORDER",
    "SM100_SCORE_MATCHES_SM90",
    "SM100_SELECTED_SUM",
    "SM100_SEGMENT_MULTIPLE",
    "SORTED_TOPK",
    "WHY_NO_TWIN_FIELD",
)
