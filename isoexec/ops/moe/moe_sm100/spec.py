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
SM100_COMBINE_ORDER = "ascending_expert_then_slot"

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

#: ``KORDER`` is the exception: this package declares TWO members of it (the forward-only
#: ``sm100_gather_korder_v1`` and the after-fc2 ``sm100_after_fc2_combine_v1``, one native combine
#: kernel), so the class has an on-arch counterparty and carries the field. ``runtimes/parallel``
#: admits EP > 1 only against a combine whose RECORD names the class, and
#: ``tests/test_ep_korder_gpu.py`` is the proof the record points at.
#:
#: Why the CROSS-ARCH class below is recorded in ``core/registry.TWIN_PROOFS`` but is not
#: spelled on an ``ImplSpec.twin_class`` field. ``ops/moe/spec.py`` is lazy: ``family_impls``
#: rebuilds the family's declarations from whichever package the LIVE arch selects, and
#: ``ops/_dispatch._assert_proven`` refuses a selected package that declares ANY impl whose
#: ``supported_archs`` omits that arch. So on a B200 the moe registry can hold this package's
#: records and no others, and the sm90 member of a shared class cannot be declared beside ours --
#: a ``twin_class`` on our record would read as a class with exactly one member, which
#: ``core/tests/test_op_gate_alignment_cpu.py`` correctly refuses ("the equivalence they claim has
#: nothing to be checked against"). The equivalence is real and it is recorded where it is
#: checkable: ``TWIN_PROOFS`` lists this battery per class, which is what ``gdn.cpr`` does for the
#: same reason. Expressing a cross-arch class ON the records needs registry vocabulary that does
#: not exist yet (README "Remaining gaps"); ``moe.expert_mlp.segment_tree_glu``, whose two members
#: are both sm100, does carry the field.
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
    # Sorted top-k with the (score desc, index asc) tie rule -- ``SORTED_TOPK``'s program, and
    # ``core/registry.TWIN_PROOFS[SORTED_TOPK]`` now names this package's battery as the sm100
    # proof of it. The ``twin_class`` FIELD is deliberately NOT set: see WHY_NO_TWIN_FIELD.
    ImplSpec(
        op="moe.route_mechanics",
        impl_id="sm100_fused_topk_v1",
        version=1,
        supported_archs=_SM100,
    ),
    # Sigmoid scores with a no-aux expert bias: a different selection function, not a different
    # implementation of the same one, so it carries no twin class.
    ImplSpec(
        op="moe.route_mechanics",
        impl_id="sm100_sigmoid_noaux_bias_select_v1",
        version=1,
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
    # home-rank fold, which is the same call on rows shipped home. ``TWIN_PROOFS[KORDER]`` names
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
    "SM100_COMBINE_ORDER",
    "SM100_DISPATCH_ORDER",
    "SM100_SCORE_MATCHES_SM90",
    "SM100_SEGMENT_MULTIPLE",
    "SORTED_TOPK",
    "WHY_NO_TWIN_FIELD",
)
