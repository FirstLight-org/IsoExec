"""Import-light declarations for SM90 experts, routing and combination.
Twin classes cover sorted-topk/O2 routing and fixed-order/EP combination;
leaf plan, leaf dtype and activation are the expert region's inputs."""


from isoexec.core.registry import ANY, ImplSpec, OneOf

#: The FC1 GLU epilogue's activation. The arithmetic is the CUDA sources' -- ``csrc/chunk_fc1.cuh``,
#: ``csrc/gathered_fc1.cuh``, ``csrc/wide_fc1.cuh`` and ``csrc/small_gemm.cuh`` each spell
#: ``x / (1 + expf(-x))`` inline -- and ``mlp.act`` owns the reference function this names.
FC1_GLU_ACT = "silu"

# Combine walks ascending expert ID, then route slot within each expert (csrc/compact_combine.cuh).
SM90_COMBINE_ORDER = "ascending_expert_then_slot"

_SM90 = frozenset({"sm90"})
KORDER = "moe.combine.korder_fp32"
SORTED_TOPK = "moe.route.sorted_topk"
#: The fc2 leaf-tree plan the caller supplies, plus the activation the expert MLP calls (`mlp.act`
#: owns its arithmetic).
_EXPERT_INPUTS = {"leaves": ANY, "leaf_dtype": OneOf("bf16", "fp32"), "act": ANY}

IMPLS: tuple[ImplSpec, ...] = (
    ImplSpec(op="moe.route_mechanics", impl_id="deterministic_softmax_topk", version=1, supported_archs=_SM90),
    ImplSpec(op="moe.expert_mlp", impl_id="cuda_wgmma_after_fc2_v1", version=1,
             supported_archs=_SM90, inputs=dict(_EXPERT_INPUTS)),
    ImplSpec(
        op="moe.route_score",
        impl_id="deterministic_fp32_gemm",
        version=1,
        supported_archs=_SM90,
    ),
    ImplSpec(
        op="moe.route_mechanics",
        impl_id="deterministic_sorted_topk",
        version=1,
        supported_archs=_SM90,
        twin_class=SORTED_TOPK,
    ),
    ImplSpec(
        op="moe.route_mechanics",
        impl_id="fused_o2",
        version=1,
        supported_archs=_SM90,
        twin_class=SORTED_TOPK,
    ),
    # Sigmoid scores with a no-aux expert bias and group top-k: a different selection function,
    # not a different implementation of the same one.
    ImplSpec(op="moe.route_mechanics", impl_id="sigmoid_noaux_bias_select", version=1, supported_archs=_SM90),
    ImplSpec(
        op="moe.expert_mlp",
        impl_id="cuda_wgmma_v1",
        version=1,
        supported_archs=_SM90,
        inputs=dict(_EXPERT_INPUTS),
    ),
    ImplSpec(
        op="moe.combine",
        impl_id="pik_leaf_tree",
        version=1,
        supported_archs=_SM90,
        twin_class=KORDER,
    ),
    ImplSpec(
        op="moe.combine",
        impl_id="pik_leaf_tree_ep_owner_fold",
        version=1,
        supported_archs=_SM90,
        twin_class=KORDER,
    ),
    ImplSpec(op="moe.shared_expert", impl_id="eager", version=1, supported_archs=_SM90),
)
