"""Import-light declarations for fixed-tree collectives and tensor/context/sequence byte movement.
Callers supply leaf plans and group degrees; sequence gather/scatter move rank-ordered dim-0 chunks
without arithmetic in either direction, while partial-gradient adjoints use the fixed tree."""

from __future__ import annotations

from isoexec.core.registry import ANY, ImplSpec, OneOf

#: NCCL moves bytes here and performs no arithmetic, so the channel count cannot move a bit.
NCCL_ROLE = "byte_movement_only"

#: The gathered result is each rank's slice in rank order along the last dimension.
ALL_GATHER_CONCAT = "rank_order_last_dim"

#: The context-parallel attention core: gather every rank's tokens, keep the owned rows' output.
#: ``cp_gather.make_cp_attention_core`` stamps this same name on the callable it returns.
CP_LAYOUT = "gather_full_keep_owned"

#: Its backward: a row's cotangent is the sum of exactly two contributors (own rank and peer).
CP_BACKWARD_REDUCTION = "sum_two_contributors"

# Equal rank-ordered dim-0 chunks; non-divisible streams refuse instead of adding padding.
SP_STREAM_SPLIT = "rank_order_dim0_chunks"

# Both directions move bytes only; P2P/NCCL equality includes signed zero and NaN payloads.
SP_SEAM_ROLE = "byte_movement_only_both_directions"

# Replicated consumers already compute complete gradients: select the local chunk without summing.
SP_GATHER_BACKWARD = "own_chunk_no_reduction"

# Gather forward and scatter backward share _gather_dim0 and one recipe version.
SP_SEAM_VERSION = 1

#: Architectures with battery evidence. sm100: tests/collectives_multirank_battery_gpu.py at TP2 and TP4
#: (every impl below, both directions, NCCL transport: sm100 does not admit P2P, see
#: pik_tp_invariant.NCCL_ONLY_ARCHS) and tests/pik_blackwell_config_invariance_gpu.py
#: (the leaf GEMM's autotune space). The tree and the byte movement are arch-independent, but the
#: row-parallel leaf GEMMs are not, so bits are equal within one arch, not across archs.
_ARCHS = frozenset({"sm90", "sm100"})
#: The leaf-tree plan row-parallel implementations take from their caller. `leaves` is config
#: (ISOEXEC_PIK_LEAVES on the bridge path, `Placement.leaves` on the unified one); `leaf_dtype` is
#: the wire dtype.
_TREE_INPUTS = {"leaves": ANY, "leaf_dtype": OneOf("bf16", "fp32")}

IMPLS: tuple[ImplSpec, ...] = (
    # Declare all-reduce and SP reduce-scatter as equivalent layouts of the same fixed-tree result.
    ImplSpec(
        op="collectives.row_parallel",
        impl_id="pik_tree",
        version=1,
        supported_archs=_ARCHS,
        inputs=dict(_TREE_INPUTS),
        twin_class="collectives.row_parallel.pik_tree",
    ),
    ImplSpec(
        op="collectives.row_parallel",
        impl_id="pik_tree_rs",
        version=1,
        supported_archs=_ARCHS,
        inputs=dict(_TREE_INPUTS),
        twin_class="collectives.row_parallel.pik_tree",
    ),
    # Trainer tensor-parallel byte movements.
    ImplSpec(
        op="collectives.tp_copy",
        impl_id="identity_fwd_tree_sum_bwd",
        version=1,
        supported_archs=_ARCHS,
        # The backward's fixed reduction tree (``collectives.tree_all_reduce``) is the impl, not an
        # input: this impl id IS "identity forward, tree sum backward".
        inputs={"tp": ANY, **_TREE_INPUTS},
    ),
    # Gather sequence-local inputs and reduce-scatter their partial gradients.
    ImplSpec(
        op="collectives.tp_copy",
        impl_id="seq_all_gather_fwd_tree_rs_bwd",
        version=1,
        supported_archs=_ARCHS,
        inputs={"tp": ANY, **_TREE_INPUTS},
    ),
    ImplSpec(
        op="collectives.all_gather",
        impl_id="nccl_bytes_tree_rs_bwd",
        version=1,
        supported_archs=_ARCHS,
        # NCCL carries bytes here and performs no arithmetic, so the channel count cannot move a bit
        # in the forward; the role, the concatenation order and the backward are all this impl's.
        inputs={"tp": ANY},
    ),
    # Sequence gather/scatter declare only TP degree: neither direction uses a leaf tree.
    # Each op has one implementation.
    ImplSpec(
        op="collectives.sp_gather",
        impl_id="seq_all_gather_replicated_fwd_slice_bwd",
        version=SP_SEAM_VERSION,
        supported_archs=_ARCHS,
        inputs={"tp": ANY},
    ),
    ImplSpec(
        op="collectives.sp_scatter",
        impl_id="seq_scatter_fwd_all_gather_bwd",
        version=SP_SEAM_VERSION,
        supported_archs=_ARCHS,
        inputs={"tp": ANY},
    ),
    # Trainer context-parallel movement for the qualified two-rank layout.
    ImplSpec(
        op="collectives.cp_gather",
        impl_id="all_gather_unzigzag_fwd_sum_rs_bwd",
        version=1,
        supported_archs=_ARCHS,
        # Only the degree is the caller's; the layout and its two-contributor backward are the impl.
        inputs={"cp": OneOf(2)},
    ),
    # The gqa_sm100 trainer core under CP: K|V gathered forward, Q|dO|statistics gathered backward, every
    # local row computed by the CP=1 tile that owns it (gqa_sm100/context_parallel.py). Bytes only.
    ImplSpec(
        op="collectives.cp_gather",
        impl_id="kv_all_gather_fwd_owner_tiles_bwd",
        version=1,
        supported_archs=frozenset({"sm100"}),
        inputs={"cp": OneOf(2)},
    ),
    # Head-wise CP of a recurrent mixer (cp_heads.py): all-to-all to (all tokens, 1/cp heads) and back.
    ImplSpec(
        op="collectives.cp_heads",
        impl_id="head_all_to_all_both_directions",
        version=1,
        supported_archs=_ARCHS,
        inputs={"cp": OneOf(2)},
    ),
    # Contiguous context parallelism (any degree): each rank attends its own query rows; lower ranks
    # send the key/value prefix of the first local sequence point-to-point (bytes only). The backward
    # returns prefix cotangents to their owners, which add them in fp32 in ascending contributor rank.
    ImplSpec(
        op="collectives.cp_gather",
        impl_id="contiguous_p2p_prefix_fwd_ranked_sum_bwd",
        version=1,
        supported_archs=_ARCHS,
        inputs={"cp": ANY},
    ),
    # Contiguous context parallelism of a recurrent mixer: the fp32 state and the conv tail of a
    # sequence cut by a shard boundary move rank r-1 -> r (bytes only); cotangents travel back.
    ImplSpec(
        op="collectives.cp_state",
        impl_id="contiguous_p2p_state_chain",
        version=1,
        supported_archs=_ARCHS,
        inputs={"cp": ANY},
    ),
)
