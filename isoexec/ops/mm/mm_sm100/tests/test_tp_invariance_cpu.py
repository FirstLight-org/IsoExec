"""The sm100 K segmentation does not move with tensor parallelism, packing or M (no CUDA, no build).

For every dense GEMM site of the released Qwen3.5 geometries and every rank of TP 1/2/4/8: a
column-parallel or replicated rank's GEMM is planned with the segmentation of its logical weight at
TP=1, whatever its local N and whatever it is packed with. A row-parallel rank shards K, which the
mm leaf never sees: pik reduces it over fixed leaves of the FULL K, and each rank's slice is a whole
aligned subtree of that one tree at every TP degree.
"""

from __future__ import annotations

import pytest

from isoexec.ops.mm.mm_sm100 import segments as seg
from isoexec.ops.mm.mm_sm100.tests import _tp_sites as T

SITES = [s for g in T.GEOMETRIES for s in T.sites(g)]
MS = (1, 8, 23, 128, 129, 512, 4096, 16384)


def _id(site):
    return f"{site.geometry}:{site.name}"


def test_every_released_geometry_presents_its_gemm_sites():
    by_geometry = {g: {s.name for s in SITES if s.geometry == g} for g in T.GEOMETRIES}
    dense = {"gdn.in_proj", "gdn.out", "attn.q", "attn.k", "attn.v", "attn.gate", "attn.qkv_gate(packed)",
             "attn.o", "lm_head"}
    assert by_geometry["qwen3.5-0.8b"] == by_geometry["qwen3.5-9b"] == dense | {"mlp.gate", "mlp.up", "mlp.down"}
    assert by_geometry["qwen3.5-35b-a3b"] == dense | {"shared.gate_up", "shared.down", "shared.expert_gate",
                                                      "router"}


@pytest.mark.parametrize("site", [s for s in SITES if s.kind != "row" and s.dtype == "bf16"
                                  and s.name != "shared.expert_gate"], ids=_id)
def test_a_column_shard_runs_the_logical_weights_segmentation(site):
    K, N = T.local_shape(site, 1, 0)
    split = seg.segment_count(K, "bf16")
    for member in site.members:  # packing: each member alone is the same program as the pack
        assert seg.plan_for(512, K, member.shape[0], "bf16").split == split, member.role
    for tp in T.TPS:
        for rank in range(tp):
            k, n = T.local_shape(site, tp, rank)
            assert k == K
            n += -n % seg.PITCH_ALIGN_N  # what ``mm_linear`` issues for a pitch-unaligned shard
            assert seg.supported(k, n), (tp, rank, n)  # mm_sm100 serves every rank: no provider switch
            for M in MS:
                plan = seg.plan_for(M, k, n, "bf16")
                assert (plan.split, plan.segment_k) == (split, K // split), (tp, rank, n, M, plan)


@pytest.mark.parametrize("site", [s for s in SITES if s.kind == "row"], ids=_id)
def test_a_row_shard_is_a_whole_subtree_of_the_full_K_leaf_tree(site):
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik
    from isoexec.runtimes.trainer_installs import PIK_LEAVES

    pik = ensure_pik()
    plan = pik.ReductionPlan(num_leaves=PIK_LEAVES)
    K = site.K
    leaf_k = plan.leaf_k(K)
    # The tree's declared largest TP degree is the leaf count: d_max shards of the policy's one segment.
    assert seg.row_parallel_leaves(K, PIK_LEAVES) == plan.num_leaves == PIK_LEAVES
    assert seg.schedule(K, PIK_LEAVES).S == leaf_k and seg.schedule(K, PIK_LEAVES).G == 1

    def tree(values):
        slots = list(values)
        for dst, lhs, rhs in pik.combine_order(len(slots)):
            slots[dst] = f"({slots[lhs]}+{slots[rhs]})"
        return slots[0]

    whole = tree(str(g) for g in range(plan.num_leaves))
    for tp in T.TPS:
        plan.validate(K, tp)
        assert plan.leaf_k(K) == leaf_k  # the leaves are the full K's, at every TP degree
        roots = []
        for rank in range(tp):
            lo, hi = T.row_range(site, tp, rank)
            first, last = plan.rank_leaf_range(rank, tp)
            assert (lo, hi) == (first * leaf_k, last * leaf_k), (tp, rank)
            roots.append(tree(str(g) for g in range(first, last)))
        # the cross-rank tree over the rank roots is the TP=1 tree over the leaves
        assert tree(roots) == whole, tp


@pytest.mark.parametrize("site", [s for s in SITES if s.kind == "replicated"], ids=_id)
def test_a_replicated_projection_is_the_same_gemm_on_every_rank(site):
    assert {T.local_shape(site, tp, rank) for tp in T.TPS for rank in range(tp)} == {(site.K, site.members[0].shape[0])}
