"""L0 (CPU): the pure slicing math behind TP placement -- col/row with the Megatron strided
convention, the below-units policy, segmented dims (GDN conv_w), contiguous EP expert ranges,
assembly, and the placement leaf plan."""

import pytest
import torch

from isoexec.runtimes import parallel as P
from isoexec.schema import col, row


def _spec(name, full_shape, axis, tp, rank, **kw):
    # The engine plan: below the granule a unit is replicated. Since M4 that is a per-tensor
    # policy carried on the LocalSpec rather than a module constant, and ``ranges_for`` refuses
    # when it is absent -- so a hand-built spec has to say which branch of decision 1 it is.
    dim = axis.dim if axis.kind in ("col", "row") else None
    below = axis.units is not None and axis.units < tp
    kw.setdefault("policy", "replicate" if below else None)
    if dim is None:
        ranges = ((0, full_shape[0]),)
    else:
        ranges = P.colwise_ranges(
            full_shape[dim], tp, rank, stride=axis.stride, units=axis.units, below_units="replicate"
        )
    return P.LocalSpec(name, tuple(full_shape), axis, dim, ranges, "bf16", tp, axis.stride, axis.units, **kw)


def test_colwise_ranges_contiguous_and_strided_match_megatron_convention():
    assert P.colwise_ranges(16, 1, 0) == ((0, 16),)
    assert P.colwise_ranges(16, 2, 1) == ((8, 16),)
    assert P.colwise_ranges(16, 4, 2) == ((8, 12),)
    # stride 2 over tp 2: rank r owns pieces [r, r+2] of 4
    assert P.colwise_ranges(16, 2, 0, stride=2) == ((0, 4), (8, 12))
    assert P.colwise_ranges(16, 2, 1, stride=2) == ((4, 8), (12, 16))
    full = torch.arange(16 * 3).reshape(16, 3)
    for tp, stride in ((2, 1), (2, 2), (4, 2)):
        for r in range(tp):
            got = P.slice_ranges(full, 0, P.colwise_ranges(16, tp, r, stride=stride), ndim=2)
            # Megatron's strided split: rank r owns pieces r, r + tp, ... of tp * stride equal pieces.
            assert torch.equal(got, torch.cat(torch.chunk(full, tp * stride, dim=0)[r::tp], dim=0))


def test_below_units_policy_replicates_whole_units():
    REP = "replicate"  # the engine branch of decision 1, and every host policy before M4
    # 2 kv heads of 64 rows at TP=4: ranks 0,1 hold head 0; ranks 2,3 hold head 1 (never a half head)
    assert [P.colwise_ranges(128, 4, r, units=2, below_units=REP) for r in range(4)] == [
        ((0, 64),),
        ((0, 64),),
        ((64, 128),),
        ((64, 128),),
    ]
    # units >= tp: an ordinary split, whole units per rank; the policy is not consulted
    assert P.colwise_ranges(128, 2, 1, units=2) == ((64, 128),)
    # the default policy refuses rather than guessing what to do below the granule
    with pytest.raises(P.ShardError, match="below tp"):
        P.colwise_ranges(128, 4, 0, units=2)
    with pytest.raises(P.ShardError, match="below_units must be"):
        P.colwise_ranges(128, 4, 0, units=2, below_units="scatter")
    # decision 1, the trainer branch: the granule is ignored and the dim takes the ordinary grid, so
    # a rank holds a fraction of a unit and ``Attention.kv_gather`` reassembles the unit's activation.
    assert [P.colwise_ranges(128, 4, r, units=2, below_units="split") for r in range(4)] == [
        ((0, 32),),
        ((32, 64),),
        ((64, 96),),
        ((96, 128),),
    ]
    # the policy is a function of the side and the row: only an attention row has a gather seam
    assert P.below_units_policy("trainer", "attn") == "split"
    assert P.below_units_policy("trainer", "gdn") == "refuse"
    assert P.below_units_policy("engine", "attn") == "replicate"
    assert P.below_units_policy("engine", "gdn") == "replicate"
    with pytest.raises(P.ShardError, match="unknown side"):
        P.below_units_policy("both", "attn")
    with pytest.raises(P.ShardError, match="unknown row kind"):
        P.below_units_policy("trainer", "col")
    # a split shard is not replicated: every rank is its own representative
    assert P.replica_ranks(4, 2, "split") == (0, 1, 2, 3)
    assert P.replica_ranks(4, 2, "replicate") == (0, 2)
    assert P.units_replicate(4, 2, "split") is False
    assert (
        P.replica_ranks(4, 2) == (0, 2) and P.replica_ranks(4, None) == (0, 1, 2, 3) and P.replica_ranks(2, 2) == (0, 1)
    )
    with pytest.raises(P.ShardError, match="not a multiple"):
        P.colwise_ranges(96, 4, 0, units=3, below_units=REP)
    with pytest.raises(P.ShardError, match="stride 1"):
        P.colwise_ranges(128, 4, 0, stride=2, units=2, below_units=REP)
    with pytest.raises(P.ShardError, match="not divisible"):
        P.colwise_ranges(10, 4, 0)
    with pytest.raises(P.ShardError, match="do not divide"):
        P.colwise_ranges(96, 2, 0, stride=2, units=6)  # 6 units over 4 pieces


def test_segmented_conv_w_and_expert_range():
    segs = ((8, 4), (8, 4), (16, 8))  # q | k | v with hk=4, hv=8
    assert P.segmented_ranges(segs, 2, 1) == ((4, 8), (12, 16), (24, 32))
    assert P.segmented_ranges(segs, 4, 3) == ((6, 8), (14, 16), (28, 32))
    # At TP=4, key_heads=2 is below granule: engines replicate q/k while v splits.
    # Trainers refuse because GDN has no head-reassembly seam.
    segs2 = ((8, 2), (8, 2), (16, 8))
    assert P.segmented_ranges(segs2, 4, 1, below_units="replicate") == ((0, 4), (8, 12), (20, 24))
    with pytest.raises(P.ShardError, match="below tp"):
        P.segmented_ranges(segs2, 4, 1)  # the default refuses rather than guessing
    assert P.below_units_policy("trainer", "gdn") == "refuse"
    from isoexec.nn.moe import expert_range

    assert expert_range(8, 2, 1, site="moe") == (4, 8) and expert_range(256, 8, 3, site="moe") == (96, 128)
    with pytest.raises(ValueError, match="do not divide"):
        expert_range(6, 4, 0, site="moe")


def test_assemble_full_round_trips_stride_replication_and_segments():
    full = torch.arange(32 * 2, dtype=torch.int32).reshape(32, 2)
    cases = [
        (col(stride=2), 4, None),
        (col(units=2), 4, None),  # replicated units
        (row(), 2, None),
        (col(), 1, None),
    ]
    for axis, tp, _ in cases:
        specs = [_spec("t", full.shape, axis, tp, r) for r in range(tp)]
        shards = [s.slice(full) for s in specs]
        assert torch.equal(P.assemble_full(shards, specs[0], tp), full), axis
        # any rank's spec reconstructs any other rank's ranges
        assert specs[1 % tp].ranges_for(0) == specs[0].ranges
    # segmented dim (conv_w-like): [q 8 | k 8 | v 16] with hk=2 (replicates at tp=4), hv=8
    segs = ((8, 2), (8, 2), (16, 8))
    for tp in (2, 4):
        specs = [
            P.LocalSpec(
                "conv",
                (32, 2),
                col(segments=segs),
                0,
                P.segmented_ranges(segs, tp, r, below_units="replicate"),
                "bf16",
                tp,
                segments=segs,
                policy="replicate",
            )
            for r in range(tp)
        ]
        shards = [s.slice(full) for s in specs]
        assert shards[0].shape[0] == 32 // tp + (8 // 2 - 8 // tp) * 2 if tp == 4 else shards[0].shape[0] == 16
        assert torch.equal(P.assemble_full(shards, specs[0], tp), full)


class _SliceReader:
    """A safetensors ``get_slice``-like handle: indexable with ints/slices, returns tensors."""

    def __init__(self, t):
        self.t = t

    def __getitem__(self, idx):
        return self.t[idx]

    def get_shape(self):
        return list(self.t.shape)


def test_placement_validation_and_expert_parallel_derivation():
    from isoexec.schema import Axis, Placement

    def pl(tp, ep=1, dp=1, leaves=8, dt="bf16", side="engine"):
        t, one = Axis(tp, 0, None), Axis(1, 0, None)
        return Placement(
            tp=t,
            ep=Axis(ep, 0, None),
            dp=Axis(dp, 0, None),
            pp=one,
            etp=t if ep == 1 else one,
            cp=one,
            side=side,
            leaves=leaves,
            leaf_dtype=dt,
        )

    facts, problems = P._validate_leaf_plan(pl(4))
    assert facts["etp"] == 4 and facts["leaves"] == 8 and problems == []
    with pytest.raises(P.ShardError, match="power of two"):
        P._validate_leaf_plan(pl(3, leaves=6))
    assert P._validate_leaf_plan(pl(4, dt="fp32"))[0]["leaf_dtype"] == "fp32"
    with pytest.raises(P.ShardError, match="must divide"):
        P._validate_leaf_plan(pl(16, leaves=8))
    # production trainer: tp=4, dp=2, ep=8 -> etp=1; engine: tp=8, ep=1 -> etp=8.
    # ``expert_parallel`` is a read now, not a derivation from tp*dp/ep: the Placement states the
    # degree and its ``__post_init__`` refuses the one grid that has no handle.
    assert P.expert_parallel(pl(4, ep=8, dp=2)).etp == 1
    assert P.expert_parallel(pl(8)).etp == 8
    assert P.expert_parallel(pl(8)).etp_rank == 0 and P.expert_parallel(pl(8)).group is None
    with pytest.raises(ValueError, match="2-D grid"):
        Placement(
            tp=Axis(4, 0, None),
            ep=Axis(2, 0, None),
            dp=Axis(1, 0, None),
            pp=Axis(1, 0, None),
            etp=Axis(4, 0, None),
            cp=Axis(1, 0, None),
            side="trainer",
            leaves=8,
        )
    with pytest.raises(ValueError, match="context parallelism"):
        Placement(
            tp=Axis(1, 0, None),
            ep=Axis(1, 0, None),
            dp=Axis(1, 0, None),
            pp=Axis(1, 0, None),
            etp=Axis(1, 0, None),
            cp=Axis(2, 0, None),
            side="trainer",
            leaves=8,
        )
