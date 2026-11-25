"""Shard-range arithmetic, placement validation and parallel-mesh admission shared by both runtimes."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, TypedDict

import torch
import torch.nn as nn

from isoexec.ops.moe.expert_parallel import EP_WIRE_DTYPE
from isoexec.schema import Axis, Placement, ShardAxis, ShardError
from isoexec.schema.run import LEAF_DTYPES

#: The op id whose impl the expert-parallel invariance check judges.
COMBINE = "moe.combine"

# Qualified combine implementations preserve ascending global-expert fp32
# accumulation with one final narrowing. Check registry twin-class membership.
COMBINE_TWIN_CLASS = "moe.combine.korder_fp32"

Ranges = tuple[tuple[int, int], ...]


# Pure shard-range calculations.


BELOW_UNITS_MODES = ("refuse", "replicate", "split")


#: Distinguish row types that permit splitting below their declared unit size.
BELOW_UNITS_ROW_KINDS = ("attn", "gdn", "other")


def below_units_policy(side: str, row_kind: str) -> str:
    """Choose whether parameters with fewer units than TP ranks are replicated, split, or rejected."""
    if side not in ("trainer", "engine"):
        raise ShardError(f"below_units_policy: unknown side {side!r}")
    if row_kind not in BELOW_UNITS_ROW_KINDS:
        raise ShardError(f"below_units_policy: unknown row kind {row_kind!r} (vocabulary {BELOW_UNITS_ROW_KINDS})")
    if side == "engine":
        return "replicate"
    return "split" if row_kind == "attn" else "refuse"


def colwise_ranges(
    dim_size: int,
    tp: int,
    rank: int,
    *,
    stride: int = 1,
    units: int | None = None,
    below_units: str = "refuse",
) -> Ranges:
    """The half-open index ranges along a sharded dim that `rank` of `tp` owns."""
    if tp < 1 or not (0 <= rank < tp):
        raise ShardError(f"rank {rank} out of range for tp={tp}")
    if stride < 1:
        raise ShardError(f"stride must be >= 1, got {stride}")
    if below_units not in BELOW_UNITS_MODES:
        raise ShardError(f"below_units must be one of {BELOW_UNITS_MODES}, got {below_units!r}")
    if tp == 1:
        return ((0, dim_size),)
    if units is not None:
        if units < 1 or dim_size % units != 0:
            raise ShardError(f"dim {dim_size} is not a whole number of {units} units")
        if units < tp:
            if below_units == "refuse":
                raise ShardError(f"{units} units below tp={tp} and the below-units policy is {below_units!r}")
            if stride != 1:
                raise ShardError(f"below-units {below_units!r} needs stride 1")
            if tp % units != 0:
                raise ShardError(f"tp={tp} is not a multiple of the {units} declared units")
            if below_units == "split":
                # The ordinary grid: the unit granule is deliberately ignored, and the activation is
                # reassembled by ``Attention.kv_gather`` over the ranks that share a unit.
                if dim_size % tp != 0:
                    raise ShardError(f"dim {dim_size} does not split over tp={tp}")
                piece = dim_size // tp
                return ((rank * piece, (rank + 1) * piece),)
            unit = dim_size // units
            u = rank // (tp // units)
            return ((u * unit, (u + 1) * unit),)
        if units % (tp * stride) != 0:
            raise ShardError(f"{units} units do not divide over tp={tp} x stride={stride}")
    if dim_size % (tp * stride) != 0:
        raise ShardError(f"dim {dim_size} is not divisible by tp={tp} x stride={stride}")
    piece = dim_size // (tp * stride)
    return tuple(((rank + j * tp) * piece, (rank + j * tp + 1) * piece) for j in range(stride))


def units_replicate(tp: int, units: int | None, policy: str = "replicate") -> bool:
    """Do ``tp // units`` ranks hold the same bytes of this dim? Only under the replicate policy."""
    return policy == "replicate" and units is not None and tp > 1 and units < tp


def replica_ranks(tp: int, units: int | None, policy: str = "replicate") -> tuple[int, ...]:
    """One representative rank per distinct shard (all ranks unless units replicate)."""
    if not units_replicate(tp, units, policy):
        return tuple(range(tp))
    assert units is not None
    step = tp // units
    return tuple(u * step for u in range(units))


def segmented_ranges(
    segments: Iterable[tuple[int, int | None]],
    tp: int,
    rank: int,
    *,
    below_units: str = "refuse",
) -> Ranges:
    """Compute local ranges for independently sharded segments of a packed dimension."""
    out: list[tuple[int, int]] = []
    off = 0
    for size, units in segments:
        out += [(off + lo, off + hi) for lo, hi in colwise_ranges(size, tp, rank, units=units, below_units=below_units)]
        off += size
    return tuple(out)


def ranges_extent(ranges: Ranges) -> int:
    return sum(hi - lo for lo, hi in ranges)


def slice_ranges(src, dim: int, ranges: Ranges, *, ndim: int) -> torch.Tensor:
    """Read the ``ranges`` of ``src`` along ``dim`` and concatenate. ``src`` is anything indexable with
    a tuple of slices to a tensor (a tensor, or a safetensors ``get_slice`` handle)."""
    pieces = []
    for lo, hi in ranges:
        idx: list[Any] = [slice(None)] * ndim
        idx[dim] = slice(lo, hi)
        pieces.append(_as_tensor(src[tuple(idx)]))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=dim)


def _as_tensor(x) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def assemble_full(shards: Mapping[int, torch.Tensor] | list, spec: "LocalSpec", tp: int) -> torch.Tensor:
    """Rebuild the full logical tensor from per-rank local shards (rank -> tensor). Handles stride,
    unit replication and segmented dims by placing each rank's ranges; duplicates (replicated units)
    are written once by their representative rank. Replicated tensors return rank 0's."""
    if isinstance(shards, list):
        shards = dict(enumerate(shards))
    if spec.dim is None:
        return shards[min(shards)]
    first = next(iter(shards.values()))
    full = torch.empty(spec.full_shape, dtype=first.dtype, device=first.device)
    # Choose one owner per distinct shard to avoid duplicate assembly.
    reps = replica_ranks(tp, spec.units if spec.segments is None else None, spec.policy or "replicate")
    for r in reps:
        piece = shards[r]
        ranges = spec.ranges_for(r, tp)
        off = 0
        for lo, hi in ranges:
            full.narrow(spec.dim, lo, hi - lo).copy_(piece.narrow(spec.dim, off, hi - lo))
            off += hi - lo
    return full


@dataclass(frozen=True)
class LocalSpec:
    """How this rank holds one logical tensor: the declared axis, the effective sharded dim and the
    index ranges it owns along it (``dim is None`` -> whole tensor), the EP ownership, and the
    ingredients to recompute any other rank's ranges (``ranges_for``)."""

    name: str
    full_shape: tuple[int, ...]
    axis: ShardAxis
    dim: int | None
    ranges: Ranges
    dtype: str
    tp: int
    stride: int = 1
    units: int | None = None
    segments: tuple[tuple[int, int | None], ...] | None = None  # conv_w: (size, units) per segment
    expert_id: int | None = None
    ep_local: bool = True
    #: Record the selected split/replication policy only when units are fewer than TP ranks.
    policy: str | None = None

    @property
    def local_shape(self) -> tuple[int, ...]:
        if self.dim is None:
            return self.full_shape
        s = list(self.full_shape)
        s[self.dim] = ranges_extent(self.ranges)
        return tuple(s)

    @property
    def sharded(self) -> bool:
        return self.dim is not None and self.local_shape != self.full_shape

    @property
    def replica_ranks(self) -> tuple[int, ...]:
        """Choose one representative rank for each distinct tensor shard."""
        if self.segments is not None:
            return tuple(range(self.tp))
        return replica_ranks(self.tp, self.units, self.policy or "replicate")

    def ranges_for(self, rank: int, tp: int | None = None) -> Ranges:
        tp = self.tp if tp is None else tp
        if self.dim is None:
            return ((0, self.full_shape[0] if self.full_shape else 1),)
        if self.segments is not None:
            return segmented_ranges(self.segments, tp, rank, below_units=self.policy or "refuse")
        return colwise_ranges(
            self.full_shape[self.dim],
            tp,
            rank,
            stride=self.stride,
            units=self.units,
            below_units=self.policy or "refuse",
        )

    def with_tp(self, tp: int, rank: int) -> "LocalSpec":
        """The same tensor's spec for another (tp, rank) of the same EP placement."""
        return dataclasses.replace(self, tp=tp, ranges=self.ranges_for(rank, tp))

    def slice(self, src) -> torch.Tensor:
        """This rank's local tensor out of a full-shape source."""
        if self.dim is None:
            return _as_tensor(src[(slice(None),) * len(self.full_shape)])
        return slice_ranges(src, self.dim, self.ranges, ndim=len(self.full_shape))


@dataclass(frozen=True)
class ExpertParallel:
    """The expert-parallel facts one call site needs, read straight off the Placement's axes."""

    ep: int
    ep_rank: int
    etp: int
    etp_rank: int
    group: object | None
    ep_group: object | None = None  # the EP process group (expert-id partition); None at ep == 1


def expert_parallel(placement: Placement) -> ExpertParallel:
    """``ExpertParallel`` for ``placement`` -- a pure read; the grid refusals live in ``Placement``."""
    etp = placement.etp
    if etp.size == 1:
        return ExpertParallel(placement.ep.size, placement.ep.rank, 1, 0, None, placement.ep.group)
    return ExpertParallel(placement.ep.size, placement.ep.rank, etp.size, etp.rank, etp.group, placement.ep.group)


class _LocalSpecBase(TypedDict):
    name: str
    full_shape: tuple[int, ...]
    axis: ShardAxis
    dtype: str
    tp: int


def axis_local_spec(
    name: str,
    full_shape: tuple[int, ...],
    axis: ShardAxis,
    dtype: str,
    *,
    tp: int,
    rank: int,
    side: str,
    kind: str,
) -> LocalSpec:
    """Plan one replicated, column, or row tensor of ``full_shape`` for ``rank`` of ``tp``.

    Units below ``tp`` follow ``below_units_policy(side, kind)``. Segmented axes reject a stride
    because ``segmented_ranges`` has no strided form.
    """
    base: _LocalSpecBase = {"name": name, "full_shape": tuple(full_shape), "axis": axis, "dtype": dtype, "tp": tp}
    if axis.kind == "replicated":
        return LocalSpec(dim=None, ranges=((0, full_shape[0] if full_shape else 1),), **base)
    if axis.kind not in ("col", "row"):
        raise ShardError(f"{name}: ShardAxis({axis.kind}) is not a replicated, column, or row axis")
    if axis.dim is None:
        raise ShardError(f"{name}: ShardAxis({axis.kind}) needs a dim")
    if axis.kind == "row" and (axis.units is not None or axis.segments is not None):
        raise ShardError(f"{name}: units / segments are a col (output-row) rule, not a K-dim rule")
    if axis.segments is not None:
        if axis.stride != 1:
            raise ShardError(f"{name}: segmented axes do not support stride={axis.stride}")
        if sum(size for size, _ in axis.segments) != full_shape[axis.dim]:
            raise ShardError(f"{name}: declared segments {axis.segments} do not sum to dim {axis.dim} of {full_shape}")
        # Shard GDN convolution segments by their own head counts; no partial-head gather is available.
        below = any(u is not None and u < tp for _, u in axis.segments)
        seg_pol = below_units_policy(side, kind) if below else None
        return LocalSpec(
            dim=axis.dim,
            ranges=segmented_ranges(axis.segments, tp, rank, below_units=seg_pol or "refuse"),
            segments=axis.segments,
            policy=seg_pol,
            **base,
        )
    # Split attention KV units only when the matching gather exists; reject unsupported GDN splits.
    pol = below_units_policy(side, kind) if (axis.units is not None and axis.units < tp) else None
    return LocalSpec(
        dim=axis.dim,
        ranges=colwise_ranges(full_shape[axis.dim], tp, rank, stride=axis.stride, units=axis.units,
                              below_units=pol or "refuse"),
        stride=axis.stride,
        units=axis.units,
        policy=pol,
        **base,
    )


# Validate placement against the resolved operator requirements.


def combine_twin_members() -> frozenset[str]:
    """Read members of the qualified combine twin class from the registry. Reject an empty class
    rather than admitting unknown implementations.
    """
    from isoexec.core.registry_build import build_registry

    registry = build_registry()
    op = registry.get_op(COMBINE)
    members = frozenset(impl.impl_id for impl in op.impls.values() if impl.twin_class == COMBINE_TWIN_CLASS)
    if not members:
        raise ShardError(
            f"no impl of {COMBINE!r} declares twin class {COMBINE_TWIN_CLASS!r} (declared: "
            f"{sorted(op.impls)}); expert parallelism cannot be admitted against a class no "
            f"declaration names"
        )
    return members


def _validate_leaf_plan(placement: Placement, *, fc2_leaves: int | None = None) -> tuple[dict[str, Any], list[str]]:
    """Validate placement-only leaf constraints and return ``(facts, problems)``. Malformed leaf
    counts, dtypes and incompatible TP degrees raise immediately; binding-dependent mismatches
    are collected.
    """
    tp, g = placement.tp.size, int(placement.leaves)
    if g < 1 or (g & (g - 1)) != 0:
        raise ShardError(f"leaves must be a power of two, got {g}")
    if placement.leaf_dtype not in LEAF_DTYPES:
        raise ShardError(f"leaf_dtype must be one of {LEAF_DTYPES}, got {placement.leaf_dtype!r}")
    if tp > g or g % tp != 0:
        raise ShardError(f"tp={tp} must divide leaves={g} (every rank owns whole leaves of the fixed tree)")
    xp = expert_parallel(placement)
    fc2 = g if fc2_leaves is None else int(fc2_leaves)
    problems: list[str] = []
    if xp.etp > fc2 or fc2 % xp.etp != 0:
        problems.append(f"expert fc2 leaves {fc2} not divisible by etp={xp.etp}")
    facts = {
        "leaves": g,
        "leaf_dtype": placement.leaf_dtype,
        "tp": tp,
        "etp": xp.etp,
        "ep": xp.ep,
        "ep_rank": xp.ep_rank,
        "fc2_leaves": fc2,
        "combine_wire_dtype": EP_WIRE_DTYPE,
    }
    return facts, problems


# Expert-parallel mesh admission.


def expert_data_parallel_degree(tp: int, dp: int, ep: int, cp: int = 1) -> int:
    """How many replicas of each expert shard the dense ``TP x CP x DP`` mesh holds (expert DP).

    With whole experts (ETP=1), Megatron forms the expert mesh ``EP x EDP`` over the same ranks as the
    dense ``TP x CP x DP`` mesh, so the only constraint is that EP divides it. EP need not be a
    multiple of TP: the dispatch (``ops/moe/expert_parallel.expert_dispatch``) sends each TP rank's
    own token slice inside that rank's EP group and restores the TP replication over the TP group,
    so an EP group may hold parts of several TP groups (TP4/EP2, TP2/EP3) or a part of one. Expert-DP
    replicas see disjoint tokens and Megatron's expert DDP group sums their gradients.
    """
    tp, dp, ep, cp = int(tp), int(dp), int(ep), int(cp)
    if min(tp, dp, ep, cp) < 1:
        raise ShardError(f"parallel degrees must be >= 1, got tp={tp} cp={cp} dp={dp} ep={ep}")
    if ep == 1:
        return 1  # no expert parallelism: experts follow the dense TP layout, nothing is replicated per shard
    dense = tp * cp * dp
    if dense % ep != 0:
        raise ShardError(
            f"expert_model_parallel_size={ep} must divide TP x CP x DP = {tp} x {cp} x {dp} = {dense}: at expert TP 1 "
            "the expert mesh EP x expert-DP covers the same ranks as the dense mesh"
        )
    return dense // ep


def expert_data_parallel_refusal(expert_dp: int, live_expert_dp: int) -> str | None:
    """Why the live expert mesh disagrees with the configuration, or ``None``.

    Every expert-DP degree is admitted (the replicas only add a gradient sum over disjoint tokens);
    what is refused is a strategy that built a different mesh than the one the degrees imply.
    """
    if int(live_expert_dp) != int(expert_dp):
        return (
            f"expert data parallelism: the config implies {expert_dp} replicas per expert shard and the live "
            f"expert-DP group has world {live_expert_dp}; the strategy initialized a different mesh"
        )
    return None


def trivial_placement(*, side: str = "trainer", leaves: int = 8, leaf_dtype: str = "bf16") -> Placement:
    """Every axis at size 1, for placing a module in one process (``isoexec.nn.place``). ``side``
    still matters -- it selects the below-units policy -- and has no silent default at any other
    call site, so it is spelled here too."""
    one = Axis(1, 0, None)
    return Placement(tp=one, ep=one, dp=one, pp=one, etp=one, cp=one, side=side, leaves=leaves, leaf_dtype=leaf_dtype)


def validate_component_placement(model: nn.Module, placement: Placement) -> dict[str, Any]:
    """Validate shared leaf constraints and each mixture's resolved combine against its qualified
    twin class. Component row-parallel pins are constructed directly from placement.
    """
    facts, problems = _validate_leaf_plan(placement)
    xp = expert_parallel(placement)
    if xp.ep > 1:
        if placement.ep.group is None:
            problems.append("ep>1 needs Placement.ep.group (placement never creates groups: host-owned handles)")
        from isoexec.nn import resolved_ops

        members = combine_twin_members()
        for site, slots in resolved_ops(model).items():
            combine = slots.get("combine")
            if combine is None or combine.impl in members:
                continue
            problems.append(
                f"{site}.combine is bound to {combine.impl!r}, which is not a member of twin class "
                f"{COMBINE_TWIN_CLASS!r} (members: {sorted(members)}); EP invariance IS that class's claim -- "
                f"a token's k contributions folded over ascending GLOBAL expert ids in an fp32 accumulator, "
                f"one narrowing at the end -- and an impl outside it may fold a different order at a "
                f"different EP layout"
            )
    if problems:
        raise ShardError(
            "refusing a Placement whose reduction plan disagrees with the composition entry (a host "
            "cannot change reduction order without rotating numerical_policy): " + "; ".join(problems)
        )
    return facts


__all__ = [
    "BELOW_UNITS_MODES",
    "BELOW_UNITS_ROW_KINDS",
    "COMBINE",
    "COMBINE_TWIN_CLASS",
    "ExpertParallel",
    "LocalSpec",
    "ShardError",
    "assemble_full",
    "axis_local_spec",
    "below_units_policy",
    "colwise_ranges",
    "combine_twin_members",
    "expert_data_parallel_degree",
    "expert_data_parallel_refusal",
    "expert_parallel",
    "ranges_extent",
    "replica_ranks",
    "segmented_ranges",
    "slice_ranges",
    "trivial_placement",
    "units_replicate",
    "validate_component_placement",
]
