"""The numerical schedule of the logprobs_sm100 log-sum-exp, stated through the policy (design 7.2).

Nothing here is on a call path: the kernels (``csrc/logprobs.cu``, ``csrc/logprobs_v2.cu``) are the tree's own
and read nothing from this module. It states their schedule as data, and ``tests/test_schedule_policy_cpu.py``
fails if the kernels' constants or the recipe table stop agreeing with it.

The reduction: one row's ``log sum_v exp(x[v])`` over the vocabulary axis. The vocabulary is the axis TP shards:
a rank of a world of ``w`` holds ``V / w`` contiguous columns, and ``w`` must divide 8 (``ops.py::supports``),
so ``d_max = 8`` and the 8 "leaves" of the kernels are the 8 shards at ``d_max`` (a rank at ``w < 8`` computes
its ``8 / w`` leaves separately). The reducers publish no cost parameter, so rule 3 of
:mod:`isoexec.ops.schedule_policy` gives ONE segment per shard: ``n = V / 8``, ``G = 1``, ``S = V / 8``, and the
shards combine by an adjacent tree. Everything below the segment is the reducer's own fixed DAG, which depends
only on a term's position inside the segment:

    strip        8 consecutive columns (one 16-byte load)
    lanes        256, whatever the segment length; a lane l >= S / 8 has no strip
    membership   lane l owns strips l, l + 256, l + 512, ... : columns 8 * (l + 256 i) + j, j = 0..7
    lane order   strip ascending, then column ascending; fp32
    v2 (two passes)   M = the segment's exact max (fmaxf; NaN never wins); F = M, or +0 when M == -inf;
                      lane chain from +0:  s = add_rn(s, expf(sub_rn(v, F)))
    v1 (online)       per lane (m, s) from (-inf, +0):  v > m: s = add_rn(mul_rn(s, expf(sub_rn(m, v))), 1); m = v
                                                        else if v != -inf: s = add_rn(s, expf(sub_rn(v, m)))
                      M = max over the lanes' m; lane value = mul_rn(s, expf(sub_rn(m, M))), +0 for a lane with
                      m == -inf and s == +0
    combine      one chain over the lanes, ascending from lane 0:  S = add_rn(S, lane[l]);  partial = (M, S)

    shard merge  (``merge_leaves``) in fp64: M = max over the 8 shards; w_g = double(S_g) * exp(double(M_g) -
                 double(M)) (+0 for an absent shard); adjacent tree (0,1) (2,3) | (01,23) (45,67) | ..., lower
                 shard on the left; logS = log(w), L32 = RN32(logS); logp = sub_rn(sub_rn(v, M), L32).

The entropy partial ``A`` (``leaf_state_ent``) runs on the same lanes with the same two chains; it is not
transcribed in ``reference.py``.

Which reducer a deployment runs is a BINDING, not a schedule: :data:`BINDING` lists the base tree's choice per
(logits dtype, vocabulary) class (``spec.py::DEFAULT_RECIPE_BY_CLASS``) and is not changed here.
"""

from __future__ import annotations

from dataclasses import dataclass

from isoexec.ops import schedule_policy as policy

#: d_max of the vocabulary axis: the largest TP world the kernels' contract allows; a kernel "leaf" is one shard.
VOCAB_SHARDS = 8
#: Columns per strip.
STRIP = 8
#: Lanes per segment.
LANES = 256


@dataclass(frozen=True)
class LaneReducer:
    """A log-sum-exp partial ``(M, S)`` over one segment on interleaved lanes. The fields are the whole DAG."""

    version: int             # the tree's recipe version
    impl_id: str             # the declared impl (spec.py)
    source: str              # the kernel source under csrc/
    normalizer: str          # "segment_max": exp(v - F), F the segment's max; "online": running max per lane
    reference: str           # the executable statement of this DAG (torch on the device, fp32, the stated order)
    strip: int = STRIP
    accumulator: str = "fp32"
    lane_start: str = "+0"
    lane_order: str = "strip ascending, then column ascending"
    combine: str = "chain over lanes, ascending from lane 0: S = add_rn(S, lane[l])"
    shard_merge: str = "fp64: w_g = S_g * exp(M_g - M); adjacent tree, lower shard on the left"

    def lanes(self, s: int) -> int:
        """Lanes as the function of the segment length that the kernel uses: a constant."""
        if s <= 0 or s % self.strip:
            raise ValueError(f"logprobs_sm100: a segment of {s} is not whole strips of {self.strip}")
        return LANES

    def members(self, s: int, lane: int) -> tuple[int, ...]:
        """The positions of ``lane`` inside the segment, in the order its chain adds them (may be empty)."""
        strips = s // self.strip
        return tuple(self.strip * q + j for q in range(lane, strips, self.lanes(s)) for j in range(self.strip))


V1 = LaneReducer(version=1, impl_id="sm100_online_leaf8_v1", source="logprobs.cu", normalizer="online",
                 reference="isoexec.ops.logprobs.logprobs_sm100.reference.segment_partial_v1")
V2 = LaneReducer(version=2, impl_id="sm100_twopass_leaf8_v2", source="logprobs_v2.cu", normalizer="segment_max",
                 reference="isoexec.ops.logprobs.logprobs_sm100.reference.segment_partial_v2")

REDUCERS = {1: V1, 2: V2}

#: The binding (``spec.py::DEFAULT_RECIPE_BY_CLASS``): (logits dtype, vocabulary) -> reducer version. A class
#: that is not listed runs :data:`DEFAULT_VERSION` (``ops.py::default_recipe``).
BINDING: dict[tuple[str, int], int] = {
    ("bf16", 248320): 2,
    ("bf16", 154880): 2,
    ("bf16", 151680): 2,
    ("fp32", 248320): 1,
    ("fp32", 154880): 1,
    ("fp32", 151680): 1,
}
DEFAULT_VERSION = 1


def reducer(dtype: str, vocab: int) -> LaneReducer:
    """The reducer bound to a deployment class (``dtype`` is ``"bf16"`` or ``"fp32"``)."""
    return REDUCERS[BINDING.get((dtype, int(vocab)), DEFAULT_VERSION)]


def supported(vocab: int) -> bool:
    """Vocabularies the kernels take: whole strips in every shard (``ops.py::supports``)."""
    return vocab > 0 and vocab % (VOCAB_SHARDS * STRIP) == 0


def segment(vocab: int) -> policy.Segments:
    """Rule 3 with no cost parameter: one segment per shard of the vocabulary axis at ``d_max``."""
    if not supported(vocab):
        raise ValueError(f"logprobs_sm100: vocabulary {vocab} is not a multiple of {VOCAB_SHARDS * STRIP}")
    return policy.reduce(vocab // VOCAB_SHARDS)


def shard_tree(shards: int = VOCAB_SHARDS) -> tuple[tuple[tuple[int, int], ...], ...]:
    """The adjacent tree over the shard partials, level by level, as (left, right) slots; the left slot keeps
    the result."""
    levels, step = [], 1
    while step < shards:
        levels.append(tuple((g, g + step) for g in range(0, shards - step, 2 * step)))
        step *= 2
    return tuple(levels)


__all__ = ["BINDING", "DEFAULT_VERSION", "LANES", "LaneReducer", "REDUCERS", "STRIP", "V1", "V2", "VOCAB_SHARDS",
           "reducer", "segment", "shard_tree", "supported"]
