"""The published K-segment table of the sm100 mm recipe, and the execution plans that serve it."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

#: Bumped when the policy changes what S any weight gets. It is part of the recipe identity.
#: v4: S is a function of K (and the dtype) ALONE. v1-v3 read the weight's N as well, and N is the one
#: dimension a deployment changes without changing the weight: a column-parallel shard holds N / tp of
#: the same rows, and a packed projection (q | k | v | gate, gate | up) concatenates several weights
#: along N. So the same output element was reduced over K in a different order at TP 1 and TP 4, or
#: packed and unpacked -- different bits from the same logical weight. Under v4 every output element
#: of every weight with the same K is the same program, whoever holds which columns. v3's per-(K, N)
#: MEASURED splits are withdrawn with the N term: a (K, N) key cannot tell a logical weight from another
#: weight's shard that happens to have the same width.
POLICY_VERSION = 4

#: Streaming multiprocessors of the deployment target (B200). A **deployment-fixed** axis: the
#: planner may read it, and a live device that disagrees is refused rather than served.
TARGET_SM_COUNT = 148

#: Segment boundaries are multiples of this many K elements. Every mainloop K block of every
#: configuration in this package divides it (64, 128 and 256), so a segment is always a whole
#: number of K blocks and the ascending 16-wide UMMA chain inside a segment is the same chain
#: whichever configuration runs it. That is what makes the tile shape free.
SEGMENT_ALIGN = 256

#: The splits the published table may contain. v4 publishes one: S = 1, the whole K axis as one
#: ascending tcgen05 chain per output element. That is what the v2 rule already gave every weight
#: wide enough to fill the machine at decode, and the narrow weights it split lose nothing by it:
#: production evaluates a split tree whole on chip, where the cluster count is the unsplit launch's,
#: so S = 2 was neither a gain nor a loss at decode (``N=1152``: 0.625 / 0.623 us at ``M=1`` for
#: S = 1 / 2) and at best free at prefill, where every doubling of S costs tensor-memory columns.
#: The leaf still compiles 2, 4 and 8 segments and the combine kernel 8 subtree roots, and the battery
#: gates every one of them: they are the EXECUTION splits an explicit caller (``plans_at_split``,
#: pik-style leaf trees) asks for, not something this table hands a weight.
LEGAL_SPLITS = (1,)

#: Output columns one CTA pair owns in the decode tile shape. The planner counts a decode launch's
#: output tiles with it, and :func:`supported` refuses a weight narrower than one.
DECODE_TILE_N = 64

_TABLE_PATH = Path(__file__).resolve().parent / "segments.json"


def _dtype_name(dtype) -> str:
    import torch

    if dtype is torch.bfloat16:
        return "bf16"
    raise ValueError(f"the sm100 mm recipe is bf16 only, not {dtype}")


def decode_clusters(N: int) -> int:
    """Output tiles a decode-shaped launch has: one M cluster, ``ceil(N / 64)`` N tiles."""
    return (N + DECODE_TILE_N - 1) // DECODE_TILE_N


def segment_count_by_policy(K: int, dtype_name: str = "bf16") -> int:
    """S for a reduction of length ``K``, from the rule that generated ``segments.json``.

    It takes no N on purpose. Tensor parallelism shards a column-parallel weight along N and packing
    concatenates weights along N, and neither may move a bit; K is the contraction itself, the same on
    every rank and in every pack. (A row-parallel layer shards K, and its cross-rank sum is pik's fixed
    leaf tree over the FULL K -- ``collectives/tp_tree.py`` -- not this kernel's segments.)
    """
    if dtype_name != "bf16":
        raise ValueError(f"the sm100 mm recipe is bf16 only, not {dtype_name}")
    return LEGAL_SPLITS[0]


@dataclass(frozen=True)
class SegmentEntry:
    K: int
    dtype: str
    split: int

    @property
    def segment_k(self) -> int:
        return self.K // self.split

    def as_row(self) -> list:
        return [self.K, self.dtype, self.split]


@lru_cache(None)
def published_table() -> dict:
    """``segments.json`` as loaded, or an empty table when the file is absent."""
    if not _TABLE_PATH.is_file():
        return {"policy_version": POLICY_VERSION, "rows": []}
    return json.loads(_TABLE_PATH.read_text())


def table_digest(table: dict | None = None) -> str:
    """Digest over the policy version and every published row, in canonical order."""
    t = published_table() if table is None else table
    rows = sorted(tuple(r) for r in t.get("rows", ()))
    payload = json.dumps(
        {"policy_version": t.get("policy_version"), "rows": [list(r) for r in rows]},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def segment_count(K: int, dtype) -> int:
    """S for a reduction of length ``K``: the published row if there is one, else the policy.

    The weight's N is not an argument: every output column of every weight with this K is
    segmented the same way, which is what makes a column shard or a packed projection the same
    program as the whole weight (:func:`segment_count_by_policy`).
    """
    name = dtype if isinstance(dtype, str) else _dtype_name(dtype)
    for k, d, s in published_table().get("rows", ()):
        if k == K and d == name:
            return int(s)
    return segment_count_by_policy(K, name)


def segment_k(K: int, dtype) -> int:
    """The segment size in K elements. ``K`` when the reduction is unsplit."""
    return K // segment_count(K, dtype)


#: The finest segment boundary an EXECUTION plan may use: the UMMA K. A segment is ``segment_k / 16``
#: ascending ``tcgen05.mma`` instructions from zero, and the leaf serves any multiple of 16 -- whole
#: mainloop K blocks and at most one TAIL block of which the issuer multiplies only the UMMAs inside
#: the segment (README section 4.17). This is what a caller that chooses its own split (pik's
#: row-parallel leaf trees: ``leaf_k = 1376``, ``64``) needs. It is NOT the published table's
#: alignment: :data:`SEGMENT_ALIGN` is policy and does not move, so :func:`supported` still declines a
#: weight whose K the POLICY cannot express.
EXECUTION_ALIGN = 16


def execution_supported(K: int, N: int, segment_k: int) -> bool:
    """Whether the leaf can run an EXECUTION plan with this segment size at all (no table involved)."""
    return (segment_k > 0 and segment_k % EXECUTION_ALIGN == 0 and K % segment_k == 0
            and N >= DECODE_TILE_N and N % 2 == 0)


#: Output columns per 16 bytes of bf16: a TMA map's row pitch must be a multiple of 16 bytes, and the
#: output is stored through one with pitch N. ``ops/mm/sm100.mm_linear`` serves a narrower multiple by
#: computing on zero rows past N, which no real column reads.
PITCH_ALIGN_N = 8


def supported(K: int, N: int) -> bool:
    """Whether this package's leaf can serve the shape at all."""
    return K % SEGMENT_ALIGN == 0 and N >= DECODE_TILE_N and N % PITCH_ALIGN_N == 0


# --------------------------------------------------------------------------- execution plans
# Everything below is ownership and scheduling. It may read M, the live machine, anything.
# Everything below is ownership and scheduling. It may read M, the live machine, anything.

#: How the remaining levels of the tree are evaluated when a CTA does not own the whole one.
#: Pure execution: all three produce the same bits, and the battery gates that.
ONCHIP = 0     #: this CTA owns every segment -- the subtree is in tensor memory, no workspace
COMBINE = 1    #: publish the FP32 subtree root, a second kernel folds the remaining levels
FOLD = 2       #: publish the FP32 root and fold in kernel: the last CTA to arrive at an output
               #: tile evaluates the remaining levels in SEGMENT INDEX order. No second launch.

#: Compiled configurations, keyed ``(family, a_rows, segments-per-CTA, publish mode)``.
#: ``a_rows`` is 0 for a configuration that fetches its tile's full A height (so it serves any M)
#: and 1, 2, 4, 8, 16 or 32 for one that fetches only that many rows of A per stage, which serves M
#: up to that height. Mirrors the table in csrc/mm_sm100_gemm.cu; ops.py checks every name exists
#: in the compiled module, so the two cannot drift.
_CONFIGS = {
    ("narrow", 0, 1, ONCHIP): "narrow_s1",
    ("narrow", 0, 1, COMBINE): "narrow_s1_part",
    ("narrow", 0, 2, ONCHIP): "narrow_s2",
    ("narrow", 0, 2, COMBINE): "narrow_s2_part",
    ("narrow", 0, 4, ONCHIP): "narrow_s4",
    ("narrow", 0, 4, COMBINE): "narrow_s4_part",
    ("narrow", 0, 8, ONCHIP): "narrow_s8",
    ("narrow", 32, 1, ONCHIP): "narrow_a32_s1",
    ("narrow", 32, 1, COMBINE): "narrow_a32_s1_part",
    ("narrow", 32, 2, ONCHIP): "narrow_a32_s2",
    ("narrow", 32, 2, COMBINE): "narrow_a32_s2_part",
    ("narrow", 32, 4, ONCHIP): "narrow_a32_s4",
    ("narrow", 32, 8, ONCHIP): "narrow_a32_s8",
    ("narrow", 16, 1, ONCHIP): "narrow_a16_s1",
    ("narrow", 16, 1, COMBINE): "narrow_a16_s1_part",
    ("narrow", 16, 2, ONCHIP): "narrow_a16_s2",
    ("narrow", 4, 1, ONCHIP): "narrow_a4_s1",
    ("narrow", 2, 1, ONCHIP): "narrow_a2_s1",
    ("narrow", 1, 1, ONCHIP): "narrow_a1_s1",
    ("narrow", 1, 2, ONCHIP): "narrow_a1_s2",
    ("narrow", 8, 1, ONCHIP): "narrow_a8_s1",
    ("narrow", 8, 1, COMBINE): "narrow_a8_s1_part",
    ("narrow", 8, 2, ONCHIP): "narrow_a8_s2",
    ("narrow", 8, 4, ONCHIP): "narrow_a8_s4",
    ("wide", 0, 1, ONCHIP): "wide_s1",
    ("wide", 0, 1, COMBINE): "wide_s1_part",
    ("wide", 0, 2, ONCHIP): "wide_s2",
    ("wide", 0, 2, COMBINE): "wide_s2_part",
    ("wide", 0, 4, ONCHIP): "wide_s4",
    ("wide", 16, 1, ONCHIP): "wide_a16_s1",
    ("wide", 16, 1, COMBINE): "wide_a16_s1_part",
    ("wide", 16, 2, ONCHIP): "wide_a16_s2",
    ("wide", 32, 1, ONCHIP): "wide_a32_s1",
    ("wide", 32, 1, COMBINE): "wide_a32_s1_part",
    ("wide", 32, 2, ONCHIP): "wide_a32_s2",
    ("wide", 32, 2, COMBINE): "wide_a32_s2_part",
    ("vwide", 0, 1, ONCHIP): "vwide_s1",
    ("vwide", 0, 1, COMBINE): "vwide_s1_part",
    ("vwide", 0, 2, ONCHIP): "vwide_s2",
    ("vwide", 16, 1, ONCHIP): "vwide_a16_s1",
    ("vwide", 16, 1, COMBINE): "vwide_a16_s1_part",
    ("vwide", 16, 2, ONCHIP): "vwide_a16_s2",
    ("vwide", 32, 1, ONCHIP): "vwide_a32_s1",
    ("vwide", 32, 1, COMBINE): "vwide_a32_s1_part",
    ("vwide", 32, 2, ONCHIP): "vwide_a32_s2",
    ("vwide", 32, 2, COMBINE): "vwide_a32_s2_part",
    ("vwide", 8, 1, ONCHIP): "vwide_a8_s1",
    ("vwide", 8, 1, COMBINE): "vwide_a8_s1_part",
    ("vwide", 8, 2, ONCHIP): "vwide_a8_s2",
    ("vwide", 1, 1, ONCHIP): "vwide_a1_s1",
    # The cta_group::1 family: one SM owns one 128x128 output tile.
    ("solo", 0, 1, ONCHIP): "solo_s1",
    ("solo", 0, 1, COMBINE): "solo_s1_part",
    ("solo", 0, 2, ONCHIP): "solo_s2",
    ("solo", 32, 1, ONCHIP): "solo_a32_s1",
    ("solo", 32, 2, ONCHIP): "solo_a32_s2",
    ("solo", 16, 1, ONCHIP): "solo_a16_s1",
    ("solo", 16, 2, ONCHIP): "solo_a16_s2",
    ("solo", 8, 1, ONCHIP): "solo_a8_s1",
    ("solo", 8, 2, ONCHIP): "solo_a8_s2",
    ("solo", 1, 1, ONCHIP): "solo_a1_s1",
    ("solo", 1, 2, ONCHIP): "solo_a1_s2",
    # The aliased-A families (README section 4.9): the same tiles with the A descriptor's
    # stride_byte_offset set to zero, so the operand's eight 8-row groups alias the single 8-row
    # group the stage holds. Exact only while M <= 8, which is why they are their own families and
    # not another A height. They are compiled and bit-gated by the plan relation; whether the
    # decode rule selects them is a measurement, not a given.
    ("narrow_al", 8, 1, ONCHIP): "narrow_al_s1",
    ("narrow_al", 8, 2, ONCHIP): "narrow_al_s2",
    ("narrow_al8", 8, 1, ONCHIP): "narrow_al8_s1",
    ("narrow_al8", 8, 2, ONCHIP): "narrow_al8_s2",
    ("narrow_al10", 8, 1, ONCHIP): "narrow_al10_s1",
    ("solo_al", 8, 1, ONCHIP): "solo_al_s1",
    ("solo_al11", 8, 1, ONCHIP): "solo_al11_s1",
    ("vwide_al", 8, 1, ONCHIP): "vwide_al_s1",
    # `solo` with a 128-deep K block: the burst-length probe for the HBM-bound M=32 N=14336 row.
    ("solo_k2", 0, 1, ONCHIP): "solo_k2_s1",
    ("solo_k2", 32, 1, ONCHIP): "solo_k2_a32_s1",
    # THE QUANTISATION FAMILIES (README section 4.13). The RL matrix showed that at
    # M in {256..16384} on the example models' real weight shapes the binding constraint is how a
    # tile quantises (M, N) onto 74 cluster slots, and that the four original families leave
    # 40-54 % of the machine idle on a weight whose N is not a multiple of 128 times a wave count.
    # Everything here is a NEW TILE SHAPE the existing epilogue already expresses -- BN a multiple
    # of 64 (one 128-byte store subtile) and, at BM 64, BN exactly 64 (the interleaved
    # tensor-memory atom). Nothing here is numerical: every one runs the published segment table as
    # a whole tree and the plan relation compares its bits against every other family's.
    ("p64", 0, 1, ONCHIP): "p64_s1",           # 256x64 pair
    ("p64", 0, 2, ONCHIP): "p64_s2",
    ("p192", 0, 1, ONCHIP): "p192_s1",         # 256x192 pair -- three store subtiles
    ("p192", 0, 2, ONCHIP): "p192_s2",
    ("p192_st4", 0, 1, ONCHIP): "p192_st4_s1",
    ("tall", 0, 1, ONCHIP): "tall_s1",         # 512x128 pair -- the first kNumMWaves = 2 tile
    ("tall", 0, 2, ONCHIP): "tall_s2",
    ("s256", 0, 1, ONCHIP): "s256_s1",         # 128x256 single CTA
    ("s256", 0, 2, ONCHIP): "s256_s2",
    ("wide_k2", 0, 1, ONCHIP): "wide_k2_s1",   # 128-deep K block
    ("wide_k2", 0, 2, ONCHIP): "wide_k2_s2",
    ("wide_st4", 0, 1, ONCHIP): "wide_st4_s1",  # four stages, for the short-K rows
    ("wide_st4", 0, 2, ONCHIP): "wide_st4_s2",
    # The raster-order axis: the same tile at a narrower cluster-M group.
    ("wide_g2", 0, 1, ONCHIP): "wide_g2_s1",
    ("wide_g4", 0, 1, ONCHIP): "wide_g4_s1",
    ("vwide_g2", 0, 1, ONCHIP): "vwide_g2_s1",
    ("vwide_g4", 0, 1, ONCHIP): "vwide_g4_s1",
    # The cluster-shape axis (README section 4.16): two CTA pairs sharing one operand by TMA
    # multicast -- `wide_mc1` a 2x2 cluster multicasting A, `wide_mc2` a 4x1 one multicasting B.
    # Removed in round 4 for producing different bits (gap 13), fixed in round 5 and back under the
    # plan relation. Never chosen by the plan rule: they are not members of `_TILE_GEOM`.
    # Publishing twins (README section 4.17): PART 1 at spc == S for the families the plan rule and
    # pik use, so an fp32 root exists wherever a whole tree does.
    ("p64", 0, 1, COMBINE): "p64_s1_part",
    ("p64", 0, 2, COMBINE): "p64_s2_part",
    ("p192", 0, 1, COMBINE): "p192_s1_part",
    ("p192", 0, 2, COMBINE): "p192_s2_part",
    ("vwide", 0, 2, COMBINE): "vwide_s2_part",
    ("wide", 0, 4, COMBINE): "wide_s4_part",
    ("narrow", 0, 8, COMBINE): "narrow_s8_part",
    ("solo", 0, 2, COMBINE): "solo_s2_part",
    # Tile generality (README section 4.18): `pN` the 256xN pair, `qN` the 128xN pair (BM 64, the
    # interleaved 2x2 accumulator), selected by coverage over the RL rows.
    ("p144", 0, 1, ONCHIP): "p144_s1",
    ("p144", 0, 2, ONCHIP): "p144_s2",
    ("p176", 0, 1, ONCHIP): "p176_s1",
    ("p176", 0, 2, ONCHIP): "p176_s2",
    ("p208", 0, 1, ONCHIP): "p208_s1",
    ("p224", 0, 1, ONCHIP): "p224_s1",
    ("p240", 0, 1, ONCHIP): "p240_s1",
    ("p240", 0, 2, ONCHIP): "p240_s2",
    ("q96", 0, 1, ONCHIP): "q96_s1",
    ("q96", 0, 2, ONCHIP): "q96_s2",
    ("q192", 0, 1, ONCHIP): "q192_s1",
    ("q192", 0, 2, ONCHIP): "q192_s2",
    # Round 6 (README section 4.25): cuBLAS's own (a1) pair tiles -- 64 tokens x BN per CTA, the
    # 2x2 atom with an 8-column store step -- and the 256x112 pair. Whole tree at S = 1 only.
    ("q112", 0, 1, ONCHIP): "q112_s1",
    ("q144", 0, 1, ONCHIP): "q144_s1",
    ("q80", 0, 1, ONCHIP): "q80_s1",
    ("p112", 0, 1, ONCHIP): "p112_s1",
    # Round 7 (README section 4.30, gap 25): deep-pipeline twins, substituted by `plan_for` at long K.
    ("q96d", 0, 1, ONCHIP): "q96d_s1",
    ("q192d", 0, 1, ONCHIP): "q192d_s1",
    ("wide_mc1", 0, 1, ONCHIP): "wide_mc1_s1",
    ("wide_mc2", 0, 1, ONCHIP): "wide_mc2_s1",
    ("vwide_mc1", 0, 1, ONCHIP): "vwide_mc1_s1",
    ("narrow", 0, 1, FOLD): "narrow_s1_fold",
    ("narrow", 0, 2, FOLD): "narrow_s2_fold",
    ("narrow", 0, 4, FOLD): "narrow_s4_fold",
    ("narrow", 32, 1, FOLD): "narrow_a32_s1_fold",
    ("narrow", 32, 2, FOLD): "narrow_a32_s2_fold",
    ("narrow", 16, 1, FOLD): "narrow_a16_s1_fold",
    ("narrow", 8, 1, FOLD): "narrow_a8_s1_fold",
    ("narrow", 8, 2, FOLD): "narrow_a8_s2_fold",
    ("wide", 0, 1, FOLD): "wide_s1_fold",
    ("wide", 0, 2, FOLD): "wide_s2_fold",
    ("wide", 32, 1, FOLD): "wide_a32_s1_fold",
    ("vwide", 0, 1, FOLD): "vwide_s1_fold",
    ("vwide", 32, 1, FOLD): "vwide_a32_s1_fold",
    ("vwide", 8, 1, FOLD): "vwide_a8_s1_fold",
    ("solo", 0, 1, FOLD): "solo_s1_fold",
    ("solo", 8, 1, FOLD): "solo_a8_s1_fold",
    # Measurement controls: the same kernel without the phantom-A skip of README section 2.8.
    # They are never planned (``_NO_SKIP`` keeps them out of the plan sweep) and exist so the skip
    # can be priced against itself.
}

#: Configurations that exist only to be measured against their twin, never to be planned.
_MEASUREMENT_ONLY = frozenset({"narrow_a8_s1_noskip", "vwide_a8_s1_noskip"})

#: Segments a CTA of each family may own on chip. Bounded by tensor memory: one FP32 accumulator
#: per segment, 512 columns in total, so a wider tile owns fewer segments.
_MAX_SPC = {"narrow": 8, "wide": 4, "vwide": 2, "solo": 2,
            "narrow_al": 2, "narrow_al8": 2, "narrow_al10": 1,
            "solo_al": 1, "solo_al11": 1, "vwide_al": 1, "solo_k2": 1,
            # The quantisation families. Bounded by tensor memory exactly as the others are:
            # kAccPerStage = spc * kNumMWaves * kTmemN and two accumulator stages need
            # 2*kAccPerStage <= 512, so BN 192 admits spc 1 with overlap and spc 2 without, and
            # BM 256 (kNumMWaves = 2) admits spc 1 at BN 128.
            "p64": 2, "p192": 2, "p192_st4": 1, "tall": 2, "s256": 2,
            "wide_k2": 2, "wide_st4": 2,
            "wide_g2": 1, "wide_g4": 1, "vwide_g2": 1, "vwide_g4": 1,
            "wide_mc1": 1, "wide_mc2": 1, "vwide_mc1": 1,
            "p144": 2, "p176": 2, "p208": 1, "p224": 1, "p240": 2,
            "q96": 2, "q192": 2,
            # Round 6 (section 4.25): compiled at S = 1 only.
            "q112": 1, "q144": 1, "q80": 1, "p112": 1,
            "q96d": 1, "q192d": 1}

#: Above this M a call is planned like prefill. Pure scheduling.
DECODE_M_MAX = 128

#: Compiled A-fetch heights. A call takes the shortest one that still covers its M, because a TMA
#: box taller than M pays for out-of-bounds zero-fill the real rows never use -- the reason an
#: M=1 call was measured SLOWER than an M=32 one on the identical configuration before this axis
#: existed. It changes no real row's arithmetic: rows below the height are fetched exactly as
#: before and rows above it are past M.
#:
#: 16 is here because MEASURED (README section 4.4): the out-of-bounds fill is charged in full
#: while the box has fewer than 16 in-bounds rows and is free at 16 or more, so at 9 <= M <= 16 the
#: 16-row box is the smallest one that covers M and it beats the 32-row box by 10 % on
#: N=4096 K=4096 (8.46 us against 9.38 at M=12). 8 stays the SELECTED floor -- not because a
#: shorter box is illegal, which it is not, but because MEASURED it is worth nothing (see below).
#: 1, 2 and 4 are COMPILED (the 8-row floor was a conflation of two different 8s: the 128-byte
#: swizzle atom is 8 rows for the UMMA *operand*, because the shared descriptor's
#: stride_byte_offset is the stride between consecutive 8-row groups of the tile's kBlockM rows,
#: while the A TMA box's outer extent is not constrained by it at all -- SWIZZLE_128B constrains
#: only the 128-byte inner extent) and are deliberately NOT SELECTED: MEASURED (README section
#: 4.8), a 1-row box is indistinguishable from an 8-row box at M=1 on all six shapes probed
#: (N=4096 K=4096: 8.48 against 8.53; N=1152: 6.384 against 6.415; N=9216: 8.77 against 8.96;
#: N=14336: 27.87 against 27.66; N=12352: 11.36 against 11.14; N=5120 K=14336: 41.38 against
#: 41.42), so the A box's cost is not proportional to its volume below 16 rows and there is no
#: gain to justify moving the selection. They stay in ``_CONFIGS`` so the plan relation gates
#: their bits.
_A_ROWS = (8, 16)

#: In-bounds rows at which the out-of-bounds fill stops being charged. MEASURED (README section
#: 4.4): below this the whole TMA box is charged against a ~9 TB/s aggregate fill bandwidth and
#: above it none of it is, so a short box is worth taking only while M is below this. It is a
#: SCHEDULING constant -- it selects which compiled A height runs and nothing numerical reads it.
FILL_FREE_ROWS = 16


def _a_rows_for(M: int) -> int:
    """Shortest compiled A fetch that covers M; 0 means the tile's full height, i.e. any M."""
    if M >= FILL_FREE_ROWS:
        return 0
    for rows in _A_ROWS:
        if M <= rows:
            return rows
    return 0


#: Highest M an aliased-A family serves. The A descriptor's stride_byte_offset is zero there, so
#: the operand's eight 8-row groups all read the one 8-row group the stage holds and instruction
#: row r reads shared row r % 8 -- exact only while every real row has r < 8.
ALIAS_M_MAX = 8

#: The aliased-A member of each decode family, and the stage depth the freed shared memory bought.
#: MEASURED at M = 1 and M = 8 (README section 4.9), against the same tile's 8-row streaming box,
#: bitwise identical on 400 comparisons: N=4096 K=4096 8.69 -> 7.39 us, N=1152 K=2048 6.54 -> 5.17,
#: N=2048 K=2048 6.61 -> 5.22, N=9216 K=2048 8.98 -> 7.65, N=5120 K=14336 41.57 -> 30.91,
#: N=14336 K=5120 27.44 -> 25.55, N=24576 K=4096 38.42 -> 34.48. Two separate effects: the
#: phantom-row zeroing disappears (worth ~1.2 us on every narrow decode row, and it is what the
#: one-cluster control isolates) and the A stage shrinks 8x, which pays for 8 stages instead of 4
#: on `narrow` and 11 instead of 6 on `solo` -- worth 10 us alone on N=5120 K=14336, the row that
#: has the longest K in the census and was the package's worst.
_ALIAS_FAMILY = {"narrow": "narrow_al8", "solo": "solo_al11"}


def _a_rows_ladder(rows: int) -> tuple[int, ...]:
    """Compiled A heights to try, shortest first, ending at the tile's full height (0)."""
    if not rows:
        return (0,)
    return tuple(r for r in _A_ROWS if r >= rows) + (0,)


@dataclass(frozen=True)
class Plan:
    """One execution plan: which configuration, which segment size, how the tree is cut."""

    config: str
    segment_k: int
    split: int
    spc: int
    #: Where the levels above this CTA's subtree are evaluated: :data:`ONCHIP` (there are none),
    #: :data:`COMBINE` (a second kernel) or :data:`FOLD` (the last CTA to arrive at the output
    #: tile, in kernel). Pure execution -- the tree, its order and the single rounding are the
    #: same in all three, and the battery gates the bits across them.
    mode: int = ONCHIP

    @property
    def groups(self) -> int:
        return self.split // self.spc

    @property
    def publishes_partials(self) -> bool:
        return self.groups > 1

    @property
    def folds_in_kernel(self) -> bool:
        return self.mode == FOLD

    def workspace_floats(self, M: int, N: int) -> int:
        return self.groups * M * N if self.publishes_partials else 0


#: N at which the "wide-N" member of each regime takes over: above it, the N axis alone already
#: fills the machine's cluster slots, so the extra parallelism a narrower tile buys is worth less
#: than the per-UMMA issue overhead it costs. ``74 * 128`` cluster slots' worth of 128-wide tiles;
#: equivalently ``decode_clusters(N) > TARGET_SM_COUNT``, since the decode tile is 64 wide.
#: MEASURED: the boundary separates all thirteen census widths probed at M = 1 and M = 32 with no
#: exception (README section 4.5).
WIDE_N = (TARGET_SM_COUNT // 2) * 128


def _decode_family(N: int, split: int) -> str:
    """Tile family at or below :data:`DECODE_M_MAX`."""
    if decode_clusters(N) > TARGET_SM_COUNT and _MAX_SPC["solo"] >= split:
        return "solo"
    return "narrow"


#: Tile geometry of every family the PREFILL rule may choose, as ``(BM, BN, CTAs per tile)``.
#: This is the *whole* input to :func:`tile_cost`; nothing else about a family enters the rule.
#: The order is the tie-break, smallest instruction first, so the rule is a total function.
_TILE_GEOM = {
    "narrow": (64, 64, 2),      # 128x64 pair
    "p64": (128, 64, 2),        # 256x64 pair
    "q96": (64, 96, 2),         # 128x96 pair  (round 5)
    "wide": (128, 128, 2),      # 256x128 pair
    "p144": (128, 144, 2),      # 256x144 pair (round 5: cuBLAS's own tile on N = 2560 / 5120)
    "p176": (128, 176, 2),      # 256x176 pair (round 5)
    "q192": (64, 192, 2),       # 128x192 pair (round 5)
    "p192": (128, 192, 2),      # 256x192 pair
    "p208": (128, 208, 2),      # 256x208 pair (round 5)
    "p224": (128, 224, 2),      # 256x224 pair (round 5)
    "p240": (128, 240, 2),      # 256x240 pair (round 5)
    "vwide": (128, 256, 2),     # 256x256 pair
    # Round 6 (section 4.25), appended so the tie-break order of the twelve is unchanged. Offered only
    # while M <= ROUND6_FAMILY_M_MAX and K >= ROUND6_FAMILY_MIN_K (see `_prefill_family_by_cost`).
    "q112": (64, 112, 2),       # 128x112 pair: nvjet 112x64_64x14_1x2_2cta (M=256 N=4096)
    "q144": (64, 144, 2),       # 128x144 pair: nvjet 144x64_64x12_1x2_2cta (M=256 N=5120)
    "q80": (64, 80, 2),         # 128x80 pair:  nvjet 80x64_64x16_2x2_2cta (M=256 N=2560)
    "p112": (128, 112, 2),      # 256x112 pair: nvjet 112x128_64x9_1x2_2cta (M=512 N=4096)
}
#: The round-6 families of section 4.25 and the bounds on where the rule may offer them. Pure
#: scheduling, deployment-fixed. MEASURED (section 4.25, the candidates sweep over all 116 measurable
#: M <= 1024 RL rows): adding them takes the rule's within-3 % agreement with the best measured
#: candidate from 81 % (all six round-6 families) / 94 of 112 picks to 105 of 112 (93.8 %), and moves
#: no M > 1024 row's plan (0 of 160) -- the M bound pins that invariant rather than enforcing a measured
#: loss. The K bound IS a measured loss: at K = 768 the model sends M=256/512 N=2560 to q80 / q144
#: (1.198 / 1.224 against the round-5 q96 / q192's 1.160 / 1.178).
ROUND6_FAMILIES = frozenset({"q112", "q144", "q80", "p112"})
ROUND6_FAMILY_M_MAX = 1024
ROUND6_FAMILY_MIN_K = 1024

#: Round 7 (section 4.30, gap 25): the K-aware stage count. A deeper pipeline pays only once the K loop is
#: long enough to fill it: MEASURED (record, cold, matched gate, one process) `q96`/`q192` at 15 / 9 stages
#: against 8 are 0.919-0.983 of the 8-stage time on the eight RL rows with K >= 2048 (median 0.969) and
#: 0.980 / 1.023 / 1.005 at K = 768. Pure scheduling: the twin is the same tile and the same chain.
DEEP_STAGE_MIN_K = 2048
_DEEP_TWIN = {"q96_s1": "q96d_s1", "q192_s1": "q192d_s1"}

#: Round 8: the MODEL TILE TABLE. Measured per audited model weight (measured and measured
#: rollout and train forward, the shape census):
#: ``(N, K) -> ((M_lo, M_hi, configuration), ...)``. Where the family rule's plan is beaten by more than
#: 1.5 % by another COMPILED whole-tree configuration of the same split (record mode, cold, ``--candidates``,
#: one lease per weight), that configuration serves the band ``(previous measured M, this M]``. Pure
#: scheduling: every entry is an already bit-gated member of ``_CONFIGS`` with ``spc == split`` (the whole
#: tree on chip), and its A-fetch height covers the band, so the bits are those of every other plan (the
#: plan relation gates them). Production calls only (no override) consult it. README section 4.31.
_MODEL_TILES: dict[tuple[int, int], tuple[tuple[int, int, str], ...]] = {
    (4096, 1280): ((9, 16, 'narrow_a32_s1'), (25, 32, 'narrow_a32_s1'), (129, 256, 'q144_s1')),
    (4096, 1536): ((1, 4, 'narrow_al_s1'), (129, 256, 'q144_s1')),
    (8192, 4096): ((9, 96, 'q112_s1'), (97, 128, 'q144_s1')),
    (25600, 6144): ((2, 4, 'narrow_al10_s1'), (9, 128, 'q192d_s1')),
    (32768, 512): ((1, 1, 'narrow_al_s1'), (2, 48, 'q192_s1')),
    (32768, 1280): ((9, 16, 'q192d_s1'), (17, 24, 'solo_a32_s1'), (25, 48, 'q192d_s1'), (1025, 2048, 'vwide_g4_s1')),
    (154880, 4096): ((1, 8, 'narrow_al8_s1'), (9, 128, 'q192d_s1')),
}
_CONFIG_KEYS = None


def _model_tile(M: int, K: int, N: int, split: int):
    """The table's configuration for this call, or None."""
    global _CONFIG_KEYS
    bands = _MODEL_TILES.get((N, K))
    if not bands:
        return None
    if _CONFIG_KEYS is None:
        _CONFIG_KEYS = {v: k for k, v in _CONFIGS.items()}
    for lo, hi, cfg in bands:
        if lo <= M <= hi:
            fam, rows, spc, mode = _CONFIG_KEYS[cfg]
            if (rows and M > rows) or split % spc:
                return None
            if spc == split and mode == ONCHIP:
                return Plan(config=cfg, segment_k=K // split, split=split, spc=spc, mode=ONCHIP)
            if spc < split and mode == COMBINE and split // spc in (2, 4, 8):
                return Plan(config=cfg, segment_k=K // split, split=split, spc=spc, mode=COMBINE)
            return None
    return None
# Round 5 (section 4.18): `q32` (128x32 pair) was in the coverage set and is deleted (step 3): MEASURED
# it wins 0 of 273 RL rows against the best of every compiled candidate and loses 20-28 % to
# `narrow_s2` on the 11 rows the cost model gave it (its per-UMMA overhead is 2/3 of its time).
# `tall` (512x128 pair), `s256` (128x256 single) and `solo` (128x128 single) are compiled and
# bit-gated and are NOT members of this set, each for a measured reason (README section 4.14):
#   * `tall` is 1.4-1.9x SLOWER than `wide` on the 61 RL rows where the two need the same wave
#     count, and it cannot help on quantisation in closed form anyway -- for any tile count t,
#     ceil(2t/74) <= 2*ceil(t/74), so doubling the tile's rows never shortens the makespan at the
#     same per-FLOP cost. It stays compiled because it is the package's only kNumMWaves = 2
#     configuration and therefore its only test of whether two interleaved accumulator chains hide
#     the per-UMMA issue latency of gap 4. They do not.
#   * `s256` and `solo`, by the same closed form on the CTA count (2t CTAs on 148 SMs is t pairs on
#     74 cluster slots), can only match a pair tile's makespan, never beat it, and halving the SMs
#     per tile doubles the per-SM B traffic. Between them they were the best of twenty candidates on
#     2 of 275 RL rows. They remain the DECODE regime's wide-N members (section 4.5), which is a
#     different rule.
# Adding them back costs 49 of the 275 rows' agreement with the measured best: 226/275 without them
# against 215/275 with, and 89 % of rows within 3 % of the twenty-candidate oracle against 85 %.

#: FLOP one SM's tensor core retires per clock on this machine, from a measurement of
#: the sustained rate (1.67 PFLOP/s over 148 SMs at ~1.46 GHz effective).
FLOP_PER_CLK_PER_SM = 7737.0

#: Flat per-instruction overhead of a ``tcgen05.mma``, INDEPENDENT of its size -- a
#: measured constant, and the reason a tile that buys parallelism by shrinking the
#: UMMA loses more in the mainloop than it gains in the fill. It is what makes this a cost model
#: and not a utilisation model: a pure utilisation rule would always pick the smallest tile.
UMMA_OVERHEAD_CLK = 17.0


def tile_cost(family: str, M: int, N: int, K: int) -> float:
    """Modelled makespan of one call under one tile family, in SM clocks."""
    bm, bn, ctas = _TILE_GEOM[family]
    tiles = -(-M // (bm * ctas)) * -(-N // bn)
    waves = -(-tiles // (TARGET_SM_COUNT // ctas))
    per_umma = (bm * ctas * bn * 16 * 2) / (ctas * FLOP_PER_CLK_PER_SM) + UMMA_OVERHEAD_CLK
    return waves * (K / 16.0) * per_umma


def _prefill_family_by_cost(M: int, N: int, split: int, K: int | None = None) -> str:
    """argmin of :func:`tile_cost` over the families this split admits."""
    avail = [f for f in _TILE_GEOM
             if _MAX_SPC.get(f, 0) >= split and (f, 0, split, ONCHIP) in _CONFIGS
             and (f not in ROUND6_FAMILIES
                  or (M <= ROUND6_FAMILY_M_MAX and (K is None or K >= ROUND6_FAMILY_MIN_K)))]
    if not avail:
        return "narrow"
    order = list(_TILE_GEOM)
    return min(avail, key=lambda f: (tile_cost(f, M, N, 1 << 20), order.index(f)))


def _prefill_family(M: int, N: int, split: int, K: int | None = None) -> str:
    """Tile family for M above the decode bound: argmin of :func:`tile_cost`.

    **What this replaced, and why.** Until this round the rule was two thresholds -- the 128x64 pair
    tile while its own cluster count fit two waves of the machine, then the 256x256 tile above
    ``WIDE_N`` and the 256x128 one below it -- fitted on four widths at two M each (section 4.5).
    The RL matrix (section 4.12: M in {256..16384} across the 39 real weight shapes the example
    models present at their deployed TP degrees) showed what that rule costs. Under the metric of
    record it put **45 of 273 rows at or below cuBLAS**, and its median ratio was **1.086**; the
    binding constraint on the bad rows is not the pipeline, the A box or the split but how the tile
    quantises ``(M, N)`` onto 74 cluster slots, and MEASURED the ratio tracks exactly that -- median
    1.030 on the rows whose utilisation is above 90 %, 1.167 on the rows between 60 and 75 %.

    So the rule is now the quantisation itself, as a formula rather than a threshold: the tile whose
    modelled makespan is least. MEASURED against a sweep of every compiled whole-tree configuration
    on all 275 rows (section 4.14), it picks the measured best family on **226 of 275** rows, is
    **within 3 % of the best of twenty candidates on 89 %** of them, and takes the median ratio from
    1.086 to **1.046** and the count at or below cuBLAS from 45 to 71. The twenty-candidate oracle
    itself is 1.038 and 86 rows, so the rule captures most of what is available to choose.

    It reads only ``(M, N)`` and the split, is a total function with a fixed tie-break, and no
    weight has an entry anywhere -- which is the property the ``_TILE_OVERRIDE`` deleted last round
    did not have, and the reason a generated per-``(N, K, M-bucket)`` plan table is not the answer
    even though it would fit these 275 rows better: it could say nothing about a weight nobody
    measured, and the census it would be fitted on is 39 of the weights a deployment can present.
    """
    return _prefill_family_by_cost(M, N, split, K)


def plan_for(M: int, K: int, N: int, dtype, *, family: str | None = None,
             spc: int | None = None, a_rows: int | None = None,
             mode: int | None = None) -> Plan:
    """The execution plan for one call."""
    split = segment_count(K, dtype)
    if family is None and spc is None and a_rows is None and mode is None:
        tabled = _model_tile(M, K, N, split)
        if tabled is not None:
            return tabled
    decode = M <= DECODE_M_MAX
    if family:
        fam = family
    elif decode:
        fam = _decode_family(N, split)
    else:
        fam = _prefill_family(M, N, split, K)
    rows = _a_rows_for(M) if a_rows is None else a_rows
    if rows and M > rows:
        rows = 0
    # Measured: the extra launch and the FP32 round trip of a cut tree cost more than the idle
    # cluster slots it recovers at these sizes, so a plan owns its whole subtree on chip wherever
    # the configuration admits it. The cut plan stays compiled and is gated by the relations.
    want = split if spc is None else spc
    # At M <= ALIAS_M_MAX the aliased-A member of the family is tried first: same tile, same bits
    # (the plan relation gates it), no phantom-row zeroing and a deeper B pipeline. It is not a
    # new A height -- it is a different shared-memory map -- so it is a family of its own, and the
    # base family is the fallback for any (split, mode) it does not compile.
    fams = [fam]
    if family is None and rows and rows <= ALIAS_M_MAX and fam in _ALIAS_FAMILY:
        fams.insert(0, _ALIAS_FAMILY[fam])
    for fam, cand_rows in [(f, r) for f in fams for r in _a_rows_ladder(rows)]:
        for cand_spc in sorted({min(want, split, _MAX_SPC[fam]), 8, 4, 2, 1}, reverse=True):
            if cand_spc < 1 or cand_spc > split or split % cand_spc:
                continue
            if split // cand_spc == 1:
                modes = (ONCHIP,)
            elif mode is not None:
                modes = (mode,)
            else:
                # MEASURED (README section 4.6): the in-kernel fold removes the second launch
                # and wins at M=1 (7.76 us against the combine kernel's 8.34 at N=1152 K=2048),
                # but the fold work for a tile is done by ONE CTA serially where the combine
                # kernel spreads it over a whole launch, so it loses from M=8 up and badly at
                # prefill (193.82 against 62.50 at M=4096). The rule never selects a cut tree at
                # all -- the whole tree on chip beats both -- so this order only decides which cut
                # form an explicit override gets, and it is the measured better one at every M
                # above 1.
                modes = (COMBINE, FOLD)
            for cand_mode in modes:
                key = (fam, cand_rows, cand_spc, cand_mode)
                if key in _CONFIGS:
                    cfg = _CONFIGS[key]
                    if family is None and K >= DEEP_STAGE_MIN_K and cfg in _DEEP_TWIN:
                        cfg = _DEEP_TWIN[cfg]
                    return Plan(config=cfg, segment_k=K // split, split=split,
                                spc=cand_spc, mode=cand_mode)
    raise ValueError(f"no compiled configuration for family={fam} M={M} split={split} "
                     f"mode={mode}")


def legal_plans(M: int, K: int, N: int, dtype) -> list[Plan]:
    """Every compiled plan that serves this shape under the published S."""
    split = segment_count(K, dtype)
    return plans_at_split(M, K, N, split)


def plans_at_split(M: int, K: int, N: int, split: int) -> list[Plan]:
    """Every compiled plan for this shape at an explicit split, ignoring the published table."""
    out: list[Plan] = []
    if split < 1 or K % split or (K // split) % EXECUTION_ALIGN:
        return out
    for (fam, rows, spc, mode), name in _CONFIGS.items():
        if rows and M > rows:
            continue
        if spc > split or split % spc:
            continue
        # The combine kernel, the in-kernel fold and the publish-only workspace are compiled for
        # 1, 2, 4 or 8 subtree roots (an EXECUTION split of 16 still has its spc >= 2 plans).
        if split // spc not in (1, 2, 4, 8):
            continue
        if (mode == ONCHIP) != (split // spc == 1):
            continue
        out.append(Plan(config=name, segment_k=K // split, split=split, spc=spc, mode=mode))
    out.sort(key=lambda p: p.config)
    return out


#: Configuration name -> its key in :data:`_CONFIGS`.
_KEY_OF = {name: key for key, name in _CONFIGS.items()}


def publishing_twin(plan: Plan) -> Plan:
    """The PART 1 configuration of ``plan``'s tile that owns the WHOLE tree and publishes its root."""
    fam, rows, _spc, _mode = _KEY_OF[plan.config]
    if plan.spc != plan.split:
        raise ValueError(
            f"{plan.config} owns {plan.spc} of {plan.split} segments per CTA; an fp32 root needs the "
            f"whole tree on one CTA (spc == split)")
    # The same tile first; then the tile's base family (an aliased-A decode family such as
    # `narrow_al8` has no PART 1 member, and its base family's is the same program -- the plan
    # relation gates every pair); then that family at its full A height, which serves any M.
    base = fam.split("_al")[0]
    # Last, any whole-tree publisher at the full A height: every legal plan has the same bits, so
    # a tile family without a PART 1 member (the round-5 quantisation families) borrows one.
    fallback = [(f, 0, plan.split, COMBINE) for f in ("wide", "vwide", "p192", "p64", "narrow", "solo")]
    for key in ((fam, rows, plan.split, COMBINE), (base, rows, plan.split, COMBINE),
                (base, 0, plan.split, COMBINE), *fallback):
        if key in _CONFIGS:
            return Plan(config=_CONFIGS[key], segment_k=plan.segment_k, split=plan.split,
                        spc=plan.split, mode=COMBINE)
    raise ValueError(f"no publishing twin of {plan.config} is compiled (split {plan.split})")


def root_plans(M: int, K: int, N: int, split: int) -> list[Plan]:
    """Every compiled plan that can publish the fp32 root of the whole ``split``-segment tree."""
    out = []
    if split < 1 or K % split or (K // split) % EXECUTION_ALIGN:
        return out
    for (fam, rows, spc, mode), name in sorted(_CONFIGS.items(), key=lambda kv: kv[1]):
        if mode != COMBINE or spc != split or (rows and M > rows):
            continue
        out.append(Plan(config=name, segment_k=K // split, split=split, spc=spc, mode=COMBINE))
    return out


def root_plan_for(M: int, K: int, N: int, dtype) -> Plan:
    """The production plan when it owns the whole tree per CTA, else a whole-tree fp32-root publisher."""
    p = plan_for(M, K, N, dtype)
    if p.spc == p.split:
        return p
    fam = _KEY_OF[p.config][0]
    roots = root_plans(M, K, N, p.split)
    if not roots:
        raise ValueError(f"no compiled whole-tree publisher at split {p.split} for M={M} K={K} N={N}")
    roots.sort(key=lambda r: (_KEY_OF[r.config][0] != fam, _KEY_OF[r.config][1] != 0, r.config))
    return roots[0]


def write_table(rows: list[SegmentEntry], path: Path | None = None) -> dict:
    """Publish ``segments.json`` from a list of entries. Regenerated, never hand-edited."""
    from .spec import RECIPE_VERSION

    t = {
        "schema": "isoexec_mm_sm100_segments_v2",
        "policy_version": POLICY_VERSION,
        "recipe": RECIPE_VERSION,
        "target_sm_count": TARGET_SM_COUNT,
        "segment_align": SEGMENT_ALIGN,
        "legal_splits": list(LEGAL_SPLITS),
        "keyed_by": "K and dtype; never N, so a column shard or a packed projection is the same program",
        "merge": "balanced adjacent-pair tree over segment index, fp32, one bf16 RNE root",
        "rows": [e.as_row() for e in rows],
    }
    t["digest"] = table_digest(t)
    (path or _TABLE_PATH).write_text(json.dumps(t, indent=2) + "\n")
    published_table.cache_clear()
    return t


# ------------------------------------------------------------------ the published weight set
# Every bf16 dense-GEMM weight the example models present, at every tensor-parallel degree the
# census covers, plus the foundation matrix. The table publishes one row per distinct K of this set
# (a row-parallel shard is a K of its own); the weights stay enumerated so the census tests can say
# which deployed weight each K serves. Sites that are fp32 (the routers), or whose K the segment
# alignment cannot express, or narrower than one store subtile, are absent: the leaf declines them.

#: (K, per-site N at TP=1) from isoexec/models/*/profile.py::gemm_census, bf16 sites only.
CENSUS_SITES = (
    # Qwen 3.5 35B A3B, hidden 2048
    (2048, 9216), (2048, 12352), (2048, 248320), (2048, 1024),
    # Qwen 3.5 9B, hidden 4096
    (4096, 10240), (4096, 12352), (4096, 248320), (4096, 24576),
    # Qwen 3.6 27B, hidden 5120
    (5120, 14336), (5120, 16480), (5120, 248320), (5120, 34816),
)

#: Sites the row-parallel and legacy tables name as (K, N); K is the shard there, so it moves
#: with TP instead of N.
ROW_PARALLEL_SITES = ((512, 2048), (1536, 4096), (2048, 2048), (14336, 5120), (4096, 4096))

#: The foundation matrix of reference GEMM shapes, as (K, N).
FOUNDATION_SITES = ((2048, 1152), (4096, 4096), (5120, 14336))

CENSUS_TPS = (1, 2, 4, 8)

#: Round 5 (README section 4.21, gap 17): the deployed weights of the models ``CENSUS_SITES`` does not
#: name, as ``(K, N, axis, tp_set)`` at TP = 1. ``axis`` says what TP shards: ``"column"`` divides N,
#: ``"row"`` divides K (the shard is K), ``"none"`` is replicated. Every ``tests._helpers.RL_CENSUS``
#: weight is generated by exactly one entry here or by the round-4 lists above; the S of each new
#: row is what the SAME policy computes (``POLICY_VERSION`` 2), so no existing row moves.
AXIS_SITES = (
    # Qwen3.5-9B row-parallel mlp.fc2: K = 12288 / tp (the 1536 shard was already published).
    (12288, 4096, "row", CENSUS_TPS),
    # GLM-4.7-Flash, hidden 2048: TP 1 and 2 are qualified; TP8 is impossible (20 MLA q heads do
    # not divide 8) and TP4 is not qualified.
    (2048, 768, "none", (1,)),        # mla.q_down, replicated
    (2048, 576, "none", (1,)),        # mla.kv_down, replicated
    (768, 5120, "column", (1, 2)),    # mla.q_up, K = q_lora_rank 768
    (2048, 20480, "column", (1, 2)),  # layer-0 dense mlp.fc1
    (2048, 3072, "column", (1, 2)),   # shared-expert fc1
    (5120, 2048, "row", (1, 2)),      # attn.o, K = 20 heads x 256
    (10240, 2048, "row", (1, 2)),     # layer-0 dense mlp.fc2
    (3072, 2048, "row", (1, 2)),      # shared-expert fc2
    (2048, 154880, "column", (1, 2)), # lm_head
    # MiMo-7B, hidden 4096, deployed TP4 (its README).
    (4096, 6144, "column", (4,)),     # attn.qkv
    (4096, 22016, "column", (4,)),    # mlp.fc1
    (11008, 4096, "row", (4,)),       # mlp.fc2: K = 2752 at TP4, declined by SEGMENT_ALIGN (policy)
    (4096, 151680, "column", (4,)),   # lm_head
)


#: Round 8 (README section 4.32): the audited model forward weights, as (K, N) at their deployed shard,
#: so the digest covers every K they run under.
MODEL_SITES = (
    (4096, 64), (4096, 128), (5120, 512), (5120, 1280), (1280, 4096), (1536, 4096),
    (8192, 4096), (4096, 8192), (512, 32768), (1280, 32768), (6144, 25600), (4096, 154880),
)


def _axis_weights() -> set[tuple[int, int]]:
    out = set()
    for K, N, axis, tps in AXIS_SITES:
        for tp in tps:
            if axis == "column" and N % tp == 0:
                out.add((K, N // tp))
            elif axis == "row" and K % tp == 0:
                out.add((K // tp, N))
            elif axis == "none":
                out.add((K, N))
    return out


def default_weight_set() -> list[tuple[int, int]]:
    """Every ``(K, N)`` the table publishes a row for, de-duplicated and sorted."""
    out = set(FOUNDATION_SITES)
    for K, N in CENSUS_SITES:
        for tp in CENSUS_TPS:
            if N % tp == 0:
                out.add((K, N // tp))
    for K, N in ROW_PARALLEL_SITES:
        for tp in CENSUS_TPS:
            if K % tp == 0:
                out.add((K // tp, N))
    out |= _axis_weights()
    out |= set(MODEL_SITES)
    return sorted(k for k in out if supported(k[0], k[1]))


def default_k_set() -> list[int]:
    """Every K the table publishes a row for: the distinct K of :func:`default_weight_set`."""
    return sorted({K for K, _N in default_weight_set()})


def regenerate(path: Path | None = None) -> dict:
    """Publish ``segments.json`` from the policy over :func:`default_k_set`."""
    rows = [SegmentEntry(K=K, dtype="bf16", split=segment_count_by_policy(K, "bf16"))
            for K in default_k_set()]
    return write_table(rows, path)


if __name__ == "__main__":  # python -m isoexec.ops.mm.mm_sm100.segments
    t = regenerate()
    print(f"published {len(t['rows'])} rows, digest {t['digest']}, policy v{t['policy_version']}")
    for K, d, S in t["rows"]:
        print(f"  K={K:6d} {d} S={S} segment_k={K // S}")
