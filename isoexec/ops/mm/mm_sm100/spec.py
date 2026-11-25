"""What this package owns on sm100."""

from __future__ import annotations

from isoexec.core.registry import ImplSpec

ARCH = "sm100"

#: The numerical program. Names the K consumption order, the segment/merge structure, the
#: accumulator precision and the rounding points -- everything that is the same for every weight.
#: Which S each K gets is NOT in here: that is the segment table, versioned separately by
#: ``segments.POLICY_VERSION`` and identified by ``segments.table_digest()``. Both the recipe
#: version and the table digest are part of the bind-time comparison; a table change moves the
#: digest and the policy version, an arithmetic change moves this string.
#:
#: v2: the segmentation of an output element is a function of K alone. v1 let it read the weight's
#: N, so a tensor-parallel column shard or a packed projection ran a different K reduction from the
#: whole weight; under v2 the program of an output element is fixed by the logical weight whatever
#: rank holds it and whatever it is packed with. That is a property of the program, not only of a
#: table row, so it moves this string as well as the policy version (``segments.POLICY_VERSION`` 4).
RECIPE_VERSION = "sm100_tcgen05_segtree_v2"

#: The registry version of :data:`IMPL_ID`, moved with :data:`RECIPE_VERSION`.
IMPL_VERSION = 2

#: The impl id. The sm90 ids (``triton_batch_invariant``, ``sm90_wgmma_segtree``) are different
#: recipes and are never renamed; ``tcgen05``'s accumulation and a K split both change the bits
#: from Hopper by design, so this is a new id, not a widening.
IMPL_ID = "sm100_tcgen05_segtree"

#: Op ids this package serves.
OPS = ("mm",)

#: The twin class this recipe opens. The mm family makes no general bitwise twin claim BETWEEN
#: PROVIDERS -- the sm90 Triton and cuBLASLt ids are different recipes and this one is a third --
#: so the class is not "every mm impl". It is this program: fixed per-K segments, each
#: accumulated from zero in ascending K by one tcgen05 chain, merged by the balanced adjacent-pair
#: tree over the segment index, one BF16 rounding at the root, the segments a function of K alone
#: (v2) so that every column shard and every packing of a weight computes it. A future member joins
#: by computing that -- an fp8 leaf, a 64-row single-CTA tile, a different split table's worth of segments --
#: and the battery this class names is what a member has to pass. The sm90 ids can never join:
#: wgmma's K step is 16 against tcgen05's 32, so the fp32 accumulation history inside a segment
#: differs and an sm90 proof does not travel.
TWIN_CLASS = "mm.dense.segtree"

#: Why :data:`TWIN_CLASS` is recorded in ``core/registry.TWIN_PROOFS`` but is NOT spelled on the
#: ``ImplSpec.twin_class`` field below, which is the same reason ``moe_sm100``'s
#: ``WHY_NO_TWIN_FIELD`` gives: the class has exactly one declarable member today, and
#: ``core/tests/test_op_gate_alignment_cpu.py`` correctly refuses a one-member class -- an
#: equivalence with nothing to be checked against is not a claim. Recording the class in
#: ``TWIN_PROOFS`` is what ``gdn.cpr`` and ``moe``'s two cross-arch classes do, and it is what makes
#: the battery the proof of the class rather than of one impl. The field goes on the record the day
#: a second member exists.
WHY_NO_TWIN_FIELD = "one declarable member; the class is recorded in TWIN_PROOFS instead"

IMPLS: tuple[ImplSpec, ...] = (
    ImplSpec(op="mm", impl_id=IMPL_ID, version=IMPL_VERSION, supported_archs=frozenset({ARCH})),
)
