"""SM100 MLA family metadata is declared by attention/spec.py."""

from __future__ import annotations

#: This package's architecture. ``ops/_dispatch.FAMILIES`` names it on the sm100 row.
ARCH = "sm100"

#: The op ids this package owes a declaration for:
#:   - attention.core.mla (declared in ops/attention/spec.py)

#: The recipe this package implements.  ``ops/attention/spec.py`` declares the ``ImplSpec`` under
#: this id; the id IS the recipe, and it is a *different* recipe from sm90's
#: ``sm90_absorbed_segtree``: tcgen05's fp32 accumulation history differs from wgmma's, the leaf's
#: row sum is reduced per tile rather than deferred to the segment end, and a segment's first PV
#: overwrites the TMEM accumulator instead of accumulating onto a zeroed one.  README.md's
#: "What changed from sm90, and why" is the list; a change to any of it is a version bump here.
IMPL_ID = "sm100_absorbed_segtree"

#: Recipe v2 (round 5): the same program with FA4's LAZY rescale -- a row keeps its running max while
#: a 64-key tile's max exceeds it by at most RESCALE_THRESHOLD = 8 log2 units of the scaled score, so
#: no O rescale runs and the tile's probabilities are taken against the kept max (<= 2^8).  Segments,
#: tree, merge operator, root and every rounding point are v1's; the decision reads only the row's own
#: running max over its own fixed tiles inside one segment, so it is a pure function of logical
#: position, exactly as gqa_sm100's sm100_gqa_segtree_v2.  A DIFFERENT recipe: its bits differ from
#: v1's, so it carries no twin class with it.  v1 stays the default and stays selectable.
IMPL_ID_V2 = "sm100_absorbed_segtree_v2"
RESCALE_THRESHOLD = 8.0

#: This recipe carries NO twin class, exactly as sm90's ``sm90_absorbed_segtree`` does not: a twin
#: class names a set of implementations that must agree bit for bit, and nothing else implements
#: this recipe.  In particular the sm90 SEG-tree recipe is deliberately not a twin of this one --
#: wgmma and tcgen05 do not accumulate a 576-wide dot product in the same order, so the registry
#: must refuse to pretend they do (docs/sm100-bringup.md's opening paragraph).  A ``TWIN_PROOFS``
#: entry is therefore neither needed nor correct here; what proves this recipe is its own battery,
#: ``tests/conformance.py``, which checks the determinism relations and the float64 reference.
TWIN_CLASS = ""


#: KV tile == page, as in mla_sm90. The CUDA side must state the same number, and a change to either
#: is a recipe version bump.
PAGE = 64

#: mla_sm90/README.md is the statement of the numerical program: QK as two ordered chains (latent,
#: then rotary) into one accumulator, fixed SEG segments merged by a balanced fp32 tree, one bf16
#: root round with L == 0 or NaN replaced by 1. Blackwell changes how it is evaluated (64 heads per
#: CTA with the accumulator in TMEM, latent-tile multicast, two-CTA NOT a decode lever) and must not
#: change what is evaluated.
