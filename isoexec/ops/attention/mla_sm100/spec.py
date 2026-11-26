"""SM100 MLA family metadata is declared by attention/spec.py."""

from __future__ import annotations

from isoexec.ops.attention import target_batch as _target_batch

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

# --- declarations the schedule policy reads (design version 7.3; fwd.schedule_for) --------------------
# Fixed when the schedule is bound, not per-call inputs, so the trainer and every engine read the same
# segment span whatever their own batch, length or tensor-parallel degree. No model or deployment config
# of this tree states the longest context or the head count for MLA (no model of the tree runs MLA), so
# they are declared here; the target batch is the launch's (``ISOEXEC_TARGET_BATCH``).

#: The declared longest context, in keys: the production ``max_model_len`` of this tree's attention
#: batteries (gqa_sm100/tests/test_seam_gpu.py), a row the removed SEG table also carried. Longer rows
#: are served exactly, with more segments.
LONGEST_CONTEXT = 10240

#: The optional target batch: the decode batch (sequences per engine replica) the schedule is tuned for.
#: ``None`` (the default: ``ISOEXEC_TARGET_BATCH`` unset or empty) declares none, and the policy then
#: schedules for throughput: ``p = 1``, one segment per row of up to ``LONGEST_CONTEXT`` keys. Every
#: batch size, including 1, runs under any value and gives the same bits as every other batch size under
#: that value; the declaration only says which batch the schedule is fast at. The trainer and every
#: engine read it from the one launch environment (``attention/target_batch.py``, ``core/flags.py``).
TARGET_BATCH: int | None = _target_batch.declared()

#: Heads of the MODEL (never a rank's): GLM-4.7-Flash, the geometry the removed SEG table was keyed on
#: (20 heads; 10 and 5 per rank at TP 2 and 4). Any head count up to 64 gives the same occupancy. It is
#: read only when a target batch is declared.
MODEL_HEADS = 20
