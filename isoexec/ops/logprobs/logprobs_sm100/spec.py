"""SM100 log-softmax scoring: row selection, the bf16 head / fp32 boundary. No build on import."""

from __future__ import annotations

from isoexec.core.registry import ImplSpec

ARCH = "sm100"

#: d_max of the vocabulary axis: a full row is LEAVES consecutive shards; a TP rank holds LEAVES / world of them.
LEAVES = 8
#: columns per load (execution; a shard must be whole strips)
STRIP = 8
#: threads of a (row, shard) CTA (execution)
LANES = 256

ACCUM = "fp32"

#: One numerical schedule (``schedule.py``; isoexec/ops/schedule_policy.py, design version 7.1) for every
#: vocabulary and logits dtype. Versions 1 (online normalizer) and 2 (two passes), both on 256 strided lanes per
#: leaf and selected per (dtype, vocabulary), are retired with their identities.
RECIPE_VERSION = 3
IMPL = "sm100_lse_segments_v3"

IMPL_BY_RECIPE: dict[int, str] = {RECIPE_VERSION: IMPL}

#: The engine's decode sampler (``sampler.py``): Gumbel-max token draw with Philox noise keyed by
#: (seed, step offset, row, column); its logprobs are the pinned recipe's, rank and top-1 re-evaluate
#: that recipe's full-row expression. An engine-side step, not a placement op, so it is not in IMPLS.
SAMPLER_IMPL = "sm100_gumbel_philox_v1"

_SM100 = frozenset({"sm100"})

IMPLS: tuple[ImplSpec, ...] = (
    ImplSpec(
        op="logprobs.head",
        impl_id="eager",
        version=1,
        supported_archs=_SM100,
    ),
    ImplSpec(
        op="logprobs.head",
        impl_id="fp32_root",
        version=1,
        supported_archs=_SM100,
    ),
    ImplSpec(
        op="logprobs.lm_head_slice",
        impl_id="identity",
        version=1,
        supported_archs=_SM100,
        twin_class="logprobs.row_select",
    ),
    ImplSpec(
        op="logprobs.lm_head_slice",
        impl_id="sampled_rows",
        version=1,
        supported_archs=_SM100,
        twin_class="logprobs.row_select",
    ),
    ImplSpec(op="logprobs.log_softmax", impl_id=IMPL, version=RECIPE_VERSION, supported_archs=_SM100),
)
