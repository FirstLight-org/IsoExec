"""SM100 log-softmax scoring: row selection, the bf16 head / fp32 boundary. No build on import."""

from __future__ import annotations

from isoexec.core.registry import ImplSpec

ARCH = "sm100"

LEAVES = 8
STRIP = 8
LANES = 256

ACCUM = "fp32"

RECIPE_VERSION = 1
IMPL = "sm100_online_leaf8_v1"

RECIPE_VERSION_V2 = 2
IMPL_V2 = "sm100_twopass_leaf8_v2"

IMPL_BY_RECIPE: dict[int, str] = {RECIPE_VERSION: IMPL, RECIPE_VERSION_V2: IMPL_V2}

#: The engine's decode sampler (``sampler.py``): Gumbel-max token draw with Philox noise keyed by
#: (seed, step offset, row, column); its logprobs are the pinned recipe's, rank and top-1 re-evaluate
#: that recipe's full-row expression. An engine-side step, not a placement op, so it is not in IMPLS.
SAMPLER_IMPL = "sm100_gumbel_philox_v1"

DEFAULT_RECIPE_BY_CLASS: dict[tuple[str, int], int] = {
    ("bf16", 248320): RECIPE_VERSION_V2,   # Qwen3.5 9B / 35B-A3B / 3.6-27B
    ("bf16", 154880): RECIPE_VERSION_V2,   # GLM4.7-flash
    ("bf16", 151680): RECIPE_VERSION_V2,   # MiMo-7B
    ("fp32", 248320): RECIPE_VERSION,
    ("fp32", 154880): RECIPE_VERSION,
    ("fp32", 151680): RECIPE_VERSION,
}

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
    ImplSpec(op="logprobs.log_softmax", impl_id=IMPL_V2, version=RECIPE_VERSION_V2, supported_archs=_SM100),
)
