"""Torch transcription of the rope_sm90/sm100 v1 rounding recipe, for explicitly requested reference execution and for the battery's independent-provider checks."""

from __future__ import annotations

from isoexec.ops.attention.rope_sm90.reference import (  # noqa: F401
    gather_cos_sin,
    inv_freq_double,
    mrope_masks,
    qk_norm_rope_float64,
    qk_norm_rope_recipe,
    rope_float64,
    rope_recipe,
    table_rows_host,
    tree_sum,
)

__all__ = [
    "gather_cos_sin",
    "inv_freq_double",
    "mrope_masks",
    "qk_norm_rope_float64",
    "qk_norm_rope_recipe",
    "rope_float64",
    "rope_recipe",
    "table_rows_host",
    "tree_sum",
]
