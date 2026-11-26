"""Executable statement of the norms_sm100 recipe -- the same recipe as norms_sm90."""

from __future__ import annotations

from isoexec.ops.norms.norms_sm90.reference import (
    add_rms_norm,
    gated_rms_norm,
    inv_rms,
    lanes,
    rms_norm,
    row_sum_of_squares,
    silu,
)

__all__ = ["add_rms_norm", "gated_rms_norm", "inv_rms", "lanes", "rms_norm", "row_sum_of_squares", "silu"]
