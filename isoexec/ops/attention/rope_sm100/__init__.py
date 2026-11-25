"""SM100 RoPE and fused q/k-norm + RoPE kernels (numerical recipe: bit-for-bit rope_sm90 v1; see README.md for what changed -- execution shape only -- and what did not."""

from isoexec.ops.attention.rope_sm100.ops import (
    RECIPE_VERSION,
    RopeTable,
    build_table,
    qk_norm_rope,
    rope,
)

__all__ = ["RECIPE_VERSION", "RopeTable", "build_table", "qk_norm_rope", "rope"]
