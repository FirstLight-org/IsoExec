"""SM90 RoPE and fused q/k-norm + RoPE kernels (numerical recipe rope_sm90 v1; see README.md).

Importing this package does not build the extension; the first kernel call does.
"""

from isoexec.ops.attention.rope_sm90.ops import RECIPE_VERSION, RopeTable, build_table, qk_norm_rope, rope

__all__ = ["RECIPE_VERSION", "RopeTable", "build_table", "qk_norm_rope", "rope"]
