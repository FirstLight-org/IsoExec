"""Constants of the SM100 four-stream mHC kernels' reduction program. Import-light: no torch, no build.

The kernels are shared by the mHC recipes that declare them (``residual/deepseek_mhc_sm100/spec.py``
declares the DeepSeek-V4.1 ops); this package declares no op of its own. ``csrc/split_fwd.cu`` and
``csrc/split_bwd.cu`` restate the constants and ``tests/test_contract_cpu.py`` holds them together.
"""

from __future__ import annotations

ARCH = "sm100"
HC_MULT = 4
NUM_OUT = 24
#: Widths the split kernels serve (hidden 4096 and DeepSeek-V4.1-Flash's 5120).
HIDDEN = (4096, 5120)

#: Forward reduction (a table over H only, never T): NUM_SLICES h-slices; slice s owns h in [s*H/32, (s+1)*H/32) of all
#: four streams. Leaf = one warp = one stream over the slice (tensor-core chain, ascending h); slice partial =
#: (l0 + l1) + (l2 + l3); row total = balanced adjacent-pair tree over the 32 slices.
NUM_SLICES = 32
LEAF = "mma.sync.m16n8k16 bf16xbf16->fp32, 16 h per step ascending (k permuted inside a step), exact products"
SUMSQ = "per-thread ascending fp32 fma chain of exact x*x, then (t0+t1)+(t2+t3)"
SLICE_TREE = "(l0+l1)+(l2+l3)"
ROW_TREE = "balanced_adjacent_pairs_over_32_slices"
#: Backward token reductions (d_fn, d_scale, d_base): fixed 512-row segments, tile order ascending, balanced tree over
#: segments (a function of T only through the segment count).
SEG_ROWS = 512
MERGE_ROW_TREE = "thread chunks of 8 h ascending; xor butterfly 16,8,4,2,1; (w0+w1)+(w2+w3)"

__all__ = ["ARCH", "HC_MULT", "HIDDEN", "LEAF", "MERGE_ROW_TREE", "NUM_OUT", "NUM_SLICES", "ROW_TREE", "SEG_ROWS",
           "SLICE_TREE", "SUMSQ"]
