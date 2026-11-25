"""SM90 MLA constants and hazards; registry entries live in ops/attention/spec.py.
Forward uses FlashMLA's 64-token tile arithmetic (masked max, 4-lane butterfly, exp2f rescaling,
bf16 probabilities and fp32 accumulators), fixed SEG segments and balanced fp32 merges, then one bf16 root
round with L==0 or NaN replaced by 1, under --fmad=false --ftz=false.
Backward is numerical only; tests/README.md maps hazards and sensitivity witnesses to checks."""

from __future__ import annotations

#: KV tile == page. The CUDA side states the same number as ``PAGE_BLOCK_SIZE`` in
#: ``csrc/fwd/config.h``; the two must agree, and a change to either is a recipe version bump.
PAGE = 64

HAZARDS = (
    "nonfinite_unused_pages",
    "logprob_contract",
    "seg_boundary_lengths",
    "phase_dependent_program",
    "null_lane_and_empty_batch",
    "occupancy_split_nondeterminism",
    "bf16_probability_cast",
)
