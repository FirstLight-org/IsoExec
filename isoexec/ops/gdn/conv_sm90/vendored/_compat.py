"""Runtime-free utilities for the vendored vLLM causal_conv1d kernels.
Triton imports and the PAD_SLOT_ID/NULL_BLOCK_ID sentinels follow the upstream revision recorded in PROVENANCE.json."""

from __future__ import annotations

import triton
import triton.language as tl

# vllm/v1/attention/backends/utils.py @ commit 6f573f486, lines 45-46 (citation; the values are
# fixed here and not read back from an installed vLLM).
PAD_SLOT_ID = -1
NULL_BLOCK_ID = 0

__all__ = ["tl", "triton", "PAD_SLOT_ID", "NULL_BLOCK_ID"]
