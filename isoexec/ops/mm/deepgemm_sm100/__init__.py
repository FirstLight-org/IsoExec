"""SM100 BF16 GEMMs on the attributed, vendored DeepGEMM kernels (``_vendor/deep_gemm``). No build on import.

* ``ops``: the loader (deterministic mode, fixed for every call) and the launch helper (programmatic dependent
  launch scoped to one launch), plus the grouped BF16 einsum ``bhr,hdr->bhd`` of the V4.1 grouped ``wo_a``.
* ``bf16``: the dense BF16 projection ``bf16_gemm_nt`` for the weights whose published ``mm_sm100`` program is one
  K chain (segment count 1).

Program (per output element): one ascending-K chain of BF16 ``tcgen05.mma`` steps from a non-accumulating first
step, FP32 accumulator, one BF16 rounding of the root; deterministic mode never splits K. That is the program of
``mm/mm_sm100`` at segment count 1 (dense) and of ``moe_sm100.fc1`` (grouped), so the bits are the incumbent
providers'; the vendored tile / swap-AB / cluster heuristics choose ownership only.

Invariance: an output depends only on its row and its weight row, so it is the same at any batch size, row
position and TP degree.
"""
