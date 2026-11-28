"""Sparse attention (and, until it is split out, indexer / top-k) constants on SM100; the numerical program is README.md.

Three sub-ops, three recipes.  Every constant here is stated in README.md with its tag
(``[assumed]`` or ``[confirmed: <source>]``); ``tests/test_contract_cpu.py`` reads README.md and
fails if the two disagree, so a change to either is a visible change to both.

Registry entries (``ImplSpec``) live in ``ops/attention/spec.py`` and are added only at 
"""

from __future__ import annotations

#: This package's architecture.
ARCH = "sm100"

#: One recipe per sub-op.  The id IS the recipe; a change to anything README.md fixes is a version bump.
IMPL_IDS = {
    "attention.dsa.indexer": "sm100_dsa_indexer_v1",
    "attention.dsa.topk": "sm100_dsa_topk_v1",
    "attention.sparse": "sm100_sparse_segtree_v1",
}
#: Recipe v2 of the sparse attention v1 with FA4's lazy rescale at a fixed
#: threshold of 8 log2 units (mla_sm100 / gqa_sm100 v2's rule) -- a separate impl id, no twin of v1.
IMPL_ID_SPARSE_V2 = "sm100_sparse_v2"
RESCALE_THRESHOLD_V2 = 8.0
#: The dual-rail leaf (README "Dual-rail recipe", dualrail.py): two-chain QK, per-row lazy max at 6 log2 units,
#: one CTA per query row, no split; with the learned sink in its epilogue the sink form. Separate ids, no twin.
IMPL_ID_SPARSE_DUALRAIL = "sm100_sparse_dualrail_v1"
IMPL_ID_SPARSE_DUALRAIL_SINK = "sm100_sparse_dualrail_sink_v1"
RESCALE_THRESHOLD_DUALRAIL = 6.0
#: No twin class for any of them: nothing else implements these recipes (README.md preamble).
TWIN_CLASS = ""

# ---- SPEC section 1: deployment-fixed geometry ---------------------------------------------------

H_Q_VALUES = (64, 128)          #: per-rank query heads the sparse attention serves
H_KV = 1
D_QK = 512                      #: V4.1: 448 NoPE + 64 RoPE inside one 512 (576 is a declared gap)
D_QK_NOPE = 448
D_QK_ROPE = 64
D_V = 512
KV_QUANT_TILE = 32              #: one ue8m0 scale per 32 e4m3 values
KV_NUM_SCALES = D_QK // KV_QUANT_TILE          # 16 = 14 NoPE + 2 RoPE tiles
KV_BYTES_PER_TOKEN = D_QK + KV_NUM_SCALES      # 528 (V41_FP8Sparse)
PAGE = 64                       #: tokens per cache page (address map only; not numerical)
TOPK = 2048                     #: K; one recipe per value
H_IDX = 64
D_IDX = 128
IDX_QUANT_TILE = 32             #: the indexer MMA's block-scale tile (UE8M0 per 32 e4m3 values)
IDX_NUM_SCALES = D_IDX // IDX_QUANT_TILE       # 4 -> one packed int32 per (token[, head])
IDX_BYTES_PER_TOKEN = D_IDX + 4                # DeepGEMM's paged FP8 layout: 128 e4m3 + 4 ue8m0
#: The V4.1 cache program (port/CACHE-SPEC.md section 3) writes ONE power-of-two scale per 128-value key (and per query
#: head), carried as four equal UE8M0 bytes, exactly as vLLM 0.33's SM100 index cache stores it.
SCORE_DTYPE = "bfloat16"
SCORE_PAD_BITS = 0xFF80         #: bf16 -inf, stored outside the window
TOPK_PAD = -1
NAN_POLICY = "nan_lowest"       #: NaN ranks below -inf and every finite score
NEXT_N = 1
MAX_CTX = 131072
TILE_SLOTS = 64                 #: list tile of the sparse attention (every boundary is a max/rescale point)

#: SPEC section 1 ``LIST_GROUPS[(H_Q, K)]``: fixed groups over list positions.  G = 1 is the unsplit
#: chain.  Regret-chosen in (README "The G table").
LIST_GROUPS = {(64, 2048): 1, (128, 2048): 1, (64, 640): 1, (128, 640): 1}

#: SPEC section 3 item 4: rows longer than this go through the multi-CTA radix path.
TOPK_ROW_SPLIT = 65536

# ---- SPEC section 4: the sparse attention leaf's constants (mla_sm100 verbatim) -----------------

MAX_INIT_VAL = -1e33            #: masked score, applied before the max
MAX_INIT_VAL_SM = -1e30         #: running-max seed; also the empty state's M
LOG2E = 1.4426950408889634

# ---- SPEC section 4.6 / dtir section 8: the one scalar profile for every kernel of every sub-op --

#: Tensor-core descriptor of the indexer's dot product (SPEC section 2 item 1).
TC_DESCRIPTOR_INDEXER = (
    "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X; A=K(e4m3,K-major,SW128) "
    "M=128 keys; B=Q(e4m3,K-major,SW128) N=64 heads; ue8m0 block 32 via TMEM scale vectors; "
    "UMMA_K=32, k-steps c=0,32,64,96 ascending; fp32 TMEM accumulator, first step ScaleOut::Zero"
)
#: Tensor-core descriptor of the sparse attention (SPEC section 4.2).
TC_DESCRIPTOR_SPARSE = (
    "tcgen05.mma.cta_group::1.kind::f16 bf16xbf16->fp32; QK: M=64 heads, N=64 slots, UMMA_K=16, "
    "32 k-steps over c=0..511 ascending, first step ScaleOut::Zero; PV: M=64, N=256 x2 halves, "
    "UMMA_K=16, 4 k-steps over the tile's 64 slots ascending, segment's first tile ScaleOut::Zero"
)

__all__ = [n for n in dir() if n.isupper()]
