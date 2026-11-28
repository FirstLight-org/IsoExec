"""Recipe identifiers and constants of the native V4.1 attention composition (``ops`` documents the composition)."""
from __future__ import annotations

ARCH = "sm100"

#: The composition recipe.  It names the rounding points of every row operation (numerics.cuh), the
#: selection rules and the sub-recipes it is assembled from; a change to any of them bumps it.
RECIPE = "deepseek_v41_attn_sm100_v3"
ROWS_RECIPE = "deepseek_v41_attn_rows_v2"
SUB_RECIPES = {
    "quant": "deepseek_v41_cache_record_quant_v1",        # quant/checkpoint_sm100 (port/CACHE-SPEC)
    "pool": "deepseek_channel_softmax_pool_fp32_v1",     # channel_pool_sm100 scalar graph
    "mm": "sm100_tcgen05_segtree_v1",                     # mm/mm_sm100 (compressor, index k/weights)
    "grouped": "sm100_tcgen05_v1",                        # moe/moe_sm100 fc1: grouped wo_a (one K chain)
    "indexer": "sm100_dsa_index_logits_dg_v1",            # dsa_sm100 (INDEX_FOLD): DeepGEMM's SM100 MQA logits fold
    "topk": "sm100_dsa_topk_v1",                          # dsa_sm100
    "sparse": "sm100_dsa_sparse_sink_v1",                 # dsa_sm100 learned-sink root, G = 1, fused epilogue
    "sparse_bwd": "sm100_dsa_sparse_sink_bwd_v2",
}
BACKWARD_RECIPE = "deepseek_v41_attn_bf16_actor_bwd_v1"

#: row-kernel instantiations (csrc/rows.cu ``row_op`` kinds)
ROW_Q_NORM = 0           # RMSNorm(q_lora_rank = 1280)
ROW_WINDOW_KV = 1        # RMSNorm(512) -> RoPE tail -> FP8 b32 UE8M0 Q/DQ
ROW_Q_ROPE = 2           # RoPE tail per head
ROW_OUT_INV_ROPE = 3     # inverse RoPE tail per head, group-major output
ROW_LATENT_NORM = 4      # RMSNorm(512) (ratio-1 compressor)
ROW_COMPRESSED_KV = 5    # RoPE tail at group start (FP32, not rounded) -> FP8 b32 UE8M0 Q/DQ (log2f exponent)
ROW_INDEX_K = 6          # RMSNorm(128) -> RoPE tail -> FP8 E4M3 codes, one UE8M0 exponent per key
ROW_INDEX_Q = 7          # RoPE tail -> FP8 E4M3 codes, one UE8M0 exponent per head (index_q_heads wide)
ROW_NORM_ROPE = 8        # qualification only
ROW_K_NORM = 9           # qualification only

#: Bytes of one cache record: window and compressed KV share the FP8 528 B record (512 E4M3 + 16 UE8M0), an index key
#: is 128 E4M3 + one exponent carried as four equal UE8M0 bytes (vLLM 0.33 FLASHMLA_SPARSE_DSV41 on SM100).
KV_RECORD_BYTES = 528
INDEX_RECORD_BYTES = 132

#: Index-query format. "fp8_b128" = vLLM's query quantizer (E4M3, one power-of-two scale per head, carried as four
#: equal UE8M0 bytes in the query's scale word). PENDING USER DECISION (port/CACHE-SPEC.md section 4): the alternative
#: "bf16" (no query quantization) needs a BF16 x FP8 index-score kernel that does not exist yet.
INDEX_QUERY_FORMAT = "fp8_b128"

#: The index-score fold. "bf16_dg" (the recipe, SUB_RECIPES["indexer"]): DeepGEMM's SM100 MQA logits program --
#: per head bf16(relu(d)) of the FP32 MMA accumulator, four BF16 fma chains over the heads 4i + c, (c0 + c2,
#: c1 + c3), low + high -- over the model's index heads (32 or 64), vLLM's own fold (port/CACHE-SPEC D8).
#: "fp32_tree": the former sm100_dsa_indexer_tree_v1 (FP32 8-leaf tree over heads padded to 64), kept selectable for
#: comparison only.
INDEX_FOLD = "bf16_dg"


def index_q_heads(heads: int) -> int:
    """Head width of the index queries / weights the selected fold reads (the tree reads heads padded to 64)."""
    return 64 if INDEX_FOLD == "fp32_tree" else heads


#: the exponent word of a padded (all-zero) index head: 2^0 in all four bytes
UE8M0_ONE_PACKED = 0x7F7F7F7F

#: deployment-fixed maximum tensor-parallel degree = the leaf count of both cross-head reductions
#: (index heads in leaves of INDEX_LEAF_HEADS, wo_b's K in TP_LEAVES equal column blocks). Every TP degree
#: dividing TP_LEAVES evaluates the same balanced FP32 tree, so outputs are TP-degree invariant.
TP_LEAVES = 8
INDEX_LEAF_HEADS = 4
INDEXER_TREE_RECIPE = "sm100_dsa_indexer_tree_v1"       # INDEX_FOLD == "fp32_tree" only

#: The index-score tree is evaluated on EVERY rank over all TP_LEAVES leaves (the indexer is replicated
#: across TP: no score collective). Its top INDEX_TREE_IEEE_LEVELS levels use IEEE FP32 adds (subnormals
#: kept), the leaf chains and the level below them the indexer kernel's FTZ adds: bit for bit the program
#: a TP = 4 host evaluated across ranks before the indexer was replicated (each rank one FTZ subtree of 2 leaves, the 4
#: FP32 roots completed by leaf_tree / detreduce, built --ftz=false), now at every TP degree. The two
#: forms differ from a fully FTZ tree only when a top-level partial sum is an FP32 subnormal.
INDEX_TREE_IEEE_LEVELS = 2

#: fixed token segment of every weight-gradient row reduction (ascending inside, segments ascending)
WGRAD_SEGMENT = 256

#: the index-weight scale must be an exact power of two so that scaling the BF16 projection is exact
def index_weight_scale(cfg) -> float:
    n = cfg.index_head_dim * cfg.index_n_heads
    root = int(round(n ** 0.5))
    if root * root != n or root & (root - 1):
        raise ValueError("index weight scale (head_dim * heads) ** -0.5 must be a power of two")
    return 1.0 / root
