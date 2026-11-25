"""Deployed BF16 dense-weight census used by contract tests."""

RL_CENSUS = (
    # --- Qwen3.5-35B-A3B, hidden 2048 -- trainer TP4 (megatron) ------------------------------
    (2304, 2048, "qwen35b.attn.qkv@tp4"),
    (3088, 2048, "qwen35b.gdn.in_proj@tp4"),
    (256, 2048, "qwen35b.shexp.fc1@tp4"),
    (2048, 1024, "qwen35b.attn.o+gdn.out@tp4"),          # row-parallel: K = 4096/4
    (62080, 2048, "qwen35b.lm_head@tp4"),
    # --- Qwen3.5-35B-A3B -- rollout engine TP8 (vLLM) ----------------------------------------
    (1152, 2048, "qwen35b.attn.qkv@tp8"),
    (1544, 2048, "qwen35b.gdn.in_proj@tp8"),
    (128, 2048, "qwen35b.shexp.fc1@tp8"),
    (2048, 512, "qwen35b.attn.o+gdn.out@tp8"),           # row-parallel: K = 4096/8
    (31040, 2048, "qwen35b.lm_head@tp8"),
    # --- Qwen3.5-9B, hidden 4096 -- TP1 ------------------------------------------------------
    (10240, 4096, "qwen9b.attn.qkv@tp1"),
    (12352, 4096, "qwen9b.gdn.in_proj@tp1"),
    (24576, 4096, "qwen9b.mlp.fc1@tp1"),
    (4096, 4096, "qwen9b.attn.o+gdn.out@tp1"),           # row-parallel: K = 4096/1
    (4096, 12288, "qwen9b.mlp.fc2@tp1"),                 # row-parallel: K = 12288/1
    (248320, 4096, "qwen9b.lm_head@tp1"),
    # --- Qwen3.5-9B -- TP4 -------------------------------------------------------------------
    (2560, 4096, "qwen9b.attn.qkv@tp4"),
    (3088, 4096, "qwen9b.gdn.in_proj@tp4"),
    (6144, 4096, "qwen9b.mlp.fc1@tp4"),
    (4096, 1024, "qwen9b.attn.o+gdn.out@tp4"),
    (4096, 3072, "qwen9b.mlp.fc2@tp4"),
    (62080, 4096, "qwen9b.lm_head@tp4"),
    # --- GLM-4.7-Flash, hidden 2048, MLA -- TP1 ----------------------------------------------
    (768, 2048, "glm.mla.q_down@tp1/2"),                 # replicated, axis none
    (576, 2048, "glm.mla.kv_down@tp1/2"),                # replicated, axis none
    (5120, 768, "glm.mla.q_up@tp1"),                     # K = q_lora_rank = 768
    (20480, 2048, "glm.mlp.fc1@tp1"),                    # layer 0, dense
    (3072, 2048, "glm.shexp.fc1@tp1"),
    (2048, 5120, "glm.attn.o@tp1"),                      # row-parallel: K = 20*256/1
    (2048, 10240, "glm.mlp.fc2@tp1"),                    # row-parallel
    (154880, 2048, "glm.lm_head@tp1"),
    # --- GLM-4.7-Flash -- TP2 ----------------------------------------------------------------
    (2560, 768, "glm.mla.q_up@tp2"),
    (10240, 2048, "glm.mlp.fc1@tp2"),
    (1536, 2048, "glm.shexp.fc1@tp2"),
    (2048, 2560, "glm.attn.o@tp2"),
    (2048, 1536, "glm.mlp.fc2@tp2"),                     # row-parallel: K = 3072/2
    (77440, 2048, "glm.lm_head@tp2"),
    # --- MiMo-7B, hidden 4096 -- TP4 ---------------------------------------------------------
    (1536, 4096, "mimo.attn.qkv@tp4"),
    (5504, 4096, "mimo.mlp.fc1@tp4"),
    # DECLINED, and kept here so the decline is on the record rather than an omission:
    # K = 11008/4 = 2752 is not a multiple of SEGMENT_ALIGN, so this package refuses it and the
    # incumbent provider serves it. `rl_shapes` filters it out; nothing below measures it.
    (4096, 2752, "mimo.mlp.fc2@tp4"),
    (37920, 4096, "mimo.lm_head@tp4"),
)
