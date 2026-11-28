"""Test geometry and modules for the native composition (not a model; synthetic BF16 weights)."""
from __future__ import annotations


import torch
from torch import nn

#: layer roles: 0 window-only; 1 ratio-2 KV+index source; 2 reuse; 3 reindex; 4 ratio-1 KV+index+
#: candidate source; 5 reuse; 6 reindex + candidate consumer
RATIOS = [0, 2, 2, 2, 1, 1, 1]


def native_cfg(*, hidden=1024, heads=16, ratios=RATIOS, kv_sources=(1, 4), index_sources=(1, 3, 4, 6),
               candidate_source=4, index_topk=512, candidate_blocks=2048, candidate_block=8, window=128):
    from ._config import config
    return config(hidden_size=hidden, num_attention_heads=heads, num_hidden_layers=len(ratios),
                  compress_ratios=list(ratios), num_nextn_predict_layers=0, kv_source_layer_ids=list(kv_sources),
                  index_source_layer_ids=list(index_sources), candidate_source_layer_id=candidate_source,
                  index_topk=index_topk, candidate_topk_blocks=candidate_blocks, candidate_block_size=candidate_block,
                  sliding_window=window, engram_layer_ids=[], engram_num_embeddings=[], dspark_target_layer_ids=[])


class Lin(nn.Module):
    def __init__(self, n_in, n_out, gen, *, gain=1.0, device="cuda"):
        super().__init__()
        w = torch.randn((n_out, n_in), generator=gen, device=device) * (gain / n_in ** 0.5)
        self.weight = nn.Parameter(w.to(torch.bfloat16))


class Norm(nn.Module):
    def __init__(self, n, gen, device="cuda"):
        super().__init__()
        self.weight = nn.Parameter((1 + 0.1 * torch.randn(n, generator=gen, device=device)).to(torch.bfloat16))


class Box(nn.Module):
    pass


def make_layer(cfg, index, seed, device="cuda"):
    g = torch.Generator(device=device).manual_seed(1000 + seed)
    hid, H, D = cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim
    m = Box()
    m.wq_a = Lin(hid, cfg.q_lora_rank, g)
    m.q_norm = Norm(cfg.q_lora_rank, g)
    m.wq_b = Lin(cfg.q_lora_rank, H * D, g, gain=2.0)
    m.wkv = Lin(hid, D, g)
    m.kv_norm = Norm(D, g)
    m.wo_a = Lin(H * D // cfg.o_groups, cfg.o_groups * cfg.o_lora_rank, g)
    m.wo_b = Lin(cfg.o_groups * cfg.o_lora_rank, hid, g)
    m.attn_sink = nn.Parameter(torch.randn(H, generator=g, device=device))
    if index in cfg.kv_source_layer_ids:
        c = Box()
        c.wkv = Lin(hid, D, g)
        if cfg.compress_ratios[index] > 1:
            c.wgate = Lin(hid, D, g, gain=2.0)
        c.norm = Norm(D, g)
        m.compressor = c
    if index in cfg.index_source_layer_ids:
        i = Box()
        i.wq_b = Lin(cfg.q_lora_rank, cfg.index_n_heads * cfg.index_head_dim, g, gain=2.0)
        i.weights_proj = Lin(hid, cfg.index_n_heads, g, gain=8.0)
        if index in cfg.kv_source_layer_ids:
            i.wk = Lin(D, cfg.index_head_dim, g, gain=2.0)
            i.k_norm = Norm(cfg.index_head_dim, g)
        m.indexer = i
    return m


def native_linear(module, x):
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops
    w = module.weight
    n = w.shape[0]
    if n < 64:   # mm_sm100 serves N >= 64: zero rows give exact zero columns, sliced off
        w = torch.cat((w, w.new_zeros((64 - n, w.shape[1]))), 0)
    y = ops.linear(x.reshape(-1, x.shape[-1]), w)[:, :n]
    return y.reshape(*x.shape[:-1], n)


def dequant(module):
    return module.weight


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()
