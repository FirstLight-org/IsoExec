"""CPU reference program of the V4.1 CED definition (the executable statement of the program).

Three formulations of one arithmetic, on CPU tensors:

* ``flat_forward``: the official reference code's SHAPE (``inference/model.py`` ``Transformer.forward``): one
  loop over every backbone layer on every token, cross-layer state handed down in shared slots keyed by the
  config's source lists. With ``decoder_window_floor`` the decoder layers after the boundary truncate their
  window at a given position (the "masked full program").
* ``ced_forward``: the CED definition as a trainer evaluates it on a packed batch: the causal encoder on every
  token, the named handoff (``EncoderOutput``), ``decoder.entry`` projecting the decoder global KV for every
  token (``DecoderGlobalKV``), then the decoder blocks over the gathered rows of ``DecoderSegments`` (window
  truncated at each row's segment start after the boundary layer).
* ``ReferenceEngine``: the CED definition as an engine evaluates it: per request, chunked prefill and decode
  steps over caches (window keys, compressed entries, index keys, the ratio-2 compressor's pending token, the
  segment-scoped decoder window, the token history of the Engram n-grams); the decoder runs on the rows the
  prompt-length rule selects (``engine_ced.plan_step``, the vLLM binding's own planner).

Arithmetic (declared, CPU): BF16 activations and FP32 arithmetic in the official reference's order of
operations; every reduction is a fixed pairwise tree over its own axis (``fsum``), every top-k a stable
descending sort (ties to the lower index), so each row's value is a function of that row's inputs only (no
dependence on batch size, chunking or packing). The cache records apply the decided quantizers of
``port/CACHE-SPEC.md`` through the op package's CPU statements (window keys: RoPE, BF16, FP8 per 32 exact
exponent; compressed KV: FP32 RoPE output, FP8 per 32 ``log2f`` exponent; index keys and index queries: FP8 per
128). Differences from the GPU recipe and from the official reference (declared, not bugs): index scores fold
in FP32 (the reference folds in BF16); Engram hashes are a synthetic integer hash of the same n-gram context
(the released multipliers belong to the embed's hash op) over BF16 tables; the MoE is dense over experts with
the ``MOE_RECIPE`` combine order. This module proves program-level equalities bitwise; it is not the GPU
recipe's bit-exact emulation (that is gate G2 on real weights).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100.reference import rope_emul, rope_frequencies
from isoexec.ops.quant.checkpoint_sm100.reference import quantize_reference

from .config import BOUNDED_REPLAY, FULL_DECODER, CedGeometry, Csa2Mode, ced_geometry
from .engine_ced import plan_step, segment_starts
from .program import DecoderSegments, EncoderOutput, Turn

BF16, F32 = torch.bfloat16, torch.float32
NEG = float("-inf")


# ---------------------------------------------------------------------------------------------- row ops
def fsum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sum over ``dim`` as a fixed pairwise tree (zero-padded to a power of two): the order depends only on the
    axis length, never on the other dimensions."""
    x = x.movedim(dim, -1)
    n = x.shape[-1]
    width = 1 << max(n - 1, 0).bit_length()
    if width != n:
        x = torch.nn.functional.pad(x, (0, width - n))
    while x.shape[-1] > 1:
        x = x[..., 0::2] + x[..., 1::2]
    return x[..., 0]


def mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``x [..., K] @ w[N, K]^T`` in FP32 with a fixed tree over K."""
    return fsum(x.float().unsqueeze(-2) * w.float(), -1)


def lin(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return mm(x, w).to(BF16)


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """The reference RMSNorm: FP32 statistics, ``weight * x`` in FP32, one BF16 rounding."""
    v = x.float()
    var = fsum(v * v) / v.shape[-1]
    return (w.float() * (v * torch.rsqrt(var + eps).unsqueeze(-1))).to(BF16)


class _QDQ(torch.autograd.Function):
    """A cache-record quantize-then-dequantize: the record's values forward, straight-through backward (the
    trainer's convention for every cache quantizer)."""

    @staticmethod
    def forward(ctx, x, block, rule):
        return quantize_reference(x.detach(), block_size=block, exponent_rule=rule, dequantize=True)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


def qdq(x: torch.Tensor, block: int, rule: str = "exact") -> torch.Tensor:
    return _QDQ.apply(x, block, rule)


def stable_top(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the ``k`` largest finite entries per row (ties to the lower index), ``-1`` padded."""
    order = torch.sort(scores, dim=-1, descending=True, stable=True).indices[..., :k]
    finite = torch.gather(scores, -1, order) > NEG
    picked = torch.where(finite, order, -1)
    if picked.shape[-1] < k:
        picked = torch.nn.functional.pad(picked, (0, k - picked.shape[-1]), value=-1)
    return picked


# ---------------------------------------------------------------------------------------------- tiny model
def synthetic_config(*, window: int = 8) -> dict:
    """A tiny CED (10 layers: encoder SWA, SWA+Engram, FULL r2, REUSE, FULL r2 + Engram; decoder FULL r1
    (candidate source), REUSE, REINDEX, REUSE, REINDEX) with the released config's field set."""
    text = {
        "vocab_size": 64, "hidden_size": 64, "moe_intermediate_size": 32, "num_hidden_layers": 10,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 128, "qk_rope_head_dim": 64,
        "q_lora_rank": 32, "o_lora_rank": 16, "o_groups": 2, "hidden_act": "silu", "swiglu_limit": 10.0,
        "rms_norm_eps": 1e-6, "attention_bias": False, "tie_word_embeddings": False,
        "max_position_embeddings": 4096, "rope_theta": 10000,
        "rope_scaling": {"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1,
                         "original_max_position_embeddings": 65536},
        "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2, "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc", "norm_topk_prob": True, "routed_scaling_factor": 1.5,
        "sliding_window": window, "compress_ratios": [0, 0, 2, 2, 2, 1, 1, 1, 1, 1],
        "compress_rope_theta": 160000, "kv_source_layer_ids": [2, 4, 5],
        "index_source_layer_ids": [2, 4, 5, 7, 9], "index_n_heads": 2, "index_head_dim": 128, "index_topk": 4,
        "candidate_source_layer_id": 5, "candidate_topk_blocks": 3, "candidate_block_size": 2,
        "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
        "engram_layer_ids": [1, 4], "engram_num_embeddings": [97, 101], "engram_max_ngram_size": 4,
        "engram_vocab_size": 128, "engram_n_heads": 2, "engram_head_dim": 32, "engram_pad_token_id": 2,
        "engram_compressed_vocab_size": 64, "num_nextn_predict_layers": 0,
    }
    return {"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "text_config": text}


def checkpoint_shapes(cfg: CedGeometry) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """``{checkpoint name: (shape, dtype)}`` of the BF16 backbone (decoded masters; Engram rows as BF16)."""
    H, D, Hd = cfg.num_attention_heads, cfg.head_dim, cfg.hidden_size
    S, E, I = cfg.hc_mult, cfg.n_routed_experts, cfg.moe_intermediate_size
    mix = (2 + S) * S
    out = {"embed.weight": ((cfg.vocab_size, Hd), BF16), "norm.weight": ((Hd,), BF16),
           "head.weight": ((cfg.vocab_size, Hd), BF16)}
    for k in cfg.kinds:
        p = f"layers.{k.layer}."
        for part in ("attn", "ffn"):
            out[p + f"hc_{part}_fn"] = ((mix, S * Hd), F32)
            out[p + f"hc_{part}_base"] = ((mix,), F32)
            out[p + f"hc_{part}_scale"] = ((3,), F32)
        out[p + "attn_norm.weight"] = ((Hd,), BF16)
        out[p + "ffn_norm.weight"] = ((Hd,), BF16)
        a = p + "attn."
        out[a + "attn_sink"] = ((H,), F32)
        out[a + "wq_a.weight"] = ((cfg.q_lora_rank, Hd), BF16)
        out[a + "q_norm.weight"] = ((cfg.q_lora_rank,), BF16)
        out[a + "wq_b.weight"] = ((H * D, cfg.q_lora_rank), BF16)
        out[a + "wkv.weight"] = ((D, Hd), BF16)
        out[a + "kv_norm.weight"] = ((D,), BF16)
        out[a + "wo_a.weight"] = ((cfg.o_groups * cfg.o_lora_rank, H * D // cfg.o_groups), BF16)
        out[a + "wo_b.weight"] = ((Hd, cfg.o_groups * cfg.o_lora_rank), BF16)
        if k.mode is Csa2Mode.FULL:
            out[a + "compressor.wkv.weight"] = ((D, Hd), BF16)
            if k.ratio > 1:
                out[a + "compressor.wgate.weight"] = ((D, Hd), BF16)
            out[a + "compressor.norm.weight"] = ((D,), BF16)
            out[a + "indexer.wk.weight"] = ((cfg.index_head_dim, D), BF16)
            out[a + "indexer.k_norm.weight"] = ((cfg.index_head_dim,), BF16)
        if k.mode in (Csa2Mode.FULL, Csa2Mode.REINDEX):
            out[a + "indexer.wq_b.weight"] = ((cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank), BF16)
            out[a + "indexer.weights_proj.weight"] = ((cfg.index_n_heads, Hd), BF16)
        f = p + "ffn."
        out[f + "gate.weight"] = ((E, Hd), BF16)
        out[f + "gate.bias"] = ((E,), F32)
        for e in [*map(str, range(E)), "shared"]:
            stem = f + ("shared_experts." if e == "shared" else f"experts.{e}.")
            out[stem + "w1.weight"] = ((I, Hd), BF16)
            out[stem + "w3.weight"] = ((I, Hd), BF16)
            out[stem + "w2.weight"] = ((Hd, I), BF16)
        if k.engram is not None:
            g = p + "engram."
            rows = cfg.engram_num_embeddings[cfg.engram_layer_ids.index(k.layer)]
            out[g + "embed.weight"] = ((rows, cfg.engram_head_dim), BF16)
            out[g + "wkv.weight"] = ((Hd * (S + 1), cfg.engram_width()), BF16)
            out[g + "q_weight"] = ((S, Hd), BF16)
            out[g + "k_weight"] = ((S, Hd), BF16)
    return out


def synthetic_params(cfg: CedGeometry, seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for name, (shape, dtype) in checkpoint_shapes(cfg).items():
        r = torch.randn(shape, generator=gen)
        if name.endswith(("norm.weight", "q_weight", "k_weight")):
            v = 1 + 0.1 * r
        elif name.endswith("_scale"):
            v = 0.5 + 0.1 * r
        elif name.endswith(("_base", "gate.bias")):
            v = 0.1 * r
        elif name.endswith("_fn"):
            v = r * (0.5 / math.sqrt(shape[-1]))
        elif name.endswith("attn_sink") or name.startswith("embed") or ".embed." in name:
            v = r
        else:
            v = r / math.sqrt(shape[-1])
        out[name] = v.to(dtype)
    return out


# ---------------------------------------------------------------------------------------------- components
class Reference:
    """Row-level components of one CED model (``cfg`` from ``ced_geometry(..., native=False)`` or real)."""

    def __init__(self, cfg: CedGeometry, params: dict[str, torch.Tensor], max_len: int = 1024):
        self.cfg, self.p = cfg, params
        self.W = cfg.replay_window
        rs = dict(cfg.rope_scaling)
        rd = cfg.qk_rope_head_dim
        self.tables = {
            "swa": rope_frequencies(rd, max_len, 0, cfg.rope_theta, rs["factor"], rs["beta_fast"], rs["beta_slow"]),
            "compressed": rope_frequencies(rd, max_len, rs["original_max_position_embeddings"],
                                           cfg.compress_rope_theta, rs["factor"], rs["beta_fast"],
                                           rs["beta_slow"]),
        }
        self.eps = cfg.rms_norm_eps
        cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
        gen = torch.Generator().manual_seed(10007)
        self.hash_mult = torch.randint(1, 1 << 20, (cols, cfg.engram_max_ngram_size), generator=gen)

    def P(self, layer: int, name: str) -> torch.Tensor:
        return self.p[f"layers.{layer}.{name}"]

    # --------------------------------------------------------------------------------- embed, mHC, head
    def embed(self, ids: torch.Tensor):
        x = self.p["embed.weight"][ids]
        X = x.unsqueeze(-2).expand(*x.shape[:-1], self.cfg.hc_mult, x.shape[-1]).contiguous()
        pre = torch.zeros(*x.shape[:-1], self.cfg.hc_mult, dtype=F32)
        pre[..., 0] = 1
        return X, pre

    def ngram_hashes(self, history: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``[R, cols]`` table rows of the n-grams ending at ``positions`` of one sequence's token ids
        ``history`` (positions before 0 read the pad id): orders 2..N, ``engram_n_heads`` heads each."""
        cfg = self.cfg
        N = cfg.engram_max_ngram_size
        back = positions.unsqueeze(-1) - torch.arange(N)                       # [R, N] newest first
        ids = torch.where(back >= 0, history[back.clamp(min=0)], cfg.engram_pad_token_id).long() + 1
        out, c = [], 0
        for n in range(2, N + 1):
            for _ in range(cfg.engram_n_heads):
                out.append((ids[:, :n] * self.hash_mult[c, :n]).sum(-1))
                c += 1
        return torch.stack(out, -1)

    def engram(self, L: int, X: torch.Tensor, hashes: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        rows = self.P(L, "engram.embed.weight")
        flat = rows[hashes % rows.shape[0]].flatten(-2)
        kv = lin(flat, self.P(L, "engram.wkv.weight"))
        S, H = cfg.hc_mult, cfg.hidden_size
        key, value = kv.split((S * H, H), dim=-1)
        key = key.float().unflatten(-1, (S, H))
        w = self.P(L, "engram.q_weight").float() * self.P(L, "engram.k_weight").float()
        h = X.float()
        rstd = torch.rsqrt(fsum(h * h) / H + self.eps) * torch.rsqrt(fsum(key * key) / H + self.eps)
        dot = fsum(h * w * key) * rstd * H ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(BF16)

    def mhc(self, L: int, part: str, X: torch.Tensor):
        cfg, n = self.cfg, self.cfg.hc_mult
        flat = X.flatten(-2).float()
        mixes = mm(flat, self.P(L, f"hc_{part}_fn")) * torch.rsqrt(fsum(flat * flat) / flat.shape[-1]
                                                                    + self.eps).unsqueeze(-1)
        scale, base, eps = self.P(L, f"hc_{part}_scale"), self.P(L, f"hc_{part}_base"), cfg.hc_eps
        pre = torch.sigmoid(mixes[..., :n] * scale[0] + base[:n]) + eps
        post = 2 * torch.sigmoid(mixes[..., n:2 * n] * scale[1] + base[n:2 * n])
        comb = (mixes[..., 2 * n:] * scale[2] + base[2 * n:]).unflatten(-1, (n, n))
        e = torch.exp(comb - comb.amax(-1, keepdim=True))
        comb = e / fsum(e).unsqueeze(-1) + eps
        comb = comb / (fsum(comb, -2).unsqueeze(-2) + eps)
        for _ in range(cfg.hc_sinkhorn_iters - 1):
            comb = comb / (fsum(comb).unsqueeze(-1) + eps)
            comb = comb / (fsum(comb, -2).unsqueeze(-2) + eps)
        return pre, post, comb

    @staticmethod
    def collapse(X, pre):
        acc = pre[..., 0:1] * X[..., 0, :].float()
        for s in range(1, X.shape[-2]):
            acc = acc + pre[..., s:s + 1] * X[..., s, :].float()
        return acc.to(BF16)

    @staticmethod
    def expand(out, X, post, comb):
        t1 = post.unsqueeze(-1) * out.float().unsqueeze(-2)
        acc = comb[..., 0, :].unsqueeze(-1) * X[..., 0, :].float().unsqueeze(-2)
        for s in range(1, X.shape[-2]):
            acc = acc + comb[..., s, :].unsqueeze(-1) * X[..., s, :].float().unsqueeze(-2)
        return (t1 + acc).to(BF16)

    def head(self, X, pre):
        h = rms(self.collapse(X, pre), self.p["norm.weight"], self.eps)
        return mm(h, self.p["head.weight"])

    @staticmethod
    def logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        m = logits.amax(-1, keepdim=True)
        lse = m.squeeze(-1) + torch.log(fsum(torch.exp(logits - m)))
        return torch.gather(logits, -1, targets.unsqueeze(-1)).squeeze(-1) - lse

    # --------------------------------------------------------------------------------- block halves
    def attn_boundary(self, L, X, pre_prev, hashes=None):
        """Engram (if any), the attention mHC coefficients and the attention input ``h``."""
        if self.cfg.kind(L).engram is not None:
            X = self.engram(L, X, hashes)
        pre_a, post_a, comb_a = self.mhc(L, "attn", X)
        h = rms(self.collapse(X, pre_prev), self.P(L, "attn_norm.weight"), self.eps)
        return X, pre_a, post_a, comb_a, h

    def block_rest(self, L, X, pre_a, post_a, comb_a, a):
        """Attention expand, the FFN boundary, the MoE and its expand: ``(X, pre_f)``."""
        X = self.expand(a, X, post_a, comb_a)
        pre_f, post_f, comb_f = self.mhc(L, "ffn", X)
        h = rms(self.collapse(X, pre_a), self.P(L, "ffn_norm.weight"), self.eps)
        return self.expand(self.moe(L, h), X, post_f, comb_f), pre_f

    # --------------------------------------------------------------------------------- attention parts
    def table(self, L):
        t = self.tables[self.cfg.kind(L).rope]
        return t.real.contiguous(), t.imag.contiguous()

    def rope(self, L, x, pos, *, inverse=False, round_bf16=True):
        cos, sin = self.table(L)
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        p = pos.reshape(-1, 1).expand(-1, flat.shape[0] // max(pos.numel(), 1)).reshape(-1) if x.ndim > 2 else pos
        y = rope_emul(flat, cos, sin, p, inverse=inverse, round_bf16=round_bf16)
        return y.reshape(shape)

    def query(self, L, h, pos):
        cfg = self.cfg
        qr = rms(lin(h, self.P(L, "attn.wq_a.weight")), self.P(L, "attn.q_norm.weight"), self.eps)
        q = lin(qr, self.P(L, "attn.wq_b.weight")).unflatten(-1, (cfg.num_attention_heads, cfg.head_dim))
        return self.rope(L, q, pos), qr

    def window_keys(self, L, h, pos):
        """The layer's window-KV records, dequantized (CACHE-SPEC section 1)."""
        kv = rms(lin(h, self.P(L, "attn.wkv.weight")), self.P(L, "attn.kv_norm.weight"), self.eps)
        return qdq(self.rope(L, kv, pos), 32)

    def compressor_rows(self, L, h):
        """Ratio 1: the latent per token. Ratio > 1: the FP32 ``(kv, score)`` rows pooled per group."""
        if self.cfg.kind(L).ratio == 1:
            return rms(lin(h, self.P(L, "attn.compressor.wkv.weight")), self.P(L, "attn.compressor.norm.weight"),
                       self.eps)
        x = h.float()
        return mm(x, self.P(L, "attn.compressor.wkv.weight")), mm(x, self.P(L, "attn.compressor.wgate.weight"))

    def pool(self, L, kv, score):
        """One complete group ``kv, score [r, D]`` -> latent ``[D]`` (per-channel softmax over the group)."""
        e = torch.exp(score - score.amax(0, keepdim=True))
        p = e / fsum(e, 0).unsqueeze(0)
        return rms(fsum(kv * p, 0).to(BF16), self.P(L, "attn.compressor.norm.weight"), self.eps)

    def compressed_entries(self, L, latent, first_pos):
        """Compressed-KV records (CACHE-SPEC section 2) and index keys (section 3) of latents whose groups
        start at ``first_pos``."""
        kv = qdq(self.rope(L, latent, first_pos, round_bf16=False), 32, "log2f")
        k = rms(lin(latent, self.P(L, "attn.indexer.wk.weight")), self.P(L, "attn.indexer.k_norm.weight"),
                self.eps)
        return kv, qdq(self.rope(L, k, first_pos), 128)

    def index_scores(self, L, h, qr, pos, keys):
        """``[R, C]`` scores of query rows against index keys ``[C, 128]``, ``-inf`` where not yet visible."""
        cfg, ratio = self.cfg, self.cfg.kind(L).ratio
        q = lin(qr, self.P(L, "attn.indexer.wq_b.weight")).unflatten(-1, (cfg.index_n_heads, cfg.index_head_dim))
        q = qdq(self.rope(L, q, pos), 128)          # FP8 queries (decided); the indexer has no gradient
        w = lin(h, self.P(L, "attn.indexer.weights_proj.weight")).float() * (
            cfg.index_head_dim ** -0.5 * cfg.index_n_heads ** -0.5)
        dots = fsum(q.float().unsqueeze(-2) * keys.float().unsqueeze(-3).unsqueeze(-3))        # [R, h, C]
        s = fsum(torch.relu(dots) * w.unsqueeze(-1), -2)                                       # [R, C]
        visible = torch.arange(keys.shape[0]).unsqueeze(0) < ((pos + 1) // ratio).unsqueeze(-1)
        return torch.where(visible, s, NEG)

    def candidates(self, scores, pos, ratio):
        """Hierarchical Sparse Indexer level one: the pool ``[R, C]`` of the top blocks (newest pinned)."""
        cfg = self.cfg
        B, C = cfg.candidate_block_size, scores.shape[-1]
        nb = -(-C // B)
        blocks = torch.nn.functional.pad(scores, (0, nb * B - C), value=NEG).unflatten(-1, (nb, B)).amax(-1)
        last = ((pos + 1) // ratio - 1) // B
        blocks = torch.where(torch.arange(nb).unsqueeze(0) == last.unsqueeze(-1), float("inf"), blocks)
        top = stable_top(blocks, min(cfg.candidate_topk_blocks, nb))
        keep = torch.zeros((scores.shape[0], nb + 1), dtype=torch.bool)
        keep.scatter_(-1, torch.where(top >= 0, top, nb), True)
        return keep[:, :nb].repeat_interleave(B, -1)[:, :C]

    def topk(self, scores):
        picked = stable_top(scores, self.cfg.index_topk)
        return torch.where(picked >= 0, torch.sort(torch.where(picked >= 0, picked, 1 << 40), -1).values, -1)

    def attend(self, L, q, keys, valid, pos):
        """Sparse sink attention over ``keys [R, S, D]`` (window slots then compressed slots), inverse RoPE,
        grouped ``wo_a``, ``wo_b``."""
        cfg = self.cfg
        keys = torch.where(valid.unsqueeze(-1), keys, torch.zeros((), dtype=keys.dtype))
        kf = keys.float()
        s = fsum(q.float().unsqueeze(-2) * kf.unsqueeze(-3)) * cfg.head_dim ** -0.5         # [R, H, S]
        s = torch.where(valid.unsqueeze(-2), s, NEG)
        sink = self.P(L, "attn.attn_sink")
        m = torch.maximum(s.amax(-1), sink)
        e = torch.exp(s - m.unsqueeze(-1))
        p = e / (fsum(e) + torch.exp(sink - m)).unsqueeze(-1)
        o = fsum(p.unsqueeze(-1) * kf.unsqueeze(-3), -2).to(BF16)                              # [R, H, D]
        o = self.rope(L, o, pos, inverse=True)
        G = cfg.o_groups
        og = o.reshape(o.shape[0], G, -1)
        wa = self.P(L, "attn.wo_a.weight").view(G, cfg.o_lora_rank, -1)
        a = fsum(og.float().unsqueeze(-2) * wa.float()).to(BF16).flatten(1)
        return lin(a, self.P(L, "attn.wo_b.weight"))

    # --------------------------------------------------------------------------------- MoE
    def moe(self, L, h):
        """``MOE_RECIPE``: sqrt-softplus scores, biased selection, unbiased normalized weights x scale; each
        selected expert's weight multiplied into its activation before FC2; FP32 add chain in ascending expert
        id, the shared expert last, one BF16 rounding."""
        cfg = self.cfg
        s = torch.sqrt(torch.nn.functional.softplus(mm(h, self.P(L, "ffn.gate.weight"))))
        sel = stable_top(s + self.P(L, "ffn.gate.bias"), cfg.num_experts_per_tok)
        w = torch.gather(s, -1, sel)
        w = w / (fsum(w) + 1e-20).unsqueeze(-1) * cfg.routed_scaling_factor
        y = torch.zeros(h.shape, dtype=F32)
        for e in range(cfg.n_routed_experts):
            hit = sel == e
            we = torch.where(hit, w, 0.0).sum(-1)        # at most one hit per row: exact
            out = self.expert(L, f"ffn.experts.{e}.", h, we)
            y = torch.where(hit.any(-1, keepdim=True), y + out.float(), y)
        return (y + self.expert(L, "ffn.shared_experts.", h, None).float()).to(BF16)

    def expert(self, L, stem, h, weight):
        lim = self.cfg.swiglu_limit
        gate = lin(h, self.P(L, stem + "w1.weight")).float().clamp(max=lim)
        up = lin(h, self.P(L, stem + "w3.weight")).float().clamp(-lim, lim)
        a = torch.nn.functional.silu(gate) * up
        if weight is not None:
            a = weight.unsqueeze(-1) * a
        return lin(a.to(BF16), self.P(L, stem + "w2.weight"))


# ---------------------------------------------------------------------------------------------- helpers
def _window(rows_of_pos, pos, floor, W):
    """``(index [R, W], valid [R, W])`` of window slots: slot j of a row at position p is position p-W+1+j,
    valid when >= max(floor, 0) and stored (``rows_of_pos(q)`` maps positions to storage rows)."""
    q = pos.unsqueeze(-1) - W + 1 + torch.arange(W)
    valid = q >= torch.as_tensor(floor).clamp(min=0).reshape(-1, 1)
    idx = rows_of_pos(q.clamp(min=0))
    return torch.where(valid, idx, 0), valid


def _compressed(entries, topk):
    valid = topk >= 0
    if entries.shape[0] == 0:            # no complete group yet: every slot is empty
        entries = torch.zeros((1, entries.shape[-1]), dtype=entries.dtype)
    return entries[torch.where(valid, topk, 0)], valid


def _sequence_entries(ref, L, h, pos):
    """All compressed entries of one sequence (positions ``pos`` = 0..T-1) of KV-source layer L."""
    k = ref.cfg.kind(L)
    rows = ref.compressor_rows(L, h)
    if k.ratio == 1:
        return ref.compressed_entries(L, rows, pos)
    kv, score = rows
    C = h.shape[0] // k.ratio
    if C == 0:
        D = ref.cfg.head_dim
        return torch.zeros((0, D), dtype=BF16), torch.zeros((0, ref.cfg.index_head_dim), dtype=BF16)
    lat = torch.stack([ref.pool(L, kv[j * k.ratio:(j + 1) * k.ratio], score[j * k.ratio:(j + 1) * k.ratio])
                       for j in range(C)])
    return ref.compressed_entries(L, lat, torch.arange(C) * k.ratio)


# ---------------------------------------------------------------------------------------------- flat program
def flat_forward(ref: Reference, ids: torch.Tensor, *, decoder_window_floor: int = 0) -> torch.Tensor:
    """The reference code's flat program on ONE sequence: every layer on every token, shared slots
    (``compress_kv``, ``index_k``, ``topk_idxs``, ``candidates``) keyed by the config's source lists, the
    layers after the last KV source truncating their window at ``decoder_window_floor``. Returns FP32 logits
    ``[T, V]``."""
    cfg, W = ref.cfg, ref.W
    T = ids.shape[0]
    pos = torch.arange(T)
    X, pre = ref.embed(ids)
    kv_src, ix_src, cand_src = cfg.kv_source_layer_ids, cfg.index_source_layer_ids, cfg.candidate_source_layer_id
    last_kv = max(kv_src)
    shared: dict = {}
    for L in range(cfg.num_hidden_layers):
        r = cfg.compress_ratios[L]
        hashes = ref.ngram_hashes(ids, pos) if L in cfg.engram_layer_ids else None
        X, pre_a, post_a, comb_a, h = ref.attn_boundary(L, X, pre, hashes)
        q, qr = ref.query(L, h, pos)
        K = ref.window_keys(L, h, pos)
        widx, wvalid = _window(lambda p: p, pos, decoder_window_floor if L > last_kv else 0, W)
        keys, valid = K[widx], wvalid
        if r:
            if L in kv_src:
                shared["compress_kv"], shared["index_k"] = _sequence_entries(ref, L, h, pos)
            if L in ix_src:
                scores = ref.index_scores(L, h, qr, pos, shared["index_k"])
                if L == cand_src:
                    shared["candidates"] = ref.candidates(scores, pos, r)
                elif cand_src < L:
                    scores = torch.where(shared["candidates"], scores, NEG)
                shared["topk_idxs"] = ref.topk(scores)
            ckeys, cvalid = _compressed(shared["compress_kv"], shared["topk_idxs"])
            keys, valid = torch.cat([keys, ckeys], 1), torch.cat([valid, cvalid], 1)
        X, pre = ref.block_rest(L, X, pre_a, post_a, comb_a, ref.attend(L, q, keys, valid, pos))
    return ref.head(X, pre)


# ---------------------------------------------------------------------------------------------- CED program
@dataclass
class CedTrace:
    """What ``ced_forward`` exposes besides logits: the handoff, the decoder global KV, per-layer decoder
    window keys at the decoder rows, and the decoder rows themselves."""

    handoff: EncoderOutput
    decoder_kv: dict
    decoder_window: dict = field(default_factory=dict)
    segments: DecoderSegments | None = None


def ced_forward(ref: Reference, sequences: Sequence[torch.Tensor], segments: DecoderSegments,
                ) -> tuple[torch.Tensor, CedTrace]:
    """The CED definition on a packed batch: logits ``[R, V]`` FP32 at the decoder rows of ``segments``."""
    cfg, W, b = ref.cfg, ref.W, ref.cfg.boundary
    cu = [0]
    for s in sequences:
        cu.append(cu[-1] + s.shape[0])
    if tuple(cu) != segments.cu_seqlens:
        raise ValueError("segments describe another packing")
    ids = torch.cat(list(sequences))
    seq_of = torch.cat([torch.full((s.shape[0],), i) for i, s in enumerate(sequences)])
    pos = torch.cat([torch.arange(s.shape[0]) for s in sequences])
    base = torch.tensor(cu[:-1])
    S = len(sequences)

    def packed(p, sq):
        return base[sq].reshape(*([-1] + [1] * (p.ndim - 1))) + p

    def per_seq(fn, *tensors):
        return [fn(i, *(t[cu[i]:cu[i + 1]] for t in tensors)) for i in range(S)]

    # ---- causal encoder: every token
    X, pre = ref.embed(ids)
    shared: dict = {}
    for k in cfg.encoder:
        L = k.layer
        hashes = (torch.cat(per_seq(lambda i, p: ref.ngram_hashes(sequences[i], p), pos))
                  if k.engram is not None else None)
        X, pre_a, post_a, comb_a, h = ref.attn_boundary(L, X, pre, hashes)
        q, qr = ref.query(L, h, pos)
        K = ref.window_keys(L, h, pos)
        widx, valid = _window(lambda p: packed(p, seq_of), pos, 0, W)
        keys = K[widx]
        if k.ratio:
            if k.mode is Csa2Mode.FULL:
                shared[L] = per_seq(lambda i, hh, p, L=L: _sequence_entries(ref, L, hh, p), h, pos)
            if k.index_source == L:
                shared[("topk", L)] = torch.cat(per_seq(
                    lambda i, hh, qq, p, L=L, src=k.kv_source: ref.topk(
                        ref.index_scores(L, hh, qq, p, shared[src][i][1])),
                    h, qr, pos))
            topk = shared[("topk", k.index_source)]
            ck = torch.cat([_compressed(shared[k.kv_source][int(seq_of[r])][0], topk[r:r + 1])[0]
                            for r in range(ids.shape[0])])
            keys, valid = torch.cat([keys, ck], 1), torch.cat([valid, topk >= 0], 1)
        X, pre = ref.block_rest(L, X, pre_a, post_a, comb_a, ref.attend(L, q, keys, valid, pos))
    handoff = EncoderOutput(streams=X, pre=pre)

    # ---- decoder.entry: the global-KV projection of the boundary layer, every token
    Xb, pre_a, post_a, comb_a, hb = ref.attn_boundary(b, handoff.streams, handoff.pre)
    Kb = ref.window_keys(b, hb, pos)
    entries = per_seq(lambda i, hh, p: _sequence_entries(ref, b, hh, p), hb, pos)
    decoder_kv = {"main_kv": [e[0] for e in entries], "index_k": [e[1] for e in entries], "window_kv": Kb,
                  "rows": {"streams": Xb, "pre_a": pre_a, "post_a": post_a, "comb_a": comb_a, "h": hb}}

    # ---- segment gather
    g = segments
    rows, dpos, dseq = g.rows, g.positions, g.sequence
    X = Xb[rows]
    trace = CedTrace(handoff, decoder_kv, segments=g)
    pool: list = []
    topks: dict = {}
    pre = None
    seg_window = g.window_rows(W)
    for k in cfg.decoder:
        L = k.layer
        if L == b:
            pa, poa, ca, h = pre_a[rows], post_a[rows], comb_a[rows], hb[rows]
            widx, valid = _window(lambda p: packed(p, dseq), dpos, 0, W)     # unclamped: token-wise keys
            keys = Kb[widx]
        else:
            X, pa, poa, ca, h = ref.attn_boundary(L, X, pre)
            K = ref.window_keys(L, h, dpos)
            valid = seg_window >= 0
            keys = K[seg_window.clamp(min=0)]
            trace.decoder_window[L] = K
        q, qr = ref.query(L, h, dpos)
        if k.index_source == L:
            # rows of different sequences see different key counts: one score row each (rows independent)
            scores = [ref.index_scores(L, h[r:r + 1], qr[r:r + 1], dpos[r:r + 1],
                                       decoder_kv["index_k"][int(dseq[r])]) for r in range(len(g))]
            if k.candidate == "select":
                pool = [ref.candidates(sc, dpos[r:r + 1], 1) for r, sc in enumerate(scores)]
            elif k.candidate == "mask":
                scores = [torch.where(pool[r], sc, NEG) for r, sc in enumerate(scores)]
            topks[L] = torch.cat([ref.topk(sc) for sc in scores])
        topk = topks[k.index_source]
        ck = torch.cat([_compressed(decoder_kv["main_kv"][int(dseq[r])], topk[r:r + 1])[0] for r in range(len(g))])
        keys, valid = torch.cat([keys, ck], 1), torch.cat([valid, topk >= 0], 1)
        X, pre = ref.block_rest(L, X, pa, poa, ca, ref.attend(L, q, keys, valid, dpos))
    return ref.head(X, pre), trace


# ---------------------------------------------------------------------------------------------- engine
@dataclass
class RequestState:
    """Per-request engine state (what crosses calls), one entry per state class of the definition."""

    ids: list = field(default_factory=list)          # token history (Engram n-grams, positions)
    window: dict = field(default_factory=dict)       # encoder layers + boundary: {L: [cap, D]} by position
    decoder_window: dict = field(default_factory=dict)  # decoder layers > b: segment-scoped {L: [cap, D]}
    main_kv: dict = field(default_factory=dict)      # {L: [entries]} encoder FULL layers and the boundary
    index_k: dict = field(default_factory=dict)
    pending: dict = field(default_factory=dict)      # ratio-2 compressor's incomplete group {L: (kv, score)}
    segment_start: int = 0
    prompt_end: int = 0


class ReferenceEngine:
    """Engine form of the CED program: ``prefill(prompt, chunks)`` then ``decode(token)``; the decoder rows of
    a chunk follow the prompt-length rule of ``mode``."""

    def __init__(self, ref: Reference, mode: str, capacity: int = 1024):
        if mode not in (BOUNDED_REPLAY, FULL_DECODER):
            raise ValueError(mode)
        self.ref, self.mode, self.cap = ref, mode, capacity

    def new_request(self, prompt_end: int) -> RequestState:
        st = RequestState(prompt_end=prompt_end)
        st.segment_start = int(segment_starts([prompt_end], self.ref.W, self.mode)[0])
        return st

    def prefill(self, st: RequestState, prompt: torch.Tensor, chunks: Sequence[int]) -> torch.Tensor:
        """Run ``prompt`` in chunks of the given sizes; returns the logits of the last prompt position."""
        if sum(chunks) != prompt.shape[0] or prompt.shape[0] < st.prompt_end:
            raise ValueError("chunks must cover the prompt (a recomputed request also feeds its outputs)")
        logits, a = None, 0
        for n in chunks:
            logits = self.step(st, prompt[a:a + n])
            a += n
        return logits

    def decode(self, st: RequestState, token: int) -> torch.Tensor:
        return self.step(st, torch.tensor([token]))

    def step(self, st: RequestState, new: torch.Tensor):
        ref, cfg, W, b = self.ref, self.ref.cfg, self.ref.W, self.ref.cfg.boundary
        a = len(st.ids)
        st.ids += [int(t) for t in new]
        c = len(st.ids)
        history = torch.tensor(st.ids)
        pos = torch.arange(a, c)
        X, pre = ref.embed(new)
        D = cfg.head_dim
        enc_topk: dict = {}                              # per forward: never crosses calls
        for k in cfg.encoder:
            L = k.layer
            hashes = ref.ngram_hashes(history, pos) if k.engram is not None else None
            X, pre_a, post_a, comb_a, h = ref.attn_boundary(L, X, pre, hashes)
            q, qr = ref.query(L, h, pos)
            win = st.window.setdefault(L, torch.zeros((self.cap, D), dtype=BF16))
            win[a:c] = ref.window_keys(L, h, pos)
            widx, valid = _window(lambda p: p, pos, 0, W)
            keys = win[widx]
            if k.ratio:
                if k.mode is Csa2Mode.FULL:
                    self._compress(st, L, h, pos)
                if k.index_source == L:
                    enc_topk[L] = ref.topk(
                        ref.index_scores(L, h, qr, pos, self._stack(st.index_k, L, cfg.index_head_dim)))
                topk = enc_topk[k.index_source]
                ck, cv = _compressed(self._stack(st.main_kv, k.kv_source, D), topk)
                keys, valid = torch.cat([keys, ck], 1), torch.cat([valid, cv], 1)
            X, pre = ref.block_rest(L, X, pre_a, post_a, comb_a, ref.attend(L, q, keys, valid, pos))
        # decoder.entry on every new token
        Xb, pre_a, post_a, comb_a, hb = ref.attn_boundary(b, X, pre)
        win = st.window.setdefault(b, torch.zeros((self.cap, D), dtype=BF16))
        win[a:c] = ref.window_keys(b, hb, pos)
        self._compress(st, b, hb, pos)
        # decoder rows of this call
        plan = plan_step([st.prompt_end], [a], [c - a], window=W, mode=self.mode)
        if plan.count == 0:
            return None
        sel = torch.from_numpy(plan.rows)
        X, dpos = Xb[sel], pos[sel]
        pool, topks, pre = None, {}, None
        for k in cfg.decoder:
            L = k.layer
            if L == b:
                pa, poa, ca, h = pre_a[sel], post_a[sel], comb_a[sel], hb[sel]
                widx, valid = _window(lambda p: p, dpos, 0, W)
                keys = st.window[b][widx]
            else:
                X, pa, poa, ca, h = ref.attn_boundary(L, X, pre)
                K = ref.window_keys(L, h, dpos)
                dw = st.decoder_window.setdefault(L, torch.zeros((self.cap, D), dtype=BF16))
                dw[dpos] = K
                widx, valid = _window(lambda p: p, dpos, st.segment_start, W)
                keys = dw[widx]
            q, qr = ref.query(L, h, dpos)
            if k.index_source == L:
                scores = ref.index_scores(L, h, qr, dpos, self._stack(st.index_k, b, cfg.index_head_dim))
                if k.candidate == "select":
                    pool = ref.candidates(scores, dpos, 1)
                elif k.candidate == "mask":
                    scores = torch.where(pool, scores, NEG)
                topks[L] = ref.topk(scores)
            ck, cv = _compressed(self._stack(st.main_kv, b, D), topks[k.index_source])
            keys, valid = torch.cat([keys, ck], 1), torch.cat([valid, cv], 1)
            X, pre = ref.block_rest(L, X, pa, poa, ca, ref.attend(L, q, keys, valid, dpos))
        return ref.head(X[-1:], pre[-1:])[0]

    @staticmethod
    def _stack(store, L, width):
        items = store.get(L, [])
        return torch.stack(items) if items else torch.zeros((0, width), dtype=BF16)

    def _compress(self, st, L, h, pos):
        """Write the entries of every group completed by these rows (a ratio-2 group may span calls)."""
        ref = self.ref
        k = ref.cfg.kind(L)
        rows = ref.compressor_rows(L, h)
        if k.ratio == 1:
            kv, ik = ref.compressed_entries(L, rows, pos)
            st.main_kv.setdefault(L, []).extend(kv)
            st.index_k.setdefault(L, []).extend(ik)
            return
        kv, score = rows
        for i, p in enumerate(pos.tolist()):
            group = st.pending.setdefault(L, [])
            group.append((kv[i], score[i]))
            if (p + 1) % k.ratio == 0:
                latent = ref.pool(L, torch.stack([g[0] for g in group]), torch.stack([g[1] for g in group]))
                e_kv, e_ik = ref.compressed_entries(L, latent.unsqueeze(0), torch.tensor([p + 1 - k.ratio]))
                st.main_kv.setdefault(L, []).append(e_kv[0])
                st.index_k.setdefault(L, []).append(e_ik[0])
                st.pending[L] = []


# ---------------------------------------------------------------------------------------------- scoring
def rollout(engine: ReferenceEngine, prompt: torch.Tensor, generated: torch.Tensor, chunks: Sequence[int]):
    """Teacher-forced generation: logits predicting each generated token ``[G, V]`` and the request state."""
    st = engine.new_request(prompt.shape[0])
    out = [engine.prefill(st, prompt, chunks)]
    for t in generated[:-1].tolist():
        out.append(engine.decode(st, t))
    return torch.stack(out), st


def trainer_logits(ref: Reference, sequences: Sequence[torch.Tensor], turns: Sequence[Sequence[Turn]],
                   mode: str):
    """Trainer scoring of a packed batch: the logits at every scored decoder row, per segment, in order, with
    the target tokens. The segments come from the rollout's turns (``DecoderSegments.for_program``)."""
    cu = [0]
    for s in sequences:
        cu.append(cu[-1] + s.shape[0])
    segs = DecoderSegments.for_program(mode, cu, turns, ref.W)
    logits, trace = ced_forward(ref, sequences, segs)
    ids = torch.cat(list(sequences))
    rows = torch.nonzero(segs.score).squeeze(-1)
    return logits[rows], ids[segs.rows[rows] + 1], segs, trace


def tiny(seed: int = 0, window: int = 8) -> Reference:
    cfg = ced_geometry(synthetic_config(window=window), native=False)
    return Reference(cfg, synthetic_params(cfg, seed))


__all__ = ["CedTrace", "Reference", "ReferenceEngine", "RequestState", "ced_forward", "checkpoint_shapes",
           "flat_forward", "fsum", "rollout", "synthetic_config", "synthetic_params", "tiny", "trainer_logits"]
