"""Deterministic test inputs shared by the DSA batteries (and by any external baseline comparison).

It stays pure ``torch`` and does not import ``isoexec``, so an interpreter with another torch build can
produce the same inputs.  The random stream is a splitmix64 hash of
the element index and the seed, evaluated with int64 tensor ops, so every interpreter builds the SAME
bytes from the same ``(seed, shape)`` regardless of its torch version's RNG -- which is what lets a
baseline's output be compared against the candidate's inside one rotation.

Layouts (the ones the baselines accept, so that data movement is identical on both sides):

* sparse attention: ``q`` bf16 ``[B, 1, h_q, 512]``; the V4.1 fp8 cache ``[num_pages, 64, 1, 528]``
  (512 e4m3 then 16 ue8m0 per token, FlashMLA ``tests/quant.py``); ``block_table`` int32
  ``[B, pages_per_req]`` (a random permutation of the pages, so pages of one request are scattered);
  logical lists ``idx`` int32 ``[B, 1, K]`` ascending, -1 padded; FlashMLA's physical indices are
  ``block_table[idx // 64] * 64 + idx % 64``.
* indexer: DeepGEMM's paged FP8 cache ``[num_pages, 64, 1, 132]`` uint8 -- per page 64 x 128 e4m3
  then 64 packed int32 ue8m0 words (four equal bytes: one exponent per key); q e4m3 ``[B, 1, 64, 128]`` + packed int32 scales ``[B, 1, 64]``;
  weights bf16 ``[B, 64]`` (row stride padded to 16 bytes).
* top-k: bf16 logits ``[rows, stride]``; vLLM takes the same values as fp32.
"""

from __future__ import annotations

import math

import torch

PAGE = 64
D_QK = 512
KV_BYTES = 528
IDX_D = 128
IDX_BYTES = 132

_M1 = 0x9E3779B97F4A7C15 - (1 << 64)   # as signed int64
_M2 = 0xBF58476D1CE4E5B9 - (1 << 64)
_M3 = 0x94D049BB133111EB - (1 << 64)


def _srl(x: torch.Tensor, k: int) -> torch.Tensor:
    """Logical right shift of an int64 tensor (torch's >> is arithmetic)."""
    return (x >> k) & ((1 << (64 - k)) - 1)


def hash64(seed: int, n: int, device="cuda", offset: int = 0) -> torch.Tensor:
    """splitmix64(seed * golden + index) for index in [offset, offset + n), int64."""
    x = torch.arange(offset, offset + n, dtype=torch.int64, device=device)
    x = x * _M1 + (seed * 0x5851F42D + 0x1234567) % (1 << 62)
    x = (x ^ _srl(x, 30)) * _M2
    x = (x ^ _srl(x, 27)) * _M3
    return x ^ _srl(x, 31)


def uniform(seed: int, n: int, device="cuda", offset: int = 0) -> torch.Tensor:
    """float32 in [0, 1) from the top 24 bits."""
    return _srl(hash64(seed, n, device, offset), 40).to(torch.float32) * (1.0 / (1 << 24))


def normal(seed: int, n: int, device="cuda") -> torch.Tensor:
    """float32 approximately N(0, 1) (Box-Muller on two hashed streams)."""
    u1 = uniform(seed, n, device).clamp_min(1e-7)
    u2 = uniform(seed + 7919, n, device)
    return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)


def rand_bytes(seed: int, n: int, device="cuda", offset: int = 0) -> torch.Tensor:
    return (_srl(hash64(seed, n, device, offset), 56)).to(torch.uint8)


# ------------------------------------------------------------------------------------------------
# sparse attention
# ------------------------------------------------------------------------------------------------

def page_table(seed: int, batch: int, pages_per_req: int, device="cuda") -> torch.Tensor:
    """A random permutation of ``batch * pages_per_req`` pages, reshaped per request."""
    n = batch * pages_per_req
    order = torch.argsort(uniform(seed, n, device))
    return order.to(torch.int32).view(batch, pages_per_req).contiguous()


def fp8_cache_random(seed: int, num_pages: int, device="cuda", scale_exp: int = -12,
                     chunk_pages: int = 4096) -> torch.Tensor:
    """A V4.1 fp8 cache of random finite e4m3 bytes with one fixed ue8m0 scale per 32-block.

    For TIMING rows only: values are e4m3 random in [-448, 448] times 2^scale_exp (|x| <= 0.11 at the
    default), NaN codes (0x7F / 0xFF) remapped to their neighbour.  Built in page chunks so a 69 GB
    cache (B = 1024, ctx = 128K) needs no full-size temporaries."""
    cache = torch.empty((num_pages, PAGE, 1, KV_BYTES), dtype=torch.uint8, device=device)
    flat = cache.view(num_pages, PAGE, KV_BYTES)
    for p0 in range(0, num_pages, chunk_pages):
        p1 = min(p0 + chunk_pages, num_pages)
        n = (p1 - p0) * PAGE * D_QK
        b = rand_bytes(seed, n, device, offset=p0 * PAGE * D_QK).view(p1 - p0, PAGE, D_QK)
        b = torch.where((b & 0x7F) == 0x7F, b ^ 0x01, b)
        flat[p0:p1, :, :D_QK] = b
        flat[p0:p1, :, D_QK:] = 127 + scale_exp
    return cache


def quantize_v41(x: torch.Tensor) -> torch.Tensor:
    """bf16 ``[num_pages, 64, 512]`` -> V4.1 fp8 cache ``[num_pages, 64, 1, 528]`` uint8.

    The V4.1 window-record quantizer (vLLM 0.33 FLASHMLA_SPARSE_DSV41 on SM100; port/CACHE-SPEC.md section 1): per
    32-block scale = 2^ceil(log2(max(amax, 1e-4) * fl(1/448))) (exact ceiling: the input is BF16), e4m3 = x / scale
    (RNE; |x / scale| <= 448)."""
    n_pages, page, d = x.shape
    out = torch.empty((n_pages, page, KV_BYTES), dtype=torch.uint8, device=x.device)
    inv448 = torch.tensor(1.0 / 448.0, dtype=torch.float32, device=x.device)
    for t in range(D_QK // 32):
        blk = x[..., t * 32:(t + 1) * 32].float()
        bits = (blk.abs().amax(dim=-1).clamp_min(1e-4) * inv448).view(torch.int32)
        e = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0).int()
        s = (e << 23).view(torch.float32)
        q = (blk / s.unsqueeze(-1)).to(torch.float8_e4m3fn)
        out[..., t * 32:(t + 1) * 32] = q.view(torch.uint8)
        out[..., D_QK + t] = e.to(torch.uint8)
    return out.view(n_pages, page, 1, KV_BYTES)


def dequantize_v41(cache: torch.Tensor) -> torch.Tensor:
    """V4.1 fp8 cache -> bf16 ``[num_pages, 64, 512]``: ``bf16(float(e4m3) * 2^(e8m0 - 127))``.

    The declared conversion point of SPEC section 4.2 (exact in bf16 for every finite in-range code:
    e4m3 has 3 mantissa bits and the scale is a power of two)."""
    n_pages = cache.shape[0]
    c = cache.view(n_pages, PAGE, KV_BYTES)
    vals = c[..., :D_QK].contiguous().view(torch.float8_e4m3fn).float()
    exp = c[..., D_QK:].to(torch.int32) - 127
    scale = torch.pow(2.0, exp.float()).repeat_interleave(32, dim=-1)
    sc = torch.where((c[..., D_QK:] == 255).repeat_interleave(32, dim=-1),
                     torch.full_like(scale, float("nan")), scale)
    return (vals * sc).to(torch.bfloat16)


def sparse_lists(seed: int, batch: int, ctx: int, topk: int, device="cuda",
                 valid: torch.Tensor | None = None) -> torch.Tensor:
    """Ascending logical lists ``[batch, 1, topk]``: ``min(topk, ctx)`` distinct positions of
    ``[0, ctx)`` per request (a uniform random subset, sorted), -1 padded."""
    n = min(topk, ctx)
    out = torch.full((batch, topk), -1, dtype=torch.int32, device=device)
    rows_per_chunk = max(1, (1 << 27) // max(ctx, 1))
    for b0 in range(0, batch, rows_per_chunk):
        b1 = min(b0 + rows_per_chunk, batch)
        u = uniform(seed, (b1 - b0) * ctx, device, offset=b0 * ctx).view(b1 - b0, ctx)
        pick = torch.topk(u, n, dim=-1, sorted=False).indices
        out[b0:b1, :n] = torch.sort(pick, dim=-1).values.to(torch.int32)
    if valid is not None:
        pos = torch.arange(topk, device=device).unsqueeze(0)
        out = torch.where(pos < valid.view(-1, 1), out, torch.full_like(out, -1))
    return out.view(batch, 1, topk).contiguous()


def physical_indices(lists: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
    """FlashMLA's ``indices_in_kvcache`` from logical lists: ``block_table[i // 64] * 64 + i % 64``."""
    b, s, k = lists.shape
    li = lists.view(b, s * k).long()
    valid = li >= 0
    safe = torch.where(valid, li, torch.zeros_like(li))
    page = torch.gather(block_table.long(), 1, safe // PAGE)
    phys = page * PAGE + safe % PAGE
    return torch.where(valid, phys, torch.full_like(phys, -1)).to(torch.int32).view(b, s, k)


def attn_decode_inputs(seed: int, batch: int, ctx: int, h_q: int, topk: int, device="cuda"):
    """The timing-row inputs of one sparse-attention decode row, identical in every interpreter."""
    pages_per_req = (ctx + PAGE - 1) // PAGE
    q = (normal(seed + 1, batch * h_q * D_QK, device) * 0.1).to(torch.bfloat16)
    q = q.view(batch, 1, h_q, D_QK)
    bt = page_table(seed + 2, batch, pages_per_req, device)
    cache = fp8_cache_random(seed + 3, batch * pages_per_req, device)
    lists = sparse_lists(seed + 4, batch, ctx, topk, device)
    return q, cache, bt, lists


def attn_prefill_inputs(seed: int, tokens: int, h_q: int, topk: int, ctx: int | None = None,
                        device="cuda"):
    """Prefill rows: ``tokens`` query rows of ONE sequence, causal: row t attends to a subset of
    ``[0, prefix + t + 1)`` where ``prefix = ctx - tokens``.  Returns bf16 ``q [T, h_q, 512]``,
    bf16 ``kv [ctx, 512]`` and ascending lists ``[T, 1, K]`` (-1 padded when a row has < K keys)."""
    ctx = tokens if ctx is None else ctx
    prefix = ctx - tokens
    q = (normal(seed + 11, tokens * h_q * D_QK, device) * 0.1).to(torch.bfloat16).view(tokens, h_q, D_QK)
    kv = (normal(seed + 12, ctx * D_QK, device) * 0.1).to(torch.bfloat16).view(ctx, D_QK)
    out = torch.full((tokens, topk), -1, dtype=torch.int32, device=device)
    for t0 in range(0, tokens, 256):
        t1 = min(t0 + 256, tokens)
        u = uniform(seed + 13, (t1 - t0) * ctx, device, offset=t0 * ctx).view(t1 - t0, ctx)
        lim = prefix + torch.arange(t0, t1, device=device) + 1          # row t sees [0, lim)
        u = torch.where(torch.arange(ctx, device=device).unsqueeze(0) < lim.unsqueeze(1), u,
                        torch.full_like(u, -1.0))
        n = min(topk, ctx)
        vals, pick = torch.topk(u, n, dim=-1, sorted=False)
        pick = torch.where(vals >= 0, pick, torch.full_like(pick, 1 << 30))
        pick = torch.sort(pick, dim=-1).values
        pick = torch.where(pick >= (1 << 30), torch.full_like(pick, -1), pick)
        out[t0:t1, :n] = pick.to(torch.int32)
    return q, kv, out.view(tokens, 1, topk).contiguous()


def attn_train_inputs(seed: int, tokens: int, h_q: int, topk: int, ctx: int | None = None,
                      d_qk: int = 576, d_v: int = 512, device="cuda"):
    """The training geometry (SPEC 6, GLM-5 / V3.2 layout): bf16 ``q [T, h_q, 576]``, bf16
    ``kv [ctx, 576]`` (latent 0..511 then rope 512..575; V = the first 512), causal ascending lists
    ``[T, 1, K]`` (-1 padded) and the upstream gradient ``dO [T, h_q, 512]`` bf16."""
    ctx = tokens if ctx is None else ctx
    q = (normal(seed + 21, tokens * h_q * d_qk, device) * 0.1).to(torch.bfloat16).view(tokens, h_q, d_qk)
    kv = (normal(seed + 22, ctx * d_qk, device) * 0.1).to(torch.bfloat16).view(ctx, d_qk)
    _, _, lists = attn_prefill_inputs(seed, tokens, 1, topk, ctx, device=device)
    do = (normal(seed + 23, tokens * h_q * d_v, device) * 0.1).to(torch.bfloat16).view(tokens, h_q, d_v)
    return q, kv, lists, do


def indexer_bwd_inputs(seed: int, tokens: int, topk: int = 2048, ctx: int | None = None, heads: int = 64,
                       dim: int = 128, device="cuda"):
    """slime's indexer-backward contract: bf16 q [T, H, 128], bf16 k [ctx, 128], fp32 w [T, H], causal
    ascending lists [T, K] (-1 pads), fp32 grad of the selected scores [T, K]."""
    ctx = tokens if ctx is None else ctx
    q = (normal(seed + 31, tokens * heads * dim, device) * 0.3).to(torch.bfloat16).view(tokens, heads, dim)
    k = (normal(seed + 32, ctx * dim, device) * 0.3).to(torch.bfloat16).view(ctx, dim)
    w = normal(seed + 33, tokens * heads, device).view(tokens, heads).contiguous()
    _, _, lists = attn_prefill_inputs(seed, tokens, 1, topk, ctx, device=device)
    g = normal(seed + 34, tokens * topk, device).view(tokens, topk).contiguous()
    return q, k, w, lists.view(tokens, topk).contiguous(), g


# ------------------------------------------------------------------------------------------------
# indexer
# ------------------------------------------------------------------------------------------------

def pack_ue8m0(exp_bytes: torch.Tensor) -> torch.Tensor:
    """uint8 ``[..., 4]`` biased exponents -> int32 ``[...]`` (little-endian, DeepGEMM's packing)."""
    e = exp_bytes.to(torch.int64)
    w = e[..., 0] | (e[..., 1] << 8) | (e[..., 2] << 16) | (e[..., 3] << 24)
    w = torch.where(w >= (1 << 31), w - (1 << 32), w)
    return w.to(torch.int32)


def fp8_row_quantize(x: torch.Tensor):
    """float ``[..., 128]`` -> (e4m3 ``[..., 128]``, packed int32 ``[...]``): the V4.1 index-key/query quantizer
    (vLLM 0.33 FLASHMLA_SPARSE_DSV41 on SM100; port/CACHE-SPEC.md section 3): one power-of-two scale per 128-value
    row, 2^ceil(log2(max(amax, 1e-4) * fl(1/448))), carried as four equal ue8m0 bytes."""
    shp = x.shape
    xv = x.float().reshape(-1, IDX_D)
    r = xv.abs().amax(dim=-1).clamp(1e-4) * torch.tensor(1.0 / 448.0, dtype=torch.float32, device=x.device)
    bits = r.view(torch.int32)
    e = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0).int()
    s = (e << 23).view(torch.float32)
    q = (xv / s.unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(shp)
    return q, pack_ue8m0(e.to(torch.uint8).unsqueeze(-1).expand(-1, 4).reshape(*shp[:-1], 4))


def indexer_decode_inputs(seed: int, batch: int, ctx: int, device="cuda", h_idx: int = 64):
    """One indexer decode row: ``next_n = 1``, every request of context ``ctx``."""
    pages_per_req = (ctx + PAGE - 1) // PAGE
    num_pages = batch * pages_per_req
    q = normal(seed + 21, batch * h_idx * IDX_D, device).view(batch, 1, h_idx, IDX_D)
    q_fp, q_sf = fp8_row_quantize(q)
    w_store = torch.zeros((batch, max(8, h_idx)), dtype=torch.bfloat16, device=device)
    w_store[:, :h_idx] = (normal(seed + 22, batch * h_idx, device) * 0.2).view(batch, h_idx)
    weights = w_store[:, :h_idx]
    bt = page_table(seed + 23, batch, pages_per_req, device)
    cache = torch.empty((num_pages, PAGE * IDX_BYTES), dtype=torch.uint8, device=device)
    chunk = 8192
    for p0 in range(0, num_pages, chunk):
        p1 = min(p0 + chunk, num_pages)
        n = (p1 - p0) * PAGE * IDX_D
        b = rand_bytes(seed + 24, n, device, offset=p0 * PAGE * IDX_D).view(p1 - p0, PAGE * IDX_D)
        b = torch.where((b & 0x7F) == 0x7F, b ^ 0x01, b)
        cache[p0:p1, :PAGE * IDX_D] = b
        cache[p0:p1, PAGE * IDX_D:] = 120      # every scale 2^-7
    cache = cache.view(num_pages, PAGE, 1, IDX_BYTES)
    ctx_lens = torch.full((batch, 1), ctx, dtype=torch.int32, device=device)
    return (q_fp, q_sf), cache, weights, ctx_lens, bt


def indexer_prefill_inputs(seed: int, tokens: int, ctx: int, device="cuda", h_idx: int = 64):
    """Prefill: ``tokens`` query rows over a contiguous key range of length ``ctx``; row t's window
    is ``[0, ctx - tokens + t + 1)`` (causal; ``ctx = T`` for a fresh prompt, larger for a chunk of a
    long prefix)."""
    q = normal(seed + 31, tokens * h_idx * IDX_D, device).view(tokens, h_idx, IDX_D)
    q_fp, q_sf = fp8_row_quantize(q)
    w_store = torch.zeros((tokens, max(8, h_idx)), dtype=torch.bfloat16, device=device)
    w_store[:, :h_idx] = (normal(seed + 32, tokens * h_idx, device) * 0.2).view(tokens, h_idx)
    weights = w_store[:, :h_idx]
    k = normal(seed + 33, ctx * IDX_D, device).view(ctx, IDX_D)
    k_fp, k_sf = fp8_row_quantize(k)
    ks = torch.zeros(tokens, dtype=torch.int32, device=device)
    ke = (torch.arange(tokens, dtype=torch.int32, device=device) + (ctx - tokens) + 1).contiguous()
    return (q_fp, q_sf), (k_fp, k_sf), weights, ks, ke


# ------------------------------------------------------------------------------------------------
# top-k
# ------------------------------------------------------------------------------------------------

def topk_logits(seed: int, rows: int, width: int, device="cuda") -> torch.Tensor:
    """bf16 ``[rows, width]`` approximately N(0, 1): heavy bf16 ties, which is the hot case."""
    out = torch.empty((rows, width), dtype=torch.bfloat16, device=device)
    per = max(1, (1 << 26) // width)
    for r0 in range(0, rows, per):
        r1 = min(r0 + per, rows)
        u1 = uniform(seed, (r1 - r0) * width, device, offset=r0 * width).clamp_min(1e-7)
        u2 = uniform(seed + 7919, (r1 - r0) * width, device, offset=r0 * width)
        z = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
        out[r0:r1] = z.view(r1 - r0, width).to(torch.bfloat16)
    return out
