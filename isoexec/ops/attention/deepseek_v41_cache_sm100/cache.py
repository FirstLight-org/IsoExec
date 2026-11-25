"""Incremental decode state of the native V4.1 attention (one token per batch row per step).

Layout (per batch slot b):

* every compressed-KV producer ``s`` owns an arena ``[B, A_s, 512]`` BF16 =
  ``L_s`` doubled window rings (``2W`` rows each, one per layer that reads ``s``) followed by
  ``C_max = max_len // ratio`` compressed KV rows. Ratio-0 layers own a ``[B, 2W, 512]`` arena.
  The doubled ring stores position ``p`` at rows ``p % W`` and ``p % W + W``, so the newest ``W``
  positions are the ASCENDING contiguous rows ``[(p-n+1) % W, (p-n+1) % W + n)``: the attention list
  of a decode row is position-for-position the list of the same row in the full-sequence program.
* ``s`` also owns the FP8 index keys (one UE8M0 exponent per key) in the DeepGEMM paged layout
  ``[B * P, 64, 1, 132]`` (identity block table), and, for ratio > 1, the FP32 partial-group
  compressor state ``[B, ratio, 1024]`` (projected kv | score of the incomplete group).

Rows advance independently (``pos[b]``); a group completes per row when ``(pos + 1) % ratio == 0``.
The cache records a fingerprint of every parameter it was filled with; decoding after any in-place
update or replacement of those parameters raises ``StaleCacheError`` (call ``reset``).
"""
from __future__ import annotations

import torch

from . import ops, spec
from ._extension import extension

__all__ = ["DecodeCache", "StaleCacheError", "attention_decode", "attention_chunk"]

_PAGE_BYTES = 64 * 132
_SF_OFFSET = 64 * 128


def _h2d(t, dev):
    """A host-built index tensor on ``dev`` in one (pinned, async) copy."""
    return t.pin_memory().to(dev, non_blocking=True)


class StaleCacheError(RuntimeError):
    pass


def _fingerprint(module):
    return tuple((id(p), p.data_ptr(), p._version) for p in module.parameters())


class DecodeCache:
    def __init__(self, cfg, batch, max_len, device="cuda", *, max_chunk=0):
        ops.supported_config(cfg)
        self.cfg, self.B, self.max_len, self.device = cfg, batch, max_len, torch.device(device)
        self.max_chunk = max_chunk
        self.W = W = cfg.sliding_window
        self.pos = torch.zeros(batch, dtype=torch.int32, device=device)
        self.pos_host = [0] * batch          # host mirror: capacity checks never synchronize
        self.layer = {}       # layer -> (arena key, window row offset)
        self.arena, self.comp_off, self.ratio = {}, {}, {}
        self.kcache, self.block_table, self.pages, self.comp_state = {}, {}, {}, {}
        members = {}
        for i in range(cfg.num_hidden_layers):
            r = cfg.compress_ratios[i]
            key = ("own", i) if r == 0 else cfg.source_layer(i, "kv")
            members.setdefault(key, []).append(i)
        for key, layers in members.items():
            r = 0 if isinstance(key, tuple) else cfg.compress_ratios[key]
            cmax = max_len // r if r else 0
            # [doubled rings | chunk staging (W + max_chunk, shared by the group's layers) | compressed]
            stage = (W + max_chunk) if max_chunk else 0
            rows = len(layers) * 2 * W + stage + cmax
            self.arena[key] = torch.zeros((batch, rows, 512), dtype=torch.bfloat16, device=device)
            self.stage_off = getattr(self, "stage_off", {})
            self.stage_off[key] = len(layers) * 2 * W
            self.comp_off[key] = len(layers) * 2 * W + stage
            self.ratio[key] = r
            for j, i in enumerate(layers):
                self.layer[i] = (key, j * 2 * W)
            if r:
                pages = max(1, (cmax + 63) // 64)
                self.pages[key] = pages
                self.kcache[key] = torch.zeros((batch * pages, 64, 1, 132), dtype=torch.uint8, device=device)
                self.block_table[key] = torch.arange(batch * pages, dtype=torch.int32,
                                                     device=device).view(batch, pages)
                if r > 1:
                    self.comp_state[key] = torch.zeros((batch, r, 1024), dtype=torch.float32, device=device)
        self.fingerprints = {}
        self.step = {}

    # -- bookkeeping ----------------------------------------------------------------------------
    def reset(self, rows=None):
        idx = list(range(self.B)) if rows is None else sorted({int(r) for r in rows})
        for r in idx:
            self.pos_host[r] = 0
        sel = torch.as_tensor(idx, device=self.device, dtype=torch.long)
        self.pos[sel] = 0
        for t in self.comp_state.values():
            t[sel] = 0
        if len(idx) == self.B:
            # a full reset is the only way to rebind changed weights (every row was filled with the old ones)
            self.fingerprints.clear()

    def advance(self, n=1, rows=None):
        if rows is None:
            self.pos += n
            self.pos_host = [p + n for p in self.pos_host]
        else:
            for r in rows:
                self.pos_host[int(r)] += n
            self.pos[torch.as_tensor(rows, device=self.device).long()] += n
        self.step = {}

    def _bind(self, module, index):
        fp = _fingerprint(module)
        old = self.fingerprints.get(index)
        if old is not None and old != fp:
            raise StaleCacheError(f"layer {index} parameters changed while other rows hold old state; reset() all rows")
        self.fingerprints[index] = fp

    def check(self, module, index):
        fp = self.fingerprints.get(index)
        if fp is None:
            raise StaleCacheError(f"layer {index} has no prefill/decode binding in this cache")
        if fp != _fingerprint(module):
            raise StaleCacheError(f"layer {index} parameters changed since this cache was filled; reset it")

    def _scatter(self, src, dst, stride_bytes, *idx, flag=None, stacked=None):
        """``stacked``: an already [sets, R] index tensor (no stacking copy)."""
        if stacked is not None:
            index = stacked
        elif len(idx) == 1:
            index = idx[0].reshape(1, -1)
        else:
            index = torch.stack(idx, 0)
        extension().scatter_rows(src.contiguous(), dst, stride_bytes, index.contiguous(), flag)

    # -- prefill publication ----------------------------------------------------------------------
    def write_prefill(self, module, index, kvbuf, state, B, T, C, rows):
        old = self.fingerprints.get(index)
        if old is not None and old != _fingerprint(module):
            raise StaleCacheError(f"layer {index} parameters changed while other rows hold old state; reset() all rows")
        cfg, W = self.cfg, self.W
        dev = kvbuf.device
        slots = (torch.arange(B, device=dev) if rows is None else torch.as_tensor(rows, device=dev)).long()
        if slots.numel() != B:
            raise ValueError("one cache slot per prefill row")
        if any(self.pos_host[int(r)] for r in (range(B) if rows is None else rows)):
            raise ValueError("prefill publishes into empty slots only (positions must be 0)")
        if T + 1 > self.max_len:
            raise ValueError("prefill exceeds the cache length")
        key, off = self.layer[index]
        arena = self.arena[key]
        A = arena.shape[1]
        S = T + C
        n = min(T, W)
        t = torch.arange(T - n, T, device=dev)
        src = kvbuf.view(B, S, 512)[:, T - n:T].reshape(B * n, 512)
        base = (slots[:, None] * A + off + (t % W)[None]).reshape(-1)
        self._scatter(src, arena, 1024, base, base + W)
        r = self.ratio[key]
        if index in cfg.kv_source_layer_ids:
            if C:
                comp = state["compress_kv_native"]
                c = torch.arange(C, device=dev)
                self._scatter(comp, arena, 1024, (slots[:, None] * A + self.comp_off[key] + c[None]).reshape(-1))
                codes, sf = state["index_k_native"]
                self._write_index_keys(key, slots[:, None].expand(B, C).reshape(-1), c.repeat(B),
                                       codes[:, :C].reshape(B * C, 128), sf[:, :C].reshape(B * C, 1))
            if r > 1 and T % r:
                proj = state["compressor_proj"].view(B, T, 1024)[:, T - T % r:]
                j = torch.arange(T % r, device=dev)
                self._scatter(proj.reshape(-1, 1024), self.comp_state[key], 4096,
                              (slots[:, None] * r + j[None]).reshape(-1))
        self._bind(module, index)

    def _write_index_keys(self, key, slot, c, codes, sf, valid=None, dev=None):
        page = slot * self.pages[key] + torch.div(c, 64, rounding_mode="floor")
        row = c % 64
        ci = page * (_PAGE_BYTES // 128) + row
        si = page * (_PAGE_BYTES // 4) + _SF_OFFSET // 4 + row
        if valid is not None:
            ci = torch.where(valid, ci, -1)
            si = torch.where(valid, si, -1)
        if dev is not None:                     # host-built indices: one copy
            both = _h2d(torch.stack((ci, si)).long(), dev)
            ci, si = both[0], both[1]
        kc = self.kcache[key]
        self._scatter(codes, kc, 128, ci)
        self._scatter(sf.view(torch.int32), kc, 4, si)


@torch.no_grad()
def attention_decode(module, x, cfg, index, cache, *, linear_fn, dequant_fn, recipe=None):
    """One token per row: ``x [B, 1, hidden]`` at positions ``cache.pos``; advance with ``cache.advance()``."""
    return ops.drive(attention_decode_gen(module, x, cfg, index, cache, linear_fn=linear_fn, dequant_fn=dequant_fn,
                                          recipe=recipe))


def attention_decode_gen(module, x, cfg, index, cache, *, linear_fn, dequant_fn, recipe=None, tp=1):
    """Decode step in the TP generator protocol (see ``ops.attention_full_gen``): index scores are local
    (``ops.index_scores``), nothing is yielded, the return value is this rank's ``wo_b`` root.

    The caller holds ``torch.no_grad`` (decode is inference)."""
    yield from ()          # host protocol: no intermediate reduction
    cache.check(module, index)
    B = x.shape[0]
    if x.shape != (cache.B, 1, cfg.hidden_size) or x.dtype != torch.bfloat16:
        raise ValueError("decode expects BF16 [cache.B, 1, hidden]")
    H, D, W, eps = cfg.num_attention_heads // tp, cfg.head_dim, cache.W, cfg.rms_norm_eps
    ratio = cfg.compress_ratios[index]
    pos = cache.pos
    if max(cache.pos_host) + 1 > cache.max_len:
        raise ValueError("decode position exceeds the cache length")
    table = ops.rope_table(cfg, ratio, cache.max_len, x.device)
    x2d = x.reshape(B, -1)
    qa = linear_fn(module.wq_a, x).reshape(B, -1)
    qr = ops._row(qa, module.q_norm.weight, spec.ROW_Q_NORM, cfg.q_lora_rank, eps=eps)
    q = linear_fn(module.wq_b, qr.view(B, 1, -1)).reshape(B * H, D)
    q = ops.q_rope(q, D, table, ops.Pos(pos=pos, div=H))
    key, off = cache.layer[index]
    arena = cache.arena[key]
    A = arena.shape[1]
    r = ratio
    comp_off = cache.comp_off.get(key, 0)
    i64 = torch.empty((5, B), dtype=torch.int64, device=x.device)
    i32 = torch.empty((2, B), dtype=torch.int32, device=x.device)
    extension().decode_prep(pos, A, off, W, comp_off, r, cache.pages.get(key, 1), i64, i32)
    kvr = linear_fn(module.wkv, x).reshape(B, D)
    kv = ops._row(kvr, module.kv_norm.weight, spec.ROW_WINDOW_KV, D, eps=eps, table=table, pos=ops.Pos(pos=pos))
    cache._scatter(kv, arena, 1024, stacked=i64[:2])
    topk = None
    if ratio:
        gpos, ke = i32[0], i32[1]
        if index in cfg.kv_source_layer_ids:
            comp = module.compressor
            latent = torch.empty((B, 512), dtype=torch.bfloat16, device=x.device)
            if r == 1:
                lat = ops.mm(x2d, ops._bf16_weight(comp, "wkv", comp.wkv.weight, False))
                latent = ops._row(lat, comp.norm.weight, spec.ROW_LATENT_NORM, 512, eps=eps)
                flag = None
            else:
                wcat = ops._cache(comp).get("wcat", (comp.wkv.weight, comp.wgate.weight),
                                            lambda: torch.cat((comp.wkv.weight, comp.wgate.weight), 0)
                                            .to(torch.bfloat16).contiguous())
                proj = ops.mm(x2d, wcat, root_fp32=True)
                flag = torch.empty(B, dtype=torch.int32, device=x.device)
                extension().pool_norm(proj, r, ops.f32_param(comp.norm.weight), float(eps), latent,
                                      0, 0, cache.comp_state[key], pos, flag)
            ckv = torch.empty((B, 512), dtype=torch.bfloat16, device=x.device)
            ops.row_op(spec.ROW_COMPRESSED_KV, latent, 512, table=table, pos=ops.Pos(pos=gpos), out=ckv, flag=flag)
            cache._scatter(ckv, arena, 1024, i64[2], flag=flag)
            if index in cfg.index_source_layer_ids:
                idx = module.indexer
                k = ops.mm(latent, ops._bf16_weight(idx, "wk", idx.wk.weight, False))
                codes = torch.empty((B, 128), dtype=torch.uint8, device=x.device)
                sf = torch.empty((B, 1), dtype=torch.int32, device=x.device)
                ops.row_op(spec.ROW_INDEX_K, k, 128, w=ops.f32_param(idx.k_norm.weight), eps=eps,
                           table=table, pos=ops.Pos(pos=gpos), codes=codes, sf=sf.view(B), flag=flag)
                kc = cache.kcache[key]
                cache._scatter(codes, kc, 128, i64[3], flag=flag)
                cache._scatter(sf, kc, 4, i64[4], flag=flag)
        if index in cfg.index_source_layer_ids:
            topk = _decode_topk(module.indexer, x, x2d, qr, cfg, index, cache, key, table, pos, ke, linear_fn)
            cache.step["topk"] = topk
        topk = cache.step.get("topk")
        if topk is None:
            raise ValueError("index source must precede its consumers in this step")
    K = W + (cfg.index_topk if ratio else 0)
    lists = torch.empty((B, K), dtype=torch.int32, device=x.device)
    extension().build_lists(topk, pos, B, 1, A, off, comp_off, W, 1 | ops.list_mode(recipe), lists)
    o = ops.sink_attention(q.view(B, H, D), arena.view(B * A, 512), lists, module.attn_sink, D ** -0.5, recipe)
    return ops.output_projection(module, o, cfg, table, ops.Pos(pos=pos, div=H), B, 1, linear_fn, dequant_fn, tp=tp)


def _decode_topk(idx, x, x2d, qr, cfg, index, cache, key, table, pos, ke, linear_fn):
    from isoexec.ops.attention.sparse_sm100 import indexer as dsa_indexer
    from isoexec.ops.attention.sparse_sm100 import topk as dsa_topk
    B = x.shape[0]
    ke = ke.contiguous()
    hi = cfg.index_n_heads                     # every rank: all index heads (ops.index_scores)
    codes, sf, w = ops.index_queries(idx, x2d, qr, cfg, table, ops.Pos(pos=pos, div=hi), linear_fn,
                                     x.shape, (B, 1, cfg.q_lora_rank))
    width = cache.pages[key] * 64
    scores = ops.index_scores(codes, sf, w, ke, width=width, cache=cache.kcache[key],
                              block_table=cache.block_table[key])
    if index == cfg.candidate_source_layer_id:
        cache.step["candidates"] = ops._candidate_select(scores, ke, cfg, dsa_topk)[0]
    elif cfg.candidate_source_layer_id < index:
        sel = cache.step.get("candidates")
        if sel is None:
            raise ValueError("candidate source must precede its consumer in this step")
        blk = cfg.candidate_block_size
        extension().candidate_mask(scores, ke, sel, blk, (width + blk - 1) // blk)
    return dsa_topk.dsa_topk(scores, ke, k=cfg.index_topk)


@torch.no_grad()
def attention_chunk(module, x, cfg, index, cache, *, rows, linear_fn, dequant_fn, recipe=None):
    """See ``attention_chunk_gen`` (this is its TP = 1 driver)."""
    return ops.drive(attention_chunk_gen(module, x, cfg, index, cache, rows=rows, linear_fn=linear_fn,
                                         dequant_fn=dequant_fn, recipe=recipe))


def attention_chunk_gen(module, x, cfg, index, cache, *, rows, linear_fn, dequant_fn, recipe=None, tp=1):
    """Continue the cached sequences of the contiguous slots ``rows = (s0, s1)`` by ``x [s1-s0, Tc, hidden]``.

    All rows of the call share their current length ``p0`` (prefill rows of different lengths in separate
    calls). Any chunking of a sequence, followed by decode, equals the full-sequence program bit for bit
    (tests/test_decode_gpu.py): window KV is staged as the ascending contiguous positions
    ``[p0 - min(p0, W), p0 + Tc)`` (previous W from the doubled ring, then the chunk) in the group's staging
    rows, so every list is position-for-position the full-sequence list; incomplete ratio > 1 groups are
    pooled from ``[FP32 state | chunk projections]`` with the full-sequence pool kernel. Advance with
    ``cache.advance(Tc, rows=range(s0, s1))`` after every layer of the chunk. TP generator protocol as
    ``attention_decode_gen`` (local index scores, nothing yielded).
    """
    yield from ()          # host protocol: no intermediate reduction
    s0, s1 = rows
    b, Tc = x.shape[0], x.shape[1]
    if b != s1 - s0 or x.dtype != torch.bfloat16 or x.shape[2] != cfg.hidden_size or Tc < 1:
        raise ValueError("chunk expects BF16 [s1 - s0, Tc >= 1, hidden]")
    if Tc > cache.max_chunk:
        raise ValueError("chunk longer than the cache's max_chunk staging")
    p0 = cache.pos_host[s0]
    if any(cache.pos_host[r] != p0 for r in range(s0, s1)):
        raise ValueError("rows of one chunk call must share their length")
    if p0 + Tc > cache.max_len:
        raise ValueError("chunk exceeds the cache length")
    if index in cache.fingerprints:
        cache.check(module, index)
    else:
        cache._bind(module, index)
    H, D, W, eps = cfg.num_attention_heads // tp, cfg.head_dim, cache.W, cfg.rms_norm_eps
    r = cfg.compress_ratios[index]
    dev = x.device
    table = ops.rope_table(cfg, r, cache.max_len, dev)
    # host-known integers: index prep built on the CPU and sent in one copy each (no tiny device kernels)
    pos = _h2d(torch.arange(p0, p0 + Tc, dtype=torch.int32).repeat(b), dev)        # [b*Tc] absolute positions
    x2d = x.reshape(b * Tc, -1)
    qa = linear_fn(module.wq_a, x).reshape(b * Tc, -1)
    qr = ops._row(qa, module.q_norm.weight, spec.ROW_Q_NORM, cfg.q_lora_rank, eps=eps)
    q = linear_fn(module.wq_b, qr.view(b, Tc, -1)).reshape(b * Tc * H, D)
    q = ops.q_rope(q, D, table, ops.Pos(pos=pos, div=H))
    key, off = cache.layer[index]
    av = cache.arena[key][s0:s1]                                                    # [b, A, 512] contiguous
    A = av.shape[1]
    st = cache.stage_off[key]
    comp_off = cache.comp_off[key]
    kvr = linear_fn(module.wkv, x).reshape(b * Tc, D)
    kv = ops._row(kvr, module.kv_norm.weight, spec.ROW_WINDOW_KV, D, eps=eps, table=table,
                  pos=ops.Pos(pos=pos)).view(b, Tc, 512)
    nprev = min(p0, W)
    if nprev:
        start = (p0 - nprev) % W
        av[:, st:st + nprev] = av[:, off + start:off + start + nprev]
    av[:, st + nprev:st + nprev + Tc] = kv
    keep = min(Tc, W)                                                              # newest positions -> ring
    q_pos = torch.arange(p0 + Tc - keep, p0 + Tc)
    slot_h = torch.arange(b)[:, None] + s0
    ring = (slot_h * A + off + (q_pos % W)[None]).reshape(-1)
    cache._scatter(kv[:, Tc - keep:].reshape(-1, 512), cache.arena[key], 1024,
                   stacked=_h2d(torch.stack((ring, ring + W)), dev))
    topk = None
    if r:
        if index in cfg.kv_source_layer_ids:
            comp = module.compressor
            c0 = p0 // r
            if r == 1:
                lat = ops.mm(x2d, ops._bf16_weight(comp, "wkv", comp.wkv.weight, False))
                latent = ops._row(lat, comp.norm.weight, spec.ROW_LATENT_NORM, 512, eps=eps)
                G = Tc
            else:
                wcat = ops._cache(comp).get("wcat", (comp.wkv.weight, comp.wgate.weight),
                                            lambda: torch.cat((comp.wkv.weight, comp.wgate.weight), 0)
                                            .to(torch.bfloat16).contiguous())
                proj = ops.mm(x2d, wcat, root_fp32=True).view(b, Tc, 1024)
                rem0 = p0 % r
                stt = cache.comp_state[key][s0:s1]
                comb = torch.cat((stt[:, :rem0], proj), 1).contiguous()             # group-aligned FP32 rows
                Tg = rem0 + Tc
                G = Tg // r
                latent = torch.empty((b * G, 512), dtype=torch.bfloat16, device=dev)
                if G:
                    extension().pool_norm(comb.view(b * Tg, 1024), r, ops.f32_param(comp.norm.weight),
                                          float(eps), latent, G, Tg, None, None, None)
                stt[:, :Tg - G * r] = comb[:, G * r:]
            if G:
                gp = ops.Pos(mod=G, mul=r, add=c0 * r)
                ckv = torch.empty((b * G, 512), dtype=torch.bfloat16, device=dev)
                ops.row_op(spec.ROW_COMPRESSED_KV, latent, 512, table=table, pos=gp, out=ckv)
                av[:, comp_off + c0:comp_off + c0 + G] = ckv.view(b, G, 512)
                idx = module.indexer
                k = ops.mm(latent, ops._bf16_weight(idx, "wk", idx.wk.weight, False))
                codes = torch.empty((b * G, 128), dtype=torch.uint8, device=dev)
                sf = torch.empty((b * G, 1), dtype=torch.int32, device=dev)
                ops.row_op(spec.ROW_INDEX_K, k, 128, w=ops.f32_param(idx.k_norm.weight), eps=eps,
                           table=table, pos=gp, codes=codes, sf=sf.view(-1))
                cpos = torch.arange(c0, c0 + G)
                cache._write_index_keys(key, slot_h.expand(b, G).reshape(-1), cpos.repeat(b), codes, sf, dev=dev)
        if index in cfg.index_source_layer_ids:
            from isoexec.ops.attention.sparse_sm100 import indexer as dsa_indexer
            from isoexec.ops.attention.sparse_sm100 import topk as dsa_topk
            ke = torch.div(pos + 1, r, rounding_mode="floor").to(torch.int32)
            hi = cfg.index_n_heads             # every rank: all index heads (ops.index_scores)
            codes, sf, w = ops.index_queries(module.indexer, x2d, qr, cfg, table, ops.Pos(pos=pos, div=hi),
                                             linear_fn, x.shape, (b, Tc, cfg.q_lora_rank))
            width = cache.pages[key] * 64
            bt = cache.block_table[key][s0:s1].repeat_interleave(Tc, 0).contiguous()
            scores = ops.index_scores(codes, sf, w, ke, width=width, cache=cache.kcache[key],
                                      block_table=bt)
            if index == cfg.candidate_source_layer_id:
                cache.step["candidates"] = ops._candidate_select(scores, ke, cfg, dsa_topk)[0]
            elif cfg.candidate_source_layer_id < index:
                sel = cache.step.get("candidates")
                if sel is None:
                    raise ValueError("candidate source must precede its consumer in this chunk")
                blk = cfg.candidate_block_size
                extension().candidate_mask(scores, ke, sel, blk, (width + blk - 1) // blk)
            cache.step["topk"] = dsa_topk.dsa_topk(scores, ke, k=cfg.index_topk)
        topk = cache.step.get("topk")
        if topk is None:
            raise ValueError("index source must precede its consumers in this chunk")
    K = W + (cfg.index_topk if r else 0)
    lists = torch.empty((b * Tc, K), dtype=torch.int32, device=dev)
    extension().build_lists(topk, pos, b * Tc, Tc, A, st - (p0 - nprev), comp_off, W, ops.list_mode(recipe), lists)
    o = ops.sink_attention(q.view(b * Tc, H, D), av.reshape(b * A, 512), lists, module.attn_sink, D ** -0.5, recipe)
    return ops.output_projection(module, o, cfg, table, ops.Pos(pos=pos, div=H), b, Tc, linear_fn, dequant_fn,
                                 tp=tp)
