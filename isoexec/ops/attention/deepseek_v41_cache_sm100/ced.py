"""The decoder of the V4.1 causal encoder-decoder (CED) at op level: the boundary layer split, the segment-truncated
window and query rows that are a subset of a sequence.

The decoder's FULL layer ``b`` (the boundary, ``b = max(kv_source_layer_ids)``, ratio 1) is split in two:

* the **entry projection** (``decoder.entry``), token-wise in the encoder output and run on EVERY token: layer b's
  window-KV record (``Q_win``), the decoder global compressed record (``Q_cmp``, one entry per token) and the index
  key (``Q_idx``) (port/CACHE-SPEC.md sections 1-3). Engine: ``entry_projection_paged`` writes the paged records of
  every scheduled token, including tokens whose decoder rows are not run. Trainer: ``entry_projection`` returns the
  quantize-then-dequantize rows (straight-through) and the FP8 index keys as tensors (``DecoderGlobal``);
* the **query side** on the decoder rows ``R``: index queries (FP8, ``spec.INDEX_QUERY_FORMAT``), index scores over
  the whole causal prefix of the global index keys, candidate-pool selection, top-k, sink attention over
  ``[layer b's unclamped window ; selected global entries]``, ``wo_a``, ``wo_b``. Engine: ``query_paged_gen`` on a
  decoder-row batch (``PagedBatch.rows_subset``); trainer: ``attention_rows_gen`` at ``index == b``.

Projection + query side on all rows is today's layer-b program, bit for bit (``ops.attention_full_gen`` /
``paged.attention_paged_gen`` at the boundary; tests/test_ced_gpu.py).

The decoder layers after ``b`` (Reuse / Reindex, ratio 1) run on the decoder rows only. Their window is truncated at
the row's decoder segment start ``s`` (bounded replay): keys at positions ``[max(s, p - W + 1), p]``; layer b's own
window and the encoder are never truncated. Engine: ``paged.attention_paged_gen`` on the decoder-row batch with
``PagedBatch.window_floor``; trainer: ``attention_rows_gen`` at ``index > b`` with ``QueryRows.floor``, the window
indexing the decoder rows of the same segment only (``QueryRows.window_end``). ``floor = 0`` (or None) is today's
program bit for bit.

Every op here is row-local: a query row's bits are a function of its own inputs, its position, its floor and its
sequence's keys, never of which other rows are in the call (row subsets == the same rows of the full computation;
tests/test_ced_gpu.py). The index path runs without gradient (the selector is frozen in the actor recipe).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import ops, spec
from ._extension import extension

__all__ = ["entry_projection", "decoder_attention_gen", "DecoderGlobal", "QueryRows", "attention_rows_gen",
           "attention_rows", "entry_projection_paged", "query_paged_gen", "query_paged", "check_boundary"]


def check_boundary(cfg, index):
    """``index`` is the CED boundary: the last KV source, ratio 1, an index source and the candidate source."""
    if not cfg.kv_source_layer_ids or index != max(cfg.kv_source_layer_ids):
        raise ValueError(f"layer {index} is not the CED boundary (the last KV source)")
    if cfg.compress_ratios[index] != 1 or index not in cfg.index_source_layer_ids:
        raise ValueError(f"CED boundary {index} must be a ratio-1 KV and index source")
    if cfg.candidate_source_layer_id != index:
        raise ValueError(f"CED boundary {index} must be the candidate source")


def _decoder_layer(cfg, index):
    """The boundary ``b`` of decoder layer ``index >= b`` (ratio 1, reading b's global KV)."""
    b = max(cfg.kv_source_layer_ids)
    check_boundary(cfg, b)
    if index < b or cfg.compress_ratios[index] != 1 or cfg.source_layer(index, "kv") != b:
        raise ValueError(f"layer {index} is not a CED decoder layer (ratio 1 at or after the boundary {b})")
    return b


# ---------------------------------------------------------------------------------------------- model-facing
def entry_projection(holder, h, cfg, layer, positions, *, linear_fn, step=None):
    """``decoder.entry``'s token-wise projection of the boundary ``layer`` on EVERY token (model/REQUESTS.md R1(a)).

    ``holder``: ``wkv``, ``kv_norm``, ``compressor.{wkv, norm}`` (ratio 1), ``indexer.{wk, k_norm}``
    (``EntryProjection``); ``h [T, hidden]`` BF16 the boundary layer's attention input of the packed tokens at absolute
    ``positions [T]``.
    Engine (``step``: ``.pools``, ``.batch`` of the step's tokens): writes layer b's window record, the global
    compressed record and the index key of every scheduled token into the pages (``entry_projection_paged``) and
    returns None. Trainer: returns ``(main_kv [T, 512], index_k, window_kv [T, 512])``: the compressed entries and
    the window keys as the leaf reads them from their records (dequant(Q_cmp), dequant(Q_win); straight-through
    differentiable) and ``index_k = (codes [Tp, 128] uint8, exponent words [Tp] int32)`` (``Tp`` = T rounded up to
    128; no gradient), row ``t`` = packed token ``t``. The arithmetic of ``ops.attention_full_gen`` at the boundary."""
    check_boundary(cfg, layer)
    if step is not None:
        entry_projection_paged(holder, h, cfg, layer, step.pools, step.batch, linear_fn=linear_fn)
        return None
    if h.ndim != 2 or h.dtype != torch.bfloat16 or not h.is_cuda or positions.numel() != h.shape[0]:
        raise ValueError("entry projection expects CUDA BF16 [tokens, hidden] and one position per token")
    T, hid = h.shape
    D, eps = cfg.head_dim, cfg.rms_norm_eps
    table = ops.rope_table(cfg, 1, max(T, 1), h.device)            # every position of a packed sequence is < T
    pos = positions.to(device=h.device, dtype=torch.int32).contiguous()
    kvr = linear_fn(holder.wkv, h.view(1, T, hid)).reshape(T, D).contiguous()
    window = ops._row(kvr, holder.kv_norm.weight, spec.ROW_WINDOW_KV, 512, eps=eps, table=table, pos=ops.Pos(pos=pos))
    latent, _ = ops.compressor_full(holder.compressor, h, 1, T, 1, eps)
    with torch.no_grad():
        codes, sf = ops.index_keys(holder.indexer, latent.detach(), cfg, 1, table, ops.Pos(pos=pos), 1, T)
    comp = ops._row(latent, None, spec.ROW_COMPRESSED_KV, 512, table=table, pos=ops.Pos(pos=pos))
    return comp, (codes[0], sf[0]), window


def decoder_attention_gen(holder, h, cfg, layer, rows, decoder_kv, shared, *, linear_fn, dequant_fn, tp=1, step=None,
                          recipe=None):
    """Decoder layer ``layer >= b`` on the decoder rows (model/REQUESTS.md R1(b), R2, R3), in the TP generator
    protocol (returns this rank's ``wo_b`` root ``[R, hidden]``; nothing is yielded).

    ``h [R, hidden]`` BF16: the layer's attention input at the decoder rows; ``rows``: ``program.DecoderRows``
    (``index [R]`` packed token row, ``positions``, ``sequence`` (packed sequence), ``window_floor`` (segment start),
    ``window_index [R, W]`` (trainer: the decoder row of each window slot of the row's segment, -1 empty), ``cu_seqlens``);
    ``decoder_kv``: ``program.DecoderGlobalKV`` of ``entry_projection``; ``shared``: one dict per decoder forward (the
    decoder chain: top-k, candidate pool; filled by the boundary and the Reindex layers, read by the Reuse layers).
    The boundary (``layer == b``, holder ``AttentionWeights(projection=False)``): its query side over its own unclamped
    window; the layers after it: their window truncated at ``window_floor``.
    Engine (``step``): ``query_paged_gen`` / ``paged.attention_paged_gen`` on the decoder-row batch of the step
    (``PagedBatch.rows_subset(rows.index, window_floor=rows.window_floor)``). Trainer: ``attention_rows_gen`` over
    ``DecoderGlobal.packed``. ``recipe``: the sparse-attention recipe, default the engine's (dual rail, 3)."""
    from . import paged
    b = _decoder_layer(cfg, layer)
    recipe = paged.RECIPE if recipe is None else recipe
    R = h.shape[0]
    if R == 0:
        yield from ()
        return torch.empty((0, h.shape[-1]), dtype=torch.bfloat16 if tp == 1 else torch.float32, device=h.device)
    if step is not None:
        if recipe != paged.RECIPE:
            raise ValueError(f"the paged engine runs sparse-attention recipe {paged.RECIPE}")
        batch = _engine_rows(step, rows, shared, layer)
        if layer == b:
            return (yield from query_paged_gen(holder, h, cfg, layer, step.pools, batch, linear_fn=linear_fn,
                                               dequant_fn=dequant_fn, tp=tp))
        return (yield from paged.attention_paged_gen(holder, h, cfg, layer, step.pools, batch, linear_fn=linear_fn,
                                                     dequant_fn=dequant_fn, tp=tp))
    glob = shared.get("_ced_global")
    if glob is None:
        glob = shared["_ced_global"] = DecoderGlobal.packed(decoder_kv.main_kv, decoder_kv.index_k,
                                                            decoder_kv.window_kv, rows.cu_seqlens, b)
    qrows = shared.get("_ced_rows")
    if qrows is None:
        qrows = shared["_ced_rows"] = QueryRows.from_decoder_rows(rows, cfg.sliding_window)
    return (yield from attention_rows_gen(holder, h, cfg, layer, shared, glob, qrows, linear_fn=linear_fn,
                                          dequant_fn=dequant_fn, recipe=recipe, tp=tp))


def _engine_rows(step, rows, shared, layer):
    """The decoder-row batch of ``step`` (built once per forward, kept in ``shared``); the engine's per-layer window
    rows / lists bound after it was built are gathered at the rows when their layer first runs."""
    batch = shared.get("_ced_batch")
    sel = rows.index.to(device=step.batch.tok_seq.device, dtype=torch.int64)
    if batch is None or shared.get("_ced_batch_of") is not step.batch:
        # every decoder layer's window rows / lists (floored after the boundary) in one launch when the step built its
        # windows in one (EngineStep.decoder_windows); else the per-layer gathers below and paged's floor
        fill = getattr(step, "decoder_windows", None)
        whole = fill is not None and getattr(step, "_win", None)
        batch = step.batch.rows_subset(rows.index, window_floor=rows.window_floor, windows=not whole)
        if whole and not fill(batch, sel.contiguous(), _decoder_layer(step.cfg, layer)):
            batch = step.batch.rows_subset(rows.index, window_floor=rows.window_floor)
        shared["_ced_batch"] = batch
        shared["_ced_batch_of"] = step.batch
    for name in ("window_rows", "window_lists"):
        full, part = getattr(step.batch, name), getattr(batch, name)
        if full is not None and layer in full and layer not in part:
            part[layer] = full[layer].index_select(0, sel).contiguous()
    return batch


# ---------------------------------------------------------------------------------------------- trainer
@dataclass
class DecoderGlobal:
    """``decoder.entry``'s output in the trainer, with the row layout of its sequences.

    ``window_kv`` ``[N, 512]`` BF16: layer b's window keys as the leaf reads them from their records (dequant(Q_win));
    ``comp_kv`` ``[N, 512]`` BF16: the global compressed entries (ratio 1: one per token; dequant(Q_cmp)); both
    straight-through differentiable; position ``p`` of sequence ``s`` is row ``kv_base[s] + p`` of both. ``codes``
    ``[K, 128]`` / ``sf`` ``[K]``: the FP8 index keys (no gradient), entry ``p`` of sequence ``s`` at row
    ``key_base[s] + p``. ``lengths[s]``: the tokens of sequence ``s`` (its key count). Host lists, device ``kv_base``."""

    window_kv: torch.Tensor
    comp_kv: torch.Tensor
    codes: torch.Tensor
    sf: torch.Tensor
    kv_base_h: tuple
    key_base_h: tuple
    lengths: tuple
    kv_base: torch.Tensor
    layer: int

    @staticmethod
    def packed(main_kv, index_k, window_kv, cu_seqlens, layer):
        """The packed form (``entry_projection``'s rows = packed tokens, sequences at ``cu_seqlens``)."""
        cu = [int(c) for c in torch.as_tensor(cu_seqlens).reshape(-1).tolist()]
        if cu[-1] != main_kv.shape[0] or window_kv.shape[0] != main_kv.shape[0] or index_k[0].shape[0] < cu[-1]:
            raise ValueError("the decoder global KV does not cover the packed sequences")
        base = tuple(cu[:-1])
        return DecoderGlobal(window_kv, main_kv, index_k[0], index_k[1], base, base,
                             tuple(z - a for a, z in zip(cu[:-1], cu[1:])),
                             torch.tensor(base, dtype=torch.int32, device=main_kv.device), layer)

    @staticmethod
    def padded(window_kv, comp_kv, codes, sf, B, T, layer):
        """The right-padded ``[B, T]`` form (``ops.attention_full_gen``'s rows; keys ``[B, Tp, 128]``)."""
        Tp = codes.shape[1]
        return DecoderGlobal(window_kv, comp_kv, codes.reshape(B * Tp, 128), sf.reshape(B * Tp),
                             tuple(b * T for b in range(B)), tuple(b * Tp for b in range(B)), (T,) * B,
                             torch.arange(B, dtype=torch.int32, device=comp_kv.device) * T, layer)

    @property
    def max_len(self):
        return max(self.lengths) if self.lengths else 1

    def keys(self, s):
        """``(codes [L, 128], sf [L], L)`` of sequence ``s`` (16-byte aligned for the indexer's TMA maps)."""
        k0, L = self.key_base_h[s], self.lengths[s]
        c, f = self.codes[k0:k0 + L], self.sf[k0:k0 + L]
        if k0 % 4:
            c, f = c.clone(), f.clone()
        return c, f, L


@dataclass
class QueryRows:
    """The decoder (query) rows of a trainer call, against a ``DecoderGlobal``.

    ``seq`` ``[R]`` int32: the row's sequence (``DecoderGlobal`` index); ``pos`` ``[R]`` int32: its absolute position;
    ``floor`` ``[R]`` int32 or None: its window floor (decoder segment start ``s``; layers after the boundary only);
    ``window_end`` ``[R]`` int32: the row of its own window key among the decoder rows (``arange(R)`` for segment rows).
    ``groups``: ``(s, rows of sequence s)`` (a slice when contiguous), host side. ``window_contiguous``: every row's
    truncated window ``[max(floor, p - W + 1), p]`` lies at the consecutive decoder rows ending at ``window_end``
    (segment rows contiguous and in position order, as ``DecoderSegments`` lays them out); required by the layers
    after the boundary, which read their own window keys by decoder row."""

    seq: torch.Tensor
    pos: torch.Tensor
    floor: torch.Tensor | None
    window_end: torch.Tensor
    groups: tuple
    window_contiguous: bool

    def __len__(self):
        return int(self.pos.numel())

    @staticmethod
    def make(seq, pos, *, floor=None, device="cuda"):
        """From host (CPU) ``seq`` / ``pos`` / ``floor`` sequences (``DecoderSegments.sequence``, ``.positions``,
        ``.seg_start``)."""
        seq_h = torch.as_tensor(seq, dtype=torch.int64).cpu().reshape(-1)
        pos_h = torch.as_tensor(pos, dtype=torch.int64).cpu().reshape(-1)
        if seq_h.numel() != pos_h.numel():
            raise ValueError("seq and pos: one entry per row")
        fl_h = None if floor is None else torch.as_tensor(floor, dtype=torch.int64).cpu().reshape(-1)
        if fl_h is not None and (fl_h.numel() != pos_h.numel() or bool((fl_h > pos_h).any())):
            raise ValueError("floor: one entry per row, at or below the row's position")
        R = pos_h.numel()
        lo = fl_h.clamp(min=0) if fl_h is not None else torch.zeros_like(pos_h)
        need = pos_h > lo
        prev_ok = torch.zeros(R, dtype=torch.bool, device="cpu")
        if R > 1:
            prev_ok[1:] = (seq_h[1:] == seq_h[:-1]) & (pos_h[1:] == pos_h[:-1] + 1) & (lo[1:] == lo[:-1])
        contiguous = bool((~need | prev_ok).all())
        groups = []
        for b in torch.unique(seq_h).tolist():
            idx = torch.nonzero(seq_h == b).reshape(-1)
            a, z = int(idx[0]), int(idx[-1]) + 1
            groups.append((b, slice(a, z) if z - a == idx.numel() else idx.to(device)))
        i32 = lambda t: t.to(device=device, dtype=torch.int32).contiguous()
        return QueryRows(i32(seq_h), i32(pos_h), None if fl_h is None else i32(fl_h),
                         torch.arange(R, dtype=torch.int32, device=device), tuple(groups), contiguous)

    @staticmethod
    def from_decoder_rows(rows, W):
        """From ``program.DecoderRows`` (trainer): its ``window_index`` must be the truncated windows at consecutive
        decoder rows (``DecoderSegments.window_rows``); checked here against ``positions`` / ``window_floor``."""
        dev = rows.positions.device
        q = QueryRows.make(rows.sequence.cpu(), rows.positions.cpu(),
                           floor=None if rows.window_floor is None else rows.window_floor.cpu(), device=dev)
        if rows.window_index is None:
            return q
        wi = rows.window_index.cpu().to(torch.int64)
        R = len(q)
        if tuple(wi.shape) != (R, W):
            raise ValueError("window_index: [rows, W]")
        pos = rows.positions.cpu().to(torch.int64)
        lo = (rows.window_floor.cpu().to(torch.int64).clamp(min=0) if rows.window_floor is not None
              else torch.zeros_like(pos))
        j = torch.arange(W)
        end = wi[:, W - 1] if R else wi.new_empty(0)
        q_pos = pos[:, None] - W + 1 + j[None]
        want = torch.where(q_pos >= lo[:, None], end[:, None] - (W - 1 - j)[None], -1)
        ok = torch.equal(want, wi) and (R == 0 or (int(end.min()) >= 0 and int(wi.max()) < R))
        if ok and R:
            # every referenced row is the same sequence and segment, at the slot's position
            seq = rows.sequence.cpu().to(torch.int64)
            ref = wi.clamp(min=0)
            live = wi >= 0
            ok = bool(((seq[ref] == seq[:, None]) & (pos[ref] == q_pos) & (lo[ref] == lo[:, None]) | ~live).all())
        if not ok:
            raise ValueError("window_index must hold each row's truncated window at consecutive decoder rows ending at "
                             "the row's own (DecoderSegments.window_rows)")
        q.window_end = end.to(device=dev, dtype=torch.int32).contiguous()
        q.window_contiguous = True
        return q


def _take(t, sel):
    return t[sel] if isinstance(sel, slice) else t.index_select(0, sel)


def _put(dst, sel, v):
    if isinstance(sel, slice):
        dst[sel] = v
    else:
        dst.index_copy_(0, sel, v)


def attention_rows(module, x, cfg, index, state, glob, rows, *, linear_fn, dequant_fn, recipe=None):
    return ops.drive(attention_rows_gen(module, x, cfg, index, state, glob, rows, linear_fn=linear_fn,
                                        dequant_fn=dequant_fn, recipe=recipe))


def attention_rows_gen(module, x, cfg, index, state, glob, rows, *, linear_fn, dequant_fn, recipe=None, tp=1):
    """Decoder layer ``index >= b`` of the trainer on the query rows ``rows`` (``QueryRows``), ``x [R, hidden]`` BF16
    (the layer's attention input at those rows); returns ``[R, hidden]`` (this rank's ``wo_b`` root, FP32 subtree
    root when ``tp > 1``, as ``ops.attention_full_gen``).

    ``index == b`` (the boundary's query side; ``module``: ``wq_a``, ``q_norm``, ``wq_b``, ``indexer.{wq_b,
    weights_proj}``, ``attn_sink``, ``wo_a``, ``wo_b``): the window is layer b's own, unclamped, from
    ``glob.window_kv``; the index scores cover the whole causal prefix of the row's sequence (``glob.index_keys``),
    the candidate pool and the top-k are published in ``state`` (``candidates_rows`` ``[R, blocks]``, ``topk_rows``
    ``[R, index_topk]``, ids of the row's own sequence).
    ``index > b``: the layer's own window keys of the rows (``wkv``, ``kv_norm``; straight-through QDQ) truncated at
    ``rows.floor``; a Reindex layer (an index source) scores the boundary's index keys under the candidate pool,
    a Reuse layer reads ``state["topk_rows"]``. ``state`` is shared by the decoder layers of one call (same rows).

    Index path detached (no gradient: the selector is frozen); gradients reach ``glob.window_kv`` / ``glob.comp_kv``
    (and through them ``entry_projection``) straight through."""
    yield from ()          # host protocol (tp.py, TreeReducer.run, drive): no intermediate reduction
    b = _decoder_layer(cfg, index)
    ops.supported_config(cfg)
    if glob.layer != b:
        raise ValueError(f"the global KV of layer {glob.layer} is not the boundary {b}'s")
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda or x.shape[0] != len(rows):
        raise ValueError("decoder rows expect CUDA BF16 [rows, hidden], one row per QueryRows entry")
    R, hid = x.shape
    H, D, W, eps = cfg.num_attention_heads // tp, cfg.head_dim, cfg.sliding_window, cfg.rms_norm_eps
    dev = x.device
    if R == 0:
        return torch.empty((0, hid), dtype=torch.bfloat16 if tp == 1 else torch.float32, device=dev)
    table = ops.rope_table(cfg, 1, glob.max_len, dev)
    pos = rows.pos
    x3 = x.view(1, R, hid)
    qa = linear_fn(module.wq_a, x3).reshape(R, -1)
    qr = ops._row(qa, module.q_norm.weight, spec.ROW_Q_NORM, cfg.q_lora_rank, eps=eps)
    q = linear_fn(module.wq_b, qr.view(1, R, -1)).reshape(R * H, D)
    q = ops.q_rope(q, D, table, ops.Pos(pos=pos, div=H))

    if index in cfg.index_source_layer_ids:
        from isoexec.ops.attention.indexer_sm100 import topk as dsa_topk
        cand_src, blk = cfg.candidate_source_layer_id, cfg.candidate_block_size
        with torch.no_grad():
            hi = cfg.index_n_heads
            codes, sf, w = ops.index_queries(module.indexer, x.detach(), qr.detach(), cfg, table, ops.Pos(pos=pos, div=hi),
                                             linear_fn, (1, R, hid), (1, R, cfg.q_lora_rank))
            ke = (pos + 1).to(torch.int32)                      # ratio 1: entries j <= p are visible
            topk = torch.full((R, cfg.index_topk), -1, dtype=torch.int32, device=dev)
            if index == cand_src:
                cands = torch.full((R, cfg.candidate_topk_blocks), -1, dtype=torch.int32, device=dev)
            else:
                cands = state.get("candidates_rows")
                if cands is None:
                    raise ValueError("the candidate source must precede its consumer on the same rows")
            for sq, sel in rows.groups:
                kc, ksf, L = glob.keys(sq)
                kg = _take(ke, sel).contiguous()
                sc = ops.index_scores(_take(codes, sel).contiguous(), _take(sf, sel).contiguous(),
                                      _take(w, sel).contiguous(), kg, width=L, keys=(kc, ksf))
                if index == cand_src:
                    _put(cands, sel, ops._candidate_select(sc, kg, cfg, dsa_topk)[0])
                else:
                    extension().candidate_mask(sc, kg, _take(cands, sel).contiguous(), blk, (L + blk - 1) // blk)
                _put(topk, sel, dsa_topk.dsa_topk(sc, kg, k=cfg.index_topk))
            if index == cand_src:
                state["candidates_rows"] = cands
            state["topk_rows"] = topk
    topk = state.get("topk_rows")
    if topk is None or topk.shape[0] != R:
        raise ValueError("the index source must precede its consumers on the same rows")

    if index == b:
        win, floor = glob.window_kv, None                         # layer b's window: token-wise, never truncated
        wend = (glob.kv_base.index_select(0, rows.seq.long()) + pos).to(torch.int32).contiguous()
    else:
        if not rows.window_contiguous:
            raise ValueError("decoder layers after the boundary need each row's window at the consecutive rows "
                             "ending at the row (segment rows contiguous, in position order)")
        kvr = linear_fn(module.wkv, x3).reshape(R, D).contiguous()
        win = ops._row(kvr, module.kv_norm.weight, spec.ROW_WINDOW_KV, 512, eps=eps, table=table,
                       pos=ops.Pos(pos=pos))
        floor, wend = rows.floor, rows.window_end
    kv2 = torch.cat((win, glob.comp_kv), 0)
    cbase = (win.shape[0] + glob.kv_base.index_select(0, rows.seq.long())).to(torch.int32).contiguous()
    lists = torch.empty((R, W + cfg.index_topk), dtype=torch.int32, device=dev)
    extension().build_lists(topk, pos, R, max(R, 1), 0, 0, 0, W, ops.list_mode(recipe), lists, floor, wend, cbase)
    o = ops.sink_attention(q.view(R, H, D), kv2, lists, module.attn_sink, D ** -0.5, recipe)
    return ops.output_projection(module, o, cfg, table, ops.Pos(pos=pos, div=H), 1, R, linear_fn, dequant_fn,
                                 tp=tp).view(R, -1)


# ---------------------------------------------------------------------------------------------- engine
def _layer_order(batch, index):
    """``paged.attention_paged_gen``'s forward bookkeeping: a layer at or below the last one seen starts a new forward."""
    last = batch.step.get("_layer")
    if last is not None and index <= last:
        batch.step.clear()
    batch.step["_layer"] = index


def _refuse_raw(linear_fn, *weights):
    """``paged._refuse_checkpoint_provider`` for the split holders (each holds part of the layer's projections)."""
    if getattr(linear_fn, "storage", None) in ("deepseek_fp8", "deepseek_fp4"):
        raise NotImplementedError("raw FP8/FP4 checkpoint projection providers are not served by this composition; "
                                  "load the checkpoint decoded to BF16 masters")
    for w in weights:
        ops._refuse_raw_checkpoint(w, "attention projection")


def _tables(cfg, index, batch):
    key = cfg.source_layer(index, "kv")
    by_source = batch.comp_bt_by_source is not None
    cbt = batch.comp_bt_by_source[key] if by_source else batch.comp_bt[1]
    ckey = (1, key) if by_source else 1
    ibt = cbt if batch.index_bt is None else batch.index_bt[1]
    if ibt is cbt and batch.comp_per != 64:
        raise ValueError("row-addressed compressed tables need their own index-key page table")
    return key, cbt, ckey, ibt


@torch.no_grad()
def entry_projection_paged(module, x, cfg, index, pools, batch, *, linear_fn):
    """``decoder.entry`` of the engine on EVERY scheduled token ``x [N, hidden]`` (the step's packed rows of ``h_b``):
    layer b's window-KV record into ``pools.window[b]``, the global compressed record into ``pools.comp[b]`` and the
    index key into ``pools.index[b]``, at the tokens' positions through the step's tables; padded rows
    (``batch.valid``) write nothing. The same kernels and bytes as ``paged.attention_paged_gen`` at the boundary
    (its one-launch fusions q_kv_post / index_kq_store write the same bytes)."""
    from . import paged
    check_boundary(cfg, index)
    N, hid = x.shape
    D, eps, dev = cfg.head_dim, cfg.rms_norm_eps, x.device
    _layer_order(batch, index)
    _refuse_raw(linear_fn, module.wkv.weight)
    table = ops.rope_table(cfg, 1, pools.max_len, dev)
    cos, sin = table
    pos = batch.tok_pos
    x3 = x.view(1, N, hid)
    win_bt = batch.win_bt[index] if isinstance(batch.win_bt, dict) else batch.win_bt
    if batch.window_rows is not None:
        window_rows = batch.window_rows[index]
    else:
        window_key = win_bt.data_ptr() if isinstance(batch.win_bt, dict) else None
        window_rows = paged._per_step(batch, ("window_rows", window_key),
                                      lambda: paged._rows(win_bt, batch.tok_seq, pos, batch.valid, batch.win_per))
    kvr = linear_fn(module.wkv, x3).reshape(N, D).contiguous()
    extension().paged_quant_write(0, kvr, ops.f32_param(module.kv_norm.weight), float(eps), cos, sin, pos, window_rows,
                                  pools.window[index])
    key, cbt, ckey, ibt = _tables(cfg, index, batch)
    comp = module.compressor
    lat = ops.mm(x.contiguous(), ops._bf16_weight(comp, "wkv", comp.wkv.weight, False))
    latent = ops._row(lat, comp.norm.weight, spec.ROW_LATENT_NORM, 512, eps=eps)
    dest = paged._per_step(batch, ("comp_rows", ckey),
                           lambda: paged._rows(cbt, batch.tok_seq, pos, batch.valid, batch.comp_per))
    idest = dest if ibt is cbt else paged._per_step(batch, ("index_rows", 1),
                                                    lambda: paged._rows(ibt, batch.tok_seq, pos, batch.valid))
    gpos = pos if (pos.dtype == torch.int32 and pos.is_contiguous()) else pos.to(torch.int32).contiguous()
    extension().paged_quant_write(1, latent, None, 0.0, cos, sin, gpos, dest, pools.comp[key])
    idx = module.indexer
    k = ops.mm(latent, ops._bf16_weight(idx, "wk", idx.wk.weight, False))
    if paged.INDEX_K_STORE and hasattr(extension(), "index_k_store"):
        extension().index_k_store(k, ops.f32_param(idx.k_norm.weight), float(eps), cos, sin, gpos, idest,
                                  pools.index[key])
    else:
        R = k.shape[0]
        codes = torch.empty((R, 128), dtype=torch.uint8, device=dev)
        sf = torch.empty((R, 1), dtype=torch.int32, device=dev)
        ops.row_op(spec.ROW_INDEX_K, k, 128, w=ops.f32_param(idx.k_norm.weight), eps=eps, table=table,
                   pos=ops.Pos(pos=gpos), codes=codes, sf=sf.view(R))
        extension().paged_index_write(codes, sf, idest, pools.index[key])


@torch.no_grad()
def query_paged(module, x, cfg, index, pools, batch, *, linear_fn, dequant_fn):
    return ops.drive(query_paged_gen(module, x, cfg, index, pools, batch, linear_fn=linear_fn, dequant_fn=dequant_fn))


def query_paged_gen(module, x, cfg, index, pools, batch, *, linear_fn, dequant_fn, tp=1):
    """The boundary's query side in the engine on the decoder rows ``x [R, hidden]`` of ``batch`` (a decoder-row batch,
    ``PagedBatch.rows_subset``; any subset of the step's tokens, after ``entry_projection_paged`` wrote the step's
    records): index queries, scores over the pages of the whole causal prefix, the candidate pool and the top-k
    (published in ``batch.step`` for the decoder layers after b, which run ``paged.attention_paged_gen`` on the same
    batch), the dual-rail sink attention over [layer b's unclamped window ; selected entries] (``window_floor`` is
    not applied: layer b's window keys are token-wise in the encoder output), ``wo_a``, ``wo_b``. The TP generator
    protocol of ``attention_paged_gen``: returns this rank's ``wo_b`` root."""
    yield from ()
    from isoexec.ops.attention.indexer_sm100 import topk as dsa_topk
    from isoexec.ops.attention.sparse_sm100.dualrail import dualrail_attn

    from . import paged
    check_boundary(cfg, index)
    R, hid = x.shape
    H, D, W, eps = cfg.num_attention_heads // tp, cfg.head_dim, cfg.sliding_window, cfg.rms_norm_eps
    dev = x.device
    _layer_order(batch, index)
    _refuse_raw(linear_fn, module.wq_a.weight, module.wq_b.weight)
    if R == 0:          # a step without decoder rows (a prefill chunk below the segment start)
        return torch.empty((0, hid), dtype=torch.bfloat16 if tp == 1 else torch.float32, device=dev)
    table = ops.rope_table(cfg, 1, pools.max_len, dev)
    pos = batch.tok_pos
    x3 = x.view(1, R, hid)
    qa = linear_fn(module.wq_a, x3).reshape(R, -1)
    qr = ops._row(qa, module.q_norm.weight, spec.ROW_Q_NORM, cfg.q_lora_rank, eps=eps)
    q = linear_fn(module.wq_b, qr.view(1, R, -1)).reshape(R * H, D)
    q = ops.q_rope(q, D, table, ops.Pos(pos=pos, div=H))
    key, cbt, ckey, ibt = _tables(cfg, index, batch)

    def make_ke():
        ke = torch.empty(R, dtype=torch.int32, device=dev)
        extension().index_bounds(pos.to(torch.int32).contiguous(), batch.valid, 1, ke)
        return ke
    ke = paged._per_step(batch, ("ke", 1), make_ke)
    hi = cfg.index_n_heads
    codes, sf, w = ops.index_queries(module.indexer, x.contiguous(), qr, cfg, table, ops.Pos(pos=pos, div=hi), linear_fn,
                                     x3.shape, (1, R, cfg.q_lora_rank))
    rbt = ibt if batch.index_read_bt is None else batch.index_read_bt[1]
    width = rbt.shape[1] * pools.index[key].shape[1]
    # the paged work division of these rows, shared with the Reindex layers of the same decoder-row batch
    # (paged.attention_paged_gen reads ("index_schedule", 1)): one launch per step, not one per consumer
    sched = paged._per_step(batch, ("index_schedule", 1), lambda: ops.index_schedule(ke, batch.tok_seq))
    scores = ops.index_scores(codes, sf, w.contiguous(), ke, width=width, cache=pools.index[key], block_table=rbt,
                              bt_rows=batch.tok_seq, schedule=sched)
    batch.step["candidates"] = ops._candidate_select(scores, ke, cfg, dsa_topk)[0]
    gen = batch.step.get("topk_generation", 0) + 1
    extra = None
    if (paged.LISTS_IN_PRODUCERS and scores.dtype == torch.bfloat16
            and hasattr(dsa_topk.extension("topk"), "topk_rows_fwd")):
        extra = batch.step[("extra_list", ckey, gen)] = torch.empty((R, cfg.index_topk), dtype=torch.int32, device=dev)
        topk = dsa_topk.dsa_topk(scores, ke, k=cfg.index_topk, rows_out=extra, rows_bt=cbt, rows_seq=batch.tok_seq,
                                 rows_per=batch.comp_per)
    else:
        topk = dsa_topk.dsa_topk(scores, ke, k=cfg.index_topk)
    batch.step["topk"] = topk
    batch.step["topk_generation"] = gen

    win_bt = batch.win_bt[index] if isinstance(batch.win_bt, dict) else batch.win_bt
    if batch.window_lists is not None and extra is not None:
        main = batch.window_lists[index]
    else:
        window_key = ("engine", batch.window_lists[index].data_ptr()) if batch.window_lists is not None else (
            win_bt.data_ptr() if isinstance(batch.win_bt, dict) else None)

        def make_lists():
            m = torch.empty((R, W), dtype=torch.int32, device=dev)
            e = torch.empty((R, cfg.index_topk), dtype=torch.int32, device=dev)
            tbl = win_bt if batch.window_lists is None else paged._zero_window_table(pools.max_len, dev)
            extension().paged_lists_masked(batch.tok_seq, pos, tbl, W, topk, cbt, m, e, batch.valid, batch.win_per,
                                           batch.comp_per, None)
            if batch.window_lists is not None:
                m = batch.window_lists[index]
            return m, e
        # the floor-free key: shared with the layers after b only when the batch carries no window floor
        main, extra = paged._per_step(batch, ("lists", window_key, ckey, gen, None), make_lists)
    o, _ = dualrail_attn(q.view(R, H, D), pools.window[index], main, D ** -0.5, sink=module.attn_sink,
                         extra_kv=pools.comp[key], extra_lists=extra,
                         extra_topk_length=paged.selected_lengths(batch, pos, 1, cfg.index_topk))
    return ops.output_projection(module, o, cfg, table, ops.Pos(pos=pos, div=H), 1, R, linear_fn, dequant_fn,
                                 tp=tp).view(R, -1)
