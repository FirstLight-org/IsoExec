"""Paged caches for engine hosting (vLLM / SGLang block tables), mixed prefill + decode batches.

Pools (engine-owned memory; ``PagedPools`` allocates them for tests and simple hosts), 64-token pages:

* window KV, one pool per layer: V4.1 FP8 records ``[pages, 64, 528]`` (512 E4M3 codes + 16 UE8M0 bytes,
  block 32) indexed by absolute position through ``win_bt`` (the engine may free pages that left the
  window);
* compressed KV, one pool per KV producer layer: the same V4.1 FP8 records ``[pages, 64, 528]`` (quantized from
  the FP32 RoPE output, port/CACHE-SPEC.md section 2) indexed by compressed position through ``comp_bt[ratio]``;
* index keys, one pool per KV producer: the DeepGEMM paged FP8 layout ``[pages, 64, 1, 132]`` (128 E4M3 codes per
  key, then one exponent word per key: four equal UE8M0 bytes), same page ids as the producer's compressed pool;
* partial-group compressor state, one pool per ratio > 1 producer: FP32 ``[slots, ratio, 1024]`` addressed
  by a per-sequence ``state_slot`` (like a recurrent-state slot).

Producer sharing is the full-sequence program's: consumers read their producer's pools; the window pool
is per layer. A step is a packed varlen batch: sequences with ``p0`` cached tokens and ``n`` new tokens
(prefill chunks and decode rows mixed, tokens of a sequence contiguous and in order). Attention is the
dual-rail sink recipe (``sm100_sparse_dualrail_sink_v1``, recipe 3) reading the FP8 window pool as
its main list (W fixed slots) and the FP8 compressed pool as its extra list -- bit for bit the contiguous
recipe-3 program (whose lists use the same fixed slots), so paged == contiguous == incremental == full.
Everything is device-resident and shape-static for a given (tokens, sequences, group bounds): a decode
step is CUDA-graph capturable once its plans are warm (tests/test_paged_gpu.py).
"""
from __future__ import annotations

import contextlib
import dataclasses
from dataclasses import dataclass, field

import os

import torch

from . import ops, spec
from ._extension import extension

__all__ = ["PagedPools", "PagedBatch", "attention_paged", "attention_paged_gen"]

RECIPE = 3


class PagedPools:
    def __init__(self, cfg, *, window_pages, comp_pages, state_slots, max_len, device="cuda", layers=None):
        ops.supported_config(cfg)
        self.cfg, self.max_len, self.device = cfg, max_len, torch.device(device)
        layers = range(cfg.num_hidden_layers) if layers is None else layers
        self.window = {i: torch.zeros((window_pages, 64, spec.KV_RECORD_BYTES), dtype=torch.uint8, device=device)
                       for i in layers}
        self.comp, self.index, self.state = {}, {}, {}
        for i in layers:
            r = cfg.compress_ratios[i]
            if r and i in cfg.kv_source_layer_ids:
                self.comp[i] = torch.zeros((comp_pages[r], 64, spec.KV_RECORD_BYTES), dtype=torch.uint8, device=device)
                self.index[i] = torch.zeros((comp_pages[r], 64, 1, 132), dtype=torch.uint8, device=device)
                if r > 1:
                    self.state[i] = torch.zeros((state_slots, r, 1024), dtype=torch.float32, device=device)


@dataclass
class PagedBatch:
    """Device metadata of one step (update in place between CUDA-graph replays)."""
    tok_seq: torch.Tensor          # [N] int32
    tok_pos: torch.Tensor          # [N] int32 absolute positions
    cu_q: torch.Tensor             # [B+1] int32
    p0: torch.Tensor               # [B] int32 tokens cached before this step
    win_bt: torch.Tensor | dict[int, torch.Tensor]  # shared or per-layer [B, Wp] int32
    comp_bt: dict                  # ratio -> [B, Cp] int32
    state_slot: torch.Tensor       # [B] int32
    groups: dict                   # ratio > 1 -> (g_seq [G] int32, g_k [G] int32): group slots (host bound)
    step: dict = field(default_factory=dict)
    valid: torch.Tensor | None = None  # [N] actual tokens; padded rows never write persistent KV
    index_bt: dict | None = None   # ratio -> [B, Cp] int32 index-key pages; None: the comp_bt page ids
    # Records per block-table entry of the window / compressed tables: 64 (entry = 64-record page id)
    # or 1 (entry = first record row of the page: block-chunk KV placement). Addressing only.
    win_per: int = 64
    comp_per: int = 64
    # Hosts whose engine computes the window addressing itself (vLLM's sliding-window metadata): per layer,
    # the physical window rows each token writes ([N] int64, -1 skips) and its main list ([N, W] int32:
    # window positions ascending as physical rows, -1 tail), the contract ``paged_lists`` produces.
    window_rows: dict | None = None
    window_lists: dict | None = None
    # ratio -> the index-key table the indexer reads: one entry per page of ``pools.index[source]``
    # (``index_bt`` is then the table of 64-row units the keys are written through). None: ``index_bt``.
    index_read_bt: dict | None = None
    # source layer -> its compressed table, for hosts whose sources of one ratio address their pages through
    # different table maps (each layer's own row view); None: ``comp_bt[ratio]``.
    comp_bt_by_source: dict | None = None
    # [N] int32 absolute window floor per row (CED bounded replay: the decoder segment start ``s``): a layer run on
    # this batch by ``attention_paged_gen`` reads window keys at positions [max(s, p - W + 1), p] only. None: no
    # floor (bit for bit the program without it). Set it on the decoder-row batch of the layers after the CED
    # boundary only (``ced.query_paged_gen``, the boundary layer, never applies it). Updated in place between
    # CUDA-graph replays like ``tok_pos``.
    window_floor: torch.Tensor | None = None
    # True: ``window_lists`` are already truncated at ``window_floor`` (a decoder-row batch whose lists the host built
    # in one launch, ``EngineStep.decoder_windows``); the layers read them as they are.
    lists_floored: bool = False

    @staticmethod
    def make(p0, n_new, win_bt, comp_bt, state_slot, ratios=(2,), device="cuda", index_bt=None, win_per=64, comp_per=64):
        """Host convenience: ``p0`` / ``n_new`` host ints per sequence."""
        tok_seq, tok_pos, cu = [], [], [0]
        for s, (a, n) in enumerate(zip(p0, n_new)):
            tok_seq += [s] * n
            tok_pos += list(range(a, a + n))
            cu.append(cu[-1] + n)
        groups = {}
        for r in ratios:
            if r > 1:
                gs, gk = [], []
                for s, n in enumerate(n_new):
                    for k in range((n + r - 1) // r):
                        gs.append(s)
                        gk.append(k)
                groups[r] = (torch.tensor(gs, dtype=torch.int32, device=device),
                             torch.tensor(gk, dtype=torch.int32, device=device))
        t = lambda v: torch.tensor(v, dtype=torch.int32, device=device)
        return PagedBatch(t(tok_seq), t(tok_pos), t(cu), t(list(p0)), ({i:t.to(torch.int32).contiguous() for i,t in win_bt.items()} if isinstance(win_bt,dict) else win_bt.to(torch.int32).contiguous()),
                          {r: b.to(torch.int32).contiguous() for r, b in comp_bt.items()}, t(list(state_slot)), groups,
                          index_bt=None if index_bt is None else {r: b.to(torch.int32).contiguous() for r, b in index_bt.items()},
                          win_per=win_per, comp_per=comp_per)

    def rows_subset(self, sel, *, window_floor=None, windows=True):
        """The batch of the rows ``sel`` ([R] int64 device indices into this step's tokens, ascending): the CED
        decoder rows of a step (``model/REQUESTS.md`` R3). Token metadata (``tok_seq``, ``tok_pos``, ``valid``, the
        engine's per-layer ``window_rows`` / ``window_lists``) is gathered at ``sel``; block tables, page geometry and
        ``state_slot`` are shared (the same sequences' pages); ``cu_q`` / ``p0`` / ``groups`` (the ratio > 1
        compressor's packing) are not defined for a subset: the batch serves ratio-1 / window layers only. A fresh
        ``step`` dict: the subset's per-step metadata (lists, top-k, candidates) never mixes with the full step's.
        Gathers only (graph-capturable). ``windows=False``: the engine's window rows / lists are left empty for the
        host to fill (``EngineStep.decoder_windows``: every decoder layer's in one launch)."""
        sel = sel.to(device=self.tok_seq.device, dtype=torch.int64)
        take = lambda t: t.index_select(0, sel).contiguous()
        floor = None
        if window_floor is not None:
            floor = window_floor.to(device=self.tok_seq.device, dtype=torch.int32).contiguous()
            if floor.numel() != sel.numel():
                raise ValueError("window_floor: one entry per selected row")
        return dataclasses.replace(
            self, tok_seq=take(self.tok_seq), tok_pos=take(self.tok_pos), cu_q=None, p0=None, groups={}, step={},
            valid=None if self.valid is None else take(self.valid),
            window_rows=None if self.window_rows is None else
            ({k: take(v) for k, v in self.window_rows.items()} if windows else {}),
            window_lists=None if self.window_lists is None else
            ({k: take(v) for k, v in self.window_lists.items()} if windows else {}),
            window_floor=floor, lists_floored=False)


def _rows(bt, seq, pos, mask=None, per=64):
    """Physical record rows ``bt[seq, pos // 64] * per + pos % 64`` as int64 (``per`` = records per table
    entry, 64 for page ids), ``-1`` where ``mask`` (bool or int32 per row) is zero: one integer-only launch
    (``paged_rows``) instead of the torch index glue."""
    out = torch.empty(seq.numel(), dtype=torch.int64, device=seq.device)
    extension().paged_rows(bt, seq.to(torch.int32).contiguous(), pos.to(torch.int32).contiguous(),
                           None if mask is None else mask.contiguous(), out, per)
    return out


_ZERO_TABLES: dict = {}


def _zero_window_table(max_len, device):
    """A zero window table of every sequence (row stride 0) wide enough for every position below ``max_len``:
    the main rows it yields are discarded (``PagedBatch.window_lists``). Allocated once per device, outside
    any capture."""
    key = (str(device), max_len)
    hit = _ZERO_TABLES.get(key)
    if hit is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("the zero window table must exist before graph capture (warm up eagerly first)")
        hit = _ZERO_TABLES[key] = torch.zeros((1, max_len // 64 + 1), dtype=torch.int32, device=device)
    return hit.expand(1 << 30, hit.shape[1])


def floor_window_lists(lists, pos, floor):
    """An engine main list (``PagedBatch.window_lists`` contract: per row the window positions ``[p - n + 1, p]``
    ascending as cache rows in slots ``[0, n)``, then -1) truncated at the per-row floor ``f``: positions
    ``[max(f, p - n + 1), p]`` moved to slots ``[0, n')``, then -1 -- the integers ``paged_lists_masked`` builds with
    the same floor. Device index arithmetic only (graph-capturable)."""
    R, W = lists.shape
    n = (lists >= 0).sum(1)
    drop = (floor.to(torch.int64) - (pos.to(torch.int64) - n + 1)).clamp(min=0)
    j = torch.arange(W, device=lists.device).unsqueeze(0) + drop.unsqueeze(1)
    kept = torch.gather(lists, 1, j.clamp(max=W - 1))
    return torch.where(j < n.unsqueeze(1), kept, -1).to(torch.int32).contiguous()


def _per_step(batch, key, make):
    """Step metadata shared by every layer of one step (block-table rows, attention lists, index bounds):
    computed by the first layer that needs it, then read by the others (``batch.step`` is cleared per
    step by every host, like ``topk``). Pure integer index math: no numerical effect."""
    hit = batch.step.get(key)
    if hit is None:
        hit = batch.step[key] = make()
    return hit


_ENCODED: dict = {}


def register_encoding(t, codes, scales, packed=None):
    """Record that ``codes``/``scales`` are the checkpoint FP8 activation encoding (``quantize_activation``,
    deepseek_fp8: 1x32 UE8M0) of the BF16 rows of ``t`` (one entry per device: the latest producer).
    A consumer finds it only for the same live storage at the same version (``encoding_of``).
    ``packed``: optionally the same scales in DeepGEMM's packed layout (the [m, K/128] int32 view
    ``layout.pack_activation_scales(scales)`` returns, byte for byte), read by ``encoding_packed_of``."""
    import weakref
    if t.is_inference():           # no version counter to validate against: never register (consumer re-quantizes)
        return
    _ENCODED[t.device] = (weakref.ref(t), t.data_ptr(), t.numel(), t._version, codes, scales)
    _PACKED[t.device] = packed


_PACKED: dict = {}


def encoding_packed_of(x):
    """The packed scales registered with ``x``'s encoding (None when absent or the encoding is stale)."""
    return _PACKED.get(x.device) if encoding_of(x) is not None else None


def encoding_of(x):
    hit = _ENCODED.get(x.device)
    if hit is None:
        return None
    ref, ptr, numel, version, codes, scales = hit
    if x.is_inference():
        return None
    t = ref()
    if (t is None or t.data_ptr() != ptr or t._version != version or x.data_ptr() != ptr or x.numel() != numel
            or x._version != version or codes.shape[-1] != x.shape[-1] or not x.is_contiguous()):
        return None
    return codes, scales


def collapse_rms_norm(streams, pre, weight, eps, *, encode=True):
    """``rms_norm(collapse(streams, pre))`` for mHC streams ``[..., 4, 5120]`` and the (shifted) ``pre
    [..., 4]`` in ONE launch: mhc_sm100's collapse program and norms_sm100 recipe v2, bit for bit
    (plus the registered FP8 encoding when ``encode``). Returns ``[..., 5120]`` or None (caller falls back)."""
    if (streams.shape[-1] != 5120 or streams.shape[-2] != 4 or weight.dtype != torch.bfloat16 or not streams.is_cuda
            or weight.numel() != 5120 or pre.dtype != torch.float32 or pre.shape != streams.shape[:-1]
            or (torch.is_grad_enabled() and (streams.requires_grad or weight.requires_grad or pre.requires_grad))):
        return None
    x = streams.contiguous().view(-1, 4, 5120)
    return _norm_launch(x, pre.contiguous().view(-1, 4), weight, eps, encode).view(*streams.shape[:-2], 5120)


def _norm_launch(x, pre, weight, eps, encode):
    rows = x.shape[0]
    # ``out`` carries a version counter even under ``inference_mode`` (vLLM's profile/dummy pass): inference
    # tensors have none, and the registered encoding is validated against it (``encoding_of``), so an in-place
    # write to ``out`` before the consumer still invalidates the encoding. No-op outside inference mode.
    with (torch.inference_mode(False) if encode and torch.is_inference_mode_enabled() else contextlib.nullcontext()):
        out = torch.empty((rows, 5120), dtype=torch.bfloat16, device=x.device)
    codes = scales = None
    if encode:
        codes = torch.empty((rows, 5120), dtype=torch.float8_e4m3fn, device=x.device)
        scales = torch.empty((rows, 160), dtype=torch.float8_e8m0fnu, device=x.device)
    extension().rms_quant_5120(x, pre, weight.contiguous(), float(eps), out, codes, scales)
    if encode:
        register_encoding(out, codes, scales)
    return out


#: The attention extra list written by the top-k emission (the selected ids as compressed-cache rows through the
#: layer's compressed table) instead of a paged_lists_masked launch; with the engine's own main list no list
#: launch remains. Same integers.
LISTS_IN_PRODUCERS = True

#: ratio > 1 compressor of a paged step in one launch (``comp_post``: pool + norm, record rows, the compressed-KV
#: record, the state tail); False: pool_varlen + state_update + paged_rows + paged_quant_write. Same bytes.
COMP_POST = True

#: index keys of a KV-source layer written by the row kernel itself into the paged pool (``index_k_store``);
#: False: row_op ROW_INDEX_K into codes / scales, then paged_index_write. Same bytes.
INDEX_K_STORE = True

#: index-source KV layers: the index keys stored by the index queries' row launch (``index_kq_store``) instead of a
#: launch of their own. Same bytes.
INDEX_KQ_STORE = True

#: The leaf rotates the query's RoPE tail itself (and, with one output group per rank, applies the output's inverse
#: RoPE): ``dualrail_attn(rope_q=True, rope_o=True)``, the same bits as ``ops.q_rope`` + ``ROW_OUT_INV_ROPE``
#: (sparse_sm100 tests/test_dualrail_fused_rope_gpu.py); two launches per layer fewer. False: the separate launches.
LEAF_ROPE = True

#: Index scores of prefill-shaped launches on the gathered route (``ops.index_scores(gather=True)``: the keys of the
#: step's sequences packed contiguous, the contiguous kernel), as native prefill gathers its keys for the non-paged
#: kernel; the paged kernel otherwise (decode). Chosen from the launch shape only (``index_gather_route``); both
#: routes give the same bits. None: the rule; True / False: always / never (where admitted).
INDEX_GATHER = None
#: the rule: at least this many rows, at least this many rows per sequence on average, and a packed key buffer
#: (min(rows, sequences) x width keys of 132 bytes) of at most this many keys
INDEX_GATHER_MIN_ROWS = 256
INDEX_GATHER_ROWS_PER_SEQ = 8
INDEX_GATHER_MAX_KEYS = 1 << 20


def index_gather_route(rows, seqs, table_rows, width):
    """The index-score route of one launch from its shape: ``rows`` query rows of ``seqs`` sequences (host shape)
    reading a ``table_rows``-row block table of ``width`` keys per row. Never data: an execution choice only."""
    from isoexec.ops.attention.indexer_sm100 import index_logits
    keys = index_logits.gather_capacity(rows, table_rows, width)
    if not (rows > 0 and keys > 0 and 1 <= table_rows <= index_logits.MAX_GATHER_SEQS
            and table_rows * width < (1 << 31)):
        return False
    if INDEX_GATHER is not None:
        return bool(INDEX_GATHER)
    return (rows >= INDEX_GATHER_MIN_ROWS and rows >= INDEX_GATHER_ROWS_PER_SEQ * seqs
            and keys <= INDEX_GATHER_MAX_KEYS)


#: q_norm + the window-KV record in one launch (``q_kv_post``); False: the separate q_norm row kernel and
#: paged_quant_write. Same output bytes either way.
BF16_Q_KV_POST = True


def _bf16_q_kv_post(module, cfg):
    """The one-launch q_norm + window-KV record applies: no-grad execution (the separate kernels carry the
    autograd form), the served geometry (q_lora_rank 1280, head_dim 512) and an extension that has it."""
    if not BF16_Q_KV_POST or cfg.q_lora_rank != 1280 or cfg.head_dim != 512:
        return False
    if torch.is_grad_enabled():
        return False
    return hasattr(extension(), "q_kv_post")


_SIDE_STREAMS: dict = {}
_AUX_STREAMS: dict = {}

#: Up to this many rows, the projections that read only the layer input x -- the compressor's [wkv | wgate]
#: (kv-source layers) and the indexer's weights_proj (index-source layers) -- run on an auxiliary stream
#: beside wq_a/wkv, the q norm and wq_b, as native vLLM does (``VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD``,
#: default 1024). Placement of work only: the same kernels on the same operands, joined before use.
MULTI_STREAM_TOKENS = int(os.environ.get("ISOEXEC_DS_MULTI_STREAM_TOKENS", "1024"))


class _Early:
    """Results of the auxiliary-stream projections; ``take`` joins the auxiliary stream into the current
    one (once) before the first use, and keeps the results' memory alive for the current stream."""

    def __init__(self, stream, values):
        self.stream, self.values, self.joined = stream, values, False

    def join(self):
        if not self.joined:
            torch.cuda.current_stream(self.stream.device).wait_stream(self.stream)
            self.joined = True

    def take(self, name):
        if name not in self.values:
            raise ValueError(f"no early projection {name!r}")
        cur = torch.cuda.current_stream(self.stream.device)
        self.join()
        t = self.values.pop(name)
        t.record_stream(cur)
        return t


def _refuse_checkpoint_provider(linear_fn, module):
    """Projections run through ``linear_fn`` on the layer's own weights: raw FP8/FP4 checkpoint providers and
    weights are refused (the decoded BF16 masters are served)."""
    if getattr(linear_fn, "storage", None) in ("deepseek_fp8", "deepseek_fp4"):
        raise NotImplementedError("raw FP8/FP4 checkpoint projection providers are not served by this composition; "
                                  "load the checkpoint decoded to BF16 masters")
    for p in (module.wq_a.weight, module.wkv.weight, module.wq_b.weight):
        ops._refuse_raw_checkpoint(p, "attention projection")


def _aux_stream(dev):
    """The per-device auxiliary stream, created OUTSIDE any CUDA-graph capture together with its mm_sm100
    arrival-counter buffer (stream-keyed; pre-sized like the capture stream's, ``mm_ops._counter_for``), so
    nothing is allocated for it inside a capture. None while capturing before it exists (the caller then runs
    the projections in place: same kernels, same bits)."""
    from isoexec.ops.residual.deepseek_mhc_sm100.ops import _side_streams_on
    if "aux" not in _side_streams_on():      # the mature single-stream order (callers run the work in place)
        return None
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    aux = _AUX_STREAMS.get(key)
    if aux is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        from isoexec.ops.mm.mm_sm100 import ops as mm_ops
        aux = torch.cuda.Stream(dev)
        with torch.cuda.stream(aux):
            mm_ops._counter_for(dev, 1 << 22)
        _AUX_STREAMS[key] = aux
    return aux


def _x_projections_aux(module, x, cfg, index, r, N, dev, weights_packed=False):
    """Issue the x-only projections on the auxiliary stream (see ``MULTI_STREAM_TOKENS``), or None.

    Ordering contract (MegaMoE, any collective): the region holds only finite compute kernels (mm_sm100 BF16
    GEMMs: no collective, no cross-rank or cross-kernel wait) and is joined into the layer's stream by
    ``_Early.join`` before the attention lists (``attention_paged_gen``), i.e. before wo_a / wo_b, the TP
    reduction of the layer output and the MoE that follow on that stream."""
    kv_src = bool(r) and index in cfg.kv_source_layer_ids
    idx_src = bool(r) and index in cfg.index_source_layer_ids and not weights_packed   # D2: weights in the one GEMM
    if not (kv_src or idx_src) or dev.type != "cuda" or torch.is_grad_enabled():
        return None
    aux = _aux_stream(dev)                 # created on the first eager source layer, whatever N is
    if aux is None or N == 0 or N > MULTI_STREAM_TOKENS:
        return None
    cur = torch.cuda.current_stream(dev)
    aux.wait_stream(cur)
    values = {}
    with torch.cuda.stream(aux):
        x2d = x.contiguous()
        if kv_src:
            comp = module.compressor
            if r == 1:
                values["comp"] = ops.mm(x2d, ops._bf16_weight(comp, "wkv", comp.wkv.weight, False))
            else:
                wcat = ops._cache(comp).get("wcat", (comp.wkv.weight, comp.wgate.weight),
                                            lambda: torch.cat((comp.wkv.weight, comp.wgate.weight), 0)
                                            .to(torch.bfloat16).contiguous())
                values["comp"] = ops.mm(x2d, wcat, root_fp32=True).contiguous()
        if idx_src:
            values["weights"] = ops.index_weights(module.indexer, x2d, cfg)
    x.record_stream(aux)
    x2d.record_stream(aux)
    return _Early(aux, values)


def _side_stream(device):
    key = device.index if device.index is not None else torch.cuda.current_device()
    stream = _SIDE_STREAMS.get(key)
    if stream is None:
        stream = _SIDE_STREAMS[key] = torch.cuda.Stream(device)
    return stream


@torch.no_grad()
def attention_paged(module, x, cfg, index, pools, batch, *, linear_fn, dequant_fn):
    return ops.drive(attention_paged_gen(module, x, cfg, index, pools, batch, linear_fn=linear_fn,
                                         dequant_fn=dequant_fn))


def attention_paged_gen(module, x, cfg, index, pools, batch, *, linear_fn, dequant_fn, tp=1):
    """One layer over a packed step ``x [N, hidden]`` in the TP generator protocol: the index scores are
    computed on every rank from all index heads (``ops.index_scores``, no collective), nothing is
    yielded, and the return value is this rank's ``wo_b`` root (reduced by the host)."""
    yield from ()          # host protocol (tp.py, TreeReducer.run, drive): no intermediate reduction
    N, hid = x.shape
    H, D, W, eps = cfg.num_attention_heads // tp, cfg.head_dim, cfg.sliding_window, cfg.rms_norm_eps
    r = cfg.compress_ratios[index]
    dev = x.device
    # Layers of one forward run in ascending index order; a layer at or below the last one seen starts a
    # new forward (a new step, a warmup repeat or a graph capture after warmup), so nothing derived by an
    # earlier forward (step metadata, top-k, candidates) can be read by this one, even when the host did
    # not clear ``batch.step``. Inside a capture this keeps every derived value a node of the graph.
    last = batch.step.get("_layer")
    if last is not None and index <= last:
        batch.step.clear()
    batch.step["_layer"] = index
    table = ops.rope_table(cfg, r, pools.max_len, dev)
    cos, sin = table
    x3 = x.view(1, N, hid)
    pos = batch.tok_pos
    fused_post = _bf16_q_kv_post(module, cfg)
    # the projections that read x as ONE GEMM when the provider packed them (``ops.pack_shared_input_rows``): wq_a |
    # wkv, and on index-source layers | the derived indexer weights (dense-lane D2: the weights are then a [N, 64]
    # column view, row stride 1856, read in place by the index-score kernel)
    packs = getattr(linear_fn, "packs", None) is not None and fused_post and not torch.is_grad_enabled()
    members = (module.wq_a, module.wkv)
    if packs and bool(r) and index in cfg.index_source_layer_ids:
        m3 = members + (ops.index_weights_holder(module.indexer, cfg),)
        if linear_fn.packs(m3):
            members = m3
    early = _x_projections_aux(module, x, cfg, index, r, N, dev, weights_packed=len(members) == 3)
    packed_weights = None
    win_bt = batch.win_bt[index] if isinstance(batch.win_bt, dict) else batch.win_bt
    window_key = win_bt.data_ptr() if isinstance(batch.win_bt, dict) else None
    window_rows_key = None
    if batch.window_rows is not None:
        # the engine's own window addressing (vLLM's sliding-window slots as rows of the pool view)
        window_rows = batch.window_rows[index]
        window_key = ("engine", batch.window_lists[index].data_ptr())
    else:
        window_rows_key = ("window_rows", window_key)
        window_rows = batch.step.get(window_rows_key)
        if window_rows is None and not (fused_post and hasattr(extension(), "q_kv_post_rows")):
            window_rows = _per_step(batch, window_rows_key,
                                    lambda: _rows(win_bt, batch.tok_seq, pos, batch.valid, batch.win_per))
    _refuse_checkpoint_provider(linear_fn, module)
    if fused_post:
        # wq_a | wkv as ONE GEMM when the provider packed them (native's fused_wqa_wkv; ``linear_fn.packed``:
        # column views of one output, each bitwise its separate projection), else wkv on the auxiliary stream
        # beside wq_a at decode sizes; then ONE launch for q_norm (BF16 qr, row_kernel ROW_Q_NORM's bytes) and
        # the window-KV record (paged_quant_write kind 0's bytes): q_kv_post (reads strided q / kv rows).
        # Without engine or step rows the same launch also computes the step's window record rows (``_rows``'
        # formula, shared by later layers of the window table). Placement and launch fusion only: every output
        # byte has the same per-element program as the separate kernels (tests/test_paged_glue_gpu.py).
        one_gemm = packs and linear_fn.packs(members)
        aux = (_aux_stream(dev) if (not one_gemm and dev.type == "cuda" and 0 < N <= MULTI_STREAM_TOKENS)
               else None)
        kv_done = None
        if one_gemm:
            outs = linear_fn.packed(members, x3)
            qa, kvr = outs[0].reshape(N, -1), outs[1].reshape(N, D)
            if len(members) == 3:
                hs = spec.index_q_heads(cfg.index_n_heads)
                packed_weights = outs[2].reshape(N, 64)[:, :hs]
        elif aux is not None:
            aux.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(aux):
                kvr = linear_fn(module.wkv, x3).reshape(N, D)
                kv_done = torch.cuda.Event()
                kv_done.record(aux)
            x3.record_stream(aux)
        if not one_gemm:
            qa = linear_fn(module.wq_a, x3).reshape(N, -1)
        if kv_done is None and not one_gemm:
            kvr = linear_fn(module.wkv, x3).reshape(N, D)
        elif kv_done is not None:
            torch.cuda.current_stream(dev).wait_event(kv_done)
            kvr.record_stream(torch.cuda.current_stream(dev))
        qr = torch.empty((N, cfg.q_lora_rank), dtype=torch.bfloat16, device=dev)
        if window_rows is None:
            window_rows = batch.step[window_rows_key] = torch.empty(N, dtype=torch.int64, device=dev)
            extension().q_kv_post_rows(qa, kvr, ops.f32_param(module.q_norm.weight),
                                       ops.f32_param(module.kv_norm.weight), float(eps), cos, sin, pos, window_rows,
                                       pools.window[index], qr, win_bt, batch.tok_seq,
                                       None if batch.valid is None else batch.valid.contiguous(), batch.win_per)
        else:
            extension().q_kv_post(qa, kvr, ops.f32_param(module.q_norm.weight), ops.f32_param(module.kv_norm.weight),
                                  float(eps), cos, sin, pos, window_rows, pools.window[index], qr)
    else:
        # separate launches (autograd through q_norm, or a build without q_kv_post): wkv + its window-KV
        # record on the auxiliary stream beside wq_a + q_norm at decode sizes, joined before anything reads the
        # pools. Placement only: the same kernels on the same operands; finite kernels only (ordering contract
        # of ``_x_projections_aux``).
        aux = _aux_stream(dev) if (dev.type == "cuda" and not torch.is_grad_enabled()
                                   and 0 < N <= MULTI_STREAM_TOKENS) else None
        kv_done = None
        if aux is not None:
            aux.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(aux):
                kvr = linear_fn(module.wkv, x3).reshape(N, D).contiguous()
                extension().paged_quant_write(0, kvr, ops.f32_param(module.kv_norm.weight), float(eps), cos, sin,
                                              pos, window_rows, pools.window[index])
                kv_done = torch.cuda.Event()
                kv_done.record(aux)
            for t in (x3, kvr, cos, sin, pos, window_rows):
                t.record_stream(aux)
        qa = linear_fn(module.wq_a, x3).reshape(N, -1)
        qr = ops._row(qa, module.q_norm.weight, spec.ROW_Q_NORM, cfg.q_lora_rank, eps=eps)
        if kv_done is None:
            kvr = linear_fn(module.wkv, x3).reshape(N, D).contiguous()
            extension().paged_quant_write(0, kvr, ops.f32_param(module.kv_norm.weight), float(eps), cos, sin,
                                          pos, window_rows, pools.window[index])
        else:
            torch.cuda.current_stream(dev).wait_event(kv_done)
    qr_linear = linear_fn
    # the leaf's fused RoPE: inference on the fp8 pools (always, on this path) with int32 positions
    # Only with one output group per rank (TP8: 8 heads): there both RoPE launches go and the Q tile is small; at 64
    # heads the leaf's warp-6 Q RoPE delays the UTCCP more than the launch it saves (runs/r5-1: +0.05-0.15 ms per step
    # at H 64, -0.02-0.10 ms at H 8), and the inverse RoPE stays a launch anyway (group-major permutation).
    leaf_rope = LEAF_ROPE and not torch.is_grad_enabled() and N > 0 and cfg.o_groups // tp == 1
    rope_o = leaf_rope
    # Index-source layers: the query projection + RoPE (needed only by the attention core) run on a side
    # stream while the compressor / indexer / score reduction run on the current one; the streams join
    # before the attention lists. Fixed program order, disjoint outputs: no bit depends on the overlap.
    from isoexec.ops.residual.deepseek_mhc_sm100.ops import _side_streams_on
    side = _side_stream(dev) if ("side" in _side_streams_on() and index in cfg.index_source_layer_ids and dev.type == "cuda"
                                 and not torch.is_grad_enabled()) else None
    # dense-lane D3: wq_b | the indexer's wq_b as ONE projection when the provider packed them (``ops.pack_qr_rows``);
    # the queries and the index queries are then column views (row stride H*D + index heads*128), read in place by the
    # leaf (fused RoPE) and the index-query row kernels; without the fused RoPE the query tails are rotated in a copy
    packed_iq = None
    if (side is not None and getattr(qr_linear, "packs", None) is not None
            and qr_linear.packs((module.wq_b, module.indexer.wq_b))):
        qf, iqf = qr_linear.packed((module.wq_b, module.indexer.wq_b), qr.view(1, N, -1))
        packed_iq = iqf.reshape(N, cfg.index_n_heads, 128)
        q = qf.reshape(N, H, D)
        if not leaf_rope:
            q = ops.q_rope(q.reshape(N * H, D), D, table, ops.Pos(pos=pos, div=H))
        side = None                     # computed on the current stream: nothing to join
    elif side is not None:
        main_stream = torch.cuda.current_stream(dev)
        side.wait_stream(main_stream)
        with torch.cuda.stream(side):
            q = qr_linear(module.wq_b, qr.view(1, N, -1)).reshape(N * H, D)
            q = q.contiguous() if leaf_rope else ops.q_rope(q, D, table, ops.Pos(pos=pos, div=H))
    else:
        q = qr_linear(module.wq_b, qr.view(1, N, -1)).reshape(N * H, D)
        q = q.contiguous() if leaf_rope else ops.q_rope(q, D, table, ops.Pos(pos=pos, div=H))
    extra = None
    if r:
        key = cfg.source_layer(index, "kv")
        by_source = batch.comp_bt_by_source is not None
        cbt = batch.comp_bt_by_source[key] if by_source else batch.comp_bt[r]
        ckey = (r, key) if by_source else r
        # Index keys: same logical pages as the compressed KV; own page ids when the host places the two pools
        # with different page strides (block-chunk KV placement). Addressing only: the same records.
        ibt = cbt if batch.index_bt is None else batch.index_bt[r]
        if ibt is cbt and batch.comp_per != 64:
            raise ValueError("row-addressed compressed tables need their own index-key page table")
        if index in cfg.kv_source_layer_ids:
            comp = module.compressor
            x2d = x.contiguous()
            if r == 1:
                lat = early.take("comp") if early is not None else ops.mm(x2d, ops._bf16_weight(comp, "wkv", comp.wkv.weight, False))
                latent = ops._row(lat, comp.norm.weight, spec.ROW_LATENT_NORM, 512, eps=eps)
                gseq, cidx = batch.tok_seq, pos
                dest = _per_step(batch, ("comp_rows", ckey), lambda: _rows(cbt, gseq, cidx, batch.valid, batch.comp_per))
                idest = dest if ibt is cbt else _per_step(batch, ("index_rows", r), lambda: _rows(ibt, gseq, cidx, batch.valid))
            else:
                wcat = ops._cache(comp).get("wcat", (comp.wkv.weight, comp.wgate.weight),
                                            lambda: torch.cat((comp.wkv.weight, comp.wgate.weight), 0)
                                            .to(torch.bfloat16).contiguous())
                proj = early.take("comp") if early is not None else ops.mm(x2d, wcat, root_fp32=True).contiguous()
                gseq, gk = batch.groups[r]
                G = gseq.numel()
                valid = torch.empty(G, dtype=torch.int32, device=dev)
                cidx = torch.empty(G, dtype=torch.int32, device=dev)
                st = pools.state[key]
                if COMP_POST and hasattr(extension(), "comp_post"):
                    # one launch: pool + RMSNorm (latent; invalid slots zero), gpos, the record rows of the
                    # compressed and index-key tables (``_rows``' formula on the same tables and ``comp_per``),
                    # the compressed-KV record and the state tail (== the six launches below, byte for byte)
                    latent = torch.empty((G, 512), dtype=torch.bfloat16, device=dev)
                    gpos = torch.empty(G, dtype=torch.int32, device=dev)
                    dest = torch.empty(G, dtype=torch.int64, device=dev)
                    idest = dest if ibt is cbt else torch.empty(G, dtype=torch.int64, device=dev)
                    extension().comp_post(proj, batch.cu_q, batch.p0, st, batch.state_slot, gseq, gk, r,
                                          ops.f32_param(comp.norm.weight), float(eps), latent, valid, cidx, gpos,
                                          cbt, batch.comp_per, dest, None if ibt is cbt else ibt,
                                          None if ibt is cbt else idest, cos, sin, pools.comp[key])
                else:
                    latent = (torch.zeros if batch.valid is not None else torch.empty)((G, 512), dtype=torch.bfloat16, device=dev)
                    extension().pool_varlen(proj, batch.cu_q, batch.p0, st, batch.state_slot, gseq, gk, r,
                                            ops.f32_param(comp.norm.weight), float(eps), latent, valid, cidx, True)
                    extension().state_update(proj, batch.cu_q, batch.p0, st, batch.state_slot, r)
                    dest = _rows(cbt, gseq, cidx, valid, batch.comp_per)
                    idest = dest if ibt is cbt else _rows(ibt, gseq, cidx, valid)
                    gpos = (cidx * r).to(torch.int32).contiguous()
                    extension().paged_quant_write(1, latent, None, 0.0, cos, sin, gpos, dest, pools.comp[key])
            if r == 1:
                # ratio 1: the group position is the token position itself (int32 contiguous: no copy launch)
                gpos = cidx if (cidx.dtype == torch.int32 and cidx.is_contiguous()) else cidx.to(torch.int32).contiguous()
                extension().paged_quant_write(1, latent, None, 0.0, cos, sin, gpos, dest, pools.comp[key])
            idx = module.indexer
            k = ops.mm(latent, ops._bf16_weight(idx, "wk", idx.wk.weight, False))
            pending_k = None
            if (INDEX_KQ_STORE and index in cfg.index_source_layer_ids and hasattr(extension(), "index_kq_store")
                    and not (torch.is_grad_enabled() and k.requires_grad)):
                # stored together with this layer's index queries, in their row launch (index_kq_store; 64- or
                # 128-entry pages addressed as paged_index_write does)
                pending_k = (k, gpos, idest, pools.index[key])
            elif INDEX_K_STORE and hasattr(extension(), "index_k_store"):
                # ROW_INDEX_K straight into the paged index-key pool (== row_op + paged_index_write, same bytes;
                # 64- or 128-entry pages addressed as paged_index_write does)
                extension().index_k_store(k, ops.f32_param(idx.k_norm.weight), float(eps), cos, sin, gpos, idest,
                                          pools.index[key])
            else:
                R = k.shape[0]
                codes = torch.empty((R, 128), dtype=torch.uint8, device=dev)
                sf = torch.empty((R, 1), dtype=torch.int32, device=dev)
                ops.row_op(spec.ROW_INDEX_K, k, 128, w=ops.f32_param(idx.k_norm.weight), eps=eps,
                           table=table, pos=ops.Pos(pos=gpos), codes=codes, sf=sf.view(R))
                extension().paged_index_write(codes, sf, idest, pools.index[key])
        if index in cfg.index_source_layer_ids:
            from isoexec.ops.attention.indexer_sm100 import topk as dsa_topk
            def make_ke():          # (pos + 1) // r, 0 on padded rows: one integer launch
                ke = torch.empty(N, dtype=torch.int32, device=dev)
                extension().index_bounds(pos.to(torch.int32).contiguous(), batch.valid, r, ke)
                return ke
            ke = _per_step(batch, ("ke", r), make_ke)
            # Every rank evaluates all index heads (ops.index_scores, the same bits at every TP degree): no score
            # collective.
            hi = cfg.index_n_heads
            pk = pending_k if index in cfg.kv_source_layer_ids else None
            if pk is not None:
                ops.check_index_query_format()
                # ops.index_queries with its row kernel shared with the index-key store: the indexer wq_b projection,
                # then ONE launch for ROW_INDEX_K (into the paged pool) + ROW_INDEX_Q (same bytes as the two launches)
                iq = (packed_iq if packed_iq is not None else
                      qr_linear(module.indexer.wq_b, qr.view(1, N, cfg.q_lora_rank)).reshape(N * hi, 128).contiguous())
                hs = spec.index_q_heads(hi)
                codes = torch.empty((N, hs, 128), dtype=torch.uint8, device=dev)
                sf = torch.empty((N, hs), dtype=torch.int32, device=dev)
                extension().index_kq_store(pk[0], ops.f32_param(module.indexer.k_norm.weight), float(eps), cos, sin,
                                           pk[1], pk[2], pk[3], iq, pos.to(torch.int32).contiguous(), hi,
                                           codes.view(N * hs, 128), sf.view(N * hs), hi, hi if hi < hs else 0)
                w = (packed_weights if packed_weights is not None else
                     early.take("weights") if early is not None and "weights" in early.values else
                     ops.index_weights(module.indexer, x.contiguous(), cfg))
            else:
                codes, sf, w = ops.index_queries(module.indexer, x.contiguous(), qr, cfg, table,
                                                 ops.Pos(pos=pos, div=hi), qr_linear, x3.shape, (1, N, cfg.q_lora_rank),
                                                 iq=packed_iq,
                                                 weights=(packed_weights if packed_weights is not None else
                                                          early.take("weights") if early is not None
                                                          and "weights" in early.values else None))
            rbt = ibt if batch.index_read_bt is None else batch.index_read_bt[r]
            width = rbt.shape[1] * pools.index[key].shape[1]
            # the token rows' block-table rows are read through tok_seq by the kernel (no per-step table copy); the
            # work division (paged) or the rows' key ranges (gathered) are one launch per step and ratio (functions of
            # ke and tok_seq only)
            seqs = batch.cu_q.numel() - 1 if batch.cu_q is not None else rbt.shape[0]
            if index_gather_route(N, seqs, rbt.shape[0], width):
                ranges = _per_step(batch, ("index_ranges", r, rbt.shape[0], width),
                                   lambda: ops.index_key_ranges(ke, batch.tok_seq, rbt.shape[0], width))
                scores = ops.index_scores(codes, sf, w, ke, width=width, cache=pools.index[key], block_table=rbt,
                                          bt_rows=batch.tok_seq, gather=True, ranges=ranges)
            else:
                sched = _per_step(batch, ("index_schedule", r), lambda: ops.index_schedule(ke, batch.tok_seq))
                scores = ops.index_scores(codes, sf, w, ke, width=width, cache=pools.index[key],
                                          block_table=rbt, bt_rows=batch.tok_seq, schedule=sched)
            if index == cfg.candidate_source_layer_id:
                batch.step["candidates"] = ops._candidate_select(scores, ke, cfg, dsa_topk)[0]
            elif cfg.candidate_source_layer_id < index:
                sel = batch.step.get("candidates")
                if sel is None:
                    raise ValueError("candidate source must precede its consumer in this step")
                blk = cfg.candidate_block_size
                extension().candidate_mask(scores, ke, sel, blk, (width + blk - 1) // blk)
            gen = batch.step.get("topk_generation", 0) + 1
            if (LISTS_IN_PRODUCERS and scores.dtype == torch.bfloat16
                    and hasattr(dsa_topk.extension("topk"), "topk_rows_fwd")):
                # the emission also writes the extra attention list of this compressed table (the selected ids as
                # cache rows ``cbt[tok_seq, j // 64] * comp_per + j % 64``; masked rows have ke = 0, i.e. all -1):
                # paged_lists_masked's extra list
                extra_list = batch.step[("extra_list", ckey, gen)] = torch.empty((N, cfg.index_topk),
                                                                                  dtype=torch.int32, device=dev)
                batch.step["topk"] = dsa_topk.dsa_topk(scores, ke, k=cfg.index_topk, rows_out=extra_list, rows_bt=cbt,
                                                       rows_seq=batch.tok_seq, rows_per=batch.comp_per)
            else:
                batch.step["topk"] = dsa_topk.dsa_topk(scores, ke, k=cfg.index_topk)
            batch.step["topk_generation"] = gen
        topk = batch.step.get("topk")
        if topk is None:
            raise ValueError("index source must precede its consumers in this step")

    if side is not None:
        main_stream.wait_stream(side)
        q.record_stream(main_stream)
    if early is not None:
        early.join()                   # every auxiliary-stream projection is consumed above; join regardless

    floor = batch.window_floor

    def engine_main():
        # the engine's own main list, truncated at the window floor when the batch carries one (floor_window_lists)
        if floor is None or batch.lists_floored:
            return batch.window_lists[index]
        return _per_step(batch, ("floored", batch.window_lists[index].data_ptr()), lambda: floor_window_lists(batch.window_lists[index], pos, floor))

    def make_lists():
        main = torch.empty((N, W), dtype=torch.int32, device=dev)
        ext = torch.empty((N, cfg.index_topk), dtype=torch.int32, device=dev) if r else None
        # with the engine's own main list the kernel builds the extra list only (its main rows go to scratch)
        table = win_bt if batch.window_lists is None else _zero_window_table(pools.max_len, dev)
        extension().paged_lists_masked(batch.tok_seq, pos, table, W, topk if r else None,
                                       cbt if r else None, main, ext, batch.valid, batch.win_per, batch.comp_per,
                                       floor)
        if batch.window_lists is not None:
            main = engine_main()
        return main, ext
    # one (main, extra) pair per (ratio, top-k selection, floor): identical for every layer reading them. With the
    # engine's main list and (compressed layers) the extra list the top-k emission wrote for this table, no list launch.
    gen = batch.step.get("topk_generation") if r else None
    extra = batch.step.get(("extra_list", ckey, gen)) if r else None
    if batch.window_lists is not None and (not r or extra is not None):
        main = engine_main()
    else:
        main, extra = _per_step(batch, ("lists", window_key, ckey if r else r, gen,
                                        None if floor is None else "floor"), make_lists)
    from isoexec.ops.attention.sparse_sm100.dualrail import dualrail_attn
    o, _ = dualrail_attn(q.view(N, H, D), pools.window[index], main, D ** -0.5, sink=module.attn_sink,
                         extra_kv=pools.comp[cfg.source_layer(index, "kv")] if r else None, extra_lists=extra,
                         rope=(cos, sin, pos) if leaf_rope else None, rope_q=leaf_rope, rope_o=rope_o)
    return ops.output_projection(module, o, cfg, table, ops.Pos(pos=pos, div=H), 1, N, linear_fn, dequant_fn,
                                 tp=tp, o_roped=rope_o).view(N, -1)
