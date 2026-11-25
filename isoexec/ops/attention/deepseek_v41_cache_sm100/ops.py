"""Native SM100 DeepSeek V4.1 attention composition (forward, decode cache, BF16-actor backward).

``attention_full(module, x, cfg, index, state, linear_fn=..., dequant_fn=...)`` is the native
counterpart of ``isoexec.nn.deepseek_attention.reference_attention`` (same arguments except the
quantizer callback: the activation quantizers are fixed by the recipe). The checkpoint projections
``wq_a``, ``wq_b``, ``wkv``, ``indexer.wq_b`` and ``wo_b`` are separate ``mm.deepseek_v41.*`` ops and
are executed through ``linear_fn``; everything else is native:

* row kernels of this package (csrc/rows.cu): RMSNorm, RoPE / inverse RoPE, the cache-record quantizer
  (``deepseek_v41_cache_record_quant_v1``, port/CACHE-SPEC.md: FP8 528 B KV records, FP8 index keys) and its
  quantize-then-dequantize form for the trainer, compressor pooling + norm, candidate blocks, lists;
* ``mm/mm_sm100`` (``sm100_tcgen05_segtree_v1``) for the BF16 projections inside the composition
  (compressor, index key/weights, grouped ``wo_a``), FP32 root for the ratio>1 compressor; at decode rows
  the same segment trees with the segments on vendored DeepGEMM (``mm/deepgemm_sm100/segtree.py``, also
  ``wo_b``'s leaf tree): the same bits;
* ``attention/sparse_sm100``: index scores (``sm100_dsa_index_logits_dg_v1``: DeepGEMM's SM100 MQA logits, vendored; FP8
  keys and queries, one exponent per row),
  top-k (``sm100_dsa_topk_v1``), learned-sink sparse attention (``sm100_sparse_sink_v1``,
  G = 1 chain) and its no-atomic backward.

Torch glue is limited to allocation, views, contiguous copies of operands, ``arange`` index prep and
one exact power-of-two scaling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import spec
from ._extension import extension

__all__ = ["attention_full", "rope_table", "row_op", "compressor_full", "attention_full_gen", "drive",
           "sink_attention", "supported_config", "DerivedWeightGuard", "WeightCache", "window_kv_qdq",
           "compressed_kv_qdq"]

_HUGE = 1 << 40


# ---------------------------------------------------------------------------------------------
# rotary table
_TABLES: dict = {}


def _rope_key(cfg, ratio):
    if ratio:
        sc = cfg.rope_scaling
        return (cfg.qk_rope_head_dim, float(cfg.compress_rope_theta),
                (float(sc["factor"]), float(sc["beta_fast"]), float(sc["beta_slow"]),
                 int(sc["original_max_position_embeddings"])))
    return (cfg.qk_rope_head_dim, float(cfg.rope_theta), None)


def rope_table(cfg, ratio, length, device):
    """(cos, sin) fp32 ``[P >= length, rd/2]``: the official ``precompute_freqs_cis`` on the CPU.

    Compressed layers use ``compress_rope_theta`` with YaRN, window-only layers ``rope_theta`` without.
    Row p is independent of the table length (``outer(arange, freqs)`` per row), so the table grows by
    recomputation without changing any existing row.
    """
    from .reference import rope_frequencies

    key = (_rope_key(cfg, ratio), str(device))
    hit = _TABLES.get(key)
    if hit is not None and hit[2] >= length:
        return hit[0], hit[1]
    cap = max(1024, 1 << max(0, int(length - 1).bit_length()))
    rd, base, scaling = key[0]
    if scaling is None:
        f = rope_frequencies(rd, cap, 0, base, 1.0, 1.0, 1.0)
    else:
        f = rope_frequencies(rd, cap, scaling[3], base, scaling[0], scaling[1], scaling[2])
    real = torch.view_as_real(f)
    cos = real[..., 0].contiguous().to(device)
    sin = real[..., 1].contiguous().to(device)
    _TABLES[key] = (cos, sin, cap)
    return cos, sin


# ---------------------------------------------------------------------------------------------
# row kernels
@dataclass(frozen=True)
class Pos:
    """pos(r) = (pos[r // div] if pos is not None else (r // div) % mod) * mul + add."""
    div: int = 1
    mod: int = _HUGE
    mul: int = 1
    add: int = 0
    pos: torch.Tensor | None = None


def row_op(kind, x, D, *, w=None, eps=0.0, table=None, pos=Pos(), out=None, codes=None, sf=None,
           dst=(_HUGE, 0, 0), group_major=(0, 1, 1), flag=None, pad_heads=0):
    cos, sin = table if table is not None else (None, None)
    extension().row_op(kind, x, D, w, float(eps), cos, sin, pos.pos, pos.div, pos.mod, pos.mul, pos.add,
                       out, codes, sf, dst[0], dst[1], dst[2], group_major[0], group_major[1], group_major[2],
                       flag, pad_heads)
    return out


def rope_tail_(x, D, table, pos, *, inverse=False):
    """In-place RoPE of the last 64 elements of each bf16 row (inference only; same scalar graph)."""
    if pos.add:
        raise ValueError("in-place rope takes no position offset")
    extension().rope_tail_inplace(x, D, table[0], table[1], pos.pos, pos.div, pos.mod, pos.mul, inverse)
    return x


def q_rope(q, D, table, pos):
    """RoPE of the query tails: in place when no gradient is recorded (the projection output is
    consumed), otherwise the out-of-place differentiable row kernel."""
    if torch.is_grad_enabled() and q.requires_grad:
        return _row(q, None, spec.ROW_Q_ROPE, D, table=table, pos=pos)
    return rope_tail_(q.contiguous(), D, table, pos)


def row_bwd(kind, x, dy, D, *, w=None, eps=0.0, table=None, pos=Pos(), dw_rows=None,
            dst=(_HUGE, 0, 0), group_major=(0, 1, 1)):
    dx = torch.empty((x.shape[0], D), dtype=torch.bfloat16, device=x.device)
    cos, sin = table if table is not None else (None, None)
    extension().row_bwd(kind, x, dy, D, w, float(eps), cos, sin, pos.pos, pos.div, pos.mod, pos.mul, pos.add,
                        dx, dw_rows, dst[0], dst[1], dst[2], group_major[0], group_major[1], group_major[2])
    return dx


_F32: dict = {}
#: bumped whenever a NEW derived-weight entry (or storage) appears; an in-place refresh keeps it
_DERIVED_GENERATION = [0]
#: bumped whenever a derived-weight entry records new source keys (derivation, re-derivation, refresh)
_DERIVED_REVISION = [0]


def _same_layout(old, new):
    if isinstance(old, (tuple, list)):
        return (isinstance(new, (tuple, list)) and len(old) == len(new)
                and all(_same_layout(a, b) for a, b in zip(old, new)))
    return (isinstance(old, torch.Tensor) and isinstance(new, torch.Tensor) and old.shape == new.shape
            and old.dtype == new.dtype and old.device == new.device and old.stride() == new.stride())


def _copy_into(old, new):
    """Write a re-derived weight into the existing storage: captured CUDA graphs keep its address."""
    if isinstance(old, (tuple, list)):
        for a, b in zip(old, new):
            _copy_into(a, b)
        return
    if old.data_ptr() == new.data_ptr():           # an alias of its own source parameter: already current
        return
    with torch.no_grad(), (torch.inference_mode() if old.is_inference() else _nullcontext()):
        old.copy_(new)


def _nullcontext():
    import contextlib
    return contextlib.nullcontext()


def f32_param(p):
    """FP32 contiguous copy of a (norm / sink) parameter, cached per live tensor object (weak reference)
    and its (data_ptr, _version, shape): one conversion per weight update instead of one per call. The
    weak reference makes a recycled ``id``/address of a freed tensor a miss, never a stale hit.
    A version miss of the same live tensor re-derives INTO the existing copy (same address), so a captured
    graph reading it observes the update once the host refreshes (``refresh_derived_weights``)."""
    import weakref
    key = (p.data_ptr(), p._version, tuple(p.shape), p.dtype)
    hit = _F32.get(id(p))
    if hit is not None and hit[0]() is p and hit[1] == key:
        return hit[2]
    with torch.no_grad():
        v = p.detach().float().contiguous()
    if hit is not None and hit[0]() is p and _same_layout(hit[2], v):
        _copy_into(hit[2], v)
        v = hit[2]
    else:
        _DERIVED_GENERATION[0] += 1
    _F32[id(p)] = (weakref.ref(p), key, v)
    _DERIVED_REVISION[0] += 1
    return v


def segment_sum(rows):
    out = torch.empty(rows.shape[1], dtype=torch.float32, device=rows.device)
    extension().segment_sum(rows.contiguous(), spec.WGRAD_SEGMENT, out)
    return out


class _RowFn(torch.autograd.Function):
    """One row kernel with an owned backward. ``w`` may be None (no norm)."""

    @staticmethod
    def forward(ctx, x, w, kind, D, eps, table, pos, out_rows, dst, group_major):
        x = x.contiguous()
        wf = None if w is None else f32_param(w)
        out = torch.empty((out_rows, D), dtype=torch.bfloat16, device=x.device)
        row_op(kind, x, D, w=wf, eps=eps, table=table, pos=pos, out=out, dst=dst, group_major=group_major)
        ctx.save_for_backward(x, wf)
        ctx.meta = (kind, D, eps, table, pos, dst, group_major, w is not None and w.requires_grad,
                    None if w is None else w.dtype)
        return out

    @staticmethod
    def backward(ctx, dy):
        x, wf = ctx.saved_tensors
        kind, D, eps, table, pos, dst, gm, need_dw, wdtype = ctx.meta
        dw_rows = torch.empty((x.shape[0], D), dtype=torch.float32, device=x.device) if need_dw else None
        dx = row_bwd(kind, x, dy.contiguous().to(torch.bfloat16), D, w=wf, eps=eps, table=table, pos=pos,
                     dw_rows=dw_rows, dst=dst, group_major=gm)
        dw = segment_sum(dw_rows).to(wdtype) if need_dw else None
        return dx, dw, None, None, None, None, None, None, None, None


def _row(x, w, kind, D, *, eps=0.0, table=None, pos=Pos(), out_rows=None, dst=(_HUGE, 0, 0),
         group_major=(0, 1, 1)):
    rows = x.shape[0] if out_rows is None else out_rows
    if torch.is_grad_enabled() and (x.requires_grad or (w is not None and w.requires_grad)):
        return _RowFn.apply(x, w, kind, D, eps, table, pos, rows, dst, group_major)
    out = torch.empty((rows, D), dtype=torch.bfloat16, device=x.device)
    wf = None if w is None else f32_param(w)
    return row_op(kind, x.contiguous(), D, w=wf, eps=eps, table=table, pos=pos, out=out, dst=dst,
                  group_major=group_major)


def window_kv_qdq(kvr, w, eps, table, pos, out=None):
    """Window KV as the sparse-attention leaf reads it back from its FP8 528 B record (CACHE-SPEC section 1): BF16
    ``[R, 512]`` = dequant(quant(RoPE(RMSNorm(kvr)))), bit for bit the values the paged writer stores. The trainer
    side of the cache program; backward is straight-through (``row_bwd``)."""
    if out is None:
        out = torch.empty((kvr.shape[0], 512), dtype=torch.bfloat16, device=kvr.device)
    return row_op(spec.ROW_WINDOW_KV, kvr.contiguous(), 512, w=f32_param(w), eps=eps, table=table, pos=pos, out=out)


def compressed_kv_qdq(latent, table, pos):
    """Compressed KV as the leaf reads it back from its FP8 528 B record (CACHE-SPEC section 2): BF16 ``[G, 512]`` =
    dequant(quant(RoPE_fp32(latent))) at the group positions ``pos``. Differentiable (straight-through)."""
    return _row(latent, None, spec.ROW_COMPRESSED_KV, 512, table=table, pos=pos)


# ---------------------------------------------------------------------------------------------
# GEMMs (mm_sm100, sm100_tcgen05_segtree_v1)
def _mm_ops():
    from isoexec.ops.mm.mm_sm100 import ops as mm_ops
    from isoexec.ops.mm.mm_sm100 import segments as mm_seg
    return mm_ops, mm_seg


def mm(a, w, *, root_fp32=False, split=None, decode_tree=True):
    """``a[M,K] @ w[N,K]^T``; BF16 in, FP32 accumulate, BF16 out (or the FP32 tree root). ``decode_tree``: up to
    ``SEGTREE_MM_ROWS`` rows the tree's segments may run on DeepGEMM (same bits); False keeps the mm_sm100 plans."""
    mm_ops, mm_seg = _mm_ops()
    M, K = a.shape
    N = w.shape[0]
    if not mm_seg.supported(K, N):
        raise ValueError(f"mm_sm100 declines K={K} N={N}")
    if M == 0:
        return torch.empty((0, N), dtype=torch.float32 if root_fp32 else torch.bfloat16, device=a.device)
    a = a.contiguous()
    if (decode_tree and split is None and M <= (SEGTREE_MM_NARROW_ROWS if N <= 64 else SEGTREE_MM_ROWS) and a.is_cuda
            and not (a.requires_grad or w.requires_grad)):
        # decode rows: the same published-S tree with one CTA per segment (DeepGEMM segment GEMMs + the
        # tree fold, ``mm/deepgemm_sm100/segtree.py``): bitwise this function's mm_sm100 plans
        from isoexec.ops.mm.deepgemm_sm100 import segtree
        S = mm_seg.segment_count(K, torch.bfloat16)
        if segtree.serves(N, K, S):
            return segtree.linear(a, w.contiguous(), S, out_dtype=torch.float32 if root_fp32 else torch.bfloat16)
    b = w.contiguous().t()
    if root_fp32 and M < _MIN_ROOT_ROWS:
        # the FP32-root publishing plans need a TMA row pitch of >= 16 rows; rows are independent in
        # mm_sm100 (a row's accumulation reads only its own A row), so zero rows change no bit
        return mm(_padded_rows(a, _MIN_ROOT_ROWS), w, root_fp32=True, split=split, decode_tree=decode_tree)[:M]
    if split is not None:
        # an explicit leaf tree (split segments of K/split), e.g. split = 1: ONE ascending FP32 chain
        if not root_fp32:
            raise ValueError("an explicit split publishes its FP32 root")
        wc = w.contiguous()
        plan, publish = _wob_plan(a, wc, split, True)
        return _wob_run(a, wc, plan, publish, True)
    # root_plan_for: a cut-tree plan (mm_sm100 policy v3) has no single-CTA root, so the fp32 root
    # comes from a whole-tree publisher at the same S (same tree, same bits: the plan relation)
    key = (root_fp32, M, K, N)
    plan = _PLANS.get(key)
    if plan is None:        # a total function of the shape and the static tables: computed once per shape
        plan = _PLANS[key] = (mm_seg.root_plan_for if root_fp32 else mm_seg.plan_for)(M, K, N, torch.bfloat16)
    return mm_ops.run_plan(a, b, plan, root_dtype=torch.float32 if root_fp32 else torch.bfloat16)


_MIN_ROOT_ROWS = 16
#: Rows up to which ``mm`` (the compressor, the indexer's weights_proj / wk) runs ``segtree`` (same bits).
SEGTREE_MM_ROWS = 128
#: The same for weights of <= 64 rows (the indexer's weights_proj): measured faster up to 1024 rows (8.7 vs 9.9 us).
SEGTREE_MM_NARROW_ROWS = 1024
#: Rows up to which ``wo_b``'s leaf tree runs ``segtree`` with the TP leaves as segments (same bits).
SEGTREE_WOB_ROWS = 16
#: Rows above SEGTREE_WOB_ROWS up to which ``wo_b`` runs DeepGEMM's on-chip leaf-tree GEMM (``segtree.leaf_linear``:
#: one FP32 accumulator per leaf in tensor memory, folded in the epilogue; same bits as mm_sm100's on-chip tree).
#: Measured faster than mm_sm100's tree at every M >= 32 (TP2: 13.8 vs 16.6 us at M = 128, 67 vs 75 at 2048).
ONCHIP_WOB_ROWS = 1 << 30
_PLANS: dict = {}
#: ``(device, stream, rows, K, dtype) -> [buffer, rows last written]``: the zero-padded A of a repeated
#: small-M FP32-root GEMM (the decode head), written in place instead of concatenated per call.
_PADDED: dict = {}


def _padded_rows(x, rows):
    """``x`` zero-padded to ``rows`` rows (``x.shape[0] < rows``). Outside a capture the padded copy is a
    persistent per-stream buffer: one copy of ``x``, and its rows past ``x`` are zero (rows a previous,
    longer ``x`` wrote are cleared), the same values as ``_pad_rows``."""
    if not x.is_cuda or torch.cuda.is_current_stream_capturing():
        return _pad_rows(x, rows)
    key = (x.device, torch.cuda.current_stream(x.device).cuda_stream, rows, x.shape[1:], x.dtype)
    hit = _PADDED.get(key)
    if hit is None:
        hit = _PADDED[key] = [x.new_zeros((rows, *x.shape[1:])), 0]
    buf, last = hit
    M = x.shape[0]
    buf[:M].copy_(x)
    if last > M:
        buf[M:last].zero_()
    hit[1] = M
    return buf


def _pad_rows(x, multiple):
    r = (-x.shape[0]) % multiple
    if not r:
        return x
    return torch.cat((x, x.new_zeros((r, *x.shape[1:]))), 0)


def mm_grads(dy, a, w, *, need_da=True, need_dw=True):
    """Backward of ``a @ w^T``: dA = dY W and dW = dY^T A by mm_sm100/general, reading W, dY^T and A in
    place (recipe ``sm100_tcgen05_ksegtree_v1``: MN-major operands, K cut into fixed segments by K only).
    The token axis (dW's K) must be a multiple of 64; otherwise dY and A are zero-padded (exact zero
    products). Deterministic, no atomics; not batch-invariant (training only)."""
    from isoexec.ops.mm.mm_sm100.general import ops as gen
    dy = dy.to(torch.bfloat16)
    if dy.stride(-1) != 1:
        dy = dy.contiguous()
    da = dw = None
    if need_da:
        wb = w if w.dtype == torch.bfloat16 else w.to(torch.bfloat16)
        if gen.supported(dy, wb):
            da = gen.mm(dy, wb)
        else:
            da = mm(dy.contiguous(), wb.t().contiguous())
    if need_dw:
        ab = a if a.dtype == torch.bfloat16 else a.to(torch.bfloat16)
        if ab.stride(-1) != 1:
            ab = ab.contiguous()
        if dy.shape[0] % 64:
            dy, ab = _pad_rows(dy.contiguous(), 64), _pad_rows(ab.contiguous(), 64)
        if gen.supported(dy.t(), ab):
            dw = gen.mm(dy.t(), ab)
        else:
            dw = mm(_pad_rows(dy, 256).t().contiguous(), _pad_rows(ab, 256).t().contiguous())
    return da, dw


class _MmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, w, root_fp32, split=None):
        ctx.save_for_backward(a, w)
        return mm(a, w, root_fp32=root_fp32, split=split)

    @staticmethod
    def backward(ctx, dy):
        a, w = ctx.saved_tensors
        da, dw = mm_grads(dy, a, w, need_da=ctx.needs_input_grad[0], need_dw=ctx.needs_input_grad[1])
        return da, (None if dw is None else dw.to(w.dtype)), None, None


def linear(a, w, *, root_fp32=False, split=None, decode_tree=True):
    if torch.is_grad_enabled() and (a.requires_grad or w.requires_grad):
        return _MmFn.apply(a, w, root_fp32, split)
    return mm(a, w, root_fp32=root_fp32, split=split, decode_tree=decode_tree)


# ---------------------------------------------------------------------------------------------
# weight cache (invalidated by any in-place parameter update or parameter replacement)
class WeightCache:
    """Derived inference weights keyed by (id, data_ptr, _version) of their source parameters.

    A key miss of an existing entry with the same layout re-derives into the existing tensors (stable addresses
    for captured graphs); :meth:`refresh` re-derives every entry in place unconditionally (the weight-update
    boundary of a graph-serving host) and :meth:`stale` names entries whose sources changed since."""

    def __init__(self):
        self._store = {}

    @staticmethod
    def _key(sources):
        return tuple((id(p), p.data_ptr(), p._version) for p in sources)

    def get(self, name, sources, make):
        key = self._key(sources)
        hit = self._store.get(name)
        if hit is not None and hit[0] == key:
            return hit[1]
        with torch.no_grad():
            value = make()
        if hit is not None and _same_layout(hit[1], value):
            _copy_into(hit[1], value)
            # the same tensors (graph-stable storage); a tuple entry is a new tuple object of them
            value = hit[1] if isinstance(hit[1], torch.Tensor) else type(hit[1])([*hit[1]])
        else:
            _DERIVED_GENERATION[0] += 1
        self._store[name] = (key, value, tuple(sources), make)
        _DERIVED_REVISION[0] += 1
        return value

    def refresh(self):
        for name, (_, value, sources, make) in list(self._store.items()):
            with torch.no_grad():
                fresh = make()
            if not _same_layout(value, fresh):
                raise RuntimeError(f'derived weight {name} changed layout; graph recapture required')
            _copy_into(value, fresh)
            self._store[name] = (self._key(sources), value, sources, make)
        _DERIVED_REVISION[0] += 1
        return len(self._store)

    def stale(self):
        return [name for name, (key, _, sources, _) in self._store.items() if self._key(sources) != key]


def _cache(module):
    c = module.__dict__.get("_v41_weight_cache")
    if c is None:
        c = WeightCache()
        module.__dict__["_v41_weight_cache"] = c
    return c


def derived_weight_entries(module):
    """The derived-weight caches of ``module``'s parameters: ``(weight caches, f32 parameter ids)``."""
    caches = [m.__dict__['_v41_weight_cache'] for m in module.modules() if '_v41_weight_cache' in m.__dict__]
    params = {id(p): p for p in (*module.parameters(), *module.buffers())}
    f32 = [k for k, hit in _F32.items() if k in params and hit[0]() is params[k]]
    return caches, f32


@torch.no_grad()
def refresh_derived_weights(module):
    """Re-derive every cached FP32/BF16/concatenated copy of ``module``'s weights IN PLACE (same storage), so
    CUDA graphs captured before a weight update read the new values. Call after the update, before replay.
    Returns the number of entries refreshed."""
    caches, f32 = derived_weight_entries(module)
    n = sum(c.refresh() for c in caches)
    for k in f32:
        ref, _, v = _F32[k]
        p = ref()
        fresh = p.detach().float().contiguous()
        if not _same_layout(v, fresh):
            raise RuntimeError('FP32 parameter copy changed layout; graph recapture required')
        _copy_into(v, fresh)
        _F32[k] = (ref, (p.data_ptr(), p._version, tuple(p.shape), p.dtype), v)
        n += 1
    _DERIVED_REVISION[0] += 1
    return n


def stale_derived_weights(entries):
    """Names of derived-weight entries (``derived_weight_entries``) whose sources changed since derivation."""
    caches, f32 = entries
    out = [name for c in caches for name in c.stale()]
    for k in f32:
        hit = _F32.get(k)
        p = hit[0]() if hit is not None else None
        if p is None or hit[1] != (p.data_ptr(), p._version, tuple(p.shape), p.dtype):
            out.append(f'f32:{k}')
    return out


def derived_generation():
    return _DERIVED_GENERATION[0]


class DerivedWeightGuard:
    """The per-step form of ``stale_derived_weights`` over the derived-weight entries of ``modules``.

    The entries are collected once (``derived_weight_entries``, a module walk) and again only when a new
    entry appears anywhere (``derived_generation``); their source keys are flattened into one list, re-read
    when an entry records new keys (a derivation, an eager re-derivation, a refresh). ``stale()`` then reads
    the live ``(data_ptr, _version)`` of every source and compares it with the recorded keys, with no module
    walk: a write that bypassed the refresh still fails closed on the next step.

    ``probe_content()`` re-derives one entry per call (round robin) on the device and returns a flag tensor
    that is nonzero when the derived copy differs from its re-derivation: the sweep that sees source writes
    no version counter records (``.data`` writes)."""

    def __init__(self, modules, names=None):
        self.modules = tuple(modules)
        self.names = tuple(names) if names is not None else tuple(str(i) for i in range(len(self.modules)))
        self._cursor = 0
        self._collect()

    @classmethod
    def combine(cls, guards):
        """One guard over the modules of ``guards`` (one flat comparison per step)."""
        guards = list(guards)
        return cls([m for g in guards for m in g.modules], [n for g in guards for n in g.names])

    def _collect(self):
        self.generation = _DERIVED_GENERATION[0]
        self._entries = [(name, derived_weight_entries(m)) for name, m in zip(self.names, self.modules)]
        self._reread()

    def _reread(self):
        self.revision = _DERIVED_REVISION[0]
        labels, sources, keys, values = [], [], [], []
        for owner, (caches, f32) in self._entries:
            for c in caches:
                for name, (key, value, srcs, make) in c._store.items():
                    values.append((f"{owner}: {name}", value, make))
                    for p, k in zip(srcs, key):
                        labels.append(f"{owner}: {name}")
                        sources.append(p)
                        keys.append((k[1], k[2]))
            for k in f32:
                hit = _F32.get(k)
                p = None if hit is None else hit[0]()
                labels.append(f"{owner}: f32:{k}")
                if p is None or hit[1] != (p.data_ptr(), p._version, tuple(p.shape), p.dtype):
                    sources.append(None)          # stale already: never equal to a key
                    keys.append(None)
                    continue
                sources.append(p)
                keys.append((hit[1][0], hit[1][1]))
                values.append((f"{owner}: f32:{k}", hit[2], _f32_make(p)))
        self._labels, self._sources, self._keys, self._values = labels, sources, keys, values
        self._dead = None in keys

    def current(self):
        """Re-collect or re-read the entries if they changed since; no walk otherwise."""
        if self.generation != _DERIVED_GENERATION[0]:
            self._collect()
        elif self.revision != _DERIVED_REVISION[0]:
            self._reread()

    def stale(self):
        self.current()
        live = [None if p is None else (p.data_ptr(), p._version) for p in self._sources]
        if live == self._keys and not self._dead:
            return []
        out = []
        for label, a, b in zip(self._labels, live, self._keys):
            if (a is None or a != b) and label not in out:
                out.append(label)
        return out

    def __len__(self):
        return len(self._sources)

    @torch.no_grad()
    def probe_content(self):
        """``(label, flag)`` of the next entry: ``flag`` is a device int32 scalar, nonzero when the derived copy
        differs from a re-derivation of its current sources. None when there are no entries."""
        self.current()
        if not self._values:
            return None
        label, value, make = self._values[self._cursor % len(self._values)]
        self._cursor += 1
        fresh = make()
        pairs = list(zip(value, fresh)) if isinstance(value, (tuple, list)) else [(value, fresh)]
        flag = None
        for old, new in pairs:
            if not _same_layout(old, new):
                return label, torch.ones((), dtype=torch.int32, device=old.device)
            differs = (_bytes(old) != _bytes(new)).any().to(torch.int32)
            flag = differs if flag is None else flag | differs
        return label, flag


def _bytes(t):
    return (t if t.is_contiguous() else t.contiguous()).reshape(-1).view(torch.uint8)


def _f32_make(p):
    import weakref
    ref = weakref.ref(p)
    return lambda: ref().detach().float().contiguous()


def _bf16_weight(module, name, p, grad):
    if grad and p.requires_grad:
        return p.to(torch.bfloat16)
    return _cache(module).get(name, (p,), lambda: p.detach().to(torch.bfloat16).contiguous())


# ---------------------------------------------------------------------------------------------
# compressor
class _PoolNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, proj, w, ratio, eps, B, T):
        C = T // ratio
        out = torch.empty((B * C, 512), dtype=torch.bfloat16, device=proj.device)
        wf = f32_param(w)
        extension().pool_norm(proj, ratio, wf, float(eps), out, C, T, None, None, None)
        ctx.save_for_backward(proj, wf)
        ctx.meta = (ratio, eps, B, T, w.requires_grad, w.dtype)
        return out

    @staticmethod
    def backward(ctx, dy):
        proj, wf = ctx.saved_tensors
        ratio, eps, B, T, need_dw, wdtype = ctx.meta
        C = T // ratio
        dproj = torch.zeros_like(proj)
        dw_rows = torch.empty((B * C, 512), dtype=torch.float32, device=proj.device)
        extension().pool_norm_bwd(proj, ratio, wf, float(eps), dy.contiguous().to(torch.bfloat16), C, T,
                                  dproj, dw_rows)
        dw = segment_sum(dw_rows).to(wdtype) if need_dw else None
        return dproj, dw, None, None, None, None


def compressor_full(comp, x2d, B, T, ratio, eps):
    """Pre-RoPE latent ``[B*C, 512]`` bf16 and the FP32 projection (ratio > 1) for decode state."""
    grad = torch.is_grad_enabled()
    if ratio == 1:
        lat = linear(x2d, _bf16_weight(comp, "wkv", comp.wkv.weight, grad))
        return _row(lat, comp.norm.weight, spec.ROW_LATENT_NORM, 512, eps=eps), None
    if grad and (comp.wkv.weight.requires_grad or comp.wgate.weight.requires_grad):
        wcat = torch.cat((comp.wkv.weight, comp.wgate.weight), 0).to(torch.bfloat16)
    else:
        wcat = _cache(comp).get("wcat", (comp.wkv.weight, comp.wgate.weight),
                                lambda: torch.cat((comp.wkv.weight, comp.wgate.weight), 0)
                                .to(torch.bfloat16).contiguous())
    proj = linear(x2d, wcat, root_fp32=True)
    if B * (T // ratio) == 0:
        return torch.empty((0, 512), dtype=torch.bfloat16, device=x2d.device), proj
    if grad and (proj.requires_grad or comp.norm.weight.requires_grad):
        return _PoolNormFn.apply(proj, comp.norm.weight, ratio, eps, B, T), proj
    out = torch.empty((B * (T // ratio), 512), dtype=torch.bfloat16, device=x2d.device)
    extension().pool_norm(proj, ratio, f32_param(comp.norm.weight), float(eps), out,
                          T // ratio, T, None, None, None)
    return out, proj


# ---------------------------------------------------------------------------------------------
# indexer (no gradient: the selector is frozen in the actor recipe)
def _sparse():
    from isoexec.ops.attention.indexer_sm100 import indexer as dsa_indexer
    from isoexec.ops.attention.indexer_sm100 import topk as dsa_topk
    from isoexec.ops.attention.sparse_sm100._extension import extension as sparse_ext
    return dsa_indexer, dsa_topk, sparse_ext


def _indexer_ext(sub):
    from isoexec.ops.attention.indexer_sm100._extension import extension
    return extension(sub)


def check_index_query_format():
    """The index-score kernel takes FP8 queries (spec.INDEX_QUERY_FORMAT); BF16 queries have no kernel yet."""
    if spec.INDEX_QUERY_FORMAT != "fp8_b128":
        raise NotImplementedError(f"index query format {spec.INDEX_QUERY_FORMAT!r}: no index-score kernel")


def index_keys(idx, latent, cfg, ratio, table, pos, B, C):
    """FP8 index keys: e4m3 codes ``[B, Cp, 128]`` + exponent words ``[B, Cp]`` (four equal UE8M0 bytes).

    ``Cp = C`` rounded up to 128 keeps every batch slice 16-byte aligned for the indexer's TMA maps.
    """
    k = mm(latent, _bf16_weight(idx, "wk", idx.wk.weight, False))
    Cp = (C + 127) // 128 * 128
    codes = torch.zeros((B, Cp, 128), dtype=torch.uint8, device=k.device)
    sf = torch.full((B, Cp), spec.UE8M0_ONE_PACKED, dtype=torch.int32, device=k.device)
    row_op(spec.ROW_INDEX_K, k, 128, w=f32_param(idx.k_norm.weight), eps=cfg.rms_norm_eps,
           table=table, pos=pos, codes=codes.view(B * Cp, 128), sf=sf.view(B * Cp), dst=(C, Cp, 0))
    return codes, sf


def _index_weight_rows_make(idx, cfg):
    w = idx.weights_proj.weight
    return lambda: torch.cat((w.to(torch.bfloat16) * spec.index_weight_scale(cfg),
                              w.new_zeros((64 - w.shape[0], w.shape[1]), dtype=torch.bfloat16)), 0).contiguous()


def index_weight_rows(idx, cfg):
    """The derived weight of ``index_weights``: weights_proj x (index_head_dim * index_heads) ** -0.5 in BF16 (exact
    powers of two), zero-padded to 64 rows: [64, K], graph-stable storage (``refresh_derived_weights``)."""
    return _cache(idx).get("weights_proj64", (idx.weights_proj.weight,), _index_weight_rows_make(idx, cfg))


class _Holder:
    """A linear-provider operand (the provider reads ``.weight``)."""

    def __init__(self, weight):
        self.weight = weight


def index_weights_holder(idx, cfg):
    """``index_weight_rows`` as a provider operand, to be packed after ``wq_a | wkv`` (``pack_shared_input_rows``)."""
    return _Holder(index_weight_rows(idx, cfg))


@torch.no_grad()
def bind_index_weight_rows(idx, cfg, rows):
    """Make ``rows`` (a [64, K] BF16 row block, e.g. rows 1792..1855 of the packed wq_a | wkv | weights storage) the
    storage of the derived index weight: derived into it now, re-derived into it by ``refresh_derived_weights``."""
    make = _index_weight_rows_make(idx, cfg)
    fresh = make()
    if not _same_layout(rows, fresh):
        raise ValueError(f"index weight rows: need a {tuple(fresh.shape)} {fresh.dtype} row block with unit column "
                         f"stride, got {tuple(rows.shape)} {rows.dtype} {rows.stride()}")
    rows.copy_(fresh)
    c = _cache(idx)
    sources = (idx.weights_proj.weight,)
    c._store["weights_proj64"] = (c._key(sources), rows, sources, make)
    _DERIVED_GENERATION[0] += 1          # a new storage: graphs captured before must be recaptured
    _DERIVED_REVISION[0] += 1
    return rows


@torch.no_grad()
def pack_shared_input_rows(module, cfg, index):
    """Engine packing of the projections that read the attention input ``x`` (dense-lane D1 / D2): ``wq_a | wkv``, and
    on index-source layers ``| index_weight_rows`` (the derived indexer weights as rows 1792..1855), as ONE contiguous
    BF16 storage: the parameters are re-pointed at their row blocks (values unchanged; later in-place updates land
    in it) and the derived rows are bound to theirs. Returns the packed [N, K] tensor, or None when the weights cannot
    share a storage (dtype, device, K)."""
    ws = [module.wq_a.weight, module.wkv.weight]
    idx_src = bool(cfg.compress_ratios[index]) and index in cfg.index_source_layer_ids
    K = ws[0].shape[1]
    if (any(w.dim() != 2 or w.shape[1] != K or w.dtype != torch.bfloat16 or w.device != ws[0].device or w.is_meta
            for w in ws)
            or (idx_src and module.indexer.weights_proj.weight.shape[1] != K)):
        return None
    n = sum(w.shape[0] for w in ws) + (64 if idx_src else 0)
    buf = torch.empty((n, K), dtype=torch.bfloat16, device=ws[0].device)
    row = 0
    for p in (module.wq_a.weight, module.wkv.weight):
        view = buf[row:row + p.shape[0]]
        view.copy_(p.detach())
        p.data = view
        row += p.shape[0]
    if idx_src:
        bind_index_weight_rows(module.indexer, cfg, buf[row:row + 64])
    return buf


@torch.no_grad()
def pack_qr_rows(module, cfg, index):
    """Engine packing of the projections that read ``qr`` on index-source layers (dense-lane D3): this rank's ``wq_b``
    rows | the indexer's ``wq_b`` (replicated) as ONE contiguous BF16 storage, the parameters re-pointed at their row
    blocks (values unchanged). The consumer then runs one projection and reads the attention queries and the index
    queries as column views (the leaf and the index-query row kernels take strided rows). Returns the packed tensor,
    or None (nothing changed) on other layers or when the weights cannot share a storage."""
    if not (bool(cfg.compress_ratios[index]) and index in cfg.index_source_layer_ids):
        return None
    ws = [module.wq_b.weight, module.indexer.wq_b.weight]
    K = ws[0].shape[1]
    if any(w.dim() != 2 or w.shape[1] != K or w.dtype != torch.bfloat16 or w.device != ws[0].device or w.is_meta
           for w in ws):
        return None
    buf = torch.empty((sum(w.shape[0] for w in ws), K), dtype=torch.bfloat16, device=ws[0].device)
    row = 0
    for p in ws:
        view = buf[row:row + p.shape[0]]
        view.copy_(p.detach())
        p.data = view
        row += p.shape[0]
    return buf


def index_weights(idx, x2d, cfg, heads=None):
    """The indexer's per-head weights [R, spec.index_q_heads(H)] BF16 of weights_proj pre-scaled by the exact power
    of two (index_head_dim * index_heads) ** -0.5: bf16(x . (W 2^-k)) == bf16(x . W) 2^-k (no subnormals). The GEMM
    always runs over the weight zero-padded to 64 rows (mm_sm100 serves N = 64, not 32 or a TP rank's 4..16) and the
    result is the view of its first Hs columns (128-byte rows: the index-score kernels take a row stride). Reads only
    x, so it may run before (or beside) the query projection."""
    H = cfg.index_n_heads if heads is None else heads
    Hs = spec.index_q_heads(H)
    w = mm(x2d, index_weight_rows(idx, cfg))
    return w if Hs == 64 else w[:, :Hs]


def index_queries(idx, x2d, qr2d, cfg, table, pos, linear_fn, x_shape, qr_shape, heads=None, weights=None, iq=None):
    """(codes [R,Hs,128], sf [R,Hs], weights [R,Hs] bf16), Hs = spec.index_q_heads(H): the index heads (padded to 64
    with zero heads for the FP32-tree fold).

    ``heads``: this rank's index heads under tensor parallelism (default: all); the weight scale is
    always the global ``(index_head_dim * index_n_heads) ** -0.5``.
    """
    H = cfg.index_n_heads if heads is None else heads
    check_index_query_format()
    if H > 64 or cfg.index_head_dim != 128:
        raise ValueError("native indexer serves index heads <= 64 at head dim 128")
    R = x2d.shape[0]
    # ``iq``: the index queries already projected, [R, H, 128] (e.g. a column view of the packed wq_b | indexer wq_b
    # projection, read in place by the row kernel)
    iq = (linear_fn(idx.wq_b, qr2d.view(qr_shape)).reshape(R * H, 128).contiguous() if iq is None else iq)
    Hs = spec.index_q_heads(H)
    codes = torch.empty((R, Hs, 128), dtype=torch.uint8, device=x2d.device)
    sf = torch.empty((R, Hs), dtype=torch.int32, device=x2d.device)
    row_op(spec.ROW_INDEX_Q, iq, 128, table=table, pos=pos, codes=codes.view(R * Hs, 128),
           sf=sf.view(R * Hs), dst=(H, Hs, 0), pad_heads=H if H < Hs else 0)
    # ``weights``: index_weights(idx, x2d, cfg, heads) computed earlier (e.g. on an auxiliary stream)
    return codes, sf, (index_weights(idx, x2d, cfg, heads) if weights is None else weights)


def _candidate_select(scores, ke, cfg, dsa_topk):
    blk = cfg.candidate_block_size
    width = scores.shape[1]
    nbmax = max(1, (width + blk - 1) // blk)
    bm = torch.empty((scores.shape[0], nbmax), dtype=torch.bfloat16, device=scores.device)
    nb = torch.empty_like(ke)
    extension().block_max(scores, ke, blk, bm, nb)
    return dsa_topk.dsa_topk(bm, nb, k=cfg.candidate_topk_blocks), nbmax


#: Paged index scores: the work-item span chosen in the kernel from the live windows (True) or fixed at 16
#: tiles (False). Execution only.
INDEX_SPAN_AUTO = True


def indexer_tree(codes, sf, w, ke, *, width, leaves=spec.TP_LEAVES, f32=False, ieee_levels=None, keys=None,
                 cache=None, block_table=None, bt_rows=None):
    """``sm100_dsa_indexer_tree_v1`` scores ``[rows, width]`` over ``leaves`` leaf groups of 4 heads.

    The default is the canonical program every host runs (``index_scores``): all ``TP_LEAVES`` leaves
    local, the top ``spec.INDEX_TREE_IEEE_LEVELS`` tree levels in IEEE adds, BF16 of the root.
    ``f32``: the FP32 subtree root of ``leaves`` leaves (all FTZ adds) -- the per-rank half of the
    retired cross-rank form, kept for its qualification (tests/test_index_local_gpu.py).
    ``bt_rows`` (paged): query row r reads ``block_table`` row ``bt_rows[r]`` -- the same page ids as
    ``block_table.index_select(0, bt_rows)``, without the copy."""
    _, _, sparse_ext = _sparse()
    if ieee_levels is None:
        ieee_levels = spec.INDEX_TREE_IEEE_LEVELS if leaves == spec.TP_LEAVES and not f32 else 0
    rows = codes.shape[0]
    stride = (width + 255) // 256 * 256
    out = torch.empty((rows, stride), dtype=torch.float32 if f32 else torch.bfloat16, device=codes.device)[:, :width]
    tiles = (width + 127) // 128
    ext = _indexer_ext("indexer")
    if cache is None:
        qb, span = 2, 1 << 20
        g = max(1, min(((rows + qb - 1) // qb) * ((tiles + span - 1) // span), 148))
        kc, ksf = keys
        ext.indexer_tree_ieee_fwd(codes.view(torch.uint8).contiguous(), sf.contiguous(),
                                  kc.view(torch.uint8).contiguous(), ksf.contiguous(), w, None, ke, None, out,
                                  qb, span, g, leaves, ieee_levels)
    else:
        span = 16
        g = max(1, min(rows * ((tiles + span - 1) // span), 148))
        if INDEX_SPAN_AUTO and getattr(ext, "SPAN_AUTO", False):
            # the kernel sizes each work item from the live windows (<= 16 tiles, about one item per CTA):
            # schedule only, the same per-tile fold (tests/test_paged_glue_gpu.py)
            span, g = -16, max(1, min(rows * tiles, 148))
        if bt_rows is not None:
            if hasattr(ext, "indexer_tree_ieee_rows_fwd"):
                ext.indexer_tree_ieee_rows_fwd(codes.view(torch.uint8).contiguous(), sf.contiguous(), cache, None, w,
                                               None, ke, block_table, out, 1, span, g, leaves, ieee_levels,
                                               bt_rows.to(torch.int32).contiguous())
                return out
            block_table = block_table.index_select(0, bt_rows)
        ext.indexer_tree_ieee_fwd(codes.view(torch.uint8).contiguous(), sf.contiguous(), cache, None, w, None, ke,
                                  block_table, out, 1, span, g, leaves, ieee_levels)
    return out


def index_schedule(ke, rows_seq):
    """The paged index-score work division of a step's rows (``indexer_sm100.index_logits.paged_schedule``): one launch
    per step and compress ratio, shared by every index layer of that ratio. None under the FP32-tree fold."""
    if spec.INDEX_FOLD == "fp32_tree":
        return None
    from isoexec.ops.attention.indexer_sm100 import index_logits
    return index_logits.paged_schedule(ke, rows_seq.to(torch.int32).contiguous())


def index_key_ranges(ke, rows_seq, seqs, width):
    """The gathered route's per-row key ranges of a step (``indexer_sm100.index_logits.key_ranges``): one launch per
    step and compress ratio, shared by every index layer of that ratio. None under the FP32-tree fold."""
    if spec.INDEX_FOLD == "fp32_tree":
        return None
    from isoexec.ops.attention.indexer_sm100 import index_logits
    return index_logits.key_ranges(ke.to(torch.int32).contiguous(), rows_seq.to(torch.int32).contiguous(), seqs,
                                   width)


def index_scores(codes, sf, w, ke, *, width, keys=None, cache=None, block_table=None, bt_rows=None, schedule=None,
                 gather=False, ranges=None):
    """The index scores of every host and TP degree: BF16 ``[rows, width]`` (entries in ``[0, ke)``)
    from ALL index heads (``index_queries`` with ``heads=None``), on the local rank, no collective.

    ``spec.INDEX_FOLD == "bf16_dg"`` (the recipe): DeepGEMM's SM100 MQA logits program per (row, key) -- the FP32 MMA
    accumulator, per head bf16(relu), four BF16 fma chains over heads 4i + c, (c0 + c2, c1 + c3), low + high -- in
    the vendored kernel (``indexer_sm100.index_logits``); the same bits from the paged (engine) and the contiguous
    (trainer) form, at every batch composition and schedule. ``bt_rows`` (paged): query row r reads block-table row
    ``bt_rows[r]``; ``schedule``: ``index_schedule(ke, bt_rows)`` of the step (computed here when None).
    ``gather`` (paged): the gathered route instead of the paged kernel -- the keys of the rows' sequences packed
    contiguous and scored by the contiguous kernel (native's prefill form; the same bits, an execution choice);
    ``ranges``: ``index_key_ranges(ke, bt_rows, block_table.shape[0], width)`` of the step (computed here when None).
    ``"fp32_tree"``: the former FP32 8-leaf tree (``indexer_tree``). Index heads, ``indexer.wq_b`` and
    ``weights_proj`` are replicated (tp.shard_layer)."""
    if spec.INDEX_FOLD == "fp32_tree":
        return indexer_tree(codes, sf, w, ke, width=width, keys=keys, cache=cache, block_table=block_table,
                            bt_rows=bt_rows)
    from isoexec.ops.attention.indexer_sm100 import index_logits
    ke = ke.to(torch.int32).contiguous()
    if cache is None:
        kc, ksf = keys
        return index_logits.index_logits(codes, sf, w, kc, ksf, ke, width=width)
    if bt_rows is None:
        bt_rows = torch.arange(codes.shape[0], dtype=torch.int32, device=codes.device)
    bt_rows = bt_rows.to(torch.int32).contiguous()
    if gather:
        return index_logits.index_logits_gathered(codes, sf, w, cache, ke, block_table.to(torch.int32), bt_rows,
                                                  width=width, ranges=ranges)
    return index_logits.index_logits_paged(codes, sf, w, cache, ke, block_table.to(torch.int32), bt_rows,
                                           width=width, schedule=schedule)


def index_scores_full(keys, q, ratio, B, T, C):
    """Per batch element the canonical index scores ``[T, C]`` (``index_scores``) and ``ke``."""
    codes, sf, w = q
    kcodes, ksf = keys
    ke = (torch.arange(1, T + 1, device=codes.device, dtype=torch.int32) // ratio).contiguous()
    scores = []
    for b in range(B):
        rows = slice(b * T, (b + 1) * T)
        scores.append(index_scores(codes[rows], sf[rows], w[rows], ke, width=C,
                                   keys=(kcodes[b, :C], ksf[b, :C])))
    return scores, ke


def select_full(scores, ke, cfg, index, state, B, T, C):
    """Candidate filter and causal top-k from the (TP-reduced) scores. [B*T, topk] int32."""
    _, dsa_topk, _ = _sparse()
    out = torch.empty((B * T, cfg.index_topk), dtype=torch.int32, device=ke.device)
    cands = state.setdefault("candidates_native", {}) if index == cfg.candidate_source_layer_id else None
    for b in range(B):
        rows = slice(b * T, (b + 1) * T)
        sc = scores[b]
        if index == cfg.candidate_source_layer_id:
            cands[b] = _candidate_select(sc, ke, cfg, dsa_topk)[0]
        elif cfg.candidate_source_layer_id < index:
            sel = state.get("candidates_native", {}).get(b)
            if sel is None:
                raise ValueError("candidate source must precede its consumer in this forward")
            extension().candidate_mask(sc, ke, sel, cfg.candidate_block_size,
                                       (C + cfg.candidate_block_size - 1) // cfg.candidate_block_size)
        dsa_topk.dsa_topk(sc, ke, k=cfg.index_topk, out=out[rows])
    return out


def drive(gen):
    """Run one composition generator without tensor parallelism: every reduction is the identity."""
    try:
        msg = next(gen)
        while True:
            msg = gen.send(msg[1])
    except StopIteration as stop:
        return stop.value


# ---------------------------------------------------------------------------------------------
# learned-sink sparse attention (sparse_sm100, G = 1)
#: the sparse-attention recipe when a caller passes ``recipe=None``: 1 = ``sm100_sparse_sink_v1`` (sparse_sm100 leaf
#: recipe 1, frozen), 2 = its lazy-rescale v2 leaf, 3 = ``sm100_sparse_dualrail_sink_v1`` (sparse_sm100 dual-rail
#: leaf). Full sequence, chunked prefill, decode and every TP rank must run the same one.
DEFAULT_SINK_RECIPE = 1


def sink_forward(q3, kv2, lists, sink, scale, recipe):
    """``sm100_sparse_sink_v1`` at G = 1 with the sink root fused into the leaf epilogue (recipe 1 / 2), or the
    dual-rail leaf with the sink in its epilogue (recipe 3).

    sparse_sm100's ``sparse_sink_fused_fwd`` evaluates the root's exact expressions on the leaf's raw
    (O, L, M) instead of after an FP32 partial round trip (tests/test_fused_sink_gpu.py: bit-identical
    to ``sparse_sink_fwd``); no ``[T, 1, H, 520]`` FP32 partials and no second launch.
    """
    _, _, sparse_ext = _sparse()
    T, h, _ = q3.shape
    out = torch.empty((T, h, 512), dtype=torch.bfloat16, device=q3.device)
    lse = torch.empty((T, h), dtype=torch.float32, device=q3.device)
    if T == 0:
        return out, lse
    if recipe is None:
        recipe = DEFAULT_SINK_RECIPE
    if recipe == 3:
        from isoexec.ops.attention.sparse_sm100.dualrail import dualrail_attn
        return dualrail_attn(q3, kv2, lists, scale, sink=sink)
    with torch.cuda.device(q3.device):
        sparse_ext("attn").sparse_sink_fused_fwd(q3, kv2, lists, float(scale), out, lse, recipe, sink)
    return out, lse


#: Trainer sparse-attention BACKWARD provider (user rulings 2026-11-19, EXECUTOR-RULES addenda 19d / 19e and the
#: correction of 2026-11-19: backward kernels must be deterministic run to run, because parameters replicated across
#: TP ranks are updated per rank from gradients every rank must compute identically).
#:   "cudnn_det"  (DEFAULT) cuDNN-frontend ``DSA.sparse_attention_backward_wrapper(..., deterministic=True)``;
#:   "isoexec"    IsoExec's own atomics-free backward (sparse_sm100.bwd): the FALLBACK where cuDNN's DSA namespace or
#:                its ``deterministic`` argument is missing;
#:   "cudnn"      deterministic=False (native's default, order-dependent fp32 atomics): REFUSED BY NAME unless the
#:                diagnostic override ``ISOEXEC_SPARSE_BWD_ALLOW_NONDET=1`` is set, and ALWAYS refused at TP > 1.
#: Selected by ``ISOEXEC_SPARSE_BWD`` or ``BWD_PROVIDER``; every backward records its program tag in
#: ``BACKWARD_PROGRAMS_USED`` (``backward_programs_used()``).
BWD_PROVIDER = None
#: TP degree of the trainer's attention, set by the caller when known (else read from Megatron's parallel state)
BWD_TP_DEGREE = None
BWD_PROGRAM_TAGS = {"cudnn_det": "cudnn_dsa_bwd_det", "isoexec": "sm100_sparse_bwd_v1",
                    "cudnn": "cudnn_dsa_bwd_nondet_DIAGNOSTIC"}
BACKWARD_PROGRAMS_USED = set()
_CUDNN_DSA = {}


class NondeterministicBackwardError(RuntimeError):
    """The non-deterministic (atomics) sparse-attention backward was requested where it is refused."""


def _cudnn_dsa():
    """cuDNN-frontend's DSA namespace and whether its backward takes ``deterministic`` (None if not importable)."""
    if "ns" not in _CUDNN_DSA:
        try:
            import inspect
            from cudnn import DSA as ns
            det = "deterministic" in inspect.signature(ns.sparse_attention_backward_wrapper).parameters
            _CUDNN_DSA["ns"], _CUDNN_DSA["det"] = ns, det
        except Exception:  # noqa: BLE001  (absent / incompatible host copy: the fallback provider runs)
            _CUDNN_DSA["ns"], _CUDNN_DSA["det"] = None, False
    return _CUDNN_DSA["ns"], _CUDNN_DSA["det"]


def _tp_degree():
    if BWD_TP_DEGREE is not None:
        return int(BWD_TP_DEGREE)
    try:
        from megatron.core import parallel_state as ps
        if ps.model_parallel_is_initialized():
            return int(ps.get_tensor_model_parallel_world_size())
    except Exception:  # noqa: BLE001
        pass
    if torch.distributed.is_available() and torch.distributed.is_initialized() and \
            torch.distributed.get_world_size() > 1:
        return None                                       # multi-process, TP degree unknown
    return 1


def backward_provider():
    """The provider the next sparse-attention backward uses."""
    import os
    want = BWD_PROVIDER or os.environ.get("ISOEXEC_SPARSE_BWD", "cudnn_det")
    if want not in BWD_PROGRAM_TAGS:
        raise ValueError(f"ISOEXEC_SPARSE_BWD / BWD_PROVIDER: {want!r} is not one of {sorted(BWD_PROGRAM_TAGS)}")
    if want == "cudnn":
        tp = _tp_degree()
        if tp != 1:
            raise NondeterministicBackwardError(
                f"sparse-attention backward 'cudnn' (non-deterministic, fp32 atomics) is refused at TP degree {tp} "
                "(replicated parameters would drift across ranks); use the default 'cudnn_det'")
        if os.environ.get("ISOEXEC_SPARSE_BWD_ALLOW_NONDET") != "1":
            raise NondeterministicBackwardError(
                "sparse-attention backward 'cudnn' (non-deterministic, fp32 atomics) is refused; it is a diagnostic "
                "only (ISOEXEC_SPARSE_BWD_ALLOW_NONDET=1, TP 1); use the default 'cudnn_det'")
    if want == "isoexec":
        return want
    ns, det = _cudnn_dsa()
    if ns is None or (want == "cudnn_det" and not det):
        return "isoexec"
    return want


def backward_programs_used():
    """Program tags of every sparse-attention backward run in this process so far."""
    return sorted(BACKWARD_PROGRAMS_USED)


#: Query heads per rank cuDNN's DSA backward serves (its own check raises otherwise); other head counts take the
#: IsoExec fallback, chosen by shape (the same on every rank of a TP group: heads per rank are equal).
CUDNN_DSA_HEADS = (16, 32, 64, 96, 128)


def provider_for_heads(provider: str, heads: int) -> str:
    """The provider a backward of ``heads`` query heads per rank runs: cuDNN's where it serves the shape, else the
    IsoExec kernel (recorded under its own program tag)."""
    return provider if provider == "isoexec" or heads in CUDNN_DSA_HEADS else "isoexec"


def _cudnn_backward(q3, kv2, lists, out, lse, d_out, scale, det):
    """cuDNN's DSA backward on OUR forward's quantities: lists compacted to cuDNN's form (ascending valid prefix +
    topk_length, as Megatron passes them); our lse already includes the sink, so cuDNN gets sink = -inf (its
    sink-folded lse then equals ours exactly) and the sink gradient comes from our own sink_grad."""
    ns, _ = _cudnn_dsa()
    compact = extension().compact_lists(lists)
    topk_length = (compact >= 0).sum(1, dtype=torch.int32)
    no_sink = torch.full((q3.shape[1],), float("-inf"), dtype=torch.float32, device=q3.device)
    kw = {"deterministic": True} if det else {}
    r = ns.sparse_attention_backward_wrapper(q3, kv2, out, d_out, lse.reshape(q3.shape[0], q3.shape[1]).float()
                                             .contiguous(), no_sink, compact, softmax_scale=scale,
                                             topk_length=topk_length, **kw)
    return r["dq"], r["dkv"]


class _SinkFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q3, kv2, lists, sink, scale, recipe):
        q3, kv2, sinkc = q3.contiguous(), kv2.contiguous(), f32_param(sink)
        out, lse = sink_forward(q3, kv2, lists, sinkc, scale, recipe)
        ctx.save_for_backward(q3, kv2, lists, sinkc, out, lse)
        ctx.scale = scale
        ctx.recipe = DEFAULT_SINK_RECIPE if recipe is None else recipe
        ctx.sink_dtype = sink.dtype
        return out

    @staticmethod
    def backward(ctx, d_out):
        from isoexec.ops.attention.sparse_sm100.bwd import sparse_attn_bwd
        _, _, sparse_ext = _sparse()
        q3, kv2, lists, sink, out, lse = ctx.saved_tensors
        d_out = d_out.contiguous().to(torch.bfloat16)
        with torch.cuda.device(q3.device):
            # Recipe 3 retains window pads before compressed entries. The
            # reverse-key reducer requires an ascending valid prefix followed
            # by pads; compact only the backward list, preserving forward bits.
            prov = provider_for_heads(backward_provider(), q3.shape[1])
            if prov == "isoexec":
                if ctx.recipe == 3:
                    lists = extension().compact_lists(lists)
                dq, dkv = sparse_attn_bwd(q3, kv2, lists, out, lse, d_out, ctx.scale, recompute_delta=True)
            else:
                dq, dkv = _cudnn_backward(q3, kv2, lists, out, lse, d_out, ctx.scale, prov == "cudnn_det")
            BACKWARD_PROGRAMS_USED.add(BWD_PROGRAM_TAGS[prov])
            dsink = torch.empty_like(sink)
            sparse_ext("attn").sink_grad(out, d_out, lse, sink, dsink)
        return dq, dkv, None, dsink.to(ctx.sink_dtype), None, None


def list_mode(recipe):
    """build_lists layout bit: the dual-rail recipe (3) uses fixed slots -- window in [0, W) with its pads in
    place, compressed from W -- exactly its paged main + extra lists, so paged == contiguous bit for bit."""
    return 2 if (DEFAULT_SINK_RECIPE if recipe is None else recipe) == 3 else 0


def sink_attention(q3, kv2, lists, sink, scale, recipe=None):
    if torch.is_grad_enabled() and (q3.requires_grad or kv2.requires_grad or sink.requires_grad):
        return _SinkFn.apply(q3, kv2, lists, sink, scale, recipe)
    return sink_forward(q3.contiguous(), kv2.contiguous(), lists, f32_param(sink), scale,
                        recipe)[0]


# ---------------------------------------------------------------------------------------------
# KV buffer: [B, S = T + C, 512], rows [0, T) window KV of this layer, [T, S) shared compressed KV
class _KVBufFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kvr, w, comp, B, T, C, eps, table):
        S = T + C
        buf = torch.empty((B * S, 512), dtype=torch.bfloat16, device=kvr.device)
        wf = f32_param(w)
        row_op(spec.ROW_WINDOW_KV, kvr, 512, w=wf, eps=eps, table=table, pos=Pos(mod=T), out=buf, dst=(T, S, 0))
        if C:
            buf.view(B, S, 512)[:, T:].copy_(comp.view(B, C, 512))
        ctx.save_for_backward(kvr, wf)
        ctx.meta = (B, T, C, eps, table, w.requires_grad, w.dtype, comp is not None)
        return buf

    @staticmethod
    def backward(ctx, dbuf):
        kvr, wf = ctx.saved_tensors
        B, T, C, eps, table, need_dw, wdtype, has_comp = ctx.meta
        S = T + C
        dbuf = dbuf.contiguous().to(torch.bfloat16)
        dw_rows = torch.empty((B * T, 512), dtype=torch.float32, device=kvr.device) if need_dw else None
        dkvr = row_bwd(spec.ROW_WINDOW_KV, kvr, dbuf, 512, w=wf, eps=eps, table=table, pos=Pos(mod=T),
                       dw_rows=dw_rows, dst=(T, S, 0))
        dcomp = dbuf.view(B, S, 512)[:, T:].reshape(B * C, 512) if has_comp else None
        dw = segment_sum(dw_rows).to(wdtype) if need_dw else None
        return dkvr, dw, dcomp, None, None, None, None, None


def _kv_buffer(kvr, w, comp, B, T, C, eps, table):
    if torch.is_grad_enabled() and (kvr.requires_grad or w.requires_grad or (comp is not None and comp.requires_grad)):
        return _KVBufFn.apply(kvr, w, comp, B, T, C, eps, table)
    return _KVBufFn.forward(_Ctx(), kvr, w, comp, B, T, C, eps, table)


class _Ctx:
    def save_for_backward(self, *a):
        pass


# ---------------------------------------------------------------------------------------------
# grouped low-rank output projection
_GROUP_PLANS: dict = {}


def _group_plan(G, M, device):
    """moe_sm100 tiles for G equal groups of M rows (graph-capturable, cached per shape)."""
    from isoexec.ops.moe.moe_sm100 import ops as moe
    key = (G, M, str(device))
    plan = _GROUP_PLANS.get(key)
    if plan is None:
        offsets = (torch.arange(G + 1, dtype=torch.int32, device=device) * M).contiguous()
        plan = moe.make_plan(offsets, G * M)
        _GROUP_PLANS[key] = plan
    return plan


def grouped_gemm(x, w, G, M):
    """``y[g] = x[g] @ w[g]^T`` for group-major ``x [G*M, K]``, ``w [G, N, K]``: moe_sm100 ``fc1``
    (``sm100_tcgen05_v1``: one ascending K chain per output, FP32 accumulator, one BF16 rounding;
    a row's bits do not depend on M)."""
    from isoexec.ops.moe.moe_sm100 import ops as moe
    with torch.no_grad():
        return moe.fc1(x, w, _group_plan(G, M, x.device))


class _GroupedOutFn(torch.autograd.Function):
    """y[:, g*R:(g+1)*R] = o_g @ Wa_g^T for the group-major o ``[G, M, Dg]``, one grouped launch.

    ``deepgemm``: the forward runs DeepGEMM's BF16 grouped einsum writing token-major rows directly
    (``mm/deepgemm_sm100.ops.bf16_grouped_bhr_hdr_bhd``): the same per-output program as ``grouped_gemm`` (one
    ascending-K BF16 chain from +0, FP32 accumulator, one BF16 RNE; GPU-gated bitwise against it), without the
    permute launch. The linear provider selects it (``grouped_deepgemm``).

    Backward (training): dO_g = dY_g Wa_g and dWa_g = dY_g^T O_g as grouped GEMMs (the token axis of
    dWa is the K chain, zero-padded to the 64 multiple: exact zero products); no atomics.
    """

    @staticmethod
    def forward(ctx, og, wa, G, R, deepgemm=False):
        M, Dg = og.shape[1], og.shape[2]
        ctx.save_for_backward(og, wa)
        ctx.meta = (G, R)
        if deepgemm and G == 1 and WOA_G1_DENSE and M:
            # one output group per rank (TP8): mm_sm100's dense GEMM on the same operands, the same per-output program
            # (one ascending-K chain, FP32 accumulator, one BF16 rounding): bitwise the grouped einsum
            # (tests/test_woa_g1_gpu.py) and faster (dense lane r16-1: 7.7-8.9 vs 9.0-9.8 us at M 1-512). Placement only.
            from isoexec.ops.mm.mm_sm100 import ops as mm_ops
            a, b = og[0], wa.reshape(R, Dg).t()
            if mm_ops.supported_shape(a, b):
                return mm_ops.mm_sm100(a, b)
        if deepgemm and _dg_grouped_ok(og, wa):
            from isoexec.ops.mm.deepgemm_sm100 import ops as dgo
            y = torch.empty((M, G, R), dtype=torch.bfloat16, device=og.device)
            if M:
                dgo.bf16_grouped_bhr_hdr_bhd(og.detach().permute(1, 0, 2), wa.detach().reshape(G, R, Dg), y)
            return y.view(M, G * R)
        y = grouped_gemm(og.reshape(G * M, Dg), wa.reshape(G, R, Dg), G, M)
        out = torch.empty((M, G * R), dtype=y.dtype, device=y.device)
        extension().permute_rows(y, out, G, M, True)        # row (g, m) -> (m, g): token-major for wo_b
        return out

    @staticmethod
    def backward(ctx, dy):
        """dO_g = dY_g Wa_g, dWa_g = dY_g^T O_g by mm_sm100/general ``bmm`` reading the token-major dY as
        [G, M, R] (row pitch G*R), Wa as [G, R, Dg] and O as [G, M, Dg] in place (ksegtree recipe)."""
        from isoexec.ops.mm.mm_sm100.general import ops as gen
        og, wa = ctx.saved_tensors
        G, R = ctx.meta
        M, Dg = og.shape[1], og.shape[2]
        dyb = dy.to(torch.bfloat16)
        if dyb.stride(-1) != 1 or dyb.stride(0) != G * R:
            dyb = dyb.contiguous()
        dyg = dyb.view(M, G, R).transpose(0, 1)                                         # [G, M, R] view
        wg = wa.reshape(G, R, Dg)
        dog = dwa = None
        if ctx.needs_input_grad[0]:
            dog = gen.bmm(dyg, wg)
        if ctx.needs_input_grad[1]:
            a, o = dyg, og
            if M % 64:
                pad = (-M) % 64
                a = torch.nn.functional.pad(dyg, (0, 0, 0, pad))
                o = torch.nn.functional.pad(og, (0, 0, 0, pad))
            dwa = gen.bmm(a.transpose(1, 2), o).reshape(G * R, Dg).to(wa.dtype)
        return dog, dwa, None, None, None


#: wo_a with one output group per rank (TP8) on mm_sm100's dense GEMM instead of the grouped einsum (dense lane D5).
#: Execution placement only: both evaluate the same per-output program.
WOA_G1_DENSE = True


def _dg_grouped_ok(og, wa):
    """DeepGEMM's grouped einsum takes these operands (CUDA BF16, unit K stride, contiguous weight)."""
    return (og.is_cuda and og.dtype == torch.bfloat16 and wa.dtype == torch.bfloat16 and og.stride(-1) == 1
            and wa.is_contiguous())


def _grouped_out(og, wa, G, R, deepgemm=False):
    if torch.is_grad_enabled() and (og.requires_grad or wa.requires_grad):
        return _GroupedOutFn.apply(og, wa, G, R, deepgemm)
    return _GroupedOutFn.forward(_Ctx(), og, wa, G, R, deepgemm)


# ---------------------------------------------------------------------------------------------
def supported_config(cfg):
    """Raise ValueError naming the first geometry the native composition does not serve."""
    if cfg.head_dim != 512 or cfg.qk_rope_head_dim != 64:
        raise ValueError("native composition serves head_dim 512 with a 64-wide RoPE tail")
    if not 1 <= cfg.num_attention_heads <= 128 or cfg.num_attention_heads % cfg.o_groups:
        raise ValueError("native sparse attention serves 1..128 heads in whole output groups")
    if cfg.q_lora_rank != 1280:
        raise ValueError("native q_norm row kernel is instantiated for q_lora_rank 1280")
    if cfg.index_head_dim != 128 or cfg.index_n_heads > 64:
        raise ValueError("native indexer serves index head dim 128 and at most 64 index heads")
    if spec.INDEX_FOLD != "fp32_tree" and cfg.index_n_heads not in (32, 64):
        raise ValueError("the index-score kernel serves 32 or 64 index heads (one 128-row Q tile per block)")
    W = cfg.sliding_window
    for ratio in set(cfg.compress_ratios):
        K = W + (cfg.index_topk if ratio else 0)
        if K % 64 or K > 2048:
            raise ValueError("window + index top-k must be a positive multiple of 64, at most 2048")
    if cfg.hidden_size % 256:
        raise ValueError("mm_sm100 requires hidden size divisible by 256")
    spec.index_weight_scale(cfg)
    if spec.INDEX_FOLD == "fp32_tree" and cfg.index_n_heads != spec.TP_LEAVES * spec.INDEX_LEAF_HEADS:
        raise ValueError("index heads must form TP_LEAVES leaves of INDEX_LEAF_HEADS heads")
    if cfg.o_groups % spec.TP_LEAVES or (cfg.o_groups * cfg.o_lora_rank // spec.TP_LEAVES) % 256:
        raise ValueError("wo_b's K must split into TP_LEAVES leaves of whole groups (256-aligned)")
    if (cfg.candidate_topk_blocks - 1) * cfg.candidate_block_size + 1 < cfg.index_topk:
        raise ValueError("candidate blocks must cover the index top-k")
    return True


def attention_full(module, x, cfg, index, state, *, linear_fn, dequant_fn, recipe=None, cache=None, cache_rows=None):
    """Full-sequence (prefill / training) forward of one layer; ``state`` is shared across layers.

    With ``cache`` (a ``cache.DecodeCache``) the layer's decode state is published for the batch
    slots ``cache_rows`` (default ``range(B)``), which must be empty (position 0).
    """
    return drive(attention_full_gen(module, x, cfg, index, state, linear_fn=linear_fn, dequant_fn=dequant_fn,
                                    recipe=recipe, cache=cache, cache_rows=cache_rows))


def attention_full_gen(module, x, cfg, index, state, *, linear_fn, dequant_fn, recipe=None, cache=None,
                       cache_rows=None, tp=1):
    """The composition as a generator over its cross-rank reductions (``tp.py``).

    ``module`` holds this rank's shard (``tp`` ranks: heads and output groups split evenly; the
    indexer and everything else replicated). The index scores are computed locally on every rank
    (``index_scores``), so it yields nothing; it returns this rank's ``wo_b`` output (an FP32 subtree
    root when tp > 1), which the host reduces.
    """
    yield from ()          # host protocol (tp.py, TreeReducer.run, drive): no intermediate reduction
    supported_config(cfg)
    if x.ndim != 3 or x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError("native attention expects CUDA BF16 [batch, sequence, hidden]")
    B, T, hid = x.shape
    H, D = cfg.num_attention_heads // tp, cfg.head_dim
    ratio, W, eps = cfg.compress_ratios[index], cfg.sliding_window, cfg.rms_norm_eps
    table = rope_table(cfg, ratio, T, x.device)
    x2d = x.reshape(B * T, hid)
    qa = linear_fn(module.wq_a, x).reshape(B * T, -1)
    qr = _row(qa, module.q_norm.weight, spec.ROW_Q_NORM, cfg.q_lora_rank, eps=eps)
    q = linear_fn(module.wq_b, qr.view(B, T, -1)).reshape(B * T * H, D)
    q = q_rope(q, D, table, Pos(div=H, mod=T))
    kvr = linear_fn(module.wkv, x).reshape(B * T, D)
    C = T // ratio if ratio else 0
    comp = None
    topk = None
    if ratio:
        latent = None
        if index in cfg.kv_source_layer_ids:
            latent, proj = compressor_full(module.compressor, x2d, B, T, ratio, eps)
            state["compressor_proj"] = proj
        if index in cfg.index_source_layer_ids:
            if C == 0:
                topk = torch.full((B * T, cfg.index_topk), -1, dtype=torch.int32, device=x.device)
            else:
                idx = module.indexer
                with torch.no_grad():
                    if index in cfg.kv_source_layer_ids:
                        state["index_k_native"] = index_keys(idx, latent.detach(), cfg, ratio, table,
                                                             Pos(mod=C, mul=ratio), B, C)
                    if "index_k_native" not in state:
                        raise ValueError("index key source must precede its consumer in this forward")
                    # every rank: all index heads, the canonical tree (index_scores); no reduction
                    hi = cfg.index_n_heads
                    iq = index_queries(idx, x2d, qr.detach(), cfg, table, Pos(div=hi, mod=T),
                                       linear_fn, x.shape, (B, T, cfg.q_lora_rank))
                    scores, ke = index_scores_full(state["index_k_native"], iq, ratio, B, T, C)
                with torch.no_grad():
                    topk = select_full(scores, ke, cfg, index, state, B, T, C)
            state["topk_native"] = topk
        if latent is not None:
            state["compress_kv_native"] = _row(latent, None, spec.ROW_COMPRESSED_KV, 512, table=table,
                                               pos=Pos(mod=max(C, 1), mul=ratio))
        if "compress_kv_native" not in state or "topk_native" not in state:
            raise ValueError("compressed KV and index sources must precede their consumers")
        comp = state["compress_kv_native"]
        topk = state["topk_native"]
    S = T + C
    kvbuf = _kv_buffer(kvr.contiguous(), module.kv_norm.weight, comp, B, T, C, eps, table)
    K = W + (cfg.index_topk if ratio else 0)
    lists = torch.empty((B * T, K), dtype=torch.int32, device=x.device)
    extension().build_lists(topk, None, B * T, T, S, 0, T, W, list_mode(recipe), lists)
    o = sink_attention(q.view(B * T, H, D), kvbuf, lists, module.attn_sink, D ** -0.5, recipe)
    if cache is not None:
        cache.write_prefill(module, index, kvbuf, state, B, T, C, cache_rows)
    return output_projection(module, o, cfg, table, Pos(div=H, mod=T), B, T, linear_fn, dequant_fn, tp=tp)


def output_projection(module, o, cfg, table, pos, B, T, linear_fn, dequant_fn, tp=1, o_roped=False):
    """Inverse RoPE (group-major), grouped low-rank wo_a (one moe_sm100 grouped GEMM, or DeepGEMM's grouped
    einsum when ``linear_fn.grouped_deepgemm``: the same program), then wo_b.

    ``o_roped``: the leaf already applied the inverse RoPE (``dualrail_attn(rope_o=True)``, the same bits as
    ``ROW_OUT_INV_ROPE``); with one output group per rank the token-major [B*T, H, D] output IS the group-major layout,
    so no launch remains."""
    H, D, G = cfg.num_attention_heads // tp, cfg.head_dim, cfg.o_groups // tp
    if o_roped:
        if G != 1 or torch.is_grad_enabled():
            raise ValueError("a leaf-rotated output serves one output group per rank, inference only")
        og = o.reshape(1, B * T, H * D)
    else:
        og = _row(o.reshape(B * T * H, D), None, spec.ROW_OUT_INV_ROPE, D, table=table, pos=pos,
                  group_major=(B * T, H, H // G)).view(G, B * T, (H // G) * D)
    if torch.is_grad_enabled() and any(p.requires_grad for p in module.wo_a.parameters()):
        wa = dequant_fn(module.wo_a).to(torch.bfloat16)
    else:
        wa = _cache(module).get("wo_a", tuple(module.wo_a.parameters()),
                                lambda: dequant_fn(module.wo_a).to(torch.bfloat16).contiguous())
    y = _grouped_out(og, wa, G, cfg.o_lora_rank, deepgemm=bool(getattr(linear_fn, "grouped_deepgemm", False)))
    if getattr(linear_fn, "supports_leaf_tree", False):
        # a caller-supplied provider of the fixed TP_LEAVES-leaf FP32 tree (per-leaf FP32 roots):
        # BF16 root at tp = 1, FP32 subtree root at tp > 1
        out = linear_fn(module.wo_b, y, leaves=spec.TP_LEAVES // tp, fp32_root=tp > 1)
    else:
        # an arbitrary linear_fn chooses its own K reduction order, which would make the output depend
        # on the TP degree; the built-in leaf tree (mm_sm100, BF16 weights) is used instead
        out = wo_b_tree(module.wo_b, y, tp)
    return out.view(B, T, -1)


class _LeafTreeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, final_bf16, *leaves):
        stacked = torch.stack([leaf.contiguous() for leaf in leaves], 0)
        out = torch.empty(leaves[0].shape, dtype=torch.bfloat16 if final_bf16 else torch.float32,
                          device=leaves[0].device)
        extension().leaf_tree(stacked, out)
        ctx.n = len(leaves)
        return out

    @staticmethod
    def backward(ctx, dy):
        g = dy.float()
        return (None, *([g] * ctx.n))


def leaf_tree(leaves, final_bf16):
    if torch.is_grad_enabled() and any(leaf.requires_grad for leaf in leaves):
        return _LeafTreeFn.apply(final_bf16, *leaves)
    return _LeafTreeFn.forward(_Ctx(), final_bf16, *leaves)


def _leaf_gemm(mod, x, k0, k1):
    """FP32 ``x[:, k0:k1] @ W[:, k0:k1]^T`` (one leaf of wo_b's K)."""
    w = mod.weight
    xs = x[:, k0:k1].contiguous()
    _refuse_raw_checkpoint(w, "wo_b")
    if torch.is_grad_enabled() and w.requires_grad:
        wl = w[:, k0:k1].to(torch.bfloat16)
    else:
        wl = _cache(mod).get(f"leaf{k0}", (w,), lambda: w[:, k0:k1].to(torch.bfloat16).contiguous())
    return linear(xs, wl, root_fp32=True, split=1)          # one leaf = one ascending FP32 chain


def wo_b_leaves(mod, y, tp=1):
    """Reference form of ``wo_b_tree``: TP_LEAVES/tp separate FP32 leaf GEMMs and the leaf_tree kernel
    (differentiable)."""
    leaves = spec.TP_LEAVES // tp
    kw = y.shape[1] // leaves
    parts = [_leaf_gemm(mod, y, l * kw, (l + 1) * kw) for l in range(leaves)]
    return leaf_tree(parts, final_bf16=tp == 1)


_CHECKPOINT_DTYPES = (torch.float8_e4m3fn, torch.int8, torch.uint8)


def _refuse_raw_checkpoint(w, name):
    """The built-in BF16 leaf tree takes BF16 (or FP32 master) weights; raw FP8/FP4 checkpoint bytes refuse."""
    if w.dtype in _CHECKPOINT_DTYPES or "float4" in str(w.dtype):
        raise NotImplementedError(f"{name}: raw {w.dtype} checkpoint weights need a checkpoint provider's fixed-leaf "
                                  "tree (linear_fn.supports_leaf_tree) or the decoded BF16 masters")


_WOB_PLANS: dict = {}


def _wob_plan(y, w, m, root_fp32):
    """An mm_sm100 plan evaluating the whole m-leaf tree on chip (``segment_k`` = one leaf): the fp32
    ROOT form or the on-chip BF16 root. Picked once per shape bucket by timing (scheduling only: every
    plan at a split returns identical bits, mm_sm100's plan relation)."""
    from isoexec.ops.mm.mm_sm100 import ops as mm_ops
    from isoexec.ops.mm.mm_sm100 import segments as seg
    M, K = y.shape
    N = w.shape[0]
    mb = max(64, 1 << max(0, (M - 1).bit_length()))
    key = (mb, K, N, m, root_fp32, str(y.device))
    hit = _WOB_PLANS.get(key)
    if hit is not None:
        return hit
    split_plans = seg.plans_at_split(mb, K, N, m)
    if root_fp32:
        cands = [(p, False) for p in seg.root_plans(mb, K, N, m)]
    else:
        # every plan at this split: on-chip whole tree, or subtree roots folded by mm_sm100's own
        # combine / in-kernel fold (the same balanced tree; plan invariance is mm_sm100's battery)
        cands = [(p, False) for p in split_plans]
    # PUBLISH form: aligned subtree roots [groups, M, N] FP32, completed by the same leaf_tree
    cands += [(p, True) for p in split_plans
              if p.mode == seg.COMBINE and p.groups in (2, 4, 8) and p.groups <= m]
    timed = []
    for p, publish in cands:
        try:
            run = lambda p=p, publish=publish: _wob_run(y, w, p, publish, root_fp32)
            run()
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(3):
                run()
            stop.record()
            torch.cuda.synchronize()
            timed.append((start.elapsed_time(stop), (p, publish)))
        except Exception:  # noqa: BLE001 -- a plan this shape cannot run is not a candidate
            continue
    if not timed:
        raise ValueError(f"no mm_sm100 plan evaluates a {m}-leaf tree at M={M} K={K} N={N}")
    best = min(timed, key=lambda tp_: tp_[0])[1]
    _WOB_PLANS[key] = best
    return best


def _wob_run(y, w, plan, publish, root_fp32):
    from isoexec.ops.mm.mm_sm100 import ops as mm_ops
    if not publish:
        return mm_ops.run_plan(y, w.t(), plan, root_dtype=torch.float32 if root_fp32 else torch.bfloat16)
    roots = mm_ops.run_plan(y, w.t(), plan, combine=False)                    # [groups, M, N] fp32
    out = torch.empty(roots.shape[1:], dtype=torch.float32 if root_fp32 else torch.bfloat16, device=y.device)
    extension().leaf_tree(roots.contiguous(), out)
    return out


_PADBUF: dict = {}


def _rows_at_least(y, rows):
    """``y`` padded with zero rows to ``rows`` through a persistent zeroed workspace (one copy)."""
    M, K = y.shape
    key = (rows, K, y.dtype, str(y.device))
    buf = _PADBUF.get(key)
    if buf is None:
        buf = _PADBUF[key] = torch.zeros((rows, K), dtype=y.dtype, device=y.device)
    buf[:M].copy_(y)
    if M < rows:
        pass        # rows >= M stay zero: only rows < M are ever written
    return buf


class _WoBFn(torch.autograd.Function):
    """Forward: the one-launch fixed-leaf tree (bit-identical to the separate-leaf form); backward:
    a linear's (the tree is a sum of leaves, each leaf a K-slice of one GEMM): dY W and dY^T Y."""

    @staticmethod
    def forward(ctx, y, w, tp):
        ctx.save_for_backward(y, w)
        with torch.no_grad():
            return _wo_b_forward(w, y, tp)

    @staticmethod
    def backward(ctx, dout):
        y, w = ctx.saved_tensors
        dy, dw = mm_grads(dout, y, w, need_da=ctx.needs_input_grad[0], need_dw=ctx.needs_input_grad[1])
        return dy, (None if dw is None else dw.to(w.dtype)), None


def _wo_b_forward(w, y, tp):
    from isoexec.ops.mm.mm_sm100 import ops as mm_ops
    wb = w if (w.dtype == torch.bfloat16 and w.is_contiguous()) else w.detach().to(torch.bfloat16).contiguous()
    M = y.shape[0]
    y = y.contiguous()
    leaves = spec.TP_LEAVES // tp
    if 0 < M <= ONCHIP_WOB_ROWS and y.is_cuda:
        from isoexec.ops.mm.deepgemm_sm100 import segtree
        if segtree.serves(wb.shape[0], wb.shape[1], leaves):
            out_dtype = torch.float32 if tp > 1 else torch.bfloat16
            if M <= SEGTREE_WOB_ROWS:
                # decode rows: one CTA per leaf (DeepGEMM leaf GEMMs, FP32 leaf roots, the same leaf_tree fold;
                # ``mm/deepgemm_sm100/segtree.py``): bitwise the on-chip tree below
                return segtree.linear(y, wb, leaves, out_dtype=out_dtype)
            # DeepGEMM's BF16 GEMM with the leaf tree in its epilogue: bitwise the on-chip tree below
            return segtree.leaf_linear(y, wb, leaves, out_dtype=out_dtype)
    if M < _MIN_ROOT_ROWS:                                       # rows are independent: zero rows change no bit
        y = _rows_at_least(y, _MIN_ROOT_ROWS)
    plan, publish = _wob_plan(y, wb, spec.TP_LEAVES // tp, tp > 1)
    return _wob_run(y, wb, plan, publish, tp > 1)[:M]


def wo_b_tree(mod, y, tp=1):
    """wo_b with its K cut into TP_LEAVES fixed column leaves, each an FP32 chain, combined by one
    balanced FP32 tree: BF16 root at tp = 1, this rank's FP32 subtree root at tp > 1.

    BF16 weights: ONE mm_sm100 launch with ``segment_k`` = one leaf (the tree on chip; bit-identical to
    ``wo_b_leaves``, tests/test_tp_gpu.py), in training too (``_WoBFn``: the whole weight, no slices).
    Raw FP8 checkpoint weights take a checkpoint provider's leaf tree (``output_projection``).
    """
    w = mod.weight
    _refuse_raw_checkpoint(w, "wo_b")
    if torch.is_grad_enabled() and (w.requires_grad or y.requires_grad):
        return _WoBFn.apply(y, w, tp)
    return _wo_b_forward(_cache(mod).get("wob_bf16", (w,), lambda: w.to(torch.bfloat16).contiguous()), y, tp)
