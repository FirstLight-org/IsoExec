"""Python entry points for the native SM100 Engram kernels (numerical program: README.md).

Torch glue is allocation, views, dtype/shape validation and the optional host-side error-flag check
(``validate=True`` costs one sync, exactly like the reference's out-of-range check; pass
``validate=False`` on a CUDA-graph / zero-sync path).
"""

from __future__ import annotations

import torch

from isoexec.ops.embed.engram_sm100._extension import extension

#: reduction program constants (README "Numerical program")
GATE_THREADS = 128
GATE_CHUNK = 8
DW_SEG_T = 64
GRAD_STRIPES = 8


def _cuda(t, name, dtype=None):
    if not isinstance(t, torch.Tensor) or not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if dtype is not None and t.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {t.dtype}")
    return t


_SCRATCH_FLAGS: dict = {}


def _flag(device, validate=True):
    """The kernels' error word. ``validate=False``: nobody reads it, so a per-device scratch word (created outside
    capture, never reset) takes the writes and the call launches no fill kernel."""
    if not validate:
        key = device.index if device.index is not None else torch.cuda.current_device()
        hit = _SCRATCH_FLAGS.get(key)
        if hit is None and not torch.cuda.is_current_stream_capturing():
            hit = _SCRATCH_FLAGS[key] = torch.zeros(1, dtype=torch.int32, device=device)
        if hit is not None:
            return hit
    return torch.zeros(1, dtype=torch.int32, device=device)


def _raise_if(err, what, validate):
    if validate and int(err.item()):
        raise ValueError(what)


# ------------------------------------------------------------------ hashing
_MAGIC: dict = {}


def _hash_magic(primes):
    """``floor((2**64 - 1) / p)`` per prime as int64 bits (uint64 in the kernel): the exact reciprocal the hash kernel
    reduces with instead of an emulated 64-bit modulo (same remainders). Built once per primes tensor outside graph
    capture (a host read); None (the kernel's plain modulo, the same value) when first seen inside a capture."""
    # an inference tensor (an engine's constants, built under inference_mode) has no version counter; its values
    # are the hash layout's fixed primes, never rewritten in place
    version = -1 if primes.is_inference() else primes._version
    key = (primes.data_ptr(), version, tuple(primes.shape), tuple(primes.stride()), primes.device)
    hit = _MAGIC.get(key)
    if hit is not None:
        return hit[1]
    if hit is None:
        if primes.is_cuda and torch.cuda.is_current_stream_capturing():
            return None
        vals = [((1 << 64) - 1) // p for p in primes.flatten().tolist()]
        hit = torch.tensor([v - (1 << 64) if v >= (1 << 63) else v for v in vals], dtype=torch.int64,
                           device=primes.device).view(primes.shape)
        # the entry holds the primes tensor itself, so its memory cannot be freed and reused by another tensor at
        # the same address (an address-keyed entry without the reference returned stale reciprocals, r21)
        _MAGIC[key] = (primes, hit)
    return hit


def ngram_hash(input_ids, token_map, primes, offsets, multipliers, pad_id, *, token_mask=None, state=None,
               return_state=False, validate=True, raw_state=None, seg_start=None):
    """Bucket indices ``int64 [B,T,L,(N-1)*H]``; bit-exact against ``EngramHash``.

    ``state`` is the ``int64 [B,N-1]`` history (compressed ids of the N-1 preceding positions, oldest
    first, -1 = absent/dead; ``init_state`` builds the empty one). With ``state=None`` the sequence starts
    at position 0. ``return_state=True`` also returns the history after this call, so chunked
    prefill/decode equals one full-sequence call. ``raw_state``: instead of ``state``, the engine's RAW lookback rows
    ``int32 [B,N-1]`` (token ids, newest first, -1 none): the kernel maps them through ``token_map`` itself, the values
    ``state = where(h >= 0, token_map[h.clamp(min=0)], -1)`` of ``h = raw_state.flip(-1)`` (no history launches).
    ``seg_start``: ``int64 [B*T]``, each token's packed-sequence start (``B = 1`` packed trainer rows): every sequence
    hashed from an empty history in ONE launch, the values of one call per sequence with ``state=None``.
    """
    _cuda(input_ids, "input_ids", torch.int64)
    if input_ids.ndim != 2:
        raise ValueError("input IDs must be int64 [B,T]")
    B, T = input_ids.shape
    L, Nm1, H = primes.shape
    N = Nm1 + 1
    ids = input_ids.contiguous()
    if token_mask is not None:
        if token_mask.shape != ids.shape or token_mask.dtype != torch.bool:
            raise ValueError("token mask must be boolean [B,T]")
        token_mask = token_mask.contiguous()
    if state is not None:
        if state.shape != (B, N - 1) or state.dtype != torch.int64:
            raise ValueError("state must be int64 [B,N-1]")
        state = state.contiguous()
    if raw_state is not None:
        if state is not None or return_state:
            raise ValueError("raw_state replaces state and returns no new state")
        if raw_state.shape != (B, N - 1) or raw_state.dtype != torch.int32:
            raise ValueError("raw_state must be int32 [B,N-1]")
        raw_state = raw_state.contiguous()
    if seg_start is not None:
        if state is not None or raw_state is not None or return_state:
            raise ValueError("seg_start hashes every packed sequence from an empty history")
        if seg_start.dtype != torch.int64 or seg_start.numel() != B * T:
            raise ValueError("seg_start must be int64 [B*T]")
        seg_start = seg_start.contiguous()
    out = torch.empty(B, T, L, Nm1 * H, dtype=torch.int64, device=ids.device)
    new_state = torch.empty(B, N - 1, dtype=torch.int64, device=ids.device) if return_state else None
    err = _flag(ids.device, validate)
    extension().hash(ids, token_map, token_mask, state, primes, offsets, multipliers, int(pad_id), out, err,
                     new_state, _hash_magic(primes), raw_state, seg_start)
    _raise_if(err, "input IDs outside tokenizer vocabulary", validate)
    return (out, new_state) if return_state else out


def init_state(batch, max_ngram_size, device):
    """Empty history: no preceding positions."""
    return torch.full((batch, max_ngram_size - 1), -1, dtype=torch.int64, device=device)


# ------------------------------------------------------------------ lookup
def fp8_rowwise_lookup(weight, scale, indices, *, out_dtype=torch.bfloat16, validate=True):
    """``out[..., :] = decode(weight[idx]) * 2**(scale[idx, d//32] - 127)`` (exact decode, one fp32 multiply,
    one narrowing). ``weight`` float8_e4m3fn [R,D], ``scale`` float8_e8m0fnu [R,D/32]."""
    _cuda(weight, "weight", torch.float8_e4m3fn)
    _cuda(scale, "scale", torch.float8_e8m0fnu)
    _cuda(indices, "indices", torch.int64)
    R, D = weight.shape
    if weight.stride(-1) != 1 or not weight.is_contiguous() or not scale.is_contiguous():
        raise ValueError("table must be contiguous")
    if scale.shape != (R, D // 32) or D % 32:
        raise ValueError("scale must be [rows, D/32] with D a multiple of 32")
    idx = indices.contiguous()
    out = torch.empty(*indices.shape, D, dtype=out_dtype, device=weight.device)
    err = _flag(weight.device, validate)
    extension().lookup_fp8(weight, scale, idx, out, err)
    _raise_if(err, "Engram row index outside the table", validate)
    return out


def fp8_rowwise_lookup_quant(weight, scale, indices, *, validate=True):
    """Lookup fused with the checkpoint activation quantiser: ``(codes e4m3 [..., D], scales UE8M0 [..., D/32])``,
    bitwise equal to the ``deepseek_v41_act_quant_v1`` FP8 1x32 UE8M0 encoding (``quant/checkpoint_sm100/reference.py``)
    of ``fp8_rowwise_lookup(...)`` (the bf16 lookup is never materialised). ``codes.view(M, K)`` /
    ``scales.view(M, K/32)`` (K = (max_ngram-1) * heads * D) are the A operand of a checkpoint FP8 GEMM."""
    _cuda(weight, "weight", torch.float8_e4m3fn)
    _cuda(scale, "scale", torch.float8_e8m0fnu)
    _cuda(indices, "indices", torch.int64)
    R, D = weight.shape
    if not weight.is_contiguous() or not scale.is_contiguous():
        raise ValueError("table must be contiguous")
    if scale.shape != (R, D // 32) or D % 32:
        raise ValueError("scale must be [rows, D/32] with D a multiple of 32")
    idx = indices.contiguous()
    codes = torch.empty(*indices.shape, D, dtype=torch.float8_e4m3fn, device=weight.device)
    qs = torch.empty(*indices.shape, D // 32, dtype=torch.float8_e8m0fnu, device=weight.device)
    err = _flag(weight.device)
    extension().lookup_fp8_quant(weight, scale, idx, codes, qs, err)
    _raise_if(err, "Engram row index outside the table", validate)
    return codes, qs


# ------------------------------------------------------------------ master -> FP8 table encode
def _table_buffers(rows, dim, device, weight, scale):
    if weight is None:
        weight = torch.empty(rows, dim, dtype=torch.float8_e4m3fn, device=device)
    if scale is None:
        scale = torch.empty(rows, dim // 32, dtype=torch.float8_e8m0fnu, device=device)
    return weight, scale


def quantize_table_fp8(master, *, weight=None, scale=None, validate=True):
    """Encode a dense fp32 master table ``[R, D]`` as the FP8 table ``(weight e4m3 [R, D], scale UE8M0 [R, D/32])``.

    Program (README "Table re-encode"): per 32-group, ``amax`` floored at ``448 * 2^-126``, ``scale = amax * float(1/448)``
    rounded UP to a power of two, ``code = e4m3_rn(clamp(div_rn(x, scale), +-448))`` -- the activation quantiser's program on
    fp32 values. A pure function of each group, so the bytes do not depend on device, schedule or run."""
    _cuda(master, "master", torch.float32)
    if master.ndim != 2 or not master.is_contiguous() or master.shape[1] % 32:
        raise ValueError("master table must be contiguous fp32 [R, D] with D a multiple of 32")
    weight, scale = _table_buffers(master.shape[0], master.shape[1], master.device, weight, scale)
    err = _flag(master.device)
    extension().quantize_table(master, None, None, None, weight, scale, err)
    _raise_if(err, "table row outside the table", validate)
    return weight, scale


def requantize_rows_(weight, scale, master, *, positions=None, nruns=None, row_ids=None, validate=True):
    """In-place sparse re-encode after an optimiser step: master rows ``positions[:nruns]`` (``nruns`` an int32 device
    scalar as returned by :func:`table_row_grad`; ``positions=None`` re-encodes every master row) are written to table
    rows ``row_ids[position]`` (``row_ids=None``: the master IS the table, position == table row). Same program and bytes
    as :func:`quantize_table_fp8` row for row; untouched rows are not written."""
    _cuda(weight, "weight", torch.float8_e4m3fn)
    _cuda(scale, "scale", torch.float8_e8m0fnu)
    _cuda(master, "master", torch.float32)
    if master.ndim != 2 or not master.is_contiguous() or master.shape[1] != weight.shape[1]:
        raise ValueError("master must be contiguous fp32 [U, D] with the table's D")
    if positions is not None:
        _cuda(positions, "positions", torch.int64)
        if nruns is None:
            raise ValueError("positions need the device count nruns")
        positions = positions.contiguous()
    if row_ids is not None:
        _cuda(row_ids, "row_ids", torch.int64)
        if row_ids.shape != (master.shape[0],):
            raise ValueError("row_ids must be int64 [U]")
        row_ids = row_ids.contiguous()
    elif master.shape[0] > weight.shape[0]:
        raise ValueError("master has more rows than the table")
    n = None if nruns is None else (nruns if isinstance(nruns, torch.Tensor) else torch.tensor([nruns], dtype=torch.int32,
                                                                                         device=weight.device))
    err = _flag(weight.device)
    extension().quantize_table(master, positions, row_ids, n, weight, scale, err)
    _raise_if(err, "table row outside the table", validate)
    return weight, scale


class _MasterLookup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, master, indices, out_dtype, validate, sparse_grad):
        _cuda(master, "master", torch.float32)
        _cuda(indices, "indices", torch.int64)
        if master.ndim != 2 or not master.is_contiguous():
            raise ValueError("master table must be contiguous fp32 [R,D]")
        idx = indices.contiguous()
        out = torch.empty(*indices.shape, master.shape[1], dtype=out_dtype, device=master.device)
        err = _flag(master.device)
        extension().lookup_master(master, idx, out, err)
        _raise_if(err, "Engram row index outside the table", validate)
        ctx.save_for_backward(idx)
        ctx.rows, ctx.validate, ctx.sparse = master.shape[0], validate, sparse_grad
        ctx.shape = master.shape
        return out

    @staticmethod
    def backward(ctx, grad):
        (idx,) = ctx.saved_tensors
        rows, g, n = table_row_grad(idx, grad, ctx.rows, validate=ctx.validate)
        n = int(n.item())
        if ctx.sparse:
            sp = torch.sparse_coo_tensor(rows[:n].unsqueeze(0), g[:n], ctx.shape, is_coalesced=True)
            return sp, None, None, None, None
        dense = torch.zeros(ctx.shape, dtype=torch.float32, device=grad.device)
        scatter_rows_(dense, rows, g, nruns=n)
        return dense, None, None, None, None


def master_rowwise_lookup(master, indices, *, out_dtype=torch.bfloat16, validate=True, sparse_grad=True):
    """Training lookup from the dense fp32 master table (the recipe's ``F.embedding(ids, master).to(dtype)``).
    Differentiable w.r.t. ``master``: the gradient is a coalesced sparse COO tensor of the unique touched rows
    (``sparse_grad=True``, required at the published table size) or a dense tensor."""
    return _MasterLookup.apply(master, indices, out_dtype, validate, sparse_grad)


# ------------------------------------------------------------------ gate + residual
def _gate_args(x, key, value, q_weight, k_weight, token_mask):
    for n, t in (("x", x), ("key", key), ("value", value)):
        _cuda(t, n, torch.bfloat16)
    if x.ndim != 4:
        raise ValueError("x must be bf16 [B,T,streams,dim]")
    if key.shape != x.shape or value.shape != (*x.shape[:-2], x.shape[-1]):
        raise ValueError("Engram key must match streams, with one shared value per token")
    S, D = x.shape[-2:]
    if D % GATE_CHUNK:
        raise ValueError("dim must be a multiple of 8")
    for n, t in (("q_weight", q_weight), ("k_weight", k_weight)):
        _cuda(t, n)
        if t.shape != (S, D) or t.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"{n} must be bf16/fp32 [streams,dim]")
    if q_weight.dtype != k_weight.dtype:
        raise ValueError("q_weight and k_weight must share a dtype")
    for n, t in (("x", x), ("key", key), ("value", value), ("q_weight", q_weight), ("k_weight", k_weight)):
        if t.data_ptr() % 16:
            raise ValueError(f"{n} must be 16-byte aligned")
    if token_mask is not None and (token_mask.shape != x.shape[:-2] or token_mask.dtype != torch.bool):
        raise ValueError("Engram token mask must be boolean [B,T]")


def _token_strided(t):
    """``t`` unchanged when the kernels can address it as strided rows ([B, T, ...] with one token stride, last dim contiguous,
    every other stride a multiple of 8 elements = 16 bytes); otherwise a contiguous copy. Values are identical either way."""
    st = t.stride()
    ok = st[-1] == 1 and all(s % 8 == 0 for s in st[:-1]) and (t.shape[0] == 1 or st[0] == t.shape[1] * st[1])
    return t if ok else t.contiguous()


class _GateResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, key, value, q_weight, k_weight, eps, token_mask):
        _gate_args(x, key, value, q_weight, k_weight, token_mask)
        x = x.contiguous()
        key, value = _token_strided(key), _token_strided(value)   # strided views are read in place (no copy)
        q_weight, k_weight = q_weight.contiguous(), k_weight.contiguous()
        mask = token_mask.contiguous() if token_mask is not None else None
        S, D = x.shape[-2:]
        out = torch.empty_like(x)
        gate = torch.empty(*x.shape[:-1], dtype=torch.float32, device=x.device)
        extension().gate_fwd(x, key, value, q_weight, k_weight, mask, out, gate, float(eps), float(D ** -0.5))
        ctx.save_for_backward(x, key, value, q_weight, k_weight, mask)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, dout):
        x, key, value, q_weight, k_weight, mask = ctx.saved_tensors
        S, D = x.shape[-2:]
        dout = dout.contiguous()
        if dout.dtype != torch.bfloat16:
            dout = dout.to(torch.bfloat16)
        T = x.numel() // (S * D)
        nseg = max(1, -(-T // DW_SEG_T))
        dx, dkey, dval = torch.empty_like(x), torch.empty_like(key), torch.empty_like(value)
        dq, dk = torch.empty_like(q_weight), torch.empty_like(k_weight)
        gate = torch.empty(T * S, dtype=torch.float32, device=x.device)
        dS = torch.empty(T * S, dtype=torch.float32, device=x.device)
        ws = torch.empty(nseg, S * D, dtype=torch.float32, device=x.device)
        extension().gate_bwd(x, key, value, q_weight, k_weight, mask, dout, dx, dkey, dval, dq, dk, gate, dS, ws,
                             float(ctx.eps), float(D ** -0.5))
        return dx, dkey, dval, dq, dk, None, None


def engram_gate_residual(x, key, value, q_weight, k_weight, eps, token_mask=None):
    """``x + gate * value`` over the four streams; bf16 in/out, fp32 internal. Differentiable w.r.t.
    x, key, value, q_weight, k_weight. See README for the declared reduction orders."""
    return _GateResidual.apply(x, key, value, q_weight, k_weight, eps, token_mask)


# ------------------------------------------------------------------ table row gradient
def table_row_grad(indices, grad, num_rows, *, validate=True):
    """Deterministic duplicate-row reduction without atomics.

    ``indices`` int64 [N], ``grad`` bf16/fp32 [N, D] (rows of dL/d(looked-up row)). Returns
    ``(rows int64 [N], g fp32 [N, D], nruns int32 [1])``: the first ``nruns`` entries are the unique touched
    rows in ascending order and their summed gradients; the rest is uninitialised padding."""
    _cuda(indices, "indices", torch.int64)
    _cuda(grad, "grad")
    idx = indices.reshape(-1).contiguous()
    D = grad.shape[-1]
    g2 = grad.reshape(-1, D).contiguous()
    if g2.shape[0] != idx.numel():
        raise ValueError("grad must have one row per index")
    if grad.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("grad must be bf16 or fp32")
    N = idx.numel()
    rows = torch.empty(N, dtype=torch.int64, device=idx.device)
    g = torch.empty(N, D, dtype=torch.float32, device=idx.device)
    nruns = torch.zeros(1, dtype=torch.int32, device=idx.device)
    err = _flag(idx.device)
    extension().table_grad(idx, g2, int(num_rows), rows, g, nruns, err)
    _raise_if(err, "Engram row index outside the table", validate)
    return rows, g, nruns


def scatter_rows_(dense, rows, g, *, nruns):
    """``dense[rows[u]] += g[u]`` for the first ``nruns`` (unique) rows; one writer per row, no atomics."""
    n = torch.as_tensor(nruns, dtype=torch.int32, device=dense.device).reshape(1) if not isinstance(
        nruns, torch.Tensor) else nruns
    extension().scatter_rows(dense, rows, g, n)
    return dense
