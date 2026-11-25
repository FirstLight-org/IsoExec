"""Reference programs of the three DSA recipes (README.md sections 2-4), numpy, no CUDA required.

Two kinds of reference per sub-op:

* a **float64** reference (``*_fp64``): the mathematical value, for tolerance checks (R-I1, R-A1);
* an **fp32 emulation** (``*_emul``): the recipe's scalar program re-stated operation by operation in
  numpy float32, whose bits must equal the kernel's (R-I2, R-A2).  numpy float32 arithmetic is IEEE
  round-to-nearest with no contraction, which is exactly what ``--fmad=false`` compiles on the device.

Four device primitives cannot be re-derived in numpy bit for bit, because their bits are defined by
the hardware: ``exp2f`` (MUFU.EX2 under ``--ftz=true``), ``logf``, the ``--prec-div=false`` division
(``div.full.f32``) and the tensor-core accumulation of one UMMA k-step.  The emulation takes them from
a ``Prims`` object.  ``CpuPrims`` is the numpy stand-in (close, not bit-exact: used by the CPU lane to
check structure); the GPU tests pass ``DevicePrims`` (``tests/_prims.py``), which evaluates exactly
those four primitives on the device under this package's profile.  Everything ELSE -- the order of
the k-steps, the masks, the max, the row-sum order, the correction, the fold over heads, the merge,
the tree, the root -- is this file's own statement and is what the bit comparison proves.

The tensor-core accumulation model (``mma_*``) is a hypothesis about the hardware, validated on the
device by ``tests/test_prims_gpu.py`` (README "Tensor-core accumulation model"): one k-step's block
dot product is formed exactly and added to the fp32 accumulator with one round-to-nearest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import spec

F32 = np.float32
MAX_INIT_VAL = F32(spec.MAX_INIT_VAL)
MAX_INIT_VAL_SM = F32(spec.MAX_INIT_VAL_SM)
LOG2E_F32 = F32(math.log2(math.e))   # (float)M_LOG2E


# =================================================================================================
# scalar helpers
# =================================================================================================

def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """IEEE round-to-nearest-even to bf16, NaN kept NaN (quiet), as ``__float2bfloat16_rn``."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    nan = np.isnan(x)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF
    out = np.where(nan, ((u >> 16) | 0x0040) & 0xFFFF, rounded)
    return out.astype(np.uint16)


def bf16_round(x: np.ndarray) -> np.ndarray:
    return bf16_bits_to_f32(f32_to_bf16_bits(x))


def e4m3_to_f64(b: np.ndarray) -> np.ndarray:
    """float8_e4m3fn bytes -> float64 (NaN for 0x7F / 0xFF; no infinities in e4m3fn)."""
    b = b.astype(np.int64)
    s = np.where(b & 0x80, -1.0, 1.0)
    e = (b >> 3) & 0xF
    m = b & 0x7
    val = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1.0 + m / 8.0) * np.exp2(e - 7.0))
    val = np.where((b & 0x7F) == 0x7F, np.nan, val)
    return s * val


def ue8m0_to_f64(b: np.ndarray) -> np.ndarray:
    b = b.astype(np.int64)
    return np.where(b == 255, np.nan, np.exp2(b - 127.0))


def unpack_ue8m0(words: np.ndarray) -> np.ndarray:
    """int32 packed scales ``[...]`` -> uint8 biased exponents ``[..., 4]`` (byte 0 = block 0)."""
    w = words.astype(np.int64) & 0xFFFFFFFF
    return np.stack([(w >> (8 * i)) & 0xFF for i in range(4)], axis=-1).astype(np.uint8)


# =================================================================================================
# device primitives
# =================================================================================================

class CpuPrims:
    """numpy stand-ins for the device primitives.  NOT bit-exact to the device; structure only."""

    exact = False

    @staticmethod
    def exp2f(x):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            r = np.exp2(np.asarray(x, dtype=np.float64)).astype(np.float32)
        return np.where(np.abs(r) < np.finfo(np.float32).tiny, F32(0), r).astype(np.float32)

    @staticmethod
    def logf(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(np.asarray(x, dtype=np.float64)).astype(np.float32)

    @staticmethod
    def divf(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (np.asarray(a, dtype=np.float32) / np.asarray(b, dtype=np.float32)).astype(np.float32)

    @staticmethod
    def mma_f16(acc, A, B):
        """One tcgen05 kind::f16 k-step, D = acc + A.B^T (A [M, 16], B [N, 16] bf16 values as float64;
        acc fp32 [M, N] or None for ScaleOut::Zero).  The model FITTED EXACTLY to the hardware by the
        one-instruction oracle ``csrc/prims/mma_probe.cu`` (``tools/fit_mma_model.py``: 0 mismatches over
        196,608 outputs, single and accumulating steps, wide and cancellation-heavy data): every
        product and the accumulator are aligned to ``E = max(max_k(e(a_k) + e(b_k)), e(acc)) + 1``
        (e = floor(log2|x|), the products' exponents taken from their factors, before the mantissa
        product's carry), truncated toward zero to the multiple of ``2^(E - 26)``, summed exactly, and
        the sum rounded toward zero to fp32.  It is NOT an exact sum rounded to nearest (that model
        mismatches 0.4% of outputs)."""
        A = np.asarray(A, np.float64)
        B = np.asarray(B, np.float64)
        P = A[:, None, :] * B[None, :, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            fa = np.where(A != 0, np.floor(np.log2(np.abs(A))), -np.inf)
            fb = np.where(B != 0, np.floor(np.log2(np.abs(B))), -np.inf)
            ep = fa[:, None, :] + fb[None, :, :]
            C = np.zeros(P.shape[:2]) if acc is None else np.asarray(acc, np.float64)
            ec = np.where(C != 0, np.floor(np.log2(np.abs(C))), -np.inf)
            E = np.maximum(ep.max(-1), ec) + 1
            Ef = np.where(np.isfinite(E), E, 0.0)
            q = np.exp2(Ef - 26)
            s = (np.trunc(P / q[..., None]) * q[..., None]).sum(-1) + np.trunc(C / q) * q
            s = np.where(np.isfinite(E), s, 0.0) if True else s
            nonfinite = ~np.isfinite(P).all(-1) | ~np.isfinite(C)
            s = np.where(nonfinite, P.sum(-1) + C, s)
        out = s.astype(np.float32)
        over = np.abs(out.astype(np.float64)) > np.abs(s)
        out = np.where(over, np.nextafter(out, np.float32(0)), out)
        return out.astype(np.float32)

    @staticmethod
    def mma_step(acc, exact_block):
        """One UMMA k-step: ``acc`` fp32 (or None for ScaleOut::Zero) plus an EXACT block dot
        product (float64 that holds it exactly), rounded once to fp32."""
        if acc is None:
            return exact_block.astype(np.float32)
        return (acc.astype(np.float64) + exact_block).astype(np.float32)


# =================================================================================================
# SPEC section 3: the top-k key map, order and output
# =================================================================================================

def key_bf16(bits: np.ndarray) -> np.ndarray:
    """kappa: bf16 bits -> uint16, monotone and total; NaN -> 0, -inf -> 0x007F, +0 -> 0x8000,
    -0 -> 0x7FFF (so -0 ranks just below +0), +inf -> 0xFF80."""
    x = bits.astype(np.uint32) & 0xFFFF
    k = np.where(x & 0x8000, (~x) & 0xFFFF, x ^ 0x8000)
    nan = (x & 0x7FFF) > 0x7F80
    return np.where(nan, 0, k).astype(np.uint32)


def key_f32(x: np.ndarray) -> np.ndarray:
    """kappa for fp32 scores: uint32, NaN -> 0."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    k = np.where(u & 0x80000000, (~u) & 0xFFFFFFFF, u ^ 0x80000000)
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    return np.where(nan, 0, k).astype(np.uint64)


def topk_ref(scores: np.ndarray, ks: np.ndarray, ke: np.ndarray, k: int, dtype: str = "bf16") -> np.ndarray:
    """SPEC section 3: per row, the K largest under (kappa desc, j asc), output ascending j, -1 pad.

    ``scores``: bf16 bits (uint16) or float32, ``[rows, width]``; ``ks``/``ke``: the window."""
    rows = scores.shape[0]
    out = np.full((rows, k), spec.TOPK_PAD, dtype=np.int32)
    keyf = key_bf16 if dtype == "bf16" else key_f32
    for r in range(rows):
        a, b = int(ks[r]), int(ke[r])
        if b <= a:
            continue
        j = np.arange(a, b, dtype=np.int64)
        kap = keyf(scores[r, a:b]).astype(np.int64)
        order = np.lexsort((j, -kap))          # primary: kappa descending; ties: j ascending
        sel = np.sort(j[order[:k]])
        out[r, :sel.size] = sel
    return out


def topk_radix_emul(scores: np.ndarray, ks: np.ndarray, ke: np.ndarray, k: int, dtype: str = "bf16",
                    digit_bits: int = 8, ctas: int = 1) -> np.ndarray:
    """The ALGORITHM CLASS of SPEC section 3 item 4 (radix select + ascending scan compaction),
    written the way the kernel does it: integer histograms per digit (optionally split over ``ctas``
    key ranges and summed -- R-T8's claim), exact threshold, then a scan in ascending j where every
    key above the threshold and the first ``K - c_gt`` keys equal to it are emitted at their prefix
    count.  Returns the same array as ``topk_ref``; the CPU contract test checks that."""
    rows = scores.shape[0]
    out = np.full((rows, k), spec.TOPK_PAD, dtype=np.int32)
    keyf = key_bf16 if dtype == "bf16" else key_f32
    total_bits = 16 if dtype == "bf16" else 32
    for r in range(rows):
        a, b = int(ks[r]), int(ke[r])
        n = b - a
        if n <= 0:
            continue
        kap = keyf(scores[r, a:b]).astype(np.int64)
        if n <= k:
            out[r, :n] = np.arange(a, b)
            continue
        prefix, prefix_mask, need = 0, 0, k      # need: how many still to select among the prefix class
        c_gt = 0
        for shift in range(total_bits - digit_bits, -1, -digit_bits):
            cand = (kap & prefix_mask) == prefix
            digit = (kap >> shift) & ((1 << digit_bits) - 1)
            hist = np.zeros(1 << digit_bits, dtype=np.int64)
            for part in np.array_split(np.arange(n), ctas):      # the multi-CTA path: same sums
                np.add.at(hist, digit[part][cand[part]], 1)
            # find the digit d* where the count of keys with digit > d* is < need <= that + hist[d*]
            above = 0
            for d in range((1 << digit_bits) - 1, -1, -1):
                if above + hist[d] >= need:
                    break
                above += hist[d]
            c_gt += above
            need -= above
            prefix |= d << shift
            prefix_mask |= ((1 << digit_bits) - 1) << shift
        theta = prefix
        gt = kap > theta
        eq = kap == theta
        eq_rank = np.cumsum(eq) - eq            # exclusive prefix count of ties
        take = gt | (eq & (eq_rank < k - c_gt))
        slots = np.cumsum(take) - take          # exclusive prefix count: the slot, never an atomic
        sel = np.nonzero(take)[0]
        out[r, slots[sel]] = sel + a
        assert take.sum() == k
    return out


# =================================================================================================
# SPEC section 2: the indexer
# =================================================================================================

def indexer_dots_exact(q_fp8, q_sf, k_fp8, k_sf) -> np.ndarray:
    """Exact per-(t, h, j) per-block dot products, float64 ``[T, H, S, 4]`` (block b covers
    c in [32b, 32b+32)); e4m3 x e4m3 x 2^(sa+sb) products are exact in float64 and a 32-term block
    sum of them is exact too (at most 2*4+1 significant bits per product, exponents within one
    block share their scale)."""
    q = e4m3_to_f64(q_fp8)                           # [T, H, 128]
    k = e4m3_to_f64(k_fp8)                           # [S, 128]
    qs = ue8m0_to_f64(unpack_ue8m0(q_sf))            # [T, H, 4]
    kscale = ue8m0_to_f64(unpack_ue8m0(k_sf))        # [S, 4]
    T, H, D = q.shape
    S = k.shape[0]
    blocks = np.empty((T, H, S, D // 32), dtype=np.float64)
    for b in range(D // 32):
        qb = q[:, :, 32 * b:32 * b + 32]
        kb = k[:, 32 * b:32 * b + 32]
        blocks[..., b] = np.einsum("thc,sc->ths", qb, kb) * qs[:, :, None, b] * kscale[None, None, :, b]
    return blocks


def indexer_dot_emul(blocks: np.ndarray, prims=CpuPrims) -> np.ndarray:
    """The tensor-core accumulation of SPEC section 2 item 1: four k-steps of 32 in ascending c, the
    first with ScaleOut::Zero, fp32 accumulator.  ``[T, H, S]`` float32."""
    acc = None
    for b in range(blocks.shape[-1]):
        acc = prims.mma_step(acc, blocks[..., b])
    return acc


def indexer_fold_emul(d: np.ndarray, w_bf16_bits: np.ndarray) -> np.ndarray:
    """SPEC section 2 items 2-4: ``r = max(d, 0)``; ``s = ((0 + r0*w0) + r1*w1) + ...`` over the
    ascending logical head index in fp32, multiply and add as two roundings; ``bf16_rne(s)``.
    ``d`` float32 ``[T, H, S]``; returns bf16 bits ``[T, S]``."""
    T, H, S = d.shape
    w = bf16_bits_to_f32(w_bf16_bits.astype(np.uint16))    # [T, H]
    s = np.zeros((T, S), dtype=np.float32)
    for h in range(H):
        r = np.fmax(d[:, h, :], F32(0.0))                    # fmaxf(d, 0): NaN -> 0, -0 -> +0
        r = np.where(r == 0, F32(0.0), r).astype(np.float32)
        s = (s + (r * w[:, h:h + 1]).astype(np.float32)).astype(np.float32)
    return f32_to_bf16_bits(s)


def indexer_emul(q_fp8, q_sf, k_fp8, k_sf, w_bits, ks, ke, prims=CpuPrims) -> np.ndarray:
    """The whole indexer program (bf16 bits ``[T, S]``), SCORE_PAD outside ``[ks, ke)``.  Keys
    outside a row's window never reach the fold: their dot is computed (the MMA may read them) and
    replaced before the relu, exactly as SPEC item 7 states."""
    blocks = indexer_dots_exact(q_fp8, q_sf, k_fp8, k_sf)
    d = indexer_dot_emul(blocks, prims)
    j = np.arange(d.shape[2])
    inwin = (j[None, :] >= ks[:, None]) & (j[None, :] < ke[:, None])
    d = np.where(inwin[:, None, :], d, F32(0.0)).astype(np.float32)
    s = indexer_fold_emul(d, w_bits)
    return np.where(inwin, s, np.uint16(spec.SCORE_PAD_BITS)).astype(np.uint16)


def indexer_fp64(q_fp8, q_sf, k_fp8, k_sf, w_bits, ks, ke) -> np.ndarray:
    """float64 value of the indexer (no intermediate rounding), -inf outside the window."""
    d = indexer_dots_exact(q_fp8, q_sf, k_fp8, k_sf).sum(-1)
    w = bf16_bits_to_f32(w_bits.astype(np.uint16)).astype(np.float64)
    s = np.einsum("ths,th->ts", np.maximum(d, 0.0), w)
    j = np.arange(s.shape[1])
    inwin = (j[None, :] >= ks[:, None]) & (j[None, :] < ke[:, None])
    return np.where(inwin, s, -np.inf)


# =================================================================================================
# SPEC section 4: the sparse attention (leaf, merge, root, group tree)
# =================================================================================================

@dataclass
class State:
    """A segment's fp32 partial ``(O[H, 512], L[H], M[H])``, never normalised."""
    O: np.ndarray
    L: np.ndarray
    M: np.ndarray

    @staticmethod
    def empty(h: int, dv: int = spec.D_V) -> "State":
        return State(np.zeros((h, dv), np.float32), np.zeros(h, np.float32),
                     np.full(h, MAX_INIT_VAL_SM, np.float32))


def scale_log2_of(softmax_scale: float) -> np.float32:
    """Host-side ``float(softmax_scale * M_LOG2E)`` (double product, one rounding), mla_sm100's."""
    return F32(float(softmax_scale) * math.log2(math.e))


def qk_exact_blocks(q: np.ndarray, k: np.ndarray, kstep: int = 16) -> np.ndarray:
    """Exact per-k-step block dots of bf16 q ``[H, D]`` and bf16 k ``[N, D]``: float64 ``[H, N, D/kstep]``.
    (bf16 x bf16 products carry 16 significant bits; a 16-term block of them is exact in float64 for
    any exponent spread below 2^36, which a finite bf16 dot of this program's magnitudes satisfies.)"""
    H, D = q.shape
    q64 = q.astype(np.float64)
    k64 = k.astype(np.float64)
    nb = D // kstep
    return np.einsum("hbc,nbc->hnb", q64.reshape(H, nb, kstep), k64.reshape(-1, nb, kstep))


def pv_exact_blocks(p: np.ndarray, v: np.ndarray, kstep: int = 16) -> np.ndarray:
    """Exact per-k-step block products of bf16 P ``[H, 64]`` and bf16 V ``[64, Dv]``: ``[H, Dv, 64/kstep]``."""
    H, N = p.shape
    nb = N // kstep
    return np.einsum("hbc,bcd->hdb", p.astype(np.float64).reshape(H, nb, kstep),
                     v.astype(np.float64).reshape(nb, kstep, -1))


def lane_row_sum(p: np.ndarray) -> np.ndarray:
    """SPEC section 4.2: the tile's full row sum.  The 16dp256b8x fragment gives lane quad c
    (c = lane % 4) the columns {8r + 2c, 8r + 2c + 1 : r = 0..7}; each lane sums them as ascending
    adjacent pairs, ``part = ((0 + (p0 + p1)) + (p8 + p9)) + ...``, then ``shfl_xor 1`` and
    ``shfl_xor 2``: ``row = (part0 + part1) + (part2 + part3)``.  ``p`` float32 ``[H, 64]``."""
    parts = []
    for c in range(4):
        part = np.zeros(p.shape[0], np.float32)
        for r in range(8):
            pair = (p[:, 8 * r + 2 * c] + p[:, 8 * r + 2 * c + 1]).astype(np.float32)
            part = (part + pair).astype(np.float32)
        parts.append(part)
    a = (parts[0] + parts[1]).astype(np.float32)
    b = (parts[2] + parts[3]).astype(np.float32)
    return (a + b).astype(np.float32)


def leaf_emul(q: np.ndarray, rows: np.ndarray, valid: np.ndarray, scale_log2: np.float32,
              prims=CpuPrims, tile: int = spec.TILE_SLOTS, lazy: bool = False) -> State:
    """One segment of SPEC section 4.2, fp32, operation by operation.

    ``q``: bf16 values as float32 ``[H, 512]``; ``rows``: the segment's gathered K/V rows AFTER the
    declared conversion (bf16 values as float32) ``[P, 512]`` in list order (P a multiple of the
    tile); ``valid``: bool ``[P]`` (False for a pad slot or a slot past topk_length).  A pad slot's
    row may hold anything (NaN included): its score is masked before the max and its V row is zeroed
    before PV, so nothing it holds can reach the state."""
    H = q.shape[0]
    P = rows.shape[0]
    assert P % tile == 0
    st = State.empty(H)
    rM, rL, O = st.M.copy(), st.L.copy(), None
    for t0 in range(0, P, tile):
        kt = rows[t0:t0 + tile]
        vt = valid[t0:t0 + tile]
        # QK: 32 k-steps of 16 over c = 0..511 ascending, first ScaleOut::Zero.  A pad slot's row is
        # gathered as zeros by the kernel (TMA zero-fill); the emulation does the same (its score is
        # masked either way).
        kz = np.where(vt[:, None], kt, np.float32(0)).astype(np.float64)
        q64 = q.astype(np.float64)
        S = None
        for c0 in range(0, q.shape[1], 16):
            S = prims.mma_f16(S, q64[:, c0:c0 + 16], kz[:, c0:c0 + 16])
        S = np.where(vt[None, :], S, MAX_INIT_VAL).astype(np.float32)          # mask before the max
        cur_max = np.fmax.reduce(S, axis=1).astype(np.float32)
        cand = np.fmax(rM, (cur_max * scale_log2).astype(np.float32)).astype(np.float32)
        if lazy:
            # recipe v2 (sm100_sparse_v2): KEEP the running max while the tile's candidate is within
            # RESCALE_THRESHOLD = 8 log2 units of it (mla_sm100 / gqa_sm100 v2's rule); kept rows have
            # scale_old exactly 1.  A segment's first tile commits (rM = MAX_INIT_VAL_SM there).
            keep = (cand - rM).astype(np.float32) <= np.float32(spec.RESCALE_THRESHOLD_V2)
            new_max = np.where(keep, rM, cand).astype(np.float32)
            scale_old = np.where(keep, np.float32(1), prims.exp2f((rM - cand).astype(np.float32))).astype(np.float32)
        else:
            new_max = cand
            scale_old = prims.exp2f((rM - new_max).astype(np.float32))
        rM = new_max
        x = ((S * scale_log2).astype(np.float32) - new_max[:, None]).astype(np.float32)
        p = prims.exp2f(x)
        p_bf = bf16_round(p)
        rL = ((rL * scale_old).astype(np.float32) + lane_row_sum(p)).astype(np.float32)
        # V of a pad slot is zeroed after QK read it and before PV
        vv = np.where(vt[:, None], kt[:, :spec.D_V], np.float32(0)).astype(np.float32)
        if O is None:                                   # the segment's first PV: ScaleOut::Zero
            acc = None
        else:                                           # O *= scale_old (x * 1.0f == x: the skip is free)
            acc = (O * scale_old[:, None]).astype(np.float32)
            # ... except under --ftz=true: the device multiplies a 16-row warp's rows whenever ANY of
            # them rescales, and a multiply flushes a denormal product to +-0 (Fable review item 12).
            # Rows of a warp that skipped keep their denormals.
            warp_rescales = np.repeat((scale_old.reshape(-1, 16) != np.float32(1)).any(axis=1), 16) \
                if scale_old.size % 16 == 0 else np.zeros(scale_old.size, bool)
            tiny = np.float32(np.finfo(np.float32).tiny)
            flush = warp_rescales[:, None] & (np.abs(acc) < tiny)
            acc = np.where(flush, np.copysign(np.float32(0), acc), acc).astype(np.float32)
        p64 = p_bf.astype(np.float64)
        v64 = vv.astype(np.float64)
        for c0 in range(0, tile, 16):                   # four K16 steps over the tile's slots, ascending
            acc = prims.mma_f16(acc, p64[:, c0:c0 + 16], v64[c0:c0 + 16, :].T)
        O = acc
    return State(O if O is not None else st.O, rL, rM)


def merge(a: State, b: State, prims=CpuPrims) -> State:
    """SPEC section 4.3, ``a`` the lower group index: ``M = max``, ``sa = exp2f(a.M - M)``,
    ``sb = exp2f(b.M - M)``, ``O = a.O*sa + b.O*sb`` and ``L = a.L*sa + b.L*sb`` (two products, one add).
    An EMPTY operand (``L == 0``, the exact emptiness test) is bypassed verbatim: the result is the
    other operand's bits, per row."""
    M = np.fmax(a.M, b.M).astype(np.float32)
    sa = prims.exp2f((a.M - M).astype(np.float32))
    sb = prims.exp2f((b.M - M).astype(np.float32))
    O = ((a.O * sa[:, None]).astype(np.float32) + (b.O * sb[:, None]).astype(np.float32)).astype(np.float32)
    L = ((a.L * sa).astype(np.float32) + (b.L * sb).astype(np.float32)).astype(np.float32)
    a_empty = a.L == 0
    b_empty = b.L == 0
    O = np.where(a_empty[:, None], b.O, np.where(b_empty[:, None], a.O, O))
    L = np.where(a_empty, b.L, np.where(b_empty, a.L, L))
    M = np.where(a_empty, b.M, np.where(b_empty, a.M, M))
    return State(O.astype(np.float32), L.astype(np.float32), M.astype(np.float32))


def root(st: State, prims=CpuPrims):
    """``L' = (L == 0 || L != L) ? 1 : L``, ``out = bf16(O / L')``, ``lse = logf(L') + M / log2(e)``.
    Returns (out bf16 bits ``[H, 512]``, lse float32 ``[H]``)."""
    Lp = np.where((st.L == 0) | np.isnan(st.L), F32(1.0), st.L).astype(np.float32)
    out = f32_to_bf16_bits(prims.divf(st.O, np.broadcast_to(Lp[:, None], st.O.shape)))
    if hasattr(prims, "lse"):
        lse = prims.lse(Lp, st.M)
    else:
        lse = (prims.logf(Lp) + prims.divf(st.M, np.full_like(st.M, LOG2E_F32))).astype(np.float32)
    return out, lse


def tree_shape(n: int):
    """The canonical adjacent-pair tree over ``n`` groups in ascending index, unpaired carried up.
    Returned as a nested tuple of leaf indices, e.g. 3 -> ((0, 1), 2)."""
    nodes = list(range(n))
    while len(nodes) > 1:
        nxt = [(nodes[i], nodes[i + 1]) for i in range(0, len(nodes) - 1, 2)]
        if len(nodes) % 2:
            nxt.append(nodes[-1])
        nodes = nxt
    return nodes[0] if nodes else None


def tree_fold(states, prims=CpuPrims, combine=None):
    """Evaluate ``tree_shape(len(states))`` level by level (the G-workers + combine form)."""
    combine = combine or (lambda a, b: merge(a, b, prims))
    nodes = list(states)
    while len(nodes) > 1:
        nxt = [combine(nodes[i], nodes[i + 1]) for i in range(0, len(nodes) - 1, 2)]
        if len(nodes) % 2:
            nxt.append(nodes[-1])
        nodes = nxt
    return nodes[0]


def tree_fold_stack(states, prims=CpuPrims, combine=None, trace=None):
    """The ONE-worker form: groups arrive in ascending order and are folded in place on a stack of
    pending nodes; the merge schedule is that of ``tree_shape`` (a node merges with its left sibling
    as soon as both exist at the same level; at the end the stack folds from the top).  Must return
    the same bits as ``tree_fold`` for every n -- ``tests/test_contract_cpu.py`` checks it."""
    combine = combine or (lambda a, b: merge(a, b, prims))
    n = len(states)
    if n == 0:
        return None
    # level-pairing tree == streaming binary counter with the tail folded right-to-left (proved by
    # the CPU test over n = 1..64); the stack holds (node, size).
    stack = []
    for i, s in enumerate(states):
        stack.append((s, 1))
        while len(stack) >= 2 and stack[-1][1] == stack[-2][1]:
            (a, za), (b, zb) = stack[-2], stack[-1]
            stack[-2:] = [(combine(a, b), za + zb)]
            if trace is not None:
                trace.append((za, zb))
    while len(stack) >= 2:
        (a, za), (b, zb) = stack[-2], stack[-1]
        stack[-2:] = [(combine(a, b), za + zb)]
        if trace is not None:
            trace.append((za, zb))
    return stack[0][0]


def sparse_attn_emul(q: np.ndarray, rows: np.ndarray, valid: np.ndarray, softmax_scale: float,
                     groups: int = 1, prims=CpuPrims, one_worker: bool = False, lazy: bool = False):
    """SPEC section 4 for one query row: ``groups`` fixed groups of ``K/G`` list positions, each a
    leaf; the group tree; the root.  Returns (out bf16 bits, lse, root State)."""
    P = rows.shape[0]
    assert P % groups == 0 and (P // groups) % spec.TILE_SLOTS == 0
    seg = P // groups
    sl = scale_log2_of(softmax_scale)
    leaves = [leaf_emul(q, rows[g * seg:(g + 1) * seg], valid[g * seg:(g + 1) * seg], sl, prims, lazy=lazy)
              for g in range(groups)]
    st = (tree_fold_stack if one_worker else tree_fold)(leaves, prims)
    out, lse = root(st, prims)
    return out, lse, st


def sparse_attn_fp64(q: np.ndarray, rows: np.ndarray, valid: np.ndarray, softmax_scale: float):
    """float64 softmax attention over the valid slots; an all-pad row gives out 0 and the declared
    empty lse ``logf(1) + MAX_INIT_VAL_SM / log2(e)``."""
    q64 = q.astype(np.float64)
    k64 = np.where(valid[:, None], rows.astype(np.float64), 0.0)
    s = q64 @ k64.T * softmax_scale
    s = np.where(valid[None, :], s, -np.inf)
    if not valid.any():
        return np.zeros((q.shape[0], spec.D_V)), np.full(q.shape[0], spec.MAX_INIT_VAL_SM / math.log2(math.e))
    m = s.max(axis=1, keepdims=True)
    e = np.exp(s - m)
    lsum = e.sum(axis=1, keepdims=True)
    out = (e / lsum) @ k64[:, :spec.D_V]
    lse = (np.log(lsum) + m)[:, 0]
    return out, lse
