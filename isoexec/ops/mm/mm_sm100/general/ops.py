"""Torch surface of mm_sm100/general: every BF16 ``mm``/``bmm`` layout the segtree leaf declines."""

from __future__ import annotations

import functools

import torch

from isoexec.ops import schedule_policy

from ..segments import REDUCER_BLOCK, REDUCER_SHORTEST
from ._extension import extension

RECIPE_CHAIN = "sm100_tcgen05_chain_v1"
RECIPE_KSEGTREE = "sm100_tcgen05_ksegtree_v1"
#: deployment constant for the tile choice only (ownership; the bits never read it)
_SLOTS = 148
# ---- the K schedule: rule 3 of ``isoexec/ops/schedule_policy.py`` on the constants this reducer publishes.
#: The reducer's block for this kernel: one K atom of 64 (four ``tcgen05.mma`` of 16), the unit its segments
#: are cut in (``REDUCER_BLOCK`` of ``mm_sm100/segments.py`` divides it).
SEGMENT_BLOCK = 64
assert SEGMENT_BLOCK % REDUCER_BLOCK == 0 and REDUCER_SHORTEST is None
#: Published shortest segment L_r and cap of the MN-major step program (an operand walked across its rows: the
#: trainer's dgrad / wgrad). Source: this kernel's own rule ("segments of at least 8192 terms, at most 16"),
#: and policy_validation/port/dense/results/bench_general_*.json: with ONE segment instead, reductions of
#: 16384-131072 terms measured 1.05-1.13 of the time at the wide shapes and 2-16 times the error against
#: float64. Occupancy: one worker per segment of one output (the split multiplies the launch's work items), so
#: p = SMs, and the cap binds first. With these constants rule 3 returns exactly the segment count and the
#: ceil-sized segments this kernel has always run; nothing here is keyed on a weight or a table.
SEGMENT_MIN_K = 8192        # L_r
MAX_SPLIT = 16              # cap
SEGMENT_OCCUPANCY = 1

__all__ = ["RECIPE_CHAIN", "RECIPE_KSEGTREE", "bmm", "mm", "plan", "recipe_for", "split_for", "supported"]


def _mn_major(t: torch.Tensor) -> bool | None:
    """Last-two-dims layout of a [.., R, C] operand: False if C is unit-stride (row-major), True if R is."""
    if t.stride(-1) == 1 and (t.size(-2) == 1 or t.stride(-2) >= t.size(-1)):
        return False
    if t.stride(-2) == 1 and (t.size(-1) == 1 or t.stride(-1) >= t.size(-2)):
        return True
    return None


def _layouts(a: torch.Tensor, b: torch.Tensor):
    """(A MN-major, B MN-major) in UMMA terms, or None when a layout is not expressible."""
    ra = _mn_major(a)  # False: a[.., M, K] K-contiguous (K-major)
    rb = _mn_major(b)  # False: b[.., K, N] N-contiguous (MN-major for UMMA B)
    if ra is None or rb is None:
        return None
    return ra, not rb


def _aligned(t: torch.Tensor) -> bool:
    if t.data_ptr() % 16:
        return False
    for d in range(t.dim()):
        s = t.stride(d)
        if s != 1 and t.size(d) > 1 and (s * 2) % 16:
            return False
    return True


def supported(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether :func:`mm` (2-D) or :func:`bmm` (3-D) serves this call."""
    if a.dtype is not torch.bfloat16 or b.dtype is not torch.bfloat16 or not (a.is_cuda and b.is_cuda):
        return False
    if a.dim() != b.dim() or a.dim() not in (2, 3):
        return False
    if a.dim() == 3 and a.size(0) != b.size(0):
        return False
    M, K = a.shape[-2:]
    if b.shape[-2] != K:
        return False
    N = b.shape[-1]
    if K <= 0 or K % 64 or M < 1 or N < 1:
        return False
    lay = _layouts(a, b)
    if lay is None or not (_aligned(a) and _aligned(b)):
        return False
    return True


@functools.lru_cache(maxsize=None)
def schedule(K: int, a_mn: bool, b_mn: bool) -> schedule_policy.Segments:
    """The policy's schedule of the K reduction. K-major x K-major operands run the chain step program, whose
    reducer publishes nothing: one segment. An MN-major operand runs the step program that publishes
    ``SEGMENT_MIN_K`` / ``MAX_SPLIT``: G = pow2floor(min(p, cap, K // L)), S = 64 ceil(K / (64 G))."""
    if not (a_mn or b_mn):
        return schedule_policy.reduce(K, b=SEGMENT_BLOCK, L=REDUCER_SHORTEST)
    return schedule_policy.reduce(K, b=SEGMENT_BLOCK, L=SEGMENT_MIN_K,
                                  p=schedule_policy.workers(_SLOTS, SEGMENT_OCCUPANCY), cap=MAX_SPLIT)


def split_for(K: int, a_mn: bool, b_mn: bool) -> int:
    """The segment count of the K reduction: the schedule policy's G (cached per (K, layout))."""
    return schedule(K, bool(a_mn), bool(b_mn)).G


def recipe_for(a: torch.Tensor, b: torch.Tensor) -> str:
    a_mn, b_mn = _layouts(a, b)
    return RECIPE_KSEGTREE if (a_mn or b_mn) else RECIPE_CHAIN


def plan(M: int, N: int, K: int, batch: int, a_mn: bool, b_mn: bool) -> tuple[int, int, int]:
    """(split, tile width, CTAs per item). Width and CTA pairing are ownership: every choice gives
    the same bits (the battery compares them)."""
    split = split_for(K, a_mn, b_mn)
    short = -(-(K // 64) // split) <= 4   # a few K atoms per item: the epilogue bounds the item
    best = None
    for cl in ((1,) if short else (1, 2)):
        rows = 128 * cl
        if cl == 2 and M <= 128:
            continue
        for w in (32, 64, 128, 256):
            if (b_mn and (w // cl) % 64) or (w == 32 and N > 32) or (cl == 2 and w < 128) or (short and w > 128):
                continue
            items = split * batch * (-(-M // rows)) * (-(-N // w))
            waves = -(-items // (_SLOTS // cl))
            # per-item cost ~ UMMA work + fixed; a pair halves each CTA's B traffic
            cost = waves * (w + 48) * (0.85 if cl == 2 else 1.0)
            if best is None or cost < best[0]:
                best = (cost, w, cl)
    return split, best[1], best[2]


#: host plan cache: (shapes, strides, dtype out) -> (a_mn, b_mn, split, width, pair). Pure host caching.
_PLANS: dict = {}
_NOWS: dict = {}


def _no_ws(device):
    ws = _NOWS.get(device)
    if ws is None:
        ws = _NOWS[device] = torch.empty(0, dtype=torch.float32, device=device)
    return ws


def _run(a3: torch.Tensor, b3: torch.Tensor, *, bn: int | None = None, grid: int = 0,
         cl: int | None = None, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    B, M, K = a3.shape
    N = b3.shape[-1]
    if bn is None and cl is None and grid == 0:
        key = (a3.shape, b3.shape, a3.stride(), b3.stride(), out_dtype)
        ent = _PLANS.get(key)
        if ent is None:
            a_mn, b_mn = _layouts(a3, b3)
            ent = _PLANS[key] = (a_mn, b_mn) + plan(M, N, K, B, a_mn, b_mn)
        a_mn, b_mn, split, width, pair = ent
        out = torch.empty((B, M, N), dtype=out_dtype, device=a3.device)
        if split > 1:
            ws = torch.empty(split * B * M * N, dtype=torch.float32, device=a3.device)
        else:
            ws = _NOWS.get(a3.device)
            if ws is None:
                ws = _NOWS[a3.device] = torch.empty(0, dtype=torch.float32, device=a3.device)
        extension().gemm(a3, b3, out, ws, split, width, 0, a_mn, b_mn, pair)
        return out
    a_mn, b_mn = _layouts(a3, b3)
    split, width, pair = plan(M, N, K, B, a_mn, b_mn)
    if bn is not None:
        width = bn
    if cl is not None:
        pair = cl
    elif bn is not None:
        pair = 1
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"mm_sm100/general returns bf16 or fp32, not {out_dtype}")
    out = torch.empty((B, M, N), dtype=out_dtype, device=a3.device)
    ws = (torch.empty(split * B * M * N, dtype=torch.float32, device=a3.device) if split > 1
          else torch.empty(0, dtype=torch.float32, device=a3.device))
    extension().gemm(a3, b3, out, ws, split, width, grid, a_mn, b_mn, pair)
    return out


_OK: set = set()


def mm(a: torch.Tensor, b: torch.Tensor, *, bn: int | None = None, grid: int = 0,
       cl: int | None = None) -> torch.Tensor:
    """``a[M,K] @ b[K,N]`` for any expressible layout of either operand (no copy)."""
    key = (a.shape, b.shape, a.stride(), b.stride(), a.dtype, b.dtype, a.device, b.device)
    if bn is None and cl is None and grid == 0:
        ent = _PLANS.get(key)
        if ent is not None and not (a.data_ptr() & 15) and not (b.data_ptr() & 15):
            a_mn, b_mn, split, width, pair = ent
            out = torch.empty((a.shape[0], b.shape[1]), dtype=torch.bfloat16, device=a.device)
            ws = (torch.empty(split * a.shape[0] * b.shape[1], dtype=torch.float32, device=a.device)
                  if split > 1 else _no_ws(a.device))
            extension().gemm(a, b, out, ws, split, width, 0, a_mn, b_mn, pair)
            return out
    if key in _OK and not (a.data_ptr() & 15) and not (b.data_ptr() & 15):
        return _run(a.unsqueeze(0), b.unsqueeze(0), bn=bn, grid=grid, cl=cl)[0]
    if not supported(a, b) or a.dim() != 2:
        raise ValueError(f"mm_sm100/general declines a={tuple(a.shape)}/{a.stride()} b={tuple(b.shape)}/{b.stride()}")
    _OK.add(key)
    if bn is None and cl is None and grid == 0:
        a_mn, b_mn = _layouts(a, b)
        _PLANS[key] = (a_mn, b_mn) + plan(a.shape[0], b.shape[1], a.shape[1], 1, a_mn, b_mn)
    return _run(a.unsqueeze(0), b.unsqueeze(0), bn=bn, grid=grid, cl=cl)[0]


def bmm(a: torch.Tensor, b: torch.Tensor, *, bn: int | None = None, grid: int = 0,
        cl: int | None = None, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """``a[B,M,K] @ b[B,K,N]``; one chain or K-segment tree per output exactly as :func:`mm`.
    ``out_dtype=float32`` returns the FP32 root unrounded (``torch.bmm(..., out_dtype=float32)``)."""
    if not supported(a, b) or a.dim() != 3:
        raise ValueError(f"mm_sm100/general declines bmm a={tuple(a.shape)}/{a.stride()} b={tuple(b.shape)}/{b.stride()}")
    return _run(a, b, bn=bn, grid=grid, cl=cl, out_dtype=out_dtype)
