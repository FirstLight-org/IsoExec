"""Fused engine routing and integer permute sort with pinned launch geometry and no FMA contraction.
The softmax uses libdevice exp, non-FTZ RN division and a power-of-two reduction matching the eager recipe.
Expert membership and index/probability pairing match the eager router; tied-column order is deterministic
but may differ, so consumers must be invariant to paired column permutations."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    from isoexec.core import triton_nonftz as _nonftz

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


_ENV = "ISOEXEC_MOE_ROUTER_O2"


def router_o2_enabled() -> bool:
    """Default OFF. Re-read per call so an in-process A/B can flip it between forwards."""
    return os.environ.get(_ENV, "0") == "1"


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


# One warp and pinned power-of-two tiles preserve the validated softmax reduction tree.
# Do not autotune: changing num_warps can change association; only the grid depends on T.
_ROUTER_WARPS = 1
_SORT_WARPS = 4
_SORT_BLOCK_T = 512


if HAVE_TRITON:

    # Use the shared non-FTZ RN divide: libdevice.div_rn can flush subnormals, while / uses approximate reciprocals.
    _div_rn_nonftz = _nonftz.div_rn

    @triton.jit
    def _router_kernel(
        LOGITS,  # [T, E] fp32
        RPROBS,  # [T, E] fp32   out: dense routing probs        (EMIT_DENSE)
        RMAP,  # [T, E] int8   out: dense routing map (viewed as bool by the caller)  (EMIT_DENSE)
        TPRB,  # [T, K] fp32   out: the selected probs, selection order   (EMIT_TOPK)
        TIDX,  # [T, K] int32  out: the selected expert ids, selection order  (EMIT_TOPK)
        T,
        E,
        stride_l,
        SCALE,  # float; 1.0 and HAS_SCALE=False when moe_router_topk_scaling_factor is unset
        K: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        EMIT_DENSE: tl.constexpr,
        EMIT_TOPK: tl.constexpr,
    ):
        """Select top-k then apply max/sub/exp/sum/div softmax for one token per program.
        Both output forms share this arithmetic: dense mode scatters to [T,E], top-k mode stores [T,k] registers."""
        t = tl.program_id(0)
        cols = tl.arange(0, BLOCK_E)
        cmask = cols < E
        s = tl.load(LOGITS + t * stride_l + cols, mask=cmask, other=float("-inf"))
        # Map NaN to -inf so max/equality selection cannot return the out-of-range E sentinel.
        s = tl.where(s != s, float("-inf"), s)

        live = cmask
        kar = tl.arange(0, BLOCK_K)
        vals = tl.full([BLOCK_K], float("-inf"), tl.float32)
        idxs = tl.zeros([BLOCK_K], tl.int32)

        # Select descending score, then lowest expert index. Membership matches eager; tied-column positions may differ.
        for j in tl.static_range(K):
            cur = tl.where(live, s, float("-inf"))
            best = tl.max(cur, axis=0)
            eq = live & (cur == best)
            bidx = tl.min(tl.where(eq, cols, E), axis=0)
            # Belt and braces: unreachable after the NaN canonicalisation, kept so "the stored index
            # is in [0, E)" is true by construction rather than by argument.
            bidx = tl.minimum(bidx, E - 1)
            vals = tl.where(kar == j, best, vals)
            idxs = tl.where(kar == j, bidx, idxs)
            live = live & (cols != bidx)

        # ---- vllm softmax_batch_invariant over the K selected scores
        kmask = kar < K
        # amax: order-free, so no reduction-order hazard here.
        vmax = tl.max(tl.where(kmask, vals, float("-inf")), axis=0)
        # libdevice.exp, NOT tl.exp: tl.exp lowers to ex2.approx and diverges from ATen.
        ex = libdevice.exp(vals - vmax)
        ex = tl.where(kmask, ex, 0.0)
        # tl.sum over a power-of-two BLOCK_K in a single-warp program is the same butterfly (half-fold)
        # tree torch.sum uses over a length-K contiguous last dim. _router_can_handle refuses
        # non-power-of-two K, where the zero padding would re-associate.
        ssum = tl.sum(ex, axis=0)
        # Neither `libdevice.div_rn` nor `/` is correct here -- see _div_rn_nonftz. `/` is an approximate
        # reciprocal; `div_rn` is correctly rounded but FLUSHES SUBNORMALS TO ZERO, which a
        # massive-activation router row produces routinely.
        p = _div_rn_nonftz(ex, ssum)
        if HAS_SCALE:
            # megatron: `probs = probs * scaling_factor` -- a plain fp32 multiply, no accumulator,
            # so nothing for FFMA contraction to reach.
            p = p * SCALE

        # Store [T,k] registers in deterministic selection order without dense [T,E] traffic.
        if EMIT_TOPK:
            tl.store(TPRB + t * K + kar, p, mask=kmask)
            tl.store(TIDX + t * K + kar, idxs, mask=kmask)

        # Dense scatter (zeros_like + index_put_) in one pass: build [BLOCK_E] dense rows by OR-ing the
        # K one-hot columns. K is constexpr, so this unrolls and stays in registers -- no [T,E] zero-fill
        # kernel and no index_put_.
        if EMIT_DENSE:
            dprob = tl.zeros([BLOCK_E], tl.float32)
            dmap = tl.zeros([BLOCK_E], tl.int8)
            for j in tl.static_range(K):
                ij = tl.sum(tl.where(kar == j, idxs, 0), axis=0)
                pj = tl.sum(tl.where(kar == j, p, 0.0), axis=0)
                hit = cols == ij
                dprob = tl.where(hit, pj, dprob)
                dmap = tl.where(hit, 1, dmap).to(tl.int8)

            tl.store(RPROBS + t * E + cols, dprob, mask=cmask)
            tl.store(RMAP + t * E + cols, dmap, mask=cmask)

    @triton.jit
    def _counts_kernel(RMAP, COUNTS, T, E, BLOCK_T: tl.constexpr):
        """``tokens_per_expert = routing_map.sum(dim=0).long()`` -- one program per expert.

        Integer sum: exact, order-free, no rounding to reproduce.
        """
        e = tl.program_id(0)
        acc = tl.zeros([BLOCK_T], tl.int32)
        for t0 in tl.range(0, T, BLOCK_T):
            ts = t0 + tl.arange(0, BLOCK_T)
            m = tl.load(RMAP + ts * E + e, mask=ts < T, other=0).to(tl.int32)
            acc += m
        tl.store(COUNTS + e, tl.sum(acc, axis=0).to(tl.int64))

    @triton.jit
    def _permute_index_kernel(
        RMAP,  # [T, E] int8
        RPROBS,  # [T, E] fp32
        COUNTS,  # [E] int64
        SORTED_IDX,  # [T*k] int64   out
        PERM_PROBS,  # [T*k] fp32    out
        T,
        E,
        BLOCK_T: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """Counting sort by expert -- the stable ``argsort(descending=True)`` of the boolean map.

        ``stable=True`` makes the True entries of ``routing_map.T.reshape(-1)`` come out in
        ascending flat index ``e*T + t``, so slot ``base[e] + rank`` holds token ``t``, where
        ``base`` is the exclusive prefix sum of the per-expert counts and ``rank`` is ``t``'s
        position among expert ``e``'s tokens in ascending ``t``. Pure integer indexing.

        ``permuted_probs[i] = probs.T.reshape(-1)[e*T + t] = routing_probs[t, e]`` -- a gather.
        """
        e = tl.program_id(0)
        ecols = tl.arange(0, BLOCK_E)
        c = tl.load(COUNTS + ecols, mask=ecols < E, other=0)
        # exclusive prefix over experts. BLOCK_E is 256 at production -- a masked sum, not a scan,
        # is both simpler and cheaper than paying for tl.cumsum here.
        base = tl.sum(tl.where(ecols < e, c, 0), axis=0)

        run = tl.zeros([], tl.int64) + 0
        for t0 in tl.range(0, T, BLOCK_T):
            ts = t0 + tl.arange(0, BLOCK_T)
            tmask = ts < T
            m = tl.load(RMAP + ts * E + e, mask=tmask, other=0).to(tl.int64)
            rank = tl.cumsum(m, axis=0) - m  # exclusive rank within this tile
            pos = base + run + rank
            hit = tmask & (m == 1)
            tl.store(SORTED_IDX + pos, ts.to(tl.int64), mask=hit)
            pr = tl.load(RPROBS + ts * E + e, mask=hit, other=0.0)
            tl.store(PERM_PROBS + pos, pr, mask=hit)
            run += tl.sum(m, axis=0)


# `K` is a constexpr driving `tl.static_range`, so the selection loop is FULLY UNROLLED and compile time
# is O(k). Past this bound torch.topk wins anyway, and a large-k compile is indistinguishable from a hang
# at engine init.
_MAX_K = 32
# A program keeps the whole row in registers; beyond this it spills. Same bound as moe_router_kernel.
_MAX_BLOCK_E = 8192
# Row offsets are int32 in Triton by default; past 2^31 elements that arithmetic wraps and the loads read
# arbitrary memory.
_MAX_ELEMS = 2**31


def _router_can_handle(logits: torch.Tensor, k: int) -> bool:
    """Host-side, shape-only envelope check. No device sync, CUDA-graph safe.

    A bare ``assert`` here would kill the Ray actor; a fallback to the eager path is merely slower
    and is bitwise-identical by construction, so fall back rather than raise.
    """
    if not HAVE_TRITON:
        return False
    if logits.dim() != 2 or not logits.is_cuda:
        return False
    if logits.dtype is not torch.float32:
        return False  # the reproduced softmax decomposition is the fp32 one
    T, E = logits.shape
    if T == 0 or E == 0 or not (0 < k <= E):
        return False
    if k > _MAX_K:
        return False
    # THE REDUCTION-ORDER CONSTRAINT. tl.sum over BLOCK_K == k is torch.sum's butterfly tree only
    # when k is a power of two; otherwise the zero padding re-associates the tree and the
    # denominator differs in the last ulp. Refuse rather than be almost right.
    if k & (k - 1) != 0:
        return False
    if _next_pow2(E) > _MAX_BLOCK_E:
        return False
    if T * E >= _MAX_ELEMS:
        return False
    return True


@torch.no_grad()
def fused_router_dense(
    logits: torch.Tensor,
    k: int,
    *,
    scaling_factor: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return fp32 routing probabilities and bool map [T,E], using the batch-invariant softmax recipe.
    Unsupported inputs use the eager reference; tie membership and paired probabilities match bitwise."""
    if not _router_can_handle(logits, k):
        return ref_router_dense(logits, k, scaling_factor=scaling_factor)
    T, E = logits.shape
    logits = logits.contiguous()
    rprobs = torch.empty(T, E, dtype=torch.float32, device=logits.device)
    rmap8 = torch.empty(T, E, dtype=torch.int8, device=logits.device)
    _router_kernel[(T,)](
        logits,
        rprobs,
        rmap8,
        rprobs,  # TPRB unused (EMIT_TOPK=False); a real pointer, never dereferenced
        rmap8,  # TIDX  unused (EMIT_TOPK=False)
        T,
        E,
        logits.stride(0),
        float(scaling_factor) if scaling_factor else 1.0,
        K=k,
        BLOCK_E=_next_pow2(E),
        BLOCK_K=k,
        HAS_SCALE=bool(scaling_factor),
        EMIT_DENSE=True,
        EMIT_TOPK=False,
        num_warps=_ROUTER_WARPS,
        enable_fp_fusion=False,  # see the module docstring; asserted non-vacuous by the harness
    )
    return rprobs, rmap8.view(torch.bool)


@torch.no_grad()
def fused_router_topk(
    logits: torch.Tensor,
    k: int,
    *,
    scaling_factor: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return fp32 probabilities and int32 expert indices [T,k] from the shared router arithmetic.
    Membership and pairings match eager; tied columns use deterministic ascending expert order,
    which consumers must treat as a paired permutation. Unsupported inputs use ref_router_topk."""
    if not _router_can_handle(logits, k):
        return ref_router_topk(logits, k, scaling_factor=scaling_factor)
    T, E = logits.shape
    logits = logits.contiguous()
    probs = torch.empty(T, k, dtype=torch.float32, device=logits.device)
    idx = torch.empty(T, k, dtype=torch.int32, device=logits.device)
    _router_kernel[(T,)](
        logits,
        probs,  # RPROBS unused (EMIT_DENSE=False); a real pointer, never dereferenced
        idx,  # RMAP   unused (EMIT_DENSE=False)
        probs,
        idx,
        T,
        E,
        logits.stride(0),
        float(scaling_factor) if scaling_factor else 1.0,
        K=k,
        BLOCK_E=_next_pow2(E),
        BLOCK_K=k,
        HAS_SCALE=bool(scaling_factor),
        EMIT_DENSE=False,
        EMIT_TOPK=True,
        num_warps=_ROUTER_WARPS,
        enable_fp_fusion=False,
    )
    return probs, idx


@torch.no_grad()
def fused_permute_index(
    routing_map: torch.Tensor,
    routing_probs: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(sorted_indices [T*k] int64, permuted_probs [T*k], tokens_per_expert [E] int64)``.

    Replaces the transposing copies + ``argsort(descending=True, stable=True)`` + ``remainder`` +
    gather + column-sum in ``moe_utils.permute`` / ``_dispatch_postprocess_fixed_shape`` with two
    kernels. Integer permutation and a gather -- no arithmetic, so nothing to round. Falls back to
    the eager index build outside the envelope, for the same reason the router does.
    """
    if not permute_can_handle(routing_map, topk):
        return ref_permute_index(routing_map, routing_probs, topk)
    T, E = routing_map.shape
    dev = routing_map.device
    rmap8 = routing_map.view(torch.int8).contiguous()
    counts = torch.empty(E, dtype=torch.int64, device=dev)
    _counts_kernel[(E,)](rmap8, counts, T, E, BLOCK_T=_SORT_BLOCK_T, num_warps=_SORT_WARPS, enable_fp_fusion=False)
    n_out = T * topk
    sorted_idx = torch.empty(n_out, dtype=torch.int64, device=dev)
    perm_probs = torch.empty(n_out, dtype=routing_probs.dtype, device=dev)
    _permute_index_kernel[(E,)](
        rmap8,
        routing_probs.contiguous(),
        counts,
        sorted_idx,
        perm_probs,
        T,
        E,
        BLOCK_T=_SORT_BLOCK_T,
        BLOCK_E=_next_pow2(E),
        num_warps=_SORT_WARPS,
        enable_fp_fusion=False,
    )
    return sorted_idx, perm_probs, counts


def permute_can_handle(routing_map: torch.Tensor, topk: int) -> bool:
    """Envelope for :func:`fused_permute_index`. Host-side, shape-only."""
    if not HAVE_TRITON:
        return False
    if routing_map.dim() != 2 or not routing_map.is_cuda or routing_map.dtype is not torch.bool:
        return False
    T, E = routing_map.shape
    if T == 0 or E == 0 or topk <= 0:
        return False
    if _next_pow2(E) > _MAX_BLOCK_E:
        return False
    if T * E >= _MAX_ELEMS:
        return False
    return True


# reference implementation -- the eager sequence, verbatim. The harness gates against THIS, and it is also
# what the installed wrapper falls back to when the envelope check fails.
def ref_router_dense(logits, k, *, scaling_factor=None):
    """megatron's eager path, transcribed. Used by the gate and as the documented fallback."""
    T = logits.shape[0]
    scores, top_indices = torch.topk(logits, k=k, dim=1, sorted=True)
    # vllm's softmax_batch_invariant, spelled out (the aten override is a python composition).
    m = torch.amax(scores, dim=-1, keepdim=True)
    ex = torch.exp(scores - m)
    probs = ex / torch.sum(ex, dim=-1, keepdim=True)
    if scaling_factor:
        probs = probs * scaling_factor
    probs = probs.type_as(logits)
    routing_probs = torch.zeros_like(logits)
    rows = torch.arange(T, device=logits.device).unsqueeze(1)
    routing_probs.index_put_((rows, top_indices), probs, accumulate=False)
    routing_map = torch.zeros_like(logits, dtype=logits.dtype)
    routing_map.index_put_((rows, top_indices), torch.ones_like(probs, dtype=routing_map.dtype), accumulate=False)
    return routing_probs, routing_map.bool()


def ref_router_topk(logits, k, *, scaling_factor=None):
    """The ``[T,k]`` eager sequence, transcribed -- the fallback and the gate's reference.

    ``megatron``'s selection half without the dense scatter, i.e. exactly what
    ``ops/moe/eager.route_mechanics_megatron`` computes, spelled out with vllm's
    ``softmax_batch_invariant`` decomposition (the aten override both runtimes install).
    """
    scores, top_indices = torch.topk(logits, k=k, dim=1, sorted=True)
    m = torch.amax(scores, dim=-1, keepdim=True)
    ex = torch.exp(scores - m)
    probs = ex / torch.sum(ex, dim=-1, keepdim=True)
    if scaling_factor:
        probs = probs * scaling_factor
    return probs.type_as(logits).contiguous(), top_indices.to(torch.int32)


def ref_permute_index(routing_map, routing_probs, topk):
    """megatron ``permute``'s index build, transcribed."""
    T = routing_map.shape[0]
    n_out = T * topk
    rt = routing_map.bool().T.contiguous()
    flat_sorted = rt.reshape(-1).argsort(descending=True, stable=True)[:n_out]
    sorted_indices = flat_sorted % T
    perm_probs = routing_probs.T.contiguous().reshape(-1)[flat_sorted]
    counts = routing_map.sum(dim=0).long()
    return sorted_indices, perm_probs, counts


# Engine-only routing binding; forward-only kernels must not reach trainer autograd.
# Paired permutations of tied k columns are invisible to consumers: expert prologues scatter
# distinct indices, EP places terms by global expert rank, and combine reads only the token count.
# Decline to the eager twin for unsupported routing, requested full scores, unsupported shapes or missing Triton.
# The served counter distinguishes actual kernel execution from total fallback.
_ROUTER_BIND_COUNTS = {"served": 0, "declined": 0, "reason": ""}


def engine_router_o2_counts() -> dict:
    return dict(_ROUTER_BIND_COUNTS)


def _bind_report() -> None:
    n = _ROUTER_BIND_COUNTS["served"] + _ROUTER_BIND_COUNTS["declined"]
    if n & (n - 1) == 0:  # powers of two only -- one line per decade, not per forward
        print(
            f"[ISOEXEC-MOE-ROUTER-O2] engine moe.route_mechanics served={_ROUTER_BIND_COUNTS['served']} "
            f"declined={_ROUTER_BIND_COUNTS['declined']} last_decline={_ROUTER_BIND_COUNTS['reason']!r}",
            flush=True,
        )


def engine_route_mechanics_o2(eager_fn):
    """``moe.route_mechanics`` for the engine table: ``fused_o2`` with ``eager_fn`` as the fallback.

    ``eager_fn`` is ``base["moe.route_mechanics"].fn`` -- the reference function itself, so a
    declined shape runs the thing the contract calls this impl bitwise-equal to.
    """
    from isoexec.ops.moe.carriers import RoutingResult

    def route_mechanics(logits, spec, want_scores_full: bool = True):
        why = None
        if want_scores_full:
            # `scores_full` is the full-row softmax, aux-only; this kernel computes the top-k row.
            # The engine passes aux=None and never asks, so this is unreachable there by construction.
            why = "want_scores_full"
        elif spec.routing != "softmax_renorm":
            why = f"routing {spec.routing!r}"
        elif not _router_can_handle(logits, int(spec.top_k)):
            why = f"outside the envelope: logits{tuple(logits.shape)} {logits.dtype} k={spec.top_k}"
        if why is not None:
            _ROUTER_BIND_COUNTS["declined"] += 1
            _ROUTER_BIND_COUNTS["reason"] = why
            _bind_report()
            return eager_fn(logits, spec, want_scores_full)
        probs, idx = fused_router_topk(logits, int(spec.top_k), scaling_factor=None)
        _ROUTER_BIND_COUNTS["served"] += 1
        _bind_report()
        return RoutingResult(indices=idx, probs=probs, scores_full=None)

    route_mechanics.__name__ = route_mechanics.__qualname__ = "engine_route_mechanics_o2"
    return route_mechanics
