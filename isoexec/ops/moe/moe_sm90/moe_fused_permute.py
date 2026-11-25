"""Fused FORWARD MoE permute for the TRAINER's alltoall dispatcher: index build + gather, no argsort.

Without Transformer Engine, megatron's ``permute`` takes its eager branch, which materializes transposed
``[E, T]`` copies and argsorts ``E*T`` keys to recover a permutation carrying only ``T*k`` entries.
Everything replaced here is integer bookkeeping plus a copy -- ``sorted_indices`` is an int64 permutation,
and both ``permuted_input`` and ``permuted_probs`` are gathers -- so the acceptance criterion is
``torch.equal``, not a tolerance. ``unpermute``'s forward and ``permute``'s backward are fp32 accumulations
and are owned elsewhere.

``moe_router_o2_kernel.fused_permute_index`` counting-sorts the same way but is unreachable from here: it
is ``@torch.no_grad()`` with no ``autograd.Function``, so on the trainer it would sever the MoE backward
while the forward-only gate stayed green; it is installed engine-only on ``MoEAllGatherTokenDispatcher``
while the trainer runs the alltoall dispatcher; and its guard requires ``ep_size == 1``. On the alltoall
path ``routing_map`` is the FULL ``[T, E]`` map and ``num_out_tokens = T * topk`` is exactly the number of
Trues, which this module asserts on device rather than assuming.

The gather's VJP stays ``moe_ordered_dispatch.ordered_dispatch_backward``, and the probs gather's VJP stays
ATen's ``index_select`` backward. When top-k does not divide ``num_out_tokens`` the ordered backward's
exact-top-k precondition cannot hold, so a grad-enabled call in that layout DECLINES to megatron rather
than guessing. No sync is added: every shape is a function of host-known ``T``, ``E`` and
``num_out_tokens``, there is no ``.item()``, ``.cpu()``, ``nonzero``, ``masked_select`` or data-dependent
output shape, and the exactness precondition uses ``torch._assert_async``, so the path stays CUDA-graph
capturable.

Rounding schedules recorded on this module's ops (moved verbatim from ``ImplSpec.rounding.documentary``, which
M6.4 deleted; the registry keeps only the pins a host must set):

``moe.route_mechanics:deterministic_sorted_topk``
    torch.topk(k, sorted=True) both sides, softmax over the top-k scores (vLLM's softmax_batch_invariant, the
    aten override both runtimes install), then megatron's stable-sort permute (expert-major, token-ascending)
    and the fixed-shape permuted-probs gather (bitwise-equal to the dispatcher's masked_select). Reference impl
    at all four sites. The counting-sort replacement for the argsort/remainder/gather index build
    (moe_router_o2_kernel.fused_permute_index) is likewise recorded here and SELECTED on the GLM recipes; it is
    reachable under ISOEXEC_MOE_ROUTER_O2 and, independently, under ISOEXEC_MOE_PERMUTE_SORT (which
    is what the GLM launchers export; the per-(op,site) neutrality proof is
    examples/isoexec/nightly/moe_routing_site_parity.py, covering engine_decode AND engine_prefill). The split
    from the router flag exists because that half never looks at a score function -- it counting-sorts a boolean
    map and gathers fp32 words -- while the O2 ROUTER half correctly declines a sigmoid+expert_bias router, so
    on GLM-4.7-Flash the coupled flag left a score-function-agnostic index build unreachable. Both predicates
    carry the same load-bearing EP=1 guard, which the proof is conditional on. The fused_topk router kernel
    (ISOEXEC_MOE_FUSED_ROUTER) is a BOTH-SIDES alternative that is NOT WIRED (default OFF, absent from the
    shipped recipe) and is bitwise-equal only ON MEMBERSHIP: it returns the raw [T,k] top-k, whose column order
    inside a tie differs from torch's.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


BANNER = "[ISOEXEC-MOE-FUSED-PERMUTE]"

# Row offsets are int32 in Triton by default; past 2^31 elements that arithmetic wraps and the loads read
# arbitrary memory.
_MAX_ELEMS = 2**31
# One program holds the whole [E] count vector to form its exclusive prefix; beyond this it spills.
_MAX_BLOCK_E = 8192
_SORT_BLOCK_T = 1024
_SORT_WARPS = 4

_COUNTS = {
    "served": 0,
    "tokens": 0,
    "routes": 0,
    "declined": 0,
    "reported": 0,
    "validated": 0,
    "rows_emitted": 0,
    "rows_consumed": 0,
}
_DECLINE_REASONS: dict = {}
_orig_permute = None


def validate_combine_rows(
    rows: Optional[torch.Tensor], sorted_indices: torch.Tensor, num_tokens: int
) -> Optional[torch.Tensor]:
    """Return producer-owned canonical combine rows, or ``None`` when they were not emitted.

    Validation reads tensor metadata only. Every structural property that could make the
    metadata unsafe is checked from tensor metadata, so consuming it introduces no device launch or
    synchronization.  If an emitting producer publishes a malformed tensor, raise: silently sorting
    would hide a broken optimization behind a correct fallback and make engagement unverifiable.
    """
    if rows is None:
        return None
    n = int(sorted_indices.numel())
    if num_tokens <= 0 or n <= 0 or n % num_tokens != 0:
        raise RuntimeError(f"{BANNER} preemitted rows attached to a non-topk layout")
    expected = (num_tokens, n // num_tokens)
    if (
        tuple(rows.shape) != expected
        or rows.dtype is not torch.int32
        or rows.device != sorted_indices.device
        or not rows.is_contiguous()
    ):
        raise RuntimeError(
            f"{BANNER} malformed preemitted rows: shape={tuple(rows.shape)} dtype={rows.dtype} "
            f"device={rows.device} contiguous={rows.is_contiguous()}, expected shape={expected} "
            f"dtype=torch.int32 device={sorted_indices.device} contiguous=True"
        )
    _COUNTS["rows_consumed"] += 1
    return rows


def _validate_budget() -> int:
    return 8


def _report() -> None:
    calls = _COUNTS["served"] + _COUNTS["declined"]
    if calls < 1 or (calls & (calls - 1)) != 0 or calls == _COUNTS["reported"]:
        return
    _COUNTS["reported"] = calls
    print(
        f"{BANNER} pid={os.getpid()} served={_COUNTS['served']} declined={_COUNTS['declined']} "
        f"tokens={_COUNTS['tokens']} routes={_COUNTS['routes']} validated={_COUNTS['validated']} "
        f"reasons={_DECLINE_REASONS}",
        flush=True,
    )


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


# The two index-build kernels. Both are pure integer bookkeeping: there is no float in either, so
# `enable_fp_fusion`, `num_warps` and the block sizes cannot move a bit. The algorithm is the counting sort
# in `moe_router_o2_kernel._permute_index_kernel`; this copy exists so the TRAINER-side module is
# self-contained and does not import an engine-marked private symbol.
if HAVE_TRITON:

    @triton.jit
    def _route_counts_kernel(RMAP, COUNTS, T, E, BLOCK_T: tl.constexpr):
        """``counts[e] = routing_map[:, e].sum()`` -- one program per expert.

        Integer sum: exact and order-free, so there is no reduction schedule to reproduce.
        """
        e = tl.program_id(0)
        acc = tl.zeros([BLOCK_T], tl.int32)
        for t0 in tl.range(0, T, BLOCK_T):
            ts = t0 + tl.arange(0, BLOCK_T)
            acc += tl.load(RMAP + ts * E + e, mask=ts < T, other=0).to(tl.int32)
        tl.store(COUNTS + e, tl.sum(acc, axis=0).to(tl.int64))

    @triton.jit
    def _route_compact_kernel(
        RMAP,  # [T, E] int8 (the bool routing map, viewed)
        COUNTS,  # [E] int64
        SORTED_IDX,  # [R] int64  out -- routed row -> token
        ROUTE_EXPERT,  # [R] int64  out -- routed row -> expert
        T,
        E,
        BLOCK_T: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """Stream-compaction of the routed ``(e, t)`` pairs into lexicographic ``(e, t)`` order.

        THE ORDERING ARGUMENT, which is the whole bitwise case. Megatron computes
        ``argsort(routing_map.T.reshape(-1), descending=True, stable=True)[:num_out_tokens]``. A
        *stable descending* argsort of a boolean array emits every True position first, in ascending
        flat index; the flat index of ``(e, t)`` in the transposed map is ``e*T + t``. When
        ``num_out_tokens`` equals the number of Trues -- asserted on device by the caller -- that
        slice IS "the routed pairs in lexicographic (e, t) order". Slot ``base[e] + rank`` therefore
        holds token ``t``, where ``base`` is the exclusive prefix sum of the per-expert counts and
        ``rank`` is ``t``'s position among expert ``e``'s tokens in ascending ``t``. No tie-breaking
        rule can differ, because a token cannot select the same expert twice and the keys are unique.

        One writer per output slot (the map ``(e, t) -> base[e] + rank`` is injective), so there are
        no atomics and the result does not depend on program scheduling.
        """
        e = tl.program_id(0)
        ecols = tl.arange(0, BLOCK_E)
        c = tl.load(COUNTS + ecols, mask=ecols < E, other=0)
        # Exclusive prefix over experts. BLOCK_E is 256 at production, so a masked sum is both
        # simpler and cheaper than paying for a scan.
        base = tl.sum(tl.where(ecols < e, c, 0), axis=0)
        expert_id = tl.full((BLOCK_T,), 0, tl.int64) + e

        run = tl.zeros([], tl.int64) + 0
        for t0 in tl.range(0, T, BLOCK_T):
            ts = t0 + tl.arange(0, BLOCK_T)
            tmask = ts < T
            m = tl.load(RMAP + ts * E + e, mask=tmask, other=0).to(tl.int64)
            rank = tl.cumsum(m, axis=0) - m  # exclusive rank within this tile
            pos = base + run + rank
            hit = tmask & (m == 1)
            tl.store(SORTED_IDX + pos, ts.to(tl.int64), mask=hit)
            tl.store(ROUTE_EXPERT + pos, expert_id, mask=hit)
            run += tl.sum(m, axis=0)

    # The same two kernels over the router's [T, K] top-k indices instead of the dense [T, E] bool
    # map (`route_to_permuted_rows`, the schema-3 prologue). Same integer bookkeeping, same
    # injective (e, t) -> base[e] + rank map, no atomics; the membership test `rmap[t, e]` becomes
    # `any_j(idx[t, j] == e)`, which is order-free in j, so the router's tie column order stays
    # unobservable exactly as the dense round trip made it. PRECONDITION, structural for both
    # producers (`torch.topk` and `_router_kernel` select without replacement): a token's K indices
    # are distinct, so every (t, e) pair has one writer and the routed row count is T*K.
    @triton.jit
    def _route_counts_topk_kernel(IDX, COUNTS, T, E, K: tl.constexpr, BLOCK_T: tl.constexpr):
        """``counts[e] = sum_t any_j(idx[t, j] == e)`` -- one program per expert."""
        e = tl.program_id(0)
        acc = tl.zeros([BLOCK_T], tl.int32)
        for t0 in tl.range(0, T, BLOCK_T):
            ts = t0 + tl.arange(0, BLOCK_T)
            tmask = ts < T
            hit = tl.zeros([BLOCK_T], tl.int32)
            for j in tl.static_range(K):
                v = tl.load(IDX + ts * K + j, mask=tmask, other=-1)
                hit = tl.where(v == e, 1, hit)
            acc += hit
        tl.store(COUNTS + e, tl.sum(acc, axis=0).to(tl.int64))

    @triton.jit
    def _route_compact_topk_kernel(
        IDX,  # [T, K] int32 -- the router's top-k expert ids
        PROBS,  # [T, K] fp32 -- the router's top-k probs, paired by column
        COUNTS,  # [E] int64
        SORTED_IDX,  # [R] int64  out -- routed row -> token
        ROUTE_EXPERT,  # [R] int64  out -- routed row -> expert
        PERM_PROBS,  # [R] fp32   out -- routed row -> prob (a SELECT of PROBS[t, j]; no arithmetic)
        T,
        E,
        K: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """`_route_compact_kernel` over [T, K] indices, emitting the permuted probs in the same launch.

        The prob is a `tl.where` select of the matching column's fp32 payload -- never `0.0 + p`,
        which would map a -0.0 payload to +0.0 -- so it is the bits `probs_dense.index_select` read.
        """
        e = tl.program_id(0)
        ecols = tl.arange(0, BLOCK_E)
        c = tl.load(COUNTS + ecols, mask=ecols < E, other=0)
        base = tl.sum(tl.where(ecols < e, c, 0), axis=0)
        expert_id = tl.full((BLOCK_T,), 0, tl.int64) + e

        run = tl.zeros([], tl.int64) + 0
        for t0 in tl.range(0, T, BLOCK_T):
            ts = t0 + tl.arange(0, BLOCK_T)
            tmask = ts < T
            m = tl.zeros([BLOCK_T], tl.int64)
            p = tl.zeros([BLOCK_T], tl.float32)
            for j in tl.static_range(K):
                v = tl.load(IDX + ts * K + j, mask=tmask, other=-1)
                hitj = v == e
                m = tl.where(hitj, 1, m)
                pj = tl.load(PROBS + ts * K + j, mask=tmask & hitj, other=0.0)
                p = tl.where(hitj, pj, p)
            rank = tl.cumsum(m, axis=0) - m  # exclusive rank within this tile
            pos = base + run + rank
            hit = tmask & (m == 1)
            tl.store(SORTED_IDX + pos, ts.to(tl.int64), mask=hit)
            tl.store(ROUTE_EXPERT + pos, expert_id, mask=hit)
            tl.store(PERM_PROBS + pos, p, mask=hit)
            run += tl.sum(m, axis=0)

    @triton.jit
    def _permute_gather_kernel(
        TOKENS,  # [T, hidden]
        SORTED_IDX,  # [R] int64
        ROUTE_EXPERT,  # [R] int64, routed row -> expert
        RMAP,  # [T, E] int8 -- or, with TOPK_IDX, the [T, K] int32 top-k indices
        OUT,  # [R, hidden]
        COMBINE_ROWS,  # [T, K] int32, token/canonical slot -> routed row
        hidden,
        E,
        K: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_E: tl.constexpr,
        EMIT_ROWS: tl.constexpr,
        TOPK_IDX: tl.constexpr = False,
    ):
        """``out[r, :] = tokens[sorted_indices[r], :]`` -- one program per (routed row, hidden tile).

        slime's ``_scatter_routes_forward_kernel`` shape, transposed to our route-major output: one
        program owns one destination row, so the map is injective and there are no atomics and no
        library call. A ``tl.load``/``tl.store`` pair at the SAME dtype moves bits; there is no
        conversion, no accumulator and no arithmetic operator anywhere in the body, which is why the
        acceptance for this kernel is ``torch.equal`` on a uint8 view and not a tolerance.
        """
        r = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < hidden
        t = tl.load(SORTED_IDX + r).to(tl.int64)
        tl.store(OUT + r * hidden + offs, tl.load(TOKENS + t * hidden + offs, mask=mask), mask=mask)

        # The routed rows are expert-major.  For route (t,e), its stable-argsort slot within token t
        # is therefore exactly the number of selected experts below e.  Emit the inverse directly
        # from information already resident in this producer.  Only hidden tile zero writes, and
        # every (t,slot) has exactly one writer because top-k experts are distinct.
        if EMIT_ROWS:
            e = tl.load(ROUTE_EXPERT + r).to(tl.int64)
            if TOPK_IDX:
                # the same count -- "selected experts below e" -- read off the K indices
                slot = tl.zeros([], tl.int32) + 0
                for j in tl.static_range(K):
                    v = tl.load(RMAP + t * K + j).to(tl.int64)
                    slot += (v < e).to(tl.int32)
            else:
                ecols = tl.arange(0, BLOCK_E)
                selected = tl.load(RMAP + t * E + ecols, mask=ecols < E, other=0).to(tl.int32)
                slot = tl.sum(tl.where(ecols < e, selected, 0), axis=0)
            tl.store(COMBINE_ROWS + t * K + slot, r.to(tl.int32), mask=tl.program_id(1) == 0)


# references -- the oracles the tests gate against and the CPU execution path
def megatron_permute_index_reference(
    routing_map: torch.Tensor, num_out_tokens: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(sorted_indices, route_expert)`` from megatron's literal eager expression.

    This is the definition, transcribed. ``flat_sorted`` holds ``e*T + t``, so the token is
    ``flat % T`` (megatron's own ``sorted_indices``) and the expert is ``flat // T``.
    """
    num_tokens = int(routing_map.shape[0])
    flat = routing_map.bool().T.contiguous().reshape(-1).argsort(descending=True, stable=True)[:num_out_tokens]
    return flat % num_tokens, flat // num_tokens


def compact_route_positions_reference(
    routing_map: torch.Tensor, num_out_tokens: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch transcription of :func:`_route_compact_kernel`. CPU path and algorithm oracle.

    Same counting sort, same exclusive prefixes, no sort anywhere -- so a CPU test of this function
    against :func:`megatron_permute_index_reference` proves the ALGORITHM, and the GPU test of the
    kernel against this function proves the KERNEL.
    """
    num_tokens, num_experts = routing_map.shape
    device = routing_map.device
    dense = routing_map.to(torch.int64)
    counts = dense.sum(dim=0)  # [E]
    base = torch.cumsum(counts, 0) - counts  # exclusive prefix over experts
    rank = torch.cumsum(dense, 0) - dense  # exclusive rank within each expert column
    pos = base.view(1, -1) + rank  # [T, E] destination slot

    sorted_indices = torch.zeros(num_out_tokens, dtype=torch.int64, device=device)
    route_expert = torch.zeros(num_out_tokens, dtype=torch.int64, device=device)
    if num_tokens == 0 or num_experts == 0 or num_out_tokens == 0:
        return sorted_indices, route_expert, counts
    sel = routing_map.bool()
    tok = torch.arange(num_tokens, device=device).view(-1, 1).expand(num_tokens, num_experts)
    exp = torch.arange(num_experts, device=device).view(1, -1).expand(num_tokens, num_experts)
    # Boolean masking here is a data-dependent shape; it is confined to the reference/CPU path and
    # is exactly what the kernel exists to avoid on the device path.
    dest = pos[sel]
    keep = dest < num_out_tokens
    sorted_indices[dest[keep]] = tok[sel][keep]
    route_expert[dest[keep]] = exp[sel][keep]
    return sorted_indices, route_expert, counts


def permute_rows_reference(tokens: torch.Tensor, sorted_indices: torch.Tensor) -> torch.Tensor:
    """The gather oracle. ``index_select`` IS the definition; the kernel must equal it bit for bit."""
    return tokens.index_select(0, sorted_indices)


def canonical_combine_rows_reference(
    routing_map: torch.Tensor, sorted_indices: torch.Tensor, route_expert: torch.Tensor
) -> torch.Tensor:
    """Literal producer-side inverse: ``rows[t,slot] = routed_row``.

    ``slot`` is the count of selected experts below the route's expert.  This is the same ordering
    a stable argsort of expert-major ``sorted_indices`` returns, without using a sort as the oracle.
    The loop is CPU/reference-only; the device path emits the same values inside the row gather.
    """
    num_tokens, num_experts = routing_map.shape
    n = int(sorted_indices.numel())
    if num_tokens <= 0 or n <= 0 or n % num_tokens != 0:
        raise ValueError("canonical combine rows require a non-empty exact-topk layout")
    k = n // num_tokens
    rows = torch.empty((num_tokens, k), dtype=torch.int32, device=sorted_indices.device)
    for r in range(n):
        t = int(sorted_indices[r])
        e = int(route_expert[r])
        if not (0 <= t < num_tokens and 0 <= e < num_experts):
            raise ValueError("route metadata is out of bounds")
        slot = int(routing_map[t, :e].sum())
        if not (0 <= slot < k):
            raise ValueError("route metadata does not describe exact top-k canonical slots")
        rows[t, slot] = r
    return rows


# device path
def _assert_exact_true_count(counts: torch.Tensor, num_out_tokens: int) -> None:
    """The one precondition of the compaction, checked WITHOUT a readback.

    megatron's ``[:num_out_tokens]`` slice equals "every True position in ascending flat index" only
    when the number of Trues is exactly ``num_out_tokens``. If it is fewer, megatron's slice spills
    into the False region and the compaction would leave that tail unwritten. ``_assert_async``
    raises on the device without a sync; the first-N-calls ``torch.equal`` validation catches the
    same condition deterministically at startup.
    """
    # `torch.eq` against a python scalar keeps this a pure device expression: no H2D copy, so the
    # check survives CUDA-graph capture, which an `as_tensor(...).to(device)` would not.
    torch._assert_async(
        torch.eq(counts.sum(), num_out_tokens),
        "[isoexec] fused MoE permute requires num_out_tokens == number of routed pairs "
        "(no token dropping, no capacity padding, full routing map)",
    )


def compact_route_positions(
    routing_map: torch.Tensor, num_out_tokens: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(sorted_indices [R] int64, route_expert [R] int64, tokens_per_expert [E] int64)``.

    Replaces megatron's transposing bool copy + ``E*T`` stable argsort + slice + remainder. Bitwise
    equal to :func:`megatron_permute_index_reference` whenever the device assertion above holds.
    """
    num_tokens, num_experts = routing_map.shape
    if not routing_map.is_cuda or not HAVE_TRITON:
        sorted_indices, route_expert, counts = compact_route_positions_reference(routing_map, num_out_tokens)
        return sorted_indices, route_expert, counts

    device = routing_map.device
    rmap8 = routing_map.contiguous().view(torch.int8)
    counts = torch.empty(num_experts, dtype=torch.int64, device=device)
    _route_counts_kernel[(num_experts,)](
        rmap8, counts, num_tokens, num_experts, BLOCK_T=_SORT_BLOCK_T, num_warps=_SORT_WARPS
    )
    _assert_exact_true_count(counts, num_out_tokens)
    sorted_indices = torch.empty(num_out_tokens, dtype=torch.int64, device=device)
    route_expert = torch.empty(num_out_tokens, dtype=torch.int64, device=device)
    _route_compact_kernel[(num_experts,)](
        rmap8,
        counts,
        sorted_indices,
        route_expert,
        num_tokens,
        num_experts,
        BLOCK_T=_SORT_BLOCK_T,
        BLOCK_E=_next_pow2(num_experts),
        num_warps=_SORT_WARPS,
    )
    return sorted_indices, route_expert, counts


def permute_rows(
    tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    *,
    routing_map: Optional[torch.Tensor] = None,
    route_expert: Optional[torch.Tensor] = None,
    emit_combine_rows: bool = False,
    topk_indices: Optional[torch.Tensor] = None,
    num_experts: Optional[int] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Byte-identical row gather, optionally emitting the canonical inverse in the same launch.

    The inverse needs each route's slot -- the number of the token's selected experts below its
    expert -- which is read from the dense ``routing_map`` [T, E] or, when ``topk_indices`` [T, K]
    is given instead (with ``num_experts``), from the K indices. Same value either way."""
    rows = None
    from_topk = topk_indices is not None
    if emit_combine_rows and route_expert is None or (emit_combine_rows and routing_map is None and not from_topk):
        raise RuntimeError(f"{BANNER} combine-row emission requires (routing_map | topk_indices) and route_expert")
    if from_topk and num_experts is None:
        raise RuntimeError(f"{BANNER} topk_indices needs num_experts")
    if not tokens.is_cuda or not HAVE_TRITON:
        out = permute_rows_reference(tokens, sorted_indices)
        if emit_combine_rows:
            rmap = routing_map if routing_map is not None else _dense_map_from_topk(topk_indices, int(num_experts))
            rows = canonical_combine_rows_reference(rmap, sorted_indices, route_expert)
            _COUNTS["rows_emitted"] += 1
        return out, rows if emit_combine_rows else None
    src = tokens.contiguous()
    n_routes = int(sorted_indices.numel())
    hidden = int(src.shape[1])
    out = torch.empty((n_routes, hidden), dtype=src.dtype, device=src.device)
    if n_routes == 0 or hidden == 0:
        return out, rows if emit_combine_rows else None
    T = int(topk_indices.shape[0]) if from_topk else (int(routing_map.shape[0]) if emit_combine_rows else 0)
    E = int(num_experts) if from_topk else (int(routing_map.shape[1]) if emit_combine_rows else 1)
    k = n_routes // T if emit_combine_rows else 1
    rows = torch.empty((T, k), dtype=torch.int32, device=src.device) if emit_combine_rows else sorted_indices
    if emit_combine_rows:
        rmap8 = topk_indices.contiguous() if from_topk else routing_map.contiguous().view(torch.int8)
    else:
        rmap8 = sorted_indices
    experts = route_expert if emit_combine_rows else sorted_indices
    block_h = min(1024, _next_pow2(hidden))
    _permute_gather_kernel[(n_routes, triton.cdiv(hidden, block_h))](
        src,
        sorted_indices,
        experts,
        rmap8,
        out,
        rows,
        hidden,
        E,
        K=k,
        BLOCK_H=block_h,
        BLOCK_E=_next_pow2(E),
        EMIT_ROWS=emit_combine_rows,
        TOPK_IDX=from_topk and emit_combine_rows,
        num_warps=4,
    )
    if emit_combine_rows:
        _COUNTS["rows_emitted"] += 1
    return out, rows if emit_combine_rows else None


class _FusedPermuteGather(torch.autograd.Function):
    """Forward = the ported gather. Backward = the already-landed atomics-free ordered VJP."""

    @staticmethod
    def forward(
        ctx,
        tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        routing_map: torch.Tensor,
        route_expert: torch.Tensor,
        emit_combine_rows: bool,
    ):
        ctx.save_for_backward(sorted_indices)
        ctx.set_materialize_grads(False)
        ctx.n_tokens = int(tokens.shape[0])
        with torch.no_grad():
            out, rows = permute_rows(
                tokens,
                sorted_indices,
                routing_map=routing_map,
                route_expert=route_expert,
                emit_combine_rows=emit_combine_rows,
            )

        if rows is not None:
            ctx.mark_non_differentiable(rows)
        return out, rows

    @staticmethod
    def backward(ctx, grad_routes: torch.Tensor, _grad_rows):
        from .moe_ordered_dispatch import ordered_dispatch_backward

        (sorted_indices,) = ctx.saved_tensors
        return ordered_dispatch_backward(grad_routes, sorted_indices, ctx.n_tokens), None, None, None, None


def _validate_against_megatron(
    tokens: torch.Tensor,
    routing_map: torch.Tensor,
    probs: Optional[torch.Tensor],
    num_out_tokens: int,
    sorted_indices: torch.Tensor,
    route_expert: torch.Tensor,
    permuted_probs: Optional[torch.Tensor],
    permuted_input: torch.Tensor,
    rows: Optional[torch.Tensor] = None,
) -> None:
    """Live byte-identity gate on the first N calls, against megatron's literal expression.

    Not a smoke test: it runs on the REAL operands, in the real process, before the arm has produced
    a step. Anything short of ``torch.equal`` raises rather than degrades -- a permute that is
    almost right is a different model, not a slower one.
    """
    with torch.no_grad():
        flat = routing_map.bool().T.contiguous().reshape(-1).argsort(descending=True, stable=True)[:num_out_tokens]
        ref_idx = flat % int(tokens.shape[0])
        if not torch.equal(sorted_indices, ref_idx):
            raise RuntimeError(f"{BANNER} sorted_indices differ from megatron's argsort permutation")
        ref_rows = tokens.detach().index_select(0, ref_idx)
        if not torch.equal(permuted_input.detach().view(torch.uint8), ref_rows.view(torch.uint8)):
            raise RuntimeError(f"{BANNER} permuted rows are not byte-identical to index_select")
        if probs is not None:
            ref_probs = probs.detach().T.contiguous().reshape(-1)[flat]
            if not torch.equal(permuted_probs.detach().view(torch.uint8), ref_probs.view(torch.uint8)):
                raise RuntimeError(f"{BANNER} permuted probs are not byte-identical to megatron's gather")
        if rows is not None:
            k = int(rows.shape[1])
            flat_rows = rows.view(-1).long()
            expected_tokens = torch.arange(int(tokens.shape[0]), device=tokens.device).repeat_interleave(k)
            if not torch.equal(sorted_indices[flat_rows], expected_tokens):
                raise RuntimeError(f"{BANNER} preemitted rows do not invert sorted_indices")
            canonical_experts = route_expert[flat_rows].view(int(tokens.shape[0]), k)
            if k > 1 and not torch.equal(
                canonical_experts[:, 1:] > canonical_experts[:, :-1],
                torch.ones_like(canonical_experts[:, 1:], dtype=torch.bool),
            ):
                raise RuntimeError(f"{BANNER} preemitted rows are not in ascending-expert order")
    _COUNTS["validated"] += 1


def _fallback(tokens, routing_map, probs, num_out_tokens, fused, drop_and_pad, tokens_per_expert, align_size):
    # `_orig_permute` used to be captured by `install_fused_permute`, which rebound megatron's
    # `moe_utils.permute` / `token_dispatcher.permute` onto `_fused_permute`. That installer was
    # bridge-era and is deleted (kernel-ownership.md §5.2), so nothing captures a binding any more
    # and a decline has nowhere to fall back TO. Fail loudly rather than call `None`: a decline path
    # that raises TypeError deep inside a step is the defect `0f30d9a` was written about.
    if _orig_permute is None:
        raise RuntimeError(
            f"{BANNER} declined and has no fallback: `_orig_permute` was captured by "
            "`install_fused_permute`, which was deleted as bridge-era dead code. `_fused_permute` is "
            "megatron's `permute` drop-in and is no longer bound to anything -- the live unified path "
            "is `route_to_permuted_rows` -> `compact_route_positions` + `permute_rows`. Bind this "
            "through an `OpBinding` (and set the fallback explicitly) before calling it in anger."
        )
    return _orig_permute(
        tokens,
        routing_map,
        probs=probs,
        num_out_tokens=num_out_tokens,
        fused=fused,
        drop_and_pad=drop_and_pad,
        tokens_per_expert=tokens_per_expert,
        align_size=align_size,
    )


def _decline_reason(tokens, routing_map, probs, num_out_tokens, fused, drop_and_pad) -> Optional[str]:
    """Host-side, shape-only admission check. No device sync, CUDA-graph safe."""
    if fused:
        return "fused"
    if drop_and_pad:
        return "drop_and_pad"
    if num_out_tokens is None:
        return "num_out_tokens=None"
    if routing_map.dim() != 2 or routing_map.dtype is not torch.bool:
        return "routing_map_not_2d_bool"
    if tokens.dim() != 2:
        return "tokens_not_2d"
    num_tokens, num_experts = routing_map.shape
    if num_tokens == 0 or num_experts == 0 or num_out_tokens == 0:
        return "empty"
    if int(tokens.shape[0]) != num_tokens:
        return "token_count_mismatch"
    if _next_pow2(num_experts) > _MAX_BLOCK_E:
        return "num_experts_too_large"
    if num_tokens * num_experts >= _MAX_ELEMS or num_out_tokens * int(tokens.shape[1]) >= _MAX_ELEMS:
        return "int32_offset_overflow"
    if probs is not None and tuple(probs.shape) != (num_tokens, num_experts):
        return "probs_shape_mismatch"
    if torch.is_grad_enabled() and num_out_tokens % num_tokens != 0:
        # The ordered backward needs [T, topk] route rows; a layout without an exact top-k has no
        # such VJP, and the pre-emitted combine rows it would publish would be malformed.
        return "grad_without_exact_topk"
    return None


def _fused_permute(
    tokens: torch.Tensor,
    routing_map: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    num_out_tokens: Optional[int] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
    tokens_per_expert: Optional[torch.Tensor] = None,
    align_size: int = 0,
):
    """Drop-in ``moe_utils.permute``: megatron's contract, its eager branch's bits, three launches."""
    if _decline_reason(tokens, routing_map, probs, num_out_tokens, fused, drop_and_pad) is not None:
        return _fallback(tokens, routing_map, probs, num_out_tokens, fused, drop_and_pad, tokens_per_expert, align_size)

    num_tokens, num_experts = routing_map.shape
    sorted_indices, route_expert, _counts = compact_route_positions(routing_map, int(num_out_tokens))

    permuted_probs = None
    if probs is not None:
        # `probs.T.contiguous().reshape(-1)[flat_sorted]` gathers `probs[t, e]`. Reaching the same
        # words through `probs.reshape(-1)[t*E + e]` deletes the [E,T] fp32 transposing copy and
        # keeps ATen's index_select VJP, which is the correct gradient for the router.
        flat_probs_index = sorted_indices * num_experts + route_expert
        permuted_probs = probs.reshape(-1).index_select(0, flat_probs_index)

    emit_combine_rows = True
    permuted_input, rows = (
        _FusedPermuteGather.apply(tokens, sorted_indices, routing_map, route_expert, emit_combine_rows)
        if torch.is_grad_enabled()
        else permute_rows(
            tokens,
            sorted_indices,
            routing_map=routing_map,
            route_expert=route_expert,
            emit_combine_rows=emit_combine_rows,
        )
    )

    if _COUNTS["validated"] < _validate_budget():
        _validate_against_megatron(
            tokens,
            routing_map,
            probs,
            int(num_out_tokens),
            sorted_indices,
            route_expert,
            permuted_probs,
            permuted_input,
            rows,
        )

    _COUNTS["served"] += 1
    _COUNTS["tokens"] += int(num_tokens)
    _COUNTS["routes"] += int(num_out_tokens)
    _report()
    return permuted_input, permuted_probs, sorted_indices, None, tokens_per_expert


# ======================================================================================================
# the schema-3 `moe.expert_mlp` PROLOGUE -- ONE transcription, shared by both impls of the region
# ======================================================================================================


def _dense_map_from_topk(idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    """The dense bool map the [T, K] indices describe (reference/CPU helper)."""
    T = int(idx.shape[0])
    return torch.zeros(T, num_experts, dtype=torch.bool, device=idx.device).scatter_(1, idx.long(), True)


def compact_route_positions_topk_reference(
    idx: torch.Tensor, probs: torch.Tensor, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch transcription of the [T, K] kernels: ``(sorted_indices, route_expert, counts,
    permuted_probs)``. Goes through the dense map on purpose -- that is the definition the kernels
    must equal, and the CPU test proves the two references agree before the GPU test proves the
    kernels against this one."""
    T, k = idx.shape
    E = int(num_experts)
    rmap = _dense_map_from_topk(idx, E)
    sorted_indices, route_expert, counts = compact_route_positions_reference(rmap, T * k)
    probs_dense = torch.zeros(T, E, dtype=probs.dtype, device=probs.device).scatter_(1, idx.long(), probs)
    permuted_probs = probs_dense.reshape(-1).index_select(0, sorted_indices * E + route_expert)
    return sorted_indices, route_expert, counts, permuted_probs


def compact_route_positions_topk(
    idx: torch.Tensor, probs: torch.Tensor, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(sorted_indices [R] int64, route_expert [R] int64, counts [E] int64, permuted_probs [R] fp32)``
    from the router's ``[T, K]`` output directly: two launches, no ``[T, E]`` tensor anywhere."""
    T, k = idx.shape
    E = int(num_experts)
    R = T * k
    if not idx.is_cuda or not HAVE_TRITON:
        return compact_route_positions_topk_reference(idx, probs, E)
    idx32 = idx.contiguous() if idx.dtype == torch.int32 else idx.contiguous().to(torch.int32)
    probs32 = probs.contiguous() if probs.dtype == torch.float32 else probs.contiguous().to(torch.float32)
    device = idx.device
    counts = torch.empty(E, dtype=torch.int64, device=device)
    _route_counts_topk_kernel[(E,)](idx32, counts, T, E, K=k, BLOCK_T=_SORT_BLOCK_T, num_warps=_SORT_WARPS)
    sorted_indices = torch.empty(R, dtype=torch.int64, device=device)
    route_expert = torch.empty(R, dtype=torch.int64, device=device)
    permuted_probs = torch.empty(R, dtype=torch.float32, device=device)
    _route_compact_topk_kernel[(E,)](
        idx32,
        probs32,
        counts,
        sorted_indices,
        route_expert,
        permuted_probs,
        T,
        E,
        K=k,
        BLOCK_T=_SORT_BLOCK_T,
        BLOCK_E=_next_pow2(E),
        num_warps=_SORT_WARPS,
    )
    return sorted_indices, route_expert, counts, permuted_probs


# The [T,K] prologue is validated against the dense one bitwise on its first calls: a per-process
# budget plus the first call at every new (k, E) class. Not per T -- a compare is a host sync, and
# the prologue's kernels have no per-T specialisation to check.
_TOPK_VALIDATED: set = set()
_TOPK_COUNTS = {"served": 0, "dense": 0, "validated": 0}


def topk_prologue_counts() -> dict:
    return dict(_TOPK_COUNTS)


def _topk_prologue_eligible(x: torch.Tensor, routing, E: int) -> bool:
    idx, probs = routing.indices, routing.probs
    if not (x.is_cuda and HAVE_TRITON and idx.is_cuda and idx.dim() == 2 and probs.shape == idx.shape):
        return False
    # Under grad the dense path's scatter_/index_select carry the probs' autograd edge to the router;
    # the kernels do not. The engine runs without grad; the trainer keeps the reference path.
    if torch.is_grad_enabled() and (x.requires_grad or probs.requires_grad):
        return False
    T, k = idx.shape
    if T == 0 or k == 0 or _next_pow2(E) > _MAX_BLOCK_E or T * k * int(x.shape[1]) >= _MAX_ELEMS:
        return False
    return True


def _route_to_permuted_rows_dense(x: torch.Tensor, routing, E: int):
    """The dense round trip: the definition, and the reference the [T, K] path is validated against."""
    T, k = routing.indices.shape
    dev = x.device
    idx_long = routing.indices.long()
    routing_map = torch.zeros(T, E, dtype=torch.bool, device=dev).scatter_(1, idx_long, True)
    probs_dense = torch.zeros(T, E, dtype=torch.float32, device=dev).scatter_(1, idx_long, routing.probs)
    R = T * k
    sorted_indices, route_expert, _counts = compact_route_positions(routing_map, R)
    # Preserve the gather's gradient edge when training through the dense route.
    if torch.is_grad_enabled() and x.requires_grad:
        permuted, rows = _FusedPermuteGather.apply(x, sorted_indices, routing_map, route_expert, True)
    else:
        permuted, rows = permute_rows(
            x, sorted_indices, routing_map=routing_map, route_expert=route_expert, emit_combine_rows=True
        )
    permuted_probs = probs_dense.reshape(-1).index_select(0, sorted_indices * E + route_expert)
    counts = routing_map.sum(dim=0).long()
    return permuted, permuted_probs, counts, sorted_indices, rows


def _validate_topk_against_dense(x, routing, E, got) -> None:
    ref = _route_to_permuted_rows_dense(x, routing, E)
    names = ("permuted", "permuted_probs", "counts", "sorted_indices", "combine_rows")
    for name, a, b in zip(names, got, ref):
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
            raise RuntimeError(f"{BANNER} [T,K] prologue: {name} differs from the dense round trip")
    _TOPK_COUNTS["validated"] += 1


def route_to_permuted_rows(x: torch.Tensor, routing, num_experts: int):
    """Return permuted [R,H], permuted_probs [R] fp32, counts [E] int64, sorted_indices [R] int64,
    and combine_rows [T,k] int32.

    The dispatch half of ``moe.expert_mlp``: re-densify the ``RoutingResult``, build the
    expert-major/token-ascending index, gather the rows and the probs, count per expert. Pure
    integer bookkeeping and two gathers -- the region declares ``index_build_bit_free`` for exactly
    this, and ``torch.equal`` (not a tolerance) is the acceptance criterion for every kernel under
    it.

    Both hosts call this shared prologue through ``integration.expert_mlp``; its
    integer routing and gathers are independent of the grouped GEMM implementation.

    THE DENSE ROUND TRIP IS WHAT MAKES THE ROUTER'S TIE COLUMN ORDER UNOBSERVABLE. ``scatter_`` with
    the top-k's DISTINCT column indices is a bijection onto the written positions, so the two dense
    tensors -- and therefore everything downstream -- are independent of the order the k columns are
    visited in. That is the discharge ``moe.route_mechanics:fused_o2`` needs to emit ``[T,k]``.

    The device path without grad (the engine) takes the [T, K] kernels instead
    (``compact_route_positions_topk`` + ``permute_rows(topk_indices=...)``), whose membership test is
    ``any_j(idx[t, j] == e)`` -- the same order-free property without materialising the map -- and
    whose outputs are validated bitwise against this round trip on their first calls. The round trip
    stays the definition, the CPU path, and the trainer's path (its ``scatter_`` / ``index_select``
    carry the probs' autograd edge to the router; the kernels do not).
    """
    E = int(num_experts)
    if not _topk_prologue_eligible(x, routing, E):
        _TOPK_COUNTS["dense"] += 1
        return _route_to_permuted_rows_dense(x, routing, E)
    # THE [T, K] FORM, no dense tensor: the engine's kernels read the router's own top-k output.
    # The dense round trip was 11 launches per MoE layer (two zeros, two scatter_, an int mul and
    # add, a .long(), a column sum, an index_select and the count assertion's eq + sum + assert);
    # this is three, and its outputs are validated bitwise against the round trip on the first calls
    # of every (k, E) class. Order-free in the K columns by construction (see the kernels), so the
    # property the round trip existed for -- the router's tie column order is unobservable -- holds.
    idx = routing.indices
    T, k = idx.shape
    sorted_indices, route_expert, counts, permuted_probs = compact_route_positions_topk(idx, routing.probs, E)
    permuted, rows = permute_rows(
        x, sorted_indices, route_expert=route_expert, emit_combine_rows=True, topk_indices=idx, num_experts=E
    )
    got = (permuted, permuted_probs, counts, sorted_indices, rows)
    key = (int(k), E)
    if (key not in _TOPK_VALIDATED or _TOPK_COUNTS["validated"] < _validate_budget()) and not (
        x.is_cuda and torch.cuda.is_current_stream_capturing()
    ):
        _validate_topk_against_dense(x, routing, E, got)
        _TOPK_VALIDATED.add(key)
    _TOPK_COUNTS["served"] += 1
    return got
