"""Forward of the deterministic absorbed-MLA op on SM100 (Blackwell)."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os

import torch

from isoexec.ops.attention.mla_sm100._extension import extension, profile_digest
from isoexec.ops.attention.mla_sm100.spec import PAGE

__all__ = [
    "MlaSm100Plan", "mla_sm100_fwd", "mla_sm100_plan", "mla_sm100_fwd_selfcheck",
    "SEG_TABLE", "EXEC_OVERRIDE_VARIANTS", "seg_table_digest", "seg_tokens_for", "profile_digest",
]

HEAD_DIM_K = 576
HEAD_DIM_V = 512
M_BLOCK = 64
PARTIAL_WIDTH = HEAD_DIM_V + 8  # O[512], L, M, 6 pad (32-byte aligned ROWS; see params.h)
#: The two recipes this leaf implements (spec.py; README.md "Round 5").  A recipe is chosen by the
#: impl id a deployment resolves, never per call by shape or phase: inside one id the program is one
#: program for every row (the logprob_contract relation), so v2 runs its lazy rescale on decode rows
#: too.  v1 is the default.
RECIPE_IDS = {1: "sm100_absorbed_segtree", 2: "sm100_absorbed_segtree_v2"}
ITEM_OVERHEAD_BLOCKS = 5  # an item's Q copy, TMEM/pipeline fill and publish, in KV blocks
FOLD_OVERHEAD_BLOCKS = 2  # an in-place segment boundary: drain the pipeline, read O out of TMEM,
                          # fold and push a 64x520 node through global memory
GRANULARITY = {"auto": 0, "single": 2, "whole": 3, "pairs": 4, "mixed": 5}
_PLAN_SMEM_INTS = 160 * 1024 // 4
_SNAKE_ITEMS_PER_WORKER = 2
_MAX_WORKER_WAVES = 4

# Execution tuning may change work partitioning, never the reduction tree or output bits.
EXEC_OVERRIDE_VARIANTS = [
    {"granularity": "single"},
    {"granularity": "whole"},
    {"granularity": "pairs"},
    {"granularity": "mixed"},
    {"granularity": "single", "num_parts": 1000},
    {"granularity": "pairs", "reverse": True},
    {"granularity": "mixed", "num_parts": 3},
    {"num_parts": 1},
    {"num_parts": 2},
    {"num_parts": 7},
    {"num_parts": 64},
    {"num_parts": 1000},
    {"reverse": True},
    {"reverse": True, "num_parts": 3},
    {"item_overhead": 0, "fold_overhead": 0},
    {"item_overhead": 100000},
    {"combine_single_group": True},
    {"combine_config": 1},
    {"combine_config": 3},
    {"no_pdl": True},
]
MAX_TREE_LEAVES = 1024
EMPTY_LSE = float(torch.tensor(-1e30, dtype=torch.float32) / torch.tensor(1.4426950408889634, dtype=torch.float32))

#: SEG, the deployment-fixed segment size, over (h_q per rank, max context).  A multiple of the
#: 64-token KV tile.  Blackwell's SEG is much smaller than sm90's 2048 because one Blackwell CTA
#: owns the whole 512-wide fp32 accumulator in TMEM and therefore cannot share an SM: the number of
#: segments IS the number of SMs the op can use, so at decode the split is the only parallelism
#: lever.  See README.md "SEG table" for the regret argument and the measurements.
SEG_TABLE = {
    (5, 8192): 512, (5, 10240): 512, (5, 32768): 512, (5, 131072): 1024, (5, 202752): 1024,
    (10, 8192): 512, (10, 10240): 512, (10, 32768): 512, (10, 131072): 1024, (10, 202752): 1024,
    (20, 8192): 512, (20, 10240): 512, (20, 32768): 512, (20, 131072): 1024, (20, 202752): 1024,
}


def seg_tokens_for(h_q: int, max_seqlen_k: int) -> int:
    """SEG for a deployment: the smallest max-context class that covers max_seqlen_k."""
    classes = sorted({c for (h, c) in SEG_TABLE if h == int(h_q)})
    if not classes:
        raise KeyError(f"no SEG table entry for h_q={h_q}")
    cls = next((c for c in classes if c >= int(max_seqlen_k)), classes[-1])
    return SEG_TABLE[(int(h_q), cls)]


def seg_table_digest() -> str:
    """Digest of the SEG table, compared at bind time."""
    return hashlib.sha256(repr(sorted(SEG_TABLE.items())).encode()).hexdigest()[:16]


def _num_sms() -> int:
    return int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)


@dataclasses.dataclass
class MlaSm100Plan:
    """Per-step plan: one work-item list per leaf worker, plus the combine rows of split trees."""

    items: torch.Tensor          # int32 [item_cap, 8]
    heads: torch.Tensor          # int32 [workers]
    combine_rows: torch.Tensor   # int32 [combine_cap, 4]
    counts: torch.Tensor         # int32 [8]
    scratch: torch.Tensor        # int32 [9*mb_cap + 3*workers]
    h_q: int
    seg_tokens: int
    num_seqs: int
    max_seqlen_q: int
    slot_cap: int
    max_n: int
    workers: int
    exact: bool
    combine_config: int | None = None
    debug_flags: int = 0
    may_empty: bool = False

    @property
    def item_cap(self) -> int:
        return self.items.shape[0]

    @property
    def combine_cap(self) -> int:
        return self.combine_rows.shape[0]

    def check(self) -> None:
        c = self.counts.tolist()
        if c[3]:
            raise RuntimeError(f"mla_sm100 plan overflow: counts={c}, item_cap={self.item_cap}, "
                               f"slot_cap={self.slot_cap}, combine_cap={self.combine_cap}")


def _exact_sizes(cu_seqlens_q, seqused_k, seg, h_q):
    """Worst-case (one item per segment) item, slot and combine-row counts from host lengths."""
    import numpy as np

    both = torch.cat((cu_seqlens_q.reshape(-1).to(torch.int64),
                      seqused_k.reshape(-1).to(torch.int64))).cpu().numpy()
    b = int(seqused_k.shape[0])
    cu, s = both[:b + 1], both[b + 1:]
    q_len = cu[1:] - cu[:-1]
    rows = q_len * h_q
    may_empty = bool((np.minimum(np.maximum(q_len - s, 0), q_len) > 0).any())
    nmb = (rows + M_BLOCK - 1) // M_BLOCK
    total_mb = int(nmb.sum())
    if total_mb == 0:
        return 0, 0, 0, 0, 0, 0, 1, may_empty
    req = np.repeat(np.arange(b), nmb)
    mb = np.arange(total_mb) - (np.cumsum(nmb) - nmb)[req]
    fr = mb * M_BLOCK
    nrows = np.minimum(rows[req] - fr, M_BLOCK)
    base = s[req] - q_len[req]
    p0 = base + fr // h_q
    p1 = base + (fr + nrows - 1) // h_q
    live = p1 >= 0
    nmax = np.where(live, p1 // seg + 1, 0)
    split = live & (nmax > 1)
    items = int(nmax[live].sum())
    slots = int((nrows * nmax)[split].sum())
    lo = np.maximum(np.maximum((seg - base) * h_q, 0), fr)
    rows_split = np.maximum(fr + nrows - lo, 0)
    combine = int(rows_split[split].sum())
    mandatory = split & ~((p0 >= 0) & (p0 // seg == p1 // seg))
    max_n = int(nmax[live].max()) if bool(live.any()) else 1
    return (total_mb, items, slots, combine, int((nrows * nmax)[mandatory].sum()),
            int(rows_split[mandatory].sum()), max_n, may_empty)


_GRAN_TARGET = 1   # plan.cu's GRAN_TARGET; a NEGATIVE target is a run-cost target (round 4)
#: A split must beat whole trees by this factor in the makespan model before it is taken: the model
#: ranks sizes by balance only, and a split adds published partials, combine work and folds.
_SPLIT_MARGIN = 0.95


def _saturated_decode_run_target(seqused_k, seg, mblocks, workers, item_overhead, fold_overhead,
                                 rows=0):
    """Run-cost target for a uniform decode batch that saturates the workers, or 0 for whole trees."""
    s = seqused_k.reshape(-1).to(torch.int64).cpu().numpy()
    live = s[s > 0]
    if live.size == 0 or not bool((live == live[0]).all()):
        return 0
    seg_blocks = seg // PAGE
    nmax = int((int(live[0]) - 1) // seg + 1)
    if nmax <= 1:
        return 0

    def cost(sz):
        return sz * seg_blocks + item_overhead + (sz - 1) * fold_overhead

    def makespan(sz):
        items = mblocks * (-(-nmax // sz))
        return (-(-items // workers)) * cost(sz)

    whole = makespan(nmax)
    best_sz, best = nmax, whole
    sz = 1
    while sz < nmax:
        ms = makespan(sz)
        if ms < best or (ms == best and sz > best_sz):
            best_sz, best = sz, ms
        sz *= 2
    if best_sz >= nmax or best >= _SPLIT_MARGIN * whole:
        return 0
    # A cut publishes every row's segments: rows x nmax slots.  If the workspace cannot hold them the
    # plan kernel would cut only the row blocks that fit and deal the rest whole -- a silent,
    # shape-dependent loss.  Round 5: say so instead.
    need = int(rows) * nmax
    have = _slot_budget()
    if need > have:
        raise ValueError(
            f"mla_sm100: this decode batch ({mblocks} row blocks of {nmax} segments, {rows} rows) needs "
            f"{need} partial slots ({need * SLOT_BYTES / 2**30:.2f} GiB at {SLOT_BYTES} B a slot) for its "
            f"makespan cut, and ISOEXEC_MLA_SM100_PARTIAL_BYTES allows {have} "
            f"({have * SLOT_BYTES / 2**30:.2f} GiB); raise it to at least {need * SLOT_BYTES} bytes")
    return int(cost(best_sz))


#: Default partial-node workspace, bytes.  One slot is one (row, segment) partial record of
#: PARTIAL_WIDTH fp32 = 2080 B.  Round 5 raised it from 1 GiB (516,222 slots) to 2 GiB (1,032,444):
#: the primary RL decode row B = 512, kv = 32768, h_q = 20 needs 655,360 slots (1.36 GB) for the
#: saturated-decode cut, and at 1 GiB the plan quietly cut only 915 of its 1024 items.  The buffer is
#: allocated per call at the size the PLAN needs (fwd.py `partials`), not at this cap, so a batch that
#: needs less costs less; the cap is what a batch may need at most.
PARTIAL_BYTES_DEFAULT = 2 << 30
SLOT_BYTES = PARTIAL_WIDTH * 4


def _slot_budget() -> int:
    """Largest partial buffer (in slots) a plan may use for published tree nodes."""
    return int(os.environ.get("ISOEXEC_MLA_SM100_PARTIAL_BYTES", str(PARTIAL_BYTES_DEFAULT))) // SLOT_BYTES


def mla_sm100_plan(cu_seqlens_q, seqused_k, seg_tokens, h_q, max_seqlen_q, capacity=None,
                   exec_override=None) -> MlaSm100Plan:
    """Build deterministic MLA work and combine plans from packed sequence lengths."""
    seg = int(seg_tokens)
    if seg <= 0 or seg % PAGE:
        raise ValueError(f"seg_tokens must be a positive multiple of {PAGE}")
    h_q = int(h_q)
    max_seqlen_q = int(max_seqlen_q)
    ov = dict(exec_override or {})
    dev = cu_seqlens_q.device
    b = int(seqused_k.shape[0])
    gran_name = ov.pop("granularity", "auto")
    if gran_name not in GRANULARITY:
        raise ValueError(f"exec_override granularity must be one of {sorted(GRANULARITY)}, got {gran_name!r}")
    granularity = GRANULARITY[gran_name]
    reverse = bool(ov.pop("reverse", False))
    item_overhead = int(ov.pop("item_overhead", ITEM_OVERHEAD_BLOCKS))
    fold_overhead = int(ov.pop("fold_overhead", FOLD_OVERHEAD_BLOCKS))
    workers_override = ov.pop("num_parts", None)
    combine_config = ov.pop("combine_config", None)
    debug_flags = ((4 << 16) if ov.pop("combine_single_group", False) else 0) \
        | (32 if ov.pop("no_pdl", False) else 0)
    if ov:
        raise ValueError(f"unknown exec_override keys {sorted(ov)}")
    workers = int(workers_override) if workers_override is not None else _num_sms()
    if workers <= 0:
        raise ValueError("num_parts must be positive")
    workers = min(workers, _MAX_WORKER_WAVES * _num_sms())
    budget = _slot_budget()
    run_target = 0
    if capacity is None:
        mblocks, items, slots, combine, mand_slots, mand_comb, max_n, may_empty = _exact_sizes(
            cu_seqlens_q, seqused_k, seg, h_q)
        exact = True
        workers = workers if workers_override is not None else max(1, min(workers, items))
        if (granularity == GRANULARITY["auto"] and max_seqlen_q == 1
                and mblocks >= _SNAKE_ITEMS_PER_WORKER * workers):
            run_target = _saturated_decode_run_target(seqused_k, seg, mblocks, workers,
                                                      item_overhead, fold_overhead,
                                                      rows=int(cu_seqlens_q.reshape(-1)[-1].item()
                                                               - cu_seqlens_q.reshape(-1)[0].item()) * h_q)
            if run_target:
                granularity = _GRAN_TARGET
        if granularity == GRANULARITY["auto"] and mblocks >= _SNAKE_ITEMS_PER_WORKER * workers:
            slots, combine = mand_slots, mand_comb
        else:
            slots = min(slots, max(budget, mand_slots))
            combine = min(combine, max(slots // 2, mand_comb))
    else:
        cap = {"max_seqlen_k": int(capacity)} if isinstance(capacity, int) else dict(capacity)
        max_k = int(cap["max_seqlen_k"])
        tokens = int(cap.get("num_tokens", b * max_seqlen_q))
        rows = tokens * h_q
        mblocks = min(b * math.ceil(max_seqlen_q * h_q / M_BLOCK), math.ceil(rows / M_BLOCK) + b)
        n_bound = max(1, math.ceil(max_k / seg))
        items = int(cap.get("items", mblocks * n_bound))
        slots = int(cap.get("slots", (rows + (M_BLOCK * b if max_seqlen_q > 1 else 0)) * n_bound
                            if n_bound > 1 else 0))
        if max_seqlen_q == 1:
            slots = min(slots, budget)
        may_empty = bool(cap.get("null_lanes", True))
        combine = int(cap.get("combine_rows", rows if n_bound > 1 else 0))
        max_n = n_bound
        exact = False
        workers = int(cap.get("workers", workers))

    if max_n > MAX_TREE_LEAVES and combine > 0:
        raise ValueError(f"{max_n} segments per row exceed the combine kernel's {MAX_TREE_LEAVES}; "
                         f"raise seg_tokens")

    plan = MlaSm100Plan(
        items=torch.zeros(max(items, 1), 8, dtype=torch.int32, device=dev),
        heads=torch.zeros(workers, dtype=torch.int32, device=dev),
        combine_rows=torch.zeros(combine, 4, dtype=torch.int32, device=dev),
        counts=torch.zeros(8, dtype=torch.int32, device=dev),
        scratch=torch.zeros(9 * max(mblocks, 1) + 3 * workers, dtype=torch.int32, device=dev),
        h_q=h_q, seg_tokens=seg, num_seqs=b, max_seqlen_q=max_seqlen_q, slot_cap=slots, max_n=max_n,
        workers=workers, exact=exact,
        combine_config=combine_config, debug_flags=debug_flags, may_empty=may_empty,
    )
    if b > 0:
        if 2 * b + 1 > _PLAN_SMEM_INTS:
            raise ValueError(f"{b} requests exceed the plan kernel's shared-memory staging")
        smem_ints = min(2 * b + 1 + 9 * max(mblocks, 1) + 3 * workers, _PLAN_SMEM_INTS)
        cost_cap = (max_n * seg + PAGE - 1) // PAGE + item_overhead + max_n * fold_overhead
        extension().plan(_i32(cu_seqlens_q), _i32(seqused_k), plan.items, plan.heads,
                         plan.combine_rows, plan.counts, plan.scratch, h_q, seg, workers,
                         granularity, -run_target, item_overhead, fold_overhead, slots, max(mblocks, 1),
                         cost_cap, reverse, smem_ints)
    return plan


def _combine_config(entries: int, max_n: int) -> int:
    """Combine worker shape from host capacities (execution only; every shape evaluates one tree)."""
    if max_n >= 16 and entries <= 128:
        return 1
    if entries >= 1024 and max_n <= 64:
        return 3
    return 2


def _i32(t: torch.Tensor) -> torch.Tensor:
    return t if (t.dtype == torch.int32 and t.is_contiguous()) else t.to(torch.int32).contiguous()


def _aligned(t: torch.Tensor) -> torch.Tensor:
    if t.data_ptr() % 16 == 0 and t.is_contiguous():
        return t
    out = torch.empty(t.numel() + 16, dtype=t.dtype, device=t.device)
    off = (-out.data_ptr()) % 16 // t.element_size()
    view = out[off: off + t.numel()].view(t.shape)
    view.copy_(t)
    return view


def mla_sm100_fwd(q, kv_cache, block_table, cu_seqlens_q, seqused_k, softmax_scale, seg_tokens,
                  max_seqlen_q, plan=None, out=None, exec_override=None, recipe=1):
    """Deterministic absorbed-MLA forward on Blackwell."""
    if q.dim() != 3 or q.shape[-1] != HEAD_DIM_K or q.dtype != torch.bfloat16:
        raise ValueError(f"q must be bf16 [T, h_q, {HEAD_DIM_K}], got {tuple(q.shape)} {q.dtype}")
    if kv_cache.dim() != 4 or tuple(kv_cache.shape[1:]) != (PAGE, 1, HEAD_DIM_K) \
            or kv_cache.dtype != torch.bfloat16:
        raise ValueError(f"kv_cache must be bf16 [num_blocks, {PAGE}, 1, {HEAD_DIM_K}], "
                         f"got {tuple(kv_cache.shape)}")
    t, h_q = int(q.shape[0]), int(q.shape[1])
    seg = int(seg_tokens)
    if out is None:
        o = torch.empty(t, h_q, HEAD_DIM_V, dtype=torch.bfloat16, device=q.device)
        lse = torch.empty(t, h_q, dtype=torch.float32, device=q.device)
    else:
        o, lse = out
        if tuple(o.shape) != (t, h_q, HEAD_DIM_V) or o.dtype != torch.bfloat16 or not o.is_contiguous():
            raise ValueError("out[0] must be contiguous bf16 [T, h_q, 512]")
        if tuple(lse.shape) != (t, h_q) or lse.dtype != torch.float32 or not lse.is_contiguous():
            raise ValueError("out[1] must be contiguous float32 [T, h_q]")
    if t == 0 or int(seqused_k.shape[0]) == 0:
        return o, lse
    if exec_override is not None or plan is None:
        plan = mla_sm100_plan(cu_seqlens_q, seqused_k, seg, h_q, max_seqlen_q,
                              exec_override=exec_override)
    if plan.h_q != h_q or plan.seg_tokens != seg or plan.num_seqs != int(seqused_k.shape[0]):
        raise ValueError(f"plan (h_q={plan.h_q}, seg={plan.seg_tokens}, B={plan.num_seqs}) does not "
                         f"match the call (h_q={h_q}, seg={seg}, B={int(seqused_k.shape[0])})")
    return _run(q, kv_cache, block_table, cu_seqlens_q, seqused_k, softmax_scale, seg, plan, o, lse,
                recipe)


def _run(q, kv_cache, block_table, cu_seqlens_q, seqused_k, softmax_scale, seg, plan, o, lse,
         recipe=1):
    if recipe not in RECIPE_IDS:
        raise ValueError(f"recipe must be one of {sorted(RECIPE_IDS)} ({RECIPE_IDS}), got {recipe!r}")
    h_q = int(q.shape[1])
    q = _aligned(q)
    if o.data_ptr() % 16:
        raise ValueError("out tensor must be 16-byte aligned")
    partials = torch.empty(max(plan.slot_cap, M_BLOCK), PARTIAL_WIDTH, dtype=torch.float32,
                           device=q.device)
    stack = torch.empty(plan.workers, max(1, (plan.max_n - 1).bit_length()), M_BLOCK,
                        PARTIAL_WIDTH, dtype=torch.float32, device=q.device)
    if plan.may_empty:
        o.zero_()
        lse.fill_(EMPTY_LSE)
    combine_entries = plan.combine_cap
    combine_config = plan.combine_config if plan.combine_config is not None else \
        _combine_config(combine_entries, plan.max_n)
    extension().fwd(q, kv_cache, _i32(block_table), _i32(cu_seqlens_q), _i32(seqused_k),
                    plan.items, plan.heads, plan.counts, plan.combine_rows, o, lse, partials,
                    _aligned(stack), float(softmax_scale), h_q, seg, plan.workers, combine_entries,
                    plan.max_n, combine_config, plan.debug_flags, int(recipe))
    return o, lse


def mla_sm100_fwd_selfcheck(seg_tokens: int, h_q: int, lengths=None, seed: int = 0, device="cuda",
                            softmax_scale: float = 256 ** -0.5) -> dict:
    """Role-free logprob contract self-check (bitwise)."""
    seg = int(seg_tokens)
    lengths = list(lengths or (seg - 1, seg + 1, 3 * seg + 37))
    g = torch.Generator().manual_seed(seed)
    results = {}

    def call(q_rows, kv_rows, spans, perm_seed, ov=None):
        B = len(spans)
        blocks = [-(-u // PAGE) for (_, _, u) in spans]
        nb = sum(blocks) + 2
        cache = torch.full((nb, PAGE, HEAD_DIM_K), float("nan"), dtype=torch.bfloat16, device="cpu")
        pg = torch.Generator().manual_seed(perm_seed)
        perm = torch.randperm(nb, generator=pg)
        table = torch.zeros(B, max(blocks), dtype=torch.int32, device="cpu")
        cur = 0
        for i, (a, e, u) in enumerate(spans):
            ids = perm[cur:cur + blocks[i]]
            cur += blocks[i]
            table[i, : blocks[i]] = ids.to(torch.int32)
            flat = cache[ids].reshape(-1, HEAD_DIM_K)
            flat[:u] = kv_rows[:u]
            cache[ids] = flat.view(-1, PAGE, HEAD_DIM_K)
        q = torch.cat([q_rows[a:e] for (a, e, _) in spans]).to(device)
        cu = torch.tensor([0] + list(torch.tensor([e - a for (a, e, _) in spans]).cumsum(0)),
                          dtype=torch.int32, device=device)
        used = torch.tensor([u for (_, _, u) in spans], dtype=torch.int32, device=device)
        o, l = mla_sm100_fwd(q, cache.unsqueeze(2).to(device), table.to(device), cu, used,
                             softmax_scale, seg, max(e - a for (a, e, _) in spans), exec_override=ov)
        torch.cuda.synchronize()
        res, off = [], 0
        for (a, e, _) in spans:
            res.append((o[off:off + e - a].cpu(), l[off:off + e - a].cpu()))
            off += e - a
        return res

    def same(x, y):
        return torch.equal(x[0].view(torch.int16), y[0].view(torch.int16)) and torch.equal(x[1], y[1])

    for L in lengths:
        kv = (torch.randn(L, HEAD_DIM_K, generator=g) * 0.6).to(torch.bfloat16)
        qr = (torch.randn(L, h_q, HEAD_DIM_K, generator=g) * 0.6).to(torch.bfloat16)
        full = call(qr, kv, [(0, L, L)], 1)[0]
        for ov in ({"granularity": "single", "num_parts": 4096},
                   {"num_parts": 3, "reverse": True, "granularity": "pairs"},
                   {"granularity": "whole", "num_parts": 1}, {"granularity": "mixed"}):
            results[f"L={L} prefill exec_override {ov}"] = same(full, call(qr, kv, [(0, L, L)], 2, ov)[0])
        for bounds in ([0, seg - 1, seg + 1, L], [0, 1, L // 2, L], [0, L - 1, L]):
            bounds = sorted({min(max(x, 0), L) for x in bounds})
            parts = [call(qr, kv, [(a, e, e)], 3 + e)[0] for a, e in zip(bounds[:-1], bounds[1:]) if e > a]
            chunked = (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]))
            results[f"L={L} chunked {bounds} == prefill"] = same(full, chunked)
        for p in sorted({0, seg - 1, seg, L - 1} & set(range(L))):
            d = call(qr, kv, [(p, p + 1, p + 1)], 7 + p)[0]
            results[f"L={L} decode p={p} == prefill"] = same((full[0][p:p + 1], full[1][p:p + 1]), d)
        k = min(3, L)
        v = call(qr, kv, [(L - k, L, L)], 11)[0]
        results[f"L={L} verify {k} rows == prefill"] = same((full[0][L - k:], full[1][L - k:]), v)
    bad = [k for k, ok in results.items() if not ok]
    if bad:
        raise AssertionError(f"mla_sm100 forward self-check failed: {bad}")
    return results
