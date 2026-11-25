"""TP-invariant GEMM kernels.

Layout convention, pinned by the contract and identical across engines: ``x`` is ``[M, K_local]`` and ``w`` is
``[N, K_local]``, both contiguous along K, with an fp32 ``[M, N]`` output because the leaf-combine tree lives
in fp32. Two schedules realize the same balanced binary tree over leaves: ``fused`` walks a CTA's m local
leaves in order, carrying subtree partials in a binary counter of accumulators (peak live is ``log2(m) + 1``,
and m == 1 is a plain GEMM), while ``splitk`` gives one CTA per (output tile, leaf) and applies the combine
tree in a second kernel, costing workspace traffic but winning at small M. Both are fully autotuned, since
block sizes, warps, stages, and grid order do not perturb the K-order.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .autotune_cache import register
from .fastlaunch import launch
from .plan import DEFAULT_PLAN, ReductionPlan


def _configs():
    """The autotune space, per architecture. Tuning over it cannot change results, only speed."""
    from .arch import current

    cfgs = []
    for bm, bn, bk, w, s in current().configs:
        for gm in (8,):
            cfgs.append(
                triton.Config(
                    {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm},
                    num_warps=w,
                    num_stages=s,
                )
            )
    return cfgs


# The smallest BLOCK_K a ``tl.dot`` will take on bf16 operands. Below it the kernel cannot be emitted at
# all, so a leaf narrower than this has NO legal Triton schedule and must be served by the sm90
# segment-tree producer (``sm90seg``), which zero-pads a narrow leaf to its K block.
MIN_BLOCK_K = 16


def triton_leaf_ok(leaf_k: int) -> bool:
    """Can ANY Triton schedule realize a leaf of this width?

    Every leaf-tree kernel walks its leaf with ``for _ in range(LEAF_K // BLOCK_K)`` -- a FLOOR. A BLOCK_K
    wider than LEAF_K makes that a ZERO-trip loop, and the kernel then stores its untouched zero accumulator:
    a GEMM that returns exact zeros, silently, at full speed. So a Triton schedule is admissible only when
    some legal BLOCK_K divides LEAF_K, and the widest legal BLOCK_K is bounded below by MIN_BLOCK_K.
    """
    return leaf_k >= MIN_BLOCK_K and leaf_k % MIN_BLOCK_K == 0


def _fallback_config(leaf_k: int) -> triton.Config:
    """The config used when the arch's autotune space holds no BLOCK_K that tiles LEAF_K.

    It MUST still satisfy ``BLOCK_K | LEAF_K`` -- that is the whole invariant ``_prune`` exists to enforce,
    and a fallback that violates it is worse than no fallback: see ``triton_leaf_ok``. The arch spaces start
    at BLOCK_K=64, so every leaf narrower than that lands here.
    """
    bk = 32
    while bk > MIN_BLOCK_K and leaf_k % bk:
        bk //= 2
    if not triton_leaf_ok(leaf_k) or leaf_k % bk:
        raise ValueError(
            f"pik: no legal BLOCK_K tiles LEAF_K={leaf_k} (tl.dot needs BLOCK_K >= {MIN_BLOCK_K} and the "
            f"leaf-tree kernels need BLOCK_K | LEAF_K). This shape has no Triton schedule; it must be served "
            f"by the sm90 segment-tree producer (sm90seg). Raising rather than emitting a zero-trip k-loop, "
            f"which would return exact zeros."
        )
    return triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": bk, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    )


def _prune(configs, nargs, **kwargs):
    """Keep only configs whose BLOCK_K tiles a leaf exactly.

    Masked k-tails would load exact zeros anyway, but requiring ``BLOCK_K | LEAF_K`` leaves the k-loop trip
    count as the only thing that varies between configs -- and, far more importantly, it is the ONLY thing
    standing between a narrow leaf and a zero-trip k-loop (``triton_leaf_ok``).
    """
    leaf_k = nargs["LEAF_K"]
    out = [c for c in configs if leaf_k % c.kwargs["BLOCK_K"] == 0]
    return out or [_fallback_config(leaf_k)]


# Triton keys its config cache on the exact values of these args, so autotune_cache.register installs a cache
# that buckets M by power of two above PIK_AUTOTUNE_M_BUCKET_FLOOR. Keep M first: the bucketing indexes key[0].
_AUTOTUNE_KEY = ["M", "N", "LEAF_K", "NUM_LOCAL_LEAVES"]


@triton.jit
def _tile_ids(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# The fused-tree kernel is generated per local-leaf count; see pik/codegen.py.


@triton.autotune(configs=_configs(), key=_AUTOTUNE_KEY, prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _leaf_partial_kernel(
    x_ptr,
    w_ptr,
    p_ptr,
    M,
    N,
    LEAF_K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_pl,
    stride_pm,
    stride_pn,
    NUM_LOCAL_LEAVES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    leaf = tl.program_id(1)
    pid_m, pid_n = _tile_ids(tl.program_id(0), M, N, BLOCK_M, BLOCK_N, GROUP_M)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    k0 = leaf * LEAF_K
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + (k0 + offs_k)[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(LEAF_K // BLOCK_K):
        a = tl.load(x_ptrs)
        b = tl.load(w_ptrs)
        acc = tl.dot(a, tl.trans(b), acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        p_ptr + leaf * stride_pl + om[:, None] * stride_pm + on[None, :] * stride_pn,
        acc,
        mask=(om[:, None] < M) & (on[None, :] < N),
    )


register("_leaf_partial", _leaf_partial_kernel)


# The elementwise tree-reduce epilogue comes from the same emitter as the fused kernel, so the two
# schedules cannot drift apart.


_WORKSPACE: dict = {}
# Only captured generations need permanent ownership: graphs hold their raw pointers. Geometric growth
# bounds their combined capacity below the current buffer. Eager-only generations use PyTorch's ordinary
# stream-aware allocator lifetime and must not accumulate over a run's increasing sequence lengths.
_WORKSPACE_RETIRED: list = []


def _workspace(leaves: int, m: int, n: int, device, dtype=torch.float32) -> torch.Tensor:
    """One growable flat buffer per (device, dtype), viewed to shape.

    Deliberately not keyed on (M, N): a serving engine sees a long tail of batch sizes, and a buffer per M
    would leak until OOM.
    """
    need = leaves * m * n
    key = (device, dtype)
    buf = _WORKSPACE.get(key)
    capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    if buf is None or buf.numel() < need:
        if capturing:
            raise RuntimeError("GEMM workspace growth during CUDA capture: warm this shape eagerly first")
        capacity = max(need, 2 * buf.numel() if buf is not None else 0)
        previous = buf
        buf = torch.empty(capacity, device=device, dtype=dtype)
        buf._pik_captured = False
        _WORKSPACE[key] = buf
        if previous is not None and previous._pik_captured:
            _WORKSPACE_RETIRED.append(previous)
    if capturing:
        buf._pik_captured = True
    elif not buf._pik_captured and buf.is_cuda:
        # The cache can be warmed on one stream and used on another. Recording eager use keeps a
        # released generation alive until its queued work finishes, without retaining it forever.
        buf.record_stream(torch.cuda.current_stream(device))
    return buf[:need].view(leaves, m, n)


def _dispatch(x, w, out, leaf_k: int, m_leaves: int, sched: str, leaf_dtype=torch.float32):
    """The one place a schedule is turned into kernel launches.

    Every branch here computes the SAME arithmetic tree and returns the SAME bits;
    they differ only in where the intermediate subtree partials live (fp32 workspace,
    registers/TMEM, or the sm90 segment-tree producer's registers).
    """
    M, _ = x.shape
    N, _ = w.shape

    if sched == "sm90seg":
        # ops/mm/mm_sm90 leaf mode: one launch writes this rank's node (the bf16 leaf itself at m == 1),
        # each leaf one ascending wgmma chain over its leaf_k slice, the tree over the m leaves in fp32.
        from isoexec.ops.mm.mm_sm90 import ops as sm90

        node_dtype = sm90_node_dtype(m_leaves, leaf_dtype)
        if out.dtype == node_dtype:
            return sm90.leaf_node(x, w, leaf_k, out, leaf_dtype=leaf_dtype)
        # The caller holds the ROOT in another dtype (TP1: the whole tree is local and pik/linear.py hands a
        # bf16 root). The producer writes its node where it must, in its own dtype, and the root is rounded
        # once into the caller's buffer: one fp32 -> bf16 RNE, the rounding every other schedule applies
        # where it stores the root -- the same bits.
        node = _workspace(1, M, N, x.device, dtype=node_dtype)[0]
        sm90.leaf_node(x, w, leaf_k, node, leaf_dtype=leaf_dtype)
        out.copy_(node)
        return out

    if sched == "fused":
        from .codegen import fused_tree_kernel

        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)  # noqa: E731
        launch(
            fused_tree_kernel(m_leaves, leaf_dtype == torch.bfloat16),
            grid,
            x,
            w,
            out,
            M,
            N,
            leaf_k,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            out.stride(0),
            out.stride(1),
            NUM_LOCAL_LEAVES=m_leaves,
        )
        return out

    if sched.startswith("split"):
        # "splitk" splits all the way (s = m); "split<s>" splits s ways, each CTA carrying m/s leaves.
        from .codegen import split_tree_kernel, tree_reduce_kernel

        s = m_leaves if sched == "splitk" else int(sched[5:])
        assert m_leaves % s == 0 and (s & (s - 1)) == 0, f"bad split {s} for m={m_leaves}"
        g = m_leaves // s

        bl = leaf_dtype == torch.bfloat16
        store_bf16 = bl and g == 1  # a g==1 partial is a LEAF; leaves may be bf16
        p = _workspace(s, M, N, x.device, dtype=torch.bfloat16 if store_bf16 else torch.float32)
        grid = lambda META: (  # noqa: E731
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            s,
        )
        launch(
            split_tree_kernel(g, bl, store_bf16),
            grid,
            x,
            w,
            p,
            M,
            N,
            leaf_k,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            p.stride(0),
            p.stride(1),
            p.stride(2),
            NUM_LOCAL_LEAVES=m_leaves,
        )
        if s == 1:
            out.copy_(p[0])
            return out
        n_elem = M * N
        launch(
            tree_reduce_kernel(s, in_bf16=store_bf16, out_bf16=(out.dtype == torch.bfloat16)),
            (triton.cdiv(n_elem, 1024),),
            p,
            out,
            n_elem,
            p.stride(0),
            BLOCK=1024,
            num_warps=4,
        )
        return out

    raise ValueError(f"unknown schedule {sched!r}")


_SCHED_CACHE: dict = {}
_BF16_LEAVES = [False]  # set by ti_gemm from the plan; see _pick_schedule


def _sched_key(M, N, leaf_k, m_leaves):
    # Bucket M by power of two so a serving engine does not re-time per batch size. Every schedule produces
    # the same bits, so a mis-bucketed choice costs time and never correctness.
    mb = 1 << max(0, (M - 1).bit_length())
    return (mb, N, leaf_k, m_leaves)


def sm90_node_dtype(m_leaves: int, leaf_dtype) -> torch.dtype:
    """The dtype the sm90 segment-tree producer writes a rank's node in (``mm_sm90.ops.leaf_node``): the bf16
    leaf itself at m == 1 under bf16 leaves, otherwise the fp32 tree node."""
    return torch.bfloat16 if (m_leaves == 1 and leaf_dtype == torch.bfloat16) else torch.float32


def _sm90seg_ok(m_leaves: int, leaf_k: int) -> bool:
    """The sm90 segment-tree GEMM can produce this leaf tree (Hopper, built, leaf in whole K blocks)."""
    try:
        from isoexec.ops.mm.mm_sm90 import ops as sm90
        from isoexec.ops.mm.mm_sm90 import segments as seg
    except Exception:  # noqa: BLE001
        return False
    return (m_leaves in (1, 2, 4, 8) and leaf_k % seg.EXECUTION_ALIGN == 0 and sm90.available())


def _sched_candidates(m_leaves: int, leaf_k: int) -> list[str]:
    """Every schedule the picker may choose from, in bench order.

    Every power-of-two split factor: s=1 is the fused tree, s=m is full split-K, and the
    interesting ones are usually in between.

    A leaf too narrow for any legal BLOCK_K (``triton_leaf_ok``) offers NO Triton schedule: the k-loop would
    run zero trips and store zeros. The sm90 segment-tree producer has no such constraint -- it zero-pads a
    leaf to its K block -- so it is the only candidate there, and where it is unavailable this raises
    instead of returning a schedule that computes nothing.
    """
    cands = []
    if triton_leaf_ok(leaf_k):
        cands = ["fused"] + [f"split{s}" for s in (1, 2, 4, 8, 16, 32) if s <= m_leaves and m_leaves % s == 0]
    if _sm90seg_ok(m_leaves, leaf_k):
        cands.insert(0, "sm90seg")
    if not cands:
        raise ValueError(
            f"pik: LEAF_K={leaf_k} admits no Triton schedule (needs BLOCK_K | LEAF_K with "
            f"BLOCK_K >= {MIN_BLOCK_K}) and the sm90 segment-tree producer is unavailable (needs a Hopper "
            f"device and LEAF_K % 8 == 0) -- there is no way to evaluate this leaf. Widen K or lower "
            f"ReductionPlan(num_leaves=...)."
        )
    return cands


# How far an admitted schedule may sit from the EAGER reference, relative to the reference's own
# magnitude. Loose on purpose: this bound is only ever consulted where no exact producer is available, and it
# exists to reject a schedule that computes something ELSE (a zero-trip k-loop stores an untouched zero
# accumulator, i.e. relative error 1.0), not to arbitrate last-bit differences between two backends.
_ADMIT_RTOL = 0.05


def _eager_leaf_reference(x, w, leaf_k: int, m_leaves: int, leaf_dtype) -> torch.Tensor:
    """The leaf tree computed with torch's own GEMM, in fp32, in the same combine order.

    NOT a bitwise reference -- torch's matmul is not pik's leaf kernel, and the two round differently --
    but a numerically faithful one, which is all `_admit_schedules` asks of it when there is no exact
    producer to compare against.
    """
    leaves = [
        x[:, j * leaf_k : (j + 1) * leaf_k].float() @ w[:, j * leaf_k : (j + 1) * leaf_k].float().T
        for j in range(m_leaves)
    ]
    if leaf_dtype == torch.bfloat16:
        # the leaf is the rounding point the plan pins; the tree above it is fp32
        leaves = [t.bfloat16().float() for t in leaves]
    while len(leaves) > 1:  # the same balanced binary tree every schedule realizes
        leaves = [leaves[i] + leaves[i + 1] for i in range(0, len(leaves), 2)]
    return leaves[0]


def _admit_schedules(x, w, out, leaf_k: int, m_leaves: int, cands: list[str], leaf_dtype) -> list[str]:
    """Drop every candidate whose output is not the reference's, BEFORE anything is timed.

    A kernel that computes nothing is the fastest kernel there is, so a picker that ranks candidates by
    wall clock alone will always choose it. That is exactly how the zero-trip k-loop shipped: at
    LEAF_K=16 `fused` and every `split*` returned exact zeros and won every bench, deterministically,
    for as long as pik has existed. The (since retired) cuBLASLt small-M path had an on-device bit-compare
    admission and therefore could never have shipped that bug; this is the same admission, for the schedule
    picker, which had no equivalent.

    The reference is the sm90 segment-tree producer (``sm90seg``, ``ops/mm/mm_sm90`` leaf mode) whenever it
    can serve the shape -- a different backend realizing the same tree -- and the comparison against it is
    BIT FOR BIT. That is not a new demand: it is the cross-backend equality `_pick_schedule`'s own docstring
    already relies on, and the one `pik_narrow_leaf_battery_gpu.py` measures. Without it there is no bitwise
    reference to be had, so `_eager_leaf_reference` stands in at `_ADMIT_RTOL` -- still decisive against a
    schedule that computes nothing, and silent about backend rounding it has no standing to judge.

    Runs once per shape bucket, on the caller's live operands, into scratch buffers: `out` is not
    touched, and the winner is re-dispatched into it afterwards exactly as before. The scratch has the
    caller's dtype and shape: every schedule, the sm90 producer included (``_dispatch`` stages its node in
    the producer's own dtype), writes the caller's dtype, so candidates are judged on the bits they
    would actually store.
    """
    ref = torch.empty(out.shape, dtype=out.dtype, device=out.device)
    got = torch.empty(out.shape, dtype=out.dtype, device=out.device)
    if _sm90seg_ok(m_leaves, leaf_k):
        ref_name, exact = "sm90seg", True
        _dispatch(x, w, ref, leaf_k, m_leaves, "sm90seg", leaf_dtype)
    else:
        ref_name, exact = "eager", False
        ref.copy_(_eager_leaf_reference(x, w, leaf_k, m_leaves, leaf_dtype))
    scale = float(ref.float().abs().max()) if not exact else 0.0

    admitted, rejected = [], {}
    for s in cands:
        try:
            _dispatch(x, w, got, leaf_k, m_leaves, s, leaf_dtype)
        except Exception as e:  # noqa: BLE001 -- a schedule may not support this shape; that is a
            # rejection like any other, and the bench used to swallow it silently
            rejected[s] = f"raised: {e!r}"
            continue
        if exact:
            ok = torch.equal(got.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8))
            why = "not bitwise equal to the sm90 segment-tree leaf producer on this shape"
        else:
            d = float((got.float() - ref.float()).abs().max())
            rel = d / scale if scale > 0 else d
            ok = rel <= _ADMIT_RTOL
            why = f"max|d| vs the eager leaf tree is {d:.6g} ({rel:.3g} relative) > {_ADMIT_RTOL}"
        (admitted.append(s) if ok else rejected.setdefault(s, why))

    if rejected:
        print(
            f"[ISOEXEC-PIK] schedule(s) REJECTED for shape (M={x.shape[0]}, N={w.shape[0]}, "
            f"leaf_k={leaf_k}, m={m_leaves}, leaf_dtype={leaf_dtype}) against the {ref_name} "
            f"reference, before any timing: "
            + "; ".join(f"{s}: {r}" for s, r in rejected.items())
            + f". Admitted: {admitted}.",
            flush=True,
        )
    if not admitted:
        raise ValueError(
            f"pik: no schedule agrees with the {ref_name} reference at (M={x.shape[0]}, N={w.shape[0]}, "
            f"leaf_k={leaf_k}, m={m_leaves}, leaf_dtype={leaf_dtype}): {rejected}. Refusing to pick one "
            f"by wall clock -- a kernel that computes nothing is the fastest kernel there is."
        )
    return admitted


def _bench_schedules(x, w, out, leaf_k: int, m_leaves: int, cands: list[str], leaf_dtype) -> str:
    """Wall-clock pick among candidate schedules. Every candidate here has already been admitted by
    `_admit_schedules`, so this decides speed only -- it never decides correctness."""
    best, best_t = None, float("inf")
    for s in cands:
        try:
            t = triton.testing.do_bench(
                lambda s=s: _dispatch(x, w, out, leaf_k, m_leaves, s, leaf_dtype),
                warmup=5,
                rep=20,
                quantiles=(0.5,),
            )
        except Exception:  # noqa: BLE001 -- a schedule may not support this shape
            continue
        if t < best_t:
            best, best_t = s, t
    # Never fall back to a schedule that is not in `cands`: at a narrow leaf "splitk" is exactly the
    # zero-trip kernel `_sched_candidates` refused to offer, and it would win any wall-clock bench.
    return best or cands[0]


def _pick_schedule(x, w, out, leaf_k: int, m_leaves: int, leaf_dtype=None) -> str:
    """Pick a schedule for this shape bucket by timing the candidates once, then cache the winner.

    All schedules realize the same combine tree by construction; whether they are also bitwise identical
    across backends (the sm90 segment-tree leaf vs Triton ``tl.dot`` chain) is a property of the (arch,
    toolchain) pair -- and `_admit_schedules` measures it, bit for bit, before anything is timed.
    Because the pick is per process, the trainer and the engine may land on different schedules, which is safe
    only while that cross-backend equality holds.

    Which one wins is shape-dependent: m == 1 is a single GEMM; decode M favours the sm90 producer; small
    leaf_k with big M favours the fused Triton kernel, which keeps the tree in TMEM and touches no workspace;
    small M favours split-K, whose leaf parallelism fills the machine.
    """

    if leaf_dtype is None:
        leaf_dtype = out.dtype if out.dtype == torch.bfloat16 else torch.float32
    # The output of multiple BF16 leaves is FP32; output dtype alone does not identify the plan.
    # Admission and timings belong to this device and precision, not just the GEMM dimensions.
    key = (
        *_sched_key(x.shape[0], w.shape[0], leaf_k, m_leaves),
        x.device,
        x.dtype,
        w.dtype,
        out.dtype,
        leaf_dtype,
    )
    hit = _SCHED_CACHE.get(key)
    if hit is not None:
        return hit

    cands = _sched_candidates(m_leaves, leaf_k)
    if m_leaves == 1 and "sm90seg" in cands:
        # Degenerates to one plain GEMM; nothing to measure, and the sm90 producer IS the reference.
        best = "sm90seg"
    else:
        # Correctness first, speed second: a candidate that does not compute the reference's answer is
        # not a slower candidate, it is not a candidate. See `_admit_schedules`.
        cands = _admit_schedules(x, w, out, leaf_k, m_leaves, cands, leaf_dtype)
        # The sm90 segment-tree producer competes like any other admitted schedule: admission has made
        # every survivor bit-equal, so the race decides speed only.
        best = _bench_schedules(x, w, out, leaf_k, m_leaves, cands, leaf_dtype)

    _SCHED_CACHE[key] = best
    return best


def ti_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    plan: ReductionPlan = DEFAULT_PLAN,
    tp_size: int = 1,
    tp_rank: int = 0,
    k_full: int | None = None,
    schedule: str | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Row-parallel TP-invariant GEMM returning this rank's subtree partial.

    ``x: [M, K_local]``, ``w: [N, K_local]`` -> ``[M, N]``. At tp_size == 1 this is already the final result;
    above it, feed the output to ``pik.allreduce.tree_all_reduce``, which applies the remaining tree levels.
    """
    assert x.ndim == 2 and w.ndim == 2, "expected 2-D x [M,K] and w [N,K]"
    M, k_local = x.shape
    N, k_local_w = w.shape
    assert k_local == k_local_w, f"K mismatch: x has {k_local}, w has {k_local_w}"

    k_full = k_full if k_full is not None else k_local * tp_size
    plan.validate(k_full, tp_size)
    assert (
        k_local == k_full // tp_size
    ), f"x/w have K_local={k_local} but plan says {k_full}//{tp_size}={k_full // tp_size}"

    leaf_k = plan.leaf_k(k_full)
    m_leaves = plan.leaves_per_rank(tp_size)

    # With bf16 leaves, a rank whose partial is itself a leaf (m == 1) stays bf16 all the way, wire included.
    # With m > 1 the partial is an internal tree node, and internal nodes are always fp32: rounding one would
    # make the result depend on where the rank boundary fell.
    if out is None:
        out_dtype = torch.bfloat16 if (plan.bf16_leaves and m_leaves == 1) else torch.float32
        out = torch.empty((M, N), device=x.device, dtype=out_dtype)

    _BF16_LEAVES[0] = plan.bf16_leaves
    sched = schedule or _pick_schedule(x, w, out, leaf_k, m_leaves, plan.leaf_dtype)
    return _dispatch(x, w, out, leaf_k, m_leaves, sched, plan.leaf_dtype)


# bf16-leaf wire for the owner all-reduce: the m leaves themselves, not their fold. On Hopper one launch of the
# sm90 segment-tree GEMM in leaf mode writes all m planes (each one ascending chain over its leaf_k slice,
# rounded once to bf16 -- the leaves every schedule's fold consumes). Elsewhere, the per-leaf Triton loop.


def ti_gemm_leaves(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    plan: ReductionPlan = DEFAULT_PLAN,
    tp_size: int = 1,
    k_full: int | None = None,
    out: torch.Tensor,
) -> torch.Tensor:
    """This rank's m bf16 LEAF partials ``out[j] = round_bf16(x_leaf_j @ w_leaf_j.T)``, ``out: [m, M, N]``.

    The fold itself is left to the consumer's tree, which is the same tree every schedule uses.
    """
    assert plan.bf16_leaves and out.dtype == torch.bfloat16, "leaf planes exist only under a bf16-leaf plan"
    M, k_local = x.shape
    N = w.shape[0]
    k_full = k_full if k_full is not None else k_local * tp_size
    plan.validate(k_full, tp_size)
    leaf_k = plan.leaf_k(k_full)
    m = plan.leaves_per_rank(tp_size)
    assert out.shape == (m, M, N) and out.is_contiguous(), f"leaf planes must be [m={m}, {M}, {N}] contiguous"
    if M == 0:
        return out
    if _sm90seg_ok(m, leaf_k):
        from isoexec.ops.mm.mm_sm90 import ops as sm90

        if sm90.leaf_supported(k_local, N, leaf_k) and x.stride(1) == 1 and w.stride(1) == 1 \
                and x.stride(0) % 8 == 0 and w.stride(0) % 8 == 0:
            return sm90.leaves(x, w, leaf_k, out)
    for j in range(m):
        sl = slice(j * leaf_k, (j + 1) * leaf_k)
        _dispatch(x[:, sl], w[:, sl], out[j], leaf_k, 1, "fused", torch.bfloat16)
    return out


