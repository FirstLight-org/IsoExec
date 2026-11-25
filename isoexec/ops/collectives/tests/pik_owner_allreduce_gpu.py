#!/usr/bin/env python
"""Bitwise battery for pik's owner tree all-reduce (pik/owner.py), against the legacy pik path and TP=1.

``--sim`` (ONE GPU): every world size W in {2, 4, 8} simulated on one device -- each virtual rank's K shard
produces its wire planes with the production GEMMs (bf16 leaf planes and fp32/bf16 node planes), the owner
kernel reduces them in local mode, and the result must equal, bit for bit, the TP=1 row-parallel projection
(``pik.linear.row_parallel_linear(tp_size=1)``) of the same global operands. Then raw-bit-pattern planes
(signed zeros, subnormals, infinities, NaNs) against the legacy Triton tree kernel + torch cast.

Multi-rank (``--world N``, one GPU per rank): for the shipped dense shapes and decode..prefill M, the owner
path at TP=N must equal the legacy pik path at TP=N and the TP=1 statement, for: p2p one-shot, p2p two-shot,
the NCCL transport, both plans (bf16 leaves G=8, fp32 leaves G=8) and the m=1 leaf wire (G=N); with rank
arrival skew; through CUDA-graph capture/replay interleaved with eager calls; under a randomized stress loop;
for the generic routed ``tree_all_reduce``; and a split flag must disable the path on every rank, not hang.

    CUDA_VISIBLE_DEVICES=0 python isoexec/ops/collectives/tests/pik_owner_allreduce_gpu.py --sim
    CUDA_VISIBLE_DEVICES=0,1 python isoexec/ops/collectives/tests/pik_owner_allreduce_gpu.py --world 2
    CUDA_VISIBLE_DEVICES=0,1,2,3 python isoexec/ops/collectives/tests/pik_owner_allreduce_gpu.py --world 4
"""

from __future__ import annotations

import argparse
import os
import random
import socket
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, *[".."] * 4))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

# (K, N): Qwen3.8-27B attention o_proj / GDN out_proj (K=24*256=48*128) and MLP fc2, plus a ragged shape.
SHAPES = {"o_proj": (6144, 5120), "fc2": (17408, 5120), "ragged": (1024, 1000)}
M_POINTS = (1, 3, 32, 128, 256, 300, 512, 2048)


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _diff(a: torch.Tensor, b: torch.Tensor) -> int:
    """Elements whose bit patterns differ (NaN vs NaN with different payloads counts as equal)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    ne = _bits(a) != _bits(b)
    both_nan = torch.isnan(a.float()) & torch.isnan(b.float())
    return int((ne & ~both_nan).sum())


def _plans(world):
    """bf16 leaves at G=8, fp32 leaves at G=8, and the leaf-wire geometries m=2 (C=2) and m=1 at this world."""
    from pik.plan import ReductionPlan  # type: ignore

    out, seen = [], set()
    for name, g, dt in (("bf16_G8", 8, torch.bfloat16), ("fp32_G8", 8, None),
                        (f"bf16_G{2 * world}", 2 * world, torch.bfloat16), (f"bf16_G{world}", world, torch.bfloat16)):
        if (g, dt) in seen or g < world:
            continue
        seen.add((g, dt))
        out.append((name, ReductionPlan(num_leaves=g, leaf_dtype=dt)))
    return out


def _operands(K, N, M, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16, generator=g)
    w = (torch.randn(N, K, device=dev, dtype=torch.bfloat16, generator=g) * 0.02).to(torch.bfloat16)
    return x, w


# ------------------------------------------------------------------------------------------------ simulation


def sim() -> int:
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik

    ensure_pik()
    from pik import owner  # type: ignore
    from pik.codegen import tree_reduce_kernel  # type: ignore
    from pik.gemm import ti_gemm, ti_gemm_leaves  # type: ignore
    from pik.linear import row_parallel_linear  # type: ignore

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    fails = checks = 0
    for sname, (K, N) in SHAPES.items():
        for M in M_POINTS:
            x, w = _operands(K, N, M, 1234 + M, dev)
            for pname, plan in _plans(8):
                if K % plan.num_leaves:
                    continue
                for out_dtype in (torch.bfloat16, torch.float32):
                    ref = row_parallel_linear(x, w, None, plan=plan, tp_size=1, k_full=K, out_dtype=out_dtype)
                    root_dt = owner.root_dtype_for(plan, None, out_dtype)
                    W = 2
                    while W <= plan.num_leaves:
                        m = plan.leaves_per_rank(W)
                        kl = K // W
                        sh = [(x[:, r * kl:(r + 1) * kl].contiguous(), w[:, r * kl:(r + 1) * kl].contiguous())
                              for r in range(W)]
                        nodes = [ti_gemm(xr, wr, plan=plan, tp_size=W, tp_rank=r, k_full=K).reshape(1, -1)
                                 for r, (xr, wr) in enumerate(sh)]
                        legs = {"node": nodes}
                        if plan.bf16_leaves:
                            legs["leaf"] = [
                                ti_gemm_leaves(xr, wr, plan=plan, tp_size=W, k_full=K,
                                               out=torch.empty((m, M, N), device=dev, dtype=torch.bfloat16)
                                               ).reshape(m, -1)
                                for xr, wr in sh
                            ]
                        for leg, ops in legs.items():
                            got = owner.reduce_local(ops, root_dt).view(M, N).to(out_dtype)
                            d = _diff(got, ref)
                            checks += 1
                            if d:
                                fails += 1
                                print(f"FAIL sim {sname} M={M} {pname} out={out_dtype} W={W} {leg}: {d} differ")
                        # the legacy arithmetic at the same W, for the record
                        st = torch.stack([t.reshape(-1).float() for t in nodes])
                        leg_root = torch.empty(M * N, device=dev, dtype=torch.float32)
                        tree_reduce_kernel(W)[((M * N + 1023) // 1024,)](st, leg_root, M * N, st.stride(0),
                                                                         BLOCK=1024, num_warps=4)
                        rr = leg_root.view(M, N)
                        rr = rr.to(torch.bfloat16) if root_dt == torch.bfloat16 else rr
                        d = _diff(rr.to(out_dtype), ref)
                        checks += 1
                        if d:
                            fails += 1
                            print(f"FAIL sim legacy-arith {sname} M={M} {pname} W={W}: {d} differ")
                        W *= 2
            print(f"sim {sname} M={M}: cumulative {checks - fails}/{checks} bitwise", flush=True)

    # raw bit patterns: every class of value the wire can carry
    g = torch.Generator(device=dev).manual_seed(7)
    for W, C, dt in ((2, 1, torch.float32), (2, 4, torch.bfloat16), (4, 2, torch.bfloat16), (4, 1, torch.float32),
                     (8, 1, torch.bfloat16), (8, 1, torch.float32), (2, 1, torch.bfloat16), (4, 1, torch.bfloat16)):
        for n in (8192, 8191):
            it = torch.int16 if dt == torch.bfloat16 else torch.int32
            lo, hi = (-(1 << 15), 1 << 15) if it == torch.int16 else (-(1 << 31), (1 << 31) - 1)
            raw = torch.randint(lo, hi, (W * C, n), device=dev, dtype=torch.int64, generator=g).to(it)
            # salt in exact specials
            sp = torch.tensor([0.0, -0.0, float("inf"), float("-inf"), 1e-40, -1e-40, 3.0e38, -3.0e38],
                              dtype=dt, device=dev)
            raw[:, :8] = sp.view(it)
            planes = raw.view(dt)
            ops = [planes[r * C:(r + 1) * C].contiguous() for r in range(W)]
            st = planes.contiguous()
            leg = torch.empty(n, device=dev, dtype=torch.float32)
            tree_reduce_kernel(W * C, in_bf16=dt == torch.bfloat16)[((n + 1023) // 1024,)](
                st, leg, n, st.stride(0), BLOCK=1024, num_warps=4)
            for rd in (torch.float32, torch.bfloat16):
                got = owner.reduce_local(ops, rd)
                d = _diff(got, leg.to(rd))
                checks += 1
                if d:
                    fails += 1
                    print(f"FAIL sim raw-bits W={W} C={C} {dt} n={n} root={rd}: {d} differ")
    print(f"SIM RESULT: {checks - fails}/{checks} bitwise-equal checks", flush=True)
    return 1 if fails else 0


# ------------------------------------------------------------------------------------------------ multi-rank


def _force(owner, group, dev, *, transport=None, oneshot_bytes=None, leaf_min_m=None, twoshot=None):
    st = owner.state(group, dev)
    assert st is not None, "owner path must be on here"
    # This battery covers the barrier protocol; the Lamport protocol has its own (pik_owner_lamport_gpu.py).
    st.lamport = owner._LamportCfg(False, False, 0, 0, 0, False, None)
    st.node_below = True  # the battery drives the one-launch node wire too
    if leaf_min_m is not None:
        st.leaf_min_m = leaf_min_m
    if twoshot is not None:
        st.twoshot = twoshot
    if transport is not None:
        if transport == "p2p":
            assert st.p2p is not None, "p2p transport unavailable"
        st.transport = transport
    if oneshot_bytes is not None:
        st.oneshot_bytes = oneshot_bytes
    return st


def _skew(rank, world, jit):
    if jit:
        torch.cuda._sleep(int(jit * 1_000_000 * (rank if jit > 0 else world - 1 - rank)))


def worker(rank: int, world: int, port: int, args) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)
    group = dist.group.WORLD
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik

    ensure_pik()
    import pik.allreduce as AR  # type: ignore
    from pik import owner  # type: ignore
    from pik.linear import row_parallel_linear  # type: ignore

    owner.set_enabled(True)
    owner.reset()
    ST = owner.state(group, dev)  # held so the legacy legs can switch it off and back on
    assert ST is not None, "owner path did not engage"
    fails = checks = 0

    def check(tag, got, ref):
        nonlocal fails, checks
        d = _diff(got, ref)
        checks += 1
        if d:
            fails += 1
            print(f"[rank {rank}] FAIL {tag}: {d if d >= 0 else 'shape/dtype'} differ", flush=True)

    def site(xs, ws, plan, K, out_dtype, bias=None):
        return row_parallel_linear(xs, ws, bias, plan=plan, tp_size=world, tp_rank=rank, k_full=K, group=group,
                                   out_dtype=out_dtype)

    def shard(x, w, K):
        kl = K // world
        return x[:, rank * kl:(rank + 1) * kl].contiguous(), w[:, rank * kl:(rank + 1) * kl].contiguous()

    # (name, transport, one-shot budget, leaf-wire min M, two-shot mode 3=push / 2=pull)
    big = 1 << 62
    legs = [("p2p_1shot_leaf", "p2p", big, 0, 3), ("p2p_push_leaf", "p2p", 0, 0, 3), ("p2p_pull_leaf", "p2p", 0, 0, 2),
            ("p2p_1shot_node", "p2p", big, big, 3), ("p2p_push_node", "p2p", 0, big, 3),
            ("nccl_leaf", "nccl", None, 0, 3), ("nccl_node", "nccl", None, big, 3)]
    if args.no_p2p:
        legs = [leg for leg in legs if leg[1] == "nccl"]
    m_points = M_POINTS if not args.quick else (1, 256, 2048)

    # 1. forward bits: owner (every leg) == legacy == TP=1, with skew
    for sname, (K, N) in SHAPES.items():
        for pname, plan in _plans(world):
            if K % plan.num_leaves:
                continue
            for M in m_points:
                x, w = _operands(K, N, M, 99 + M, dev)
                xs, ws = shard(x, w, K)
                bias = (torch.randn(N, device=dev, dtype=torch.bfloat16) * 0.1) if M == 3 else None
                for out_dtype in ((torch.bfloat16, torch.float32) if M in (3, 256) else (torch.bfloat16,)):
                    ref1 = row_parallel_linear(x, w, bias, plan=plan, tp_size=1, k_full=K, out_dtype=out_dtype)
                    ST.on = False  # this group runs the legacy site (state() returns None) ...
                    legacy = site(xs, ws, plan, K, out_dtype, bias)
                    ST.on = True  # ... and the owner path again
                    check(f"legacy {sname} {pname} M={M} {out_dtype} vs TP1", legacy, ref1)
                    for lname, tr, ob, lm, ts in legs:
                        _force(owner, group, dev, transport=tr, oneshot_bytes=ob, leaf_min_m=lm, twoshot=ts)
                        for jit in (0.0, 0.02, -0.02):
                            _skew(rank, world, jit)
                            got = site(xs, ws, plan, K, out_dtype, bias)
                            check(f"owner[{lname}] {sname} {pname} M={M} {out_dtype} jit={jit} vs legacy", got,
                                  legacy)
                            check(f"owner[{lname}] {sname} {pname} M={M} {out_dtype} jit={jit} vs TP1", got, ref1)
            if rank == 0:
                print(f"{sname} {pname}: {checks - fails}/{checks} bitwise so far; owner={owner.counts()['admitted']} "
                      f"admitted classes, legacy fallbacks={owner.counts()['legacy_fallbacks']}", flush=True)

    os.environ["ISOEXEC_PIK_OWNER_ROUTE"] = "1"
    # 2. generic routed tree_all_reduce (odd n -> vec 1; fp32 and bf16 partials; both roots)
    for shape in ((5, 333), (64, 5120), (1, 7)):
        for dt in (torch.float32, torch.bfloat16):
            for rd in (None, torch.float32, torch.bfloat16):
                torch.manual_seed(1000 + rank)
                p = torch.randn(*shape, device=dev).to(dt)
                ref = AR._tree_all_reduce_legacy(p, group, None, None, rd)
                for lname, tr, ob, lm, ts in legs[:3]:
                    _force(owner, group, dev, transport=tr, oneshot_bytes=ob, leaf_min_m=lm, twoshot=ts)
                    got = AR.tree_all_reduce(p, group=group, root_dtype=rd)
                    check(f"routed tree_all_reduce[{lname}] {shape} {dt} root={rd}", got, ref)

    # 3. CUDA graphs: several sites (one-shot + two-shot classes) captured, replayed with fresh data,
    #    interleaved with eager calls of other shapes (epochs shared by graph and eager launches).
    if not args.no_p2p:
        from pik.plan import ReductionPlan  # type: ignore

        plan = ReductionPlan(num_leaves=8, leaf_dtype=torch.bfloat16)
        # leaf wire from M=64: the graphs hold node-wire (M=1, 32) and leaf-wire (M=256) sites together
        _force(owner, group, dev, transport="p2p", oneshot_bytes=owner._DEFAULT_ONESHOT_BYTES.get(world, 1 << 17),
               leaf_min_m=64, twoshot=3)
        K1, N1 = SHAPES["o_proj"]
        K2, N2 = SHAPES["fc2"]
        Ms = (1, 32, 256)
        static, weights = {}, {}
        for M in Ms:
            x1, w1 = _operands(K1, N1, M, 5 + M, dev)
            x2, w2 = _operands(K2, N2, M, 6 + M, dev)
            static[M] = (shard(x1, w1, K1), shard(x2, w2, K2))
            weights[M] = (w1, w2)
        for M in Ms:  # eager warmup (admission, pools, schedules) BEFORE capture
            (a1, b1), (a2, b2) = static[M]
            site(a1, b1, plan, K1, torch.bfloat16)
            site(a2, b2, plan, K2, torch.bfloat16)
        torch.cuda.synchronize()
        graphs = {}
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for M in Ms:
                (a1, b1), (a2, b2) = static[M]
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    y1 = site(a1, b1, plan, K1, torch.bfloat16)
                    y2 = site(a2, b2, plan, K2, torch.bfloat16)
                    y3 = site(a1, b1, plan, K1, torch.bfloat16)
                graphs[M] = (g, y1, y2, y3)
        torch.cuda.current_stream().wait_stream(s)
        for trial in range(args.graph_trials):
            for M in Ms:
                (a1, b1), (a2, b2) = static[M]
                w1, w2 = weights[M]
                x1 = _operands(K1, N1, M, 1000 * trial + M, dev)[0]
                x2 = _operands(K2, N2, M, 2000 * trial + M, dev)[0]
                a1.copy_(shard(x1, w1, K1)[0])
                a2.copy_(shard(x2, w2, K2)[0])
                _skew(rank, world, 0.01 if trial % 2 else -0.01)
                g, y1, y2, y3 = graphs[M]
                g.replay()
                e1 = site(a1, b1, plan, K1, torch.bfloat16)  # eager, same data, between replays
                e2 = site(a2, b2, plan, K2, torch.bfloat16)
                torch.cuda.synchronize()
                r1 = row_parallel_linear(x1, w1, None, plan=plan, tp_size=1, k_full=K1, out_dtype=torch.bfloat16)
                r2 = row_parallel_linear(x2, w2, None, plan=plan, tp_size=1, k_full=K2, out_dtype=torch.bfloat16)
                check(f"graph M={M} trial={trial} o_proj", y1, r1)
                check(f"graph M={M} trial={trial} fc2", y2, r2)
                check(f"graph M={M} trial={trial} o_proj#2", y3, r1)
                check(f"eager-after-graph M={M} trial={trial}", e1, r1)
                check(f"eager-after-graph M={M} trial={trial} fc2", e2, r2)
        if rank == 0:
            print(f"graphs: {checks - fails}/{checks} bitwise so far", flush=True)

    # 4. stress: random shapes (same sequence on every rank), random per-rank skew, both branches
    from pik.plan import ReductionPlan  # type: ignore

    plan = ReductionPlan(num_leaves=8, leaf_dtype=torch.bfloat16)
    K, N = SHAPES["o_proj"]
    refs = {}
    for M in (1, 8, 64, 200, 256, 1024):
        x, w = _operands(K, N, M, 77 + M, dev)
        refs[M] = (shard(x, w, K), row_parallel_linear(x, w, None, plan=plan, tp_size=1, k_full=K,
                                                       out_dtype=torch.bfloat16))
    shared_rng = random.Random(4242)
    local_rng = random.Random(rank)
    for lname, tr, ob, lm, ts in legs:
        _force(owner, group, dev, transport=tr, oneshot_bytes=ob if ob is not None else 1 << 18, leaf_min_m=lm,
               twoshot=ts)
        outs = []
        for it in range(args.stress):
            M = shared_rng.choice(sorted(refs))
            if local_rng.random() < 0.3:
                torch.cuda._sleep(local_rng.randint(0, 200_000))
            (xs, ws), r = refs[M]
            outs.append((M, site(xs, ws, plan, K, torch.bfloat16)))
        torch.cuda.synchronize()
        bad = sum(_diff(y, refs[M][1]) != 0 for M, y in outs)
        checks += 1
        if bad:
            fails += 1
            print(f"[rank {rank}] FAIL stress[{lname}]: {bad}/{len(outs)} calls differ", flush=True)

    # 5. a split flag disables everywhere instead of hanging
    owner.set_enabled(rank == 0)
    owner.reset()
    st = owner.state(group, dev)
    checks += 1
    if st is not None:
        fails += 1
        print(f"[rank {rank}] FAIL split flag did not disable the owner path", flush=True)
    owner.set_enabled(True)
    owner.reset()

    t = torch.tensor([fails, checks], device=dev)
    dist.all_reduce(t)
    if rank == 0:
        print(f"MULTIRANK RESULT world={world}: {int(t[1] - t[0])}/{int(t[1])} bitwise-equal checks "
              f"({'PASS' if int(t[0]) == 0 else 'FAIL'})", flush=True)
        print(f"owner counts (rank 0): {owner.counts()}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if int(t[0]):
        sys.exit(1)


def _port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-p2p", action="store_true")
    ap.add_argument("--stress", type=int, default=400)
    ap.add_argument("--graph-trials", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    if args.sim:
        rc = sim()
    else:
        mp.spawn(worker, args=(args.world, _port(), args), nprocs=args.world, join=True)
        rc = 0
    print(f"done in {time.time() - t0:.0f}s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
