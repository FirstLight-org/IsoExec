#!/usr/bin/env python
"""Bitwise battery for the owner Lamport all-reduce and its fused residual-add + RMSNorm epilogue
(pik/owner.py ``lamport_*`` / ``RootHandoff``, csrc/owner_lamport.cu).

``--sim`` (ONE GPU):
  * the hardware fact the sentinel substitution rests on: an fp32 tree add with a NaN operand returns the same
    bits whatever the NaN's payload (every bf16 / fp32 NaN pattern, both tree positions);
  * the kernel's arithmetic in local mode for simulated worlds W in {2, 4, 8} (production GEMM leaves/nodes and
    raw bit patterns) equals the barrier owner kernel / legacy tree, and with the fused epilogue equals the root
    followed by ``norms_sm90`` add_rms_norm (and the TP=1 projection followed by it), for every recipe-v1 width,
    bf16/fp32 gamma, plain/zero-centred, raw-bit residuals.

Multi-rank (``--world N``, one GPU per rank): row-parallel sites through the Lamport protocol (one-shot and
two-shot) equal the legacy pik path and TP=1; the deferred root + fused add+norm pair equals the unfused site;
NaN patterns equal to the sentinel cross the wire exactly; CUDA-graph capture/replay interleaved with eager
calls of other sizes (generations rotate through graph and eager launches); rank skew; a randomized stress loop;
PDL on and off; handoff misuse raises; a split Lamport flag disables it on every rank instead of hanging.

    CUDA_VISIBLE_DEVICES=0 python isoexec/ops/collectives/tests/pik_owner_lamport_gpu.py --sim
    CUDA_VISIBLE_DEVICES=0,1,2,3 python isoexec/ops/collectives/tests/pik_owner_lamport_gpu.py --world 4
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

SHAPES = {"o_proj": (6144, 5120), "fc2": (17408, 5120), "w1024": (2048, 1024), "ragged": (1024, 1000)}
M_POINTS = (1, 3, 26, 128, 175, 256, 300, 2048)
EPS = 1e-6


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _diff(a: torch.Tensor, b: torch.Tensor) -> int:
    """Elements whose bit patterns differ -- exact, NaN payloads included."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    return int((_bits(a) != _bits(b)).sum())


def _plans(world):
    from pik.plan import ReductionPlan  # type: ignore

    out, seen = [], set()
    for name, g, dt in (("bf16_G8", 8, torch.bfloat16), ("fp32_G8", 8, None),
                        (f"bf16_G{world}", world, torch.bfloat16)):
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


def _norm_operands(M, N, seed, dev, wdt=torch.bfloat16, raw=False):
    g = torch.Generator(device=dev).manual_seed(seed)
    if raw:  # every bf16 bit pattern class, specials salted in
        r = torch.randint(-(1 << 15), 1 << 15, (M, N), device=dev, dtype=torch.int64, generator=g).to(torch.int16)
        r = r.view(torch.bfloat16)
        sp = torch.tensor([0.0, -0.0, float("inf"), float("-inf"), 1e-40, -1e-40, 3.0e38, float("nan")],
                          dtype=torch.bfloat16, device=dev)
        r[:, :8] = sp
    else:
        r = torch.randn(M, N, device=dev, dtype=torch.bfloat16, generator=g)
    w = (torch.randn(N, device=dev, generator=g) * 0.1).to(wdt)
    return r.contiguous(), w


def _add_rms(x, r, w, zc):
    from isoexec.ops.norms.norms_sm90.ops import add_rms_norm_fused

    y, h = add_rms_norm_fused(x, r, w, EPS, zc)
    return y, h


# ------------------------------------------------------------------------------------------------ simulation


def sim() -> int:
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik

    ensure_pik()
    from pik import owner  # type: ignore
    from pik.gemm import ti_gemm, ti_gemm_leaves  # type: ignore
    from pik.linear import row_parallel_linear  # type: ignore

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    fails = checks = 0

    def check(tag, got, ref):
        nonlocal fails, checks
        d = _diff(got, ref)
        checks += 1
        if d:
            fails += 1
            print(f"FAIL {tag}: {d if d >= 0 else 'shape/dtype'} differ", flush=True)

    # 1. NaN payloads never reach the root's bits (the sentinel substitution's premise).
    for dt, it, nbits in ((torch.bfloat16, torch.int16, 16), (torch.float32, torch.int32, 32)):
        if nbits == 16:
            pats = torch.arange(-(1 << 15), 1 << 15, device=dev, dtype=torch.int32).to(torch.int16)
        else:
            gg = torch.Generator(device=dev).manual_seed(3)
            pats = torch.randint(-(1 << 31), (1 << 31) - 1, (1 << 20,), device=dev, dtype=torch.int64,
                                 generator=gg).to(torch.int32)
            pats = torch.cat([pats, torch.tensor([-1, 0x7FFFFFFF, 0x7FC00000, -0x400000, 0x7F800001],
                                                 device=dev, dtype=torch.int32)])
        vals = pats.view(dt)
        nan = torch.isnan(vals.float())
        nans = vals[nan]
        canon = torch.full_like(nans, float("nan"))
        other = torch.randn(nans.numel(), device=dev).to(dt)
        for rd in (torch.float32, torch.bfloat16):
            for order in (0, 1):
                a = [nans.view(1, -1).contiguous(), other.view(1, -1).contiguous()]
                b = [canon.view(1, -1).contiguous(), other.view(1, -1).contiguous()]
                if order:
                    a, b = a[::-1], b[::-1]
                check(f"nan-payload {dt} root={rd} order={order}", owner.reduce_local(a, rd), owner.reduce_local(b, rd))

    # 2. local-mode root arithmetic == the barrier owner kernel == TP=1 (production GEMM planes)
    for sname, (K, N) in SHAPES.items():
        for M in (1, 3, 128, 300):
            x, w = _operands(K, N, M, 1234 + M, dev)
            for pname, plan in _plans(8):
                if K % plan.num_leaves:
                    continue
                ref = row_parallel_linear(x, w, None, plan=plan, tp_size=1, k_full=K, out_dtype=torch.bfloat16)
                root_dt = owner.root_dtype_for(plan, None, torch.bfloat16)
                W = 2
                while W <= min(8, plan.num_leaves):
                    m = plan.leaves_per_rank(W)
                    kl = K // W
                    sh = [(x[:, r * kl:(r + 1) * kl].contiguous(), w[:, r * kl:(r + 1) * kl].contiguous())
                          for r in range(W)]
                    planes, wdt = owner.wire(plan, W)
                    if plan.bf16_leaves and planes == m:
                        ops = [ti_gemm_leaves(xr, wr, plan=plan, tp_size=W, k_full=K,
                                              out=torch.empty((m, M, N), device=dev, dtype=torch.bfloat16))
                               for xr, wr in sh]
                    else:
                        ops = [ti_gemm(xr, wr, plan=plan, tp_size=W, tp_rank=r, k_full=K).reshape(1, M, N)
                               .to(wdt).contiguous() for r, (xr, wr) in enumerate(sh)]
                    if N % 8 == 0:
                        got, _ = owner.lamport_local(ops, root_dt)
                        check(f"sim-root {sname} M={M} {pname} W={W}", got.to(torch.bfloat16), ref)
                        bar = owner.reduce_local([o.reshape(o.shape[0], -1) for o in ops], root_dt).view(M, N)
                        check(f"sim-root-vs-barrier {sname} M={M} {pname} W={W}", got, bar)
                        if N % 256 == 0 and root_dt == torch.bfloat16:
                            for zc in (True, False):
                                for wt in (torch.bfloat16, torch.float32):
                                    r, g = _norm_operands(M, N, 77 + M + W, dev, wt)
                                    h, y = owner.lamport_local(ops, root_dt, resid=r, weight=g, eps=EPS,
                                                               norm_mode=2 if zc else 1)
                                    ry, rh = _add_rms(ref, r, g, zc)
                                    tag = f"sim-norm {sname} M={M} {pname} W={W} zc={zc} w={wt}"
                                    check(tag + " h", h, rh)
                                    check(tag + " y", y, ry)
                    W *= 2
        print(f"sim {sname}: {checks - fails}/{checks} bitwise", flush=True)

    # 3. raw bit patterns: planes and residuals of every class
    g = torch.Generator(device=dev).manual_seed(7)
    for W, C, dt in ((2, 1, torch.float32), (4, 2, torch.bfloat16), (4, 1, torch.float32), (8, 1, torch.bfloat16),
                     (2, 2, torch.bfloat16), (16, 1, torch.bfloat16)):
        for M, N in ((5, 5120), (33, 256), (2, 8192), (7, 1000)):
            it = torch.int16 if dt == torch.bfloat16 else torch.int32
            lo, hi = (-(1 << 15), 1 << 15) if it == torch.int16 else (-(1 << 31), (1 << 31) - 1)
            raw = torch.randint(lo, hi, (W * C, M, N), device=dev, dtype=torch.int64, generator=g).to(it)
            sp = torch.tensor([0.0, -0.0, float("inf"), float("-inf"), 1e-40, -1e-40, 3.0e38, -3.0e38],
                              dtype=dt, device=dev)
            raw[:, :, :8] = sp.view(it)
            raw[:, 0, 8:16] = -1  # the sentinel pattern itself
            planes = raw.view(dt)
            ops = [planes[r * C:(r + 1) * C].contiguous() for r in range(W)]
            for rd in ((torch.float32, torch.bfloat16) if dt == torch.float32 else (torch.bfloat16,)):
                got, _ = owner.lamport_local(ops, rd)
                bar = owner.reduce_local([o.reshape(C, -1) for o in ops], rd).view(M, N)
                check(f"sim-raw W={W} C={C} {dt} M={M} N={N} root={rd}", got, bar)
                if rd == torch.bfloat16 and N % 256 == 0:
                    r, gm = _norm_operands(M, N, 5 + W, dev, torch.float32, raw=True)
                    h, y = owner.lamport_local(ops, rd, resid=r, weight=gm, eps=EPS, norm_mode=2)
                    ry, rh = _add_rms(bar, r, gm, True)
                    check(f"sim-raw-norm W={W} C={C} M={M} N={N} h", h, rh)
                    check(f"sim-raw-norm W={W} C={C} M={M} N={N} y", y, ry)
    print(f"SIM RESULT: {checks - fails}/{checks} bitwise-equal checks", flush=True)
    return 1 if fails else 0


# ------------------------------------------------------------------------------------------------ multi-rank


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
    from pik import owner  # type: ignore
    from pik.linear import row_parallel_linear  # type: ignore

    owner.set_enabled(True)
    owner.reset()
    ST = owner.state(group, dev)
    assert ST is not None and ST.transport == "p2p", "owner p2p path did not engage"
    CFG = owner.lamport_state(ST, group, dev)
    assert CFG is not None, "Lamport path did not engage"
    OFF = owner._LamportCfg(False, False, 0, 0, 0, False, None)
    fails = checks = 0

    def check(tag, got, ref):
        nonlocal fails, checks
        d = _diff(got, ref)
        checks += 1
        if d:
            fails += 1
            print(f"[rank {rank}] FAIL {tag}: {d if d >= 0 else 'shape/dtype'} differ", flush=True)

    def site(xs, ws, plan, K, out_dtype=torch.bfloat16):
        return row_parallel_linear(xs, ws, None, plan=plan, tp_size=world, tp_rank=rank, k_full=K, group=group,
                                   out_dtype=out_dtype)

    def fused_site(xs, ws, plan, K, r, g, zc, handoff=None):
        """The engine's paired bindings: deferred row-parallel root, then the fused add+norm."""
        handoff = handoff or owner.RootHandoff("battery")
        nm = 2 if zc else 1
        x = owner.row_parallel_deferred(xs, ws, None, plan=plan, tp_size=world, tp_rank=rank, k_full=K,
                                        group=group, out_dtype=torch.bfloat16, handoff=handoff)
        if x is None:
            x = site(xs, ws, plan, K)
        return owner.add_norm_deferred(x, r, g, EPS, norm_mode=nm, impl="sm90_add_rms_v1",
                                       unfused=lambda a, b, c, d: _add_rms(a, b, c, zc), handoff=handoff)

    def shard(x, w, K):
        kl = K // world
        return x[:, rank * kl:(rank + 1) * kl].contiguous(), w[:, rank * kl:(rank + 1) * kl].contiguous()

    def legacy_site(xs, ws, plan, K):
        ST.on = False
        try:
            return site(xs, ws, plan, K)
        finally:
            ST.on = True

    big = 1 << 62
    legs = [("lamport_1shot", big, True), ("lamport_2shot", 0, True), ("lamport_2shot_nopdl", 0, False)]
    m_points = M_POINTS if not args.quick else (1, 175, 2048)

    # 1. roots and fused add+norm: Lamport (each leg) == legacy == TP=1, with skew
    for sname, (K, N) in SHAPES.items():
        for pname, plan in _plans(world):
            if K % plan.num_leaves:
                continue
            for M in m_points:
                x, w = _operands(K, N, M, 99 + M, dev)
                xs, ws = shard(x, w, K)
                ref1 = row_parallel_linear(x, w, None, plan=plan, tp_size=1, k_full=K, out_dtype=torch.bfloat16)
                leg_root = legacy_site(xs, ws, plan, K)
                check(f"legacy {sname} {pname} M={M} vs TP1", leg_root, ref1)
                fusable = N % 256 == 0
                if fusable:
                    zc = M % 2 == 1
                    r, g = _norm_operands(M, N, 31 + M, dev, torch.float32 if M == 26 else torch.bfloat16)
                    ry, rh = _add_rms(ref1, r, g, zc)
                for lname, ob, pdl in legs:
                    CFG.oneshot_bytes, CFG.pdl = ob, pdl
                    for jit in (0.0, 0.02, -0.02):
                        _skew(rank, world, jit)
                        got = site(xs, ws, plan, K)
                        check(f"root[{lname}] {sname} {pname} M={M} jit={jit} vs legacy", got, leg_root)
                        if fusable:
                            _skew(rank, world, -jit)
                            y, h = fused_site(xs, ws, plan, K, r, g, zc)
                            check(f"fused[{lname}] {sname} {pname} M={M} jit={jit} h", h, rh)
                            check(f"fused[{lname}] {sname} {pname} M={M} jit={jit} y", y, ry)
            if rank == 0:
                print(f"{sname} {pname}: {checks - fails}/{checks} bitwise; lamport {owner.lamport_counts()}",
                      flush=True)
    CFG.oneshot_bytes, CFG.pdl = owner._DEFAULT_LAMPORT_ONESHOT_BYTES.get(world, 1 << 15), True

    # 2. the sentinel and other NaN payloads cross the wire exactly (raw planes straight into the protocol)
    gen = torch.Generator(device=dev).manual_seed(1000 + rank)
    for C, dt in ((2, torch.bfloat16), (1, torch.float32), (1, torch.bfloat16)):
        for M, N in ((3, 5120), (64, 1024), (40, 1000)):
            it = torch.int16 if dt == torch.bfloat16 else torch.int32
            lo, hi = (-(1 << 15), 1 << 15) if it == torch.int16 else (-(1 << 31), (1 << 31) - 1)
            raw = torch.randint(lo, hi, (C, M, N), device=dev, dtype=torch.int64, generator=gen).to(it)
            raw[:, :, :16] = -1  # sentinel
            staged = owner.stage(ST, C, M, N, dt, dev)
            for rd in ((torch.float32, torch.bfloat16) if dt == torch.float32 else (torch.bfloat16,)):
                # admit each class on clean operands first (admission compares with the legacy path, whose NaN
                # root pattern is torch's, not the kernels'), then push raw patterns through the admitted class
                staged.copy_(torch.randn(C, M, N, device=dev).to(dt))
                for lname, ob, pdl in legs:
                    CFG.oneshot_bytes, CFG.pdl = ob, pdl
                    owner.lamport_all_reduce(staged, group, rd, st=ST)
                staged.copy_(raw.view(dt))
                # the barrier owner kernel (production bits; its NaN root is the cvt.rn canonical 0x7FFF, which
                # torch's own bf16 cast does not produce, so the legacy path is not an exact NaN reference)
                ref = owner._run_p2p(ST, staged.reshape(C, -1), rd, 3).view(M, N).clone()
                for lname, ob, pdl in legs:
                    CFG.oneshot_bytes, CFG.pdl = ob, pdl
                    got = owner.lamport_all_reduce(staged, group, rd, st=ST)
                    if got is None:
                        checks += 1
                        fails += 1
                        print(f"[rank {rank}] FAIL raw[{lname}] C={C} {dt} M={M} N={N}: Lamport refused", flush=True)
                        continue
                    check(f"raw[{lname}] C={C} {dt} M={M} N={N} root={rd}", got, ref)
    CFG.oneshot_bytes, CFG.pdl = owner._DEFAULT_LAMPORT_ONESHOT_BYTES.get(world, 1 << 15), True

    # 3. CUDA graphs: fused and root sites of several sizes (both branches) captured together, replayed with
    #    fresh data a number of times not divisible by 3, interleaved with eager calls of other sizes.
    from pik.plan import ReductionPlan  # type: ignore

    plan = ReductionPlan(num_leaves=8, leaf_dtype=torch.bfloat16)
    K1, N1 = SHAPES["o_proj"]
    K2, N2 = SHAPES["fc2"]
    Ms = (1, 26, 175, 256)
    static = {}
    for M in Ms:
        x1, w1 = _operands(K1, N1, M, 5 + M, dev)
        x2, w2 = _operands(K2, N2, M, 6 + M, dev)
        r, g = _norm_operands(M, N1, 9 + M, dev)
        static[M] = [shard(x1, w1, K1), shard(x2, w2, K2), r, g, w1, w2]
    for M in Ms:  # eager warmup (admission) before capture
        (a1, b1), (a2, b2), r, g, _, _ = static[M]
        fused_site(a1, b1, plan, K1, r, g, True)
        site(a2, b2, plan, K2)
    torch.cuda.synchronize()
    graphs = {}
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for M in Ms:
            (a1, b1), (a2, b2), r, g, _, _ = static[M]
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                y1, h1 = fused_site(a1, b1, plan, K1, r, g, True)
                z2 = site(a2, b2, plan, K2)
                y3, h3 = fused_site(a2, b2, plan, K2, h1, g, False)
            graphs[M] = (gr, y1, h1, z2, y3, h3)
    torch.cuda.current_stream().wait_stream(s)
    for trial in range(args.graph_trials):
        for M in Ms:
            (a1, b1), (a2, b2), r, g, w1, w2 = static[M]
            x1 = _operands(K1, N1, M, 1000 * trial + M, dev)[0]
            x2 = _operands(K2, N2, M, 2000 * trial + M, dev)[0]
            a1.copy_(shard(x1, w1, K1)[0])
            a2.copy_(shard(x2, w2, K2)[0])
            r.copy_(_norm_operands(M, N1, 3000 * trial + M, dev)[0])
            _skew(rank, world, 0.01 if trial % 2 else -0.01)
            gr, y1, h1, z2, y3, h3 = graphs[M]
            for _ in range(1 + trial % 2):
                gr.replay()
            e1 = site(a1, b1, plan, K1)  # eager between replays (another size class too)
            ef = fused_site(a2, b2, plan, K2, r, g, False) if trial % 3 == 0 else None
            torch.cuda.synchronize()
            r1 = row_parallel_linear(x1, w1, None, plan=plan, tp_size=1, k_full=K1, out_dtype=torch.bfloat16)
            r2 = row_parallel_linear(x2, w2, None, plan=plan, tp_size=1, k_full=K2, out_dtype=torch.bfloat16)
            ry1, rh1 = _add_rms(r1, r, g, True)
            ry3, rh3 = _add_rms(r2, rh1, g, False)
            check(f"graph M={M} t={trial} y1", y1, ry1)
            check(f"graph M={M} t={trial} h1", h1, rh1)
            check(f"graph M={M} t={trial} z2", z2, r2)
            check(f"graph M={M} t={trial} y3", y3, ry3)
            check(f"graph M={M} t={trial} h3", h3, rh3)
            check(f"eager-after-graph M={M} t={trial}", e1, r1)
            if ef is not None:
                rye, rhe = _add_rms(r2, r, g, False)
                check(f"eager-fused-after-graph M={M} t={trial} y", ef[0], rye)
                check(f"eager-fused-after-graph M={M} t={trial} h", ef[1], rhe)
    if rank == 0:
        print(f"graphs: {checks - fails}/{checks} bitwise so far", flush=True)

    # 4. stress: random sizes (same sequence on every rank), random per-rank skew, fused and root interleaved
    K, N = SHAPES["o_proj"]
    refs = {}
    for M in (1, 8, 26, 64, 175, 200, 256, 409, 1024):
        x, w = _operands(K, N, M, 77 + M, dev)
        r, g = _norm_operands(M, N, 78 + M, dev)
        ref = row_parallel_linear(x, w, None, plan=plan, tp_size=1, k_full=K, out_dtype=torch.bfloat16)
        refs[M] = (shard(x, w, K), ref, r, g, _add_rms(ref, r, g, True))
    shared_rng = random.Random(4242)
    local_rng = random.Random(rank)
    outs = []
    for it in range(args.stress):
        M = shared_rng.choice(sorted(refs))
        CFG.oneshot_bytes = shared_rng.choice((0, 1 << 20, owner._DEFAULT_LAMPORT_ONESHOT_BYTES.get(world, 1 << 15)))
        CFG.pdl = shared_rng.random() < 0.8
        if local_rng.random() < 0.3:
            torch.cuda._sleep(local_rng.randint(0, 200_000))
        (xs, ws), ref, r, g, _ = refs[M]
        if shared_rng.random() < 0.5:
            outs.append((M, "root", site(xs, ws, plan, K)))
        else:
            outs.append((M, "fused", fused_site(xs, ws, plan, K, r, g, True)))
    torch.cuda.synchronize()
    bad = 0
    for M, kind, got in outs:
        _, ref, _, _, (ry, rh) = refs[M]
        bad += (_diff(got, ref) != 0) if kind == "root" else (_diff(got[0], ry) != 0 or _diff(got[1], rh) != 0)
    checks += 1
    if bad:
        fails += 1
        print(f"[rank {rank}] FAIL stress: {bad}/{len(outs)} calls differ", flush=True)
    CFG.oneshot_bytes, CFG.pdl = owner._DEFAULT_LAMPORT_ONESHOT_BYTES.get(world, 1 << 15), True

    # 5. handoff misuse is a wiring error, never silent garbage
    (xs, ws), ref, r, g, _ = refs[8]
    ho = owner.RootHandoff("battery-misuse")
    owner.row_parallel_deferred(xs, ws, None, plan=plan, tp_size=world, tp_rank=rank, k_full=K, group=group,
                                out_dtype=torch.bfloat16, handoff=ho)
    checks += 1
    try:
        owner.add_norm_deferred(ref.clone(), r, g, EPS, norm_mode=2, impl="sm90_add_rms_v1",
                                unfused=lambda a, b, c, d: _add_rms(a, b, c, True), handoff=ho)
        fails += 1
        print(f"[rank {rank}] FAIL handoff misuse did not raise", flush=True)
    except RuntimeError:
        pass
    # the deferred site left unconsumed must not have advanced the protocol: a normal call still agrees
    check("after-misuse root", site(xs, ws, plan, K), ref)

    # 6. Lamport off on this state -> the barrier owner kernel serves every class, same bits
    ST.lamport = OFF
    check("lamport-off root", site(xs, ws, plan, K), ref)
    ST.lamport = CFG

    # 7. a split Lamport flag disables it everywhere instead of hanging
    os.environ["ISOEXEC_PIK_OWNER_LAMPORT"] = "1" if rank == 0 else "0"
    owner.reset()
    st2 = owner.state(group, dev)
    cfg2 = owner.lamport_state(st2, group, dev)
    checks += 1
    if cfg2 is not None:
        fails += 1
        print(f"[rank {rank}] FAIL split Lamport flag did not disable the path", flush=True)
    check("split-flag root", site(xs, ws, plan, K), ref)
    os.environ["ISOEXEC_PIK_OWNER_LAMPORT"] = "1"

    t = torch.tensor([fails, checks], device=dev)
    dist.all_reduce(t)
    if rank == 0:
        print(f"MULTIRANK RESULT world={world}: {int(t[1] - t[0])}/{int(t[1])} bitwise-equal checks "
              f"({'PASS' if int(t[0]) == 0 else 'FAIL'})", flush=True)
        print(f"lamport counts (rank 0): {owner.lamport_counts()}", flush=True)
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
    ap.add_argument("--stress", type=int, default=400)
    ap.add_argument("--graph-trials", type=int, default=5)
    args = ap.parse_args()
    os.environ.setdefault("ISOEXEC_PIK_OWNER_AR", "1")
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
