#!/usr/bin/env python
"""Bitwise battery for pik's comm lane (pik/lane.py): pipelined row-parallel sites, early dgrad tree sums
under wgrad, physical-identity P2P admission, and private (non-symmetric) outputs.

``--sim`` (ONE GPU): the pipelined site's arithmetic at every world W in {1, 2, 4, 8}: each virtual rank's
leaf planes from the production leaf GEMM on ROW CHUNKS (the pipeline's cuts, with and without a persistent
CTA cap) reduced by the owner kernel in local mode must equal, bit for bit, the unchunked planes and the
TP=1 row-parallel projection of the same global operands.

Multi-rank (``--world N``, one GPU per rank; ``--ray-sim`` makes every rank see ONLY its own GPU as
``cuda:0``, as Ray trainer actors do):
  * the physical-identity probe admits the group (P2P) under both launch styles;
  * pipelined row-parallel (chunks 2/3/4, CTA reservation 0/32, p2p and nccl transports) == unpipelined
    owner site == TP=1 projection, bitwise, over decode..prefill M;
  * tp_copy -> mm forward/backward with the early dgrad tree sum: dx and dw == the serial schedule, for a
    sole consumer (consumed early), two consumers and an extra in-place contribution (discarded early,
    still exact), under activation checkpointing (reentrant and not), with SP (seq all-gather adjoint);
  * every gradient handed to autograd is a caching-allocator tensor: ``untyped_storage().resize_(0)`` on it
    (what verl's checkpoint backward does) is legal;
  * a randomized interleaving of pipelined sites, serial sites, routed all-reduces and SP reduce-scatters on
    the caller's stream and the lane stays exact (fences keep symmetric buffers single-owner).

    CUDA_VISIBLE_DEVICES=0 python isoexec/ops/collectives/tests/pik_overlap_gpu.py --sim
    CUDA_VISIBLE_DEVICES=0,1,2,3 python isoexec/ops/collectives/tests/pik_overlap_gpu.py --world 4
    CUDA_VISIBLE_DEVICES=0,1,2,3 python isoexec/ops/collectives/tests/pik_overlap_gpu.py --world 4 --ray-sim
"""

from __future__ import annotations

import argparse
import os
import random
import socket
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, *[".."] * 4))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

SHAPES = {"o_proj": (6144, 5120), "fc2": (17408, 5120)}
M_POINTS = (1, 300, 2048, 2049, 4097, 5791)


def _bits(t):
    t = t.contiguous()
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _diff(a, b) -> int:
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    return int((_bits(a) != _bits(b)).sum())


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
    from pik import pipeline as PIPE  # type: ignore
    from pik.gemm import ti_gemm_leaves  # type: ignore
    from pik.linear import row_parallel_linear  # type: ignore
    from pik.plan import ReductionPlan  # type: ignore

    try:
        from isoexec.ops.mm.mm_sm90 import ops as sm90
    except Exception:  # noqa: BLE001
        sm90 = None
    dev = torch.device("cuda", 0)
    plan = ReductionPlan(num_leaves=8, leaf_dtype=torch.bfloat16)
    fails = checks = 0
    for name, (K, N) in SHAPES.items():
        for M in (2048, 4097, 5791):
            x, w = _operands(K, N, M, 7 + M, dev)
            ref = row_parallel_linear(x, w, None, plan=plan, tp_size=1, tp_rank=0, k_full=K, group=None,
                                      out_dtype=torch.bfloat16)
            for W in (1, 2, 4, 8):
                kl = K // W
                m = plan.leaves_per_rank(W)
                for chunks in (1, 2, 3, 4, 5, 7, 8):
                    for cap in ((0, 100) if sm90 is not None else (0,)):
                        prev = sm90.set_sm_limit(cap) if sm90 is not None else None
                        try:
                            bounds = PIPE.pipeline_bounds(M, chunks, 2048)
                            out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
                            for r0, r1 in bounds:
                                planes = [torch.empty(m, r1 - r0, N, dtype=torch.bfloat16, device=dev)
                                          for _ in range(W)]
                                for r in range(W):
                                    ti_gemm_leaves(x[r0:r1, r * kl:(r + 1) * kl], w[:, r * kl:(r + 1) * kl].contiguous(),
                                                   plan=plan, tp_size=W, k_full=K, out=planes[r])
                                owner.reduce_local([p.view(m, -1) for p in planes], torch.bfloat16,
                                                   out=out[r0:r1].view(-1))
                        finally:
                            if sm90 is not None:
                                sm90.set_sm_limit(prev)
                        d = _diff(out, ref)
                        checks += 1
                        if d:
                            fails += 1
                            print(f"FAIL sim {name} M={M} W={W} chunks={chunks} cap={cap}: {d} elements differ")
    print(f"[pik_overlap --sim] {checks - fails}/{checks} configurations bitwise == TP1", flush=True)
    return 1 if fails else 0


# ---------------------------------------------------------------------------------------------- multi-rank


def worker(rank: int, world: int, port: int, args, fails) -> None:
    # Overlap engages only with more than one hardware queue (pik.lane.overlap_agreed); this battery tests
    # the overlapped schedules, so it runs with the CUDA default whatever the caller's environment says.
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
    if args.ray_sim:
        phys = os.environ.get("_PIK_PHYS", ",".join(str(i) for i in range(world))).split(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = phys[rank]
        local = 0
    else:
        local = rank
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)
    group = dist.group.WORLD

    from isoexec.ops.collectives.pik_bootstrap import ensure_pik

    ensure_pik()
    import pik.allreduce as AR  # type: ignore
    import pik.lane as LANE  # type: ignore
    import pik.owner as OWNER  # type: ignore
    import pik.pipeline as PIPE  # type: ignore
    from pik.linear import row_parallel_linear  # type: ignore
    from pik.plan import ReductionPlan  # type: ignore

    from isoexec.ops.collectives import tp_bytes as TB
    from isoexec.ops.collectives.pik_tp_invariant import select_trainer_transport
    from isoexec.ops.mm.trainer import mm_trainer

    bad = []

    def check(tag, got, ref):
        d = _diff(got, ref)
        if d:
            bad.append(f"rank{rank} {tag}: {d} elements differ")

    os.environ["ISOEXEC_PIK_OWNER_LEAF_MIN_M"] = "1"
    os.environ["ISOEXEC_PIK_OWNER_ROUTE"] = "1"  # the early dgrad sum runs only the owner P2P kernel
    OWNER.set_enabled(True)
    OWNER._DEFAULT_LEAF_MIN_M = 1
    sel = select_trainer_transport("TRAINER")
    if sel != "p2p_physical":
        bad.append(f"rank{rank} trainer transport selection {sel!r} != p2p_physical")
    if not AR.p2p_available(group):
        bad.append(f"rank{rank} physical probe REFUSED a single-node group (ray_sim={args.ray_sim})")
    plan = ReductionPlan(num_leaves=8, leaf_dtype=torch.bfloat16)
    try:
        from isoexec.ops.mm.mm_sm90 import ops as sm90
    except Exception:  # noqa: BLE001
        sm90 = None

    # ---- 1. pipelined row-parallel == unpipelined == TP1, both transports
    for transport in ("p2p", "nccl"):
        OWNER.reset()
        LANE._AGREED.clear()
        st = OWNER.state(group, dev)
        if transport == "nccl":
            st.transport, st.p2p = "nccl", None
        for name, (K, N) in SHAPES.items():
            kl = K // world
            for M in M_POINTS:
                x, w = _operands(K, N, M, 1000 + M, dev)
                ref = row_parallel_linear(x, w, None, plan=plan, tp_size=1, tp_rank=0, k_full=K, group=None,
                                          out_dtype=torch.bfloat16)
                xs, ws = x[:, rank * kl:(rank + 1) * kl].contiguous(), w[:, rank * kl:(rank + 1) * kl].contiguous()
                os.environ["ISOEXEC_PIK_OVERLAP"] = "0"
                LANE._AGREED.clear()
                serial = row_parallel_linear(xs, ws, None, plan=plan, tp_size=world, tp_rank=rank, k_full=K,
                                             group=group, out_dtype=torch.bfloat16).clone()
                check(f"{transport} serial {name} M={M} vs TP1", serial, ref)
                for chunks in (-1, 2, 3, 4):  # -1 = derived depth
                    for sms in (0, 32):
                        os.environ.update(ISOEXEC_PIK_OVERLAP="1", ISOEXEC_PIK_OVERLAP_CHUNKS=str(chunks),
                                          ISOEXEC_PIK_OVERLAP_COMM_SMS=str(sms))
                        LANE._AGREED.clear()
                        got = row_parallel_linear(xs, ws, None, plan=plan, tp_size=world, tp_rank=rank,
                                                  k_full=K, group=group, out_dtype=torch.bfloat16)
                        check(f"{transport} pipelined chunks={chunks} sms={sms} {name} M={M}", got, ref)
                        if AR._PRIVATE_OUTPUTS:
                            got.untyped_storage().resize_(0)  # a caching-allocator tensor, not a pool view
        torch.cuda.synchronize()
    OWNER.reset()
    LANE._AGREED.clear()
    os.environ.update(ISOEXEC_PIK_OVERLAP="1", ISOEXEC_PIK_OVERLAP_CHUNKS="4", ISOEXEC_PIK_OVERLAP_COMM_SMS="0")

    # ---- 2. tp_copy -> mm backward: early dgrad tree sum == serial, private grads
    tp_copy = TB._make_tp_copy(world, group)
    seq_gather = TB._make_seq_all_gather(world, group)

    def run(kind, overlap, T, Kin, Nout, seed, ckpt=None):
        os.environ["ISOEXEC_PIK_OVERLAP"] = "1" if overlap else "0"
        LANE._AGREED.clear()
        g = torch.Generator(device=dev).manual_seed(seed)
        rows = T // world if kind == "sp" else T
        x0 = torch.randn(rows, Kin, device=dev, dtype=torch.bfloat16, generator=g)
        x0 = x0 + rank * 0  # same input on every rank for tp; SP slices differ by construction below
        if kind == "sp":
            full = torch.randn(T, Kin, device=dev, dtype=torch.bfloat16, generator=g)
            x0 = full[rank * rows:(rank + 1) * rows].clone()
        w1 = (torch.randn(Nout, Kin, device=dev, dtype=torch.bfloat16, generator=g) * 0.05 * (rank + 1))
        w2 = (torch.randn(Nout, Kin, device=dev, dtype=torch.bfloat16, generator=g) * 0.05 * (rank + 1))
        dy = torch.randn(T, Nout, device=dev, dtype=torch.bfloat16, generator=g)
        x = x0.clone().requires_grad_(True)
        a = w1.clone().requires_grad_(True)
        b = w2.clone().requires_grad_(True)

        def body(xx):
            xc = seq_gather(xx) if kind == "sp" else tp_copy(xx)
            if kind == "two":  # two GEMM consumers: autograd sums their dgrads before the tree
                return mm_trainer(xc, a) + mm_trainer(xc, b)
            if kind == "inplace":  # a second, non-GEMM consumer adds into the gradient
                return mm_trainer(xc, a) + xc[:, :Nout] * 0.5
            return mm_trainer(xc, a)

        if ckpt is None:
            y = body(x)
        else:
            from torch.utils.checkpoint import checkpoint

            y = checkpoint(body, x, use_reentrant=(ckpt == "reentrant"))
        y.backward(dy)
        torch.cuda.synchronize()
        out = (x.grad.clone(), a.grad.clone(), b.grad.clone() if b.grad is not None else None)
        x.grad.untyped_storage().resize_(0)  # verl patch_backward: must be legal on every gradient
        return out

    for kind in ("sole", "two", "inplace", "sp"):
        for ckpt in (None, "reentrant", "nonreentrant"):
            for T in (256, 2048, 4096):
                Kin, Nout = 1024, 1024 if kind != "inplace" else 512
                # Serial first: it admits the class on the caller's stream (the lane never admits).
                slow = run(kind, False, T, Kin, Nout, 7 * T + 3, ckpt)
                before = dict(TB._LINK_COUNTS)
                fast = run(kind, True, T, Kin, Nout, 7 * T + 3, ckpt)
                after = dict(TB._LINK_COUNTS)
                for i, (p, q) in enumerate(zip(fast, slow)):
                    if p is not None:
                        check(f"bwd {kind} ckpt={ckpt} T={T} grad{i}", p, q)
                early = after["early"] - before["early"]
                consumed = after["consumed"] - before["consumed"]
                if kind in ("sole", "sp") and not (early >= 1 and consumed == early):
                    bad.append(f"rank{rank} bwd {kind} ckpt={ckpt} T={T}: early={early} consumed={consumed}")
                if kind == "two" and early:
                    bad.append(f"rank{rank} bwd two consumers launched early ({early})")

    # ---- 3. randomized interleaving on both streams
    rng = random.Random(1234)
    for it in range(args.stress):
        op = rng.choice(["pipe", "serial", "ar", "rs", "bwd"])
        M = rng.choice([256, 2048, 3000, 4096])
        K, N = 2048, 1024
        kl = K // world
        x, w = _operands(K, N, M, it, dev)
        xs, ws = x[:, rank * kl:(rank + 1) * kl].contiguous(), w[:, rank * kl:(rank + 1) * kl].contiguous()
        if op in ("pipe", "serial"):
            os.environ["ISOEXEC_PIK_OVERLAP"] = "1" if op == "pipe" else "0"
            LANE._AGREED.clear()
            ref = row_parallel_linear(x, w, None, plan=plan, tp_size=1, tp_rank=0, k_full=K, group=None,
                                      out_dtype=torch.bfloat16)
            got = row_parallel_linear(xs, ws, None, plan=plan, tp_size=world, tp_rank=rank, k_full=K,
                                      group=group, out_dtype=torch.bfloat16)
            check(f"stress{it} {op} M={M}", got, ref)
        elif op == "ar":
            p = (xs[:, :N].float() * (rank + 1)).to(torch.bfloat16).contiguous()
            got = TB.tree_grad_sum(p, group)
            ref = OWNER.legacy(p.view(1, -1), group, torch.bfloat16).view(p.shape)
            check(f"stress{it} ar M={M}", got, ref)
        elif op == "rs":
            M4 = (M // world) * world
            p = (xs[:M4, :N].float() * (rank + 1)).to(torch.bfloat16).contiguous()
            got = TB.tree_grad_reduce_scatter(p, group)
            full = OWNER.legacy(p.view(1, -1), group, torch.bfloat16).view(p.shape)
            check(f"stress{it} rs M={M}", got, full[rank * (M4 // world):(rank + 1) * (M4 // world)])
        else:
            os.environ["ISOEXEC_PIK_OVERLAP"] = "1"
            LANE._AGREED.clear()
            xx = xs[:, :1024].clone().requires_grad_(True)
            ww = ws[:256, :1024].clone().requires_grad_(True)
            y = mm_trainer(tp_copy(xx), ww)
            dy = torch.ones_like(y)
            y.backward(dy)
            ref = OWNER.legacy((dy @ ww.detach()).view(1, -1), group, torch.bfloat16).view(xx.shape)
            check(f"stress{it} bwd M={M}", xx.grad, ref)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(f"[pik_overlap --world {world}{' --ray-sim' if args.ray_sim else ''}] link counts "
              f"{TB.tp_copy_link_counts()} pipeline {PIPE.PIPE_COUNTS} peers {LANE.peer_counts()}", flush=True)
    for b in bad[:20]:
        print("FAIL", b, flush=True)
    fails[rank] = len(bad)
    dist.destroy_process_group()


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
    ap.add_argument("--ray-sim", action="store_true")
    ap.add_argument("--stress", type=int, default=200)
    args = ap.parse_args()
    if args.sim:
        return sim()
    if args.ray_sim:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["_PIK_PHYS"] = vis if vis else ",".join(str(i) for i in range(args.world))
    fails = mp.Manager().list([0] * args.world)
    mp.spawn(worker, args=(args.world, _port(), args, fails), nprocs=args.world, join=True)
    total = sum(fails)
    print(f"[pik_overlap --world {args.world}] {'PASS' if not total else f'FAIL ({total})'}", flush=True)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
