#!/usr/bin/env python
"""pik's symmetric-memory P2P tree all-reduce vs its NCCL transport, bitwise, across process groups.

One GPU per rank. Every call reduces fp32 subtree partials (the row-parallel site's node wire) and must give
the NCCL transport's bits, on every rank, for:

  1. the world group (TP = world), eager and through CUDA-graph capture/replay;
  2. two families of half-size groups in the same process -- {0,1},{2,3} and {0,2},{1,3} at world 4 -- used
     alternately, then one group alone. A rank then holds two groups of one size; symmetric pools, flag
     pools and admission verdicts keyed by group size (not by the group) shared one allocation between
     them: alternating calls read the other group's peers (wrong bits), and a lone group's barrier waited
     on the other group's ranks (the B200 TP4 hang). Last, a pool setup only some ranks take part in, then
     one they share: torch numbers rendezvous per process, so pik must refuse it (NCCL, same bits)
     rather than wait on keys the other rank never writes;
  3. the TP-invariance statement at the row-parallel site: TP = 1, 2 and world give identical bits.

A watchdog names the phase and exits non-zero instead of hanging.

    CUDA_VISIBLE_DEVICES=0,1 python isoexec/ops/collectives/tests/pik_p2p_groups_gpu.py --world 2
    CUDA_VISIBLE_DEVICES=0,1,2,3 python isoexec/ops/collectives/tests/pik_p2p_groups_gpu.py --world 4
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, *[".."] * 4))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

SIZES = (1024, 256 * 5120, 999 * 1024)  # two-shot needs numel divisible by world
K, N = 6144, 5120  # Qwen3.6-27B o_proj
M_POINTS = (1, 64, 256)


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


class _Watchdog:
    def __init__(self, rank: int, seconds: float):
        self.rank, self.seconds, self.phase, self.t = rank, seconds, "start", time.time()
        threading.Thread(target=self._run, daemon=True).start()

    def enter(self, phase: str) -> None:
        self.phase, self.t = phase, time.time()

    def _run(self) -> None:
        while True:
            time.sleep(1)
            if time.time() - self.t > self.seconds:
                print(f"[rank {self.rank}] HANG in phase {self.phase!r} (> {self.seconds:.0f}s)", flush=True)
                os._exit(3)


def worker(rank: int, world: int, port: int, args) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)
    dog = _Watchdog(rank, args.timeout)
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik

    ensure_pik()
    import pik.allreduce as AR  # type: ignore
    from pik.linear import row_parallel_linear  # type: ignore
    from pik.plan import ReductionPlan  # type: ignore

    fails = checks = 0

    def check(tag, got, ref):
        nonlocal fails, checks
        checks += 1
        if not torch.equal(_bits(got), _bits(ref)):
            fails += 1
            print(f"[rank {rank}] FAIL {tag}", flush=True)

    def both(partial, group, tag):
        ref = AR.tree_all_reduce(partial, group=group, backend="nccl")
        got = AR.tree_all_reduce(partial, group=group, backend="p2p")
        check(tag, got, ref)

    # 1. the world group, eager, then captured and replayed
    dog.enter("world eager")
    g = torch.Generator(device=dev).manual_seed(100 + rank)
    for n in SIZES:
        both(torch.randn(n, device=dev, generator=g), dist.group.WORLD, f"world n={n}")
    dog.enter("world graph")
    x = torch.randn(256 * 5120, device=dev, generator=g)
    AR.tree_all_reduce(x, group=dist.group.WORLD, backend="p2p")  # pools sized eagerly
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = AR.tree_all_reduce(x, group=dist.group.WORLD, backend="p2p")
    for trial in range(3):
        x.copy_(torch.randn(x.shape, device=dev, generator=g))
        graph.replay()
        check(f"world graph trial {trial}", y, AR.tree_all_reduce(x, group=dist.group.WORLD, backend="nccl"))

    # 2. two families of half-size groups in one process, used alternately
    if world >= 4:
        half = world // 2
        blocks = [list(range(i, i + half)) for i in range(0, world, half)]           # {0,1},{2,3}
        strided = [list(range(i, world, 2)) for i in range(2)]                          # {0,2},{1,3}
        mine = {}
        for name, family in (("block", blocks), ("strided", strided)):
            for ranks in family:
                grp = dist.new_group(ranks)  # every rank creates every group, in the same order
                if rank in ranks:
                    mine[name] = grp
        for it in range(args.iters):
            for name in ("block", "strided"):
                dog.enter(f"half-size groups {name} iter {it}")
                for n in SIZES:
                    both(torch.randn(n, device=dev, generator=g), mine[name], f"{name} n={n} iter {it}")
        # Then one family alone, its pools already rendezvoused (the warmed steady state): the even-rank
        # strided group calls while the odd ranks are elsewhere, as a DP peer would be. A pool tied to the
        # block group's ranks would wait on an odd rank that never arrives.
        dog.enter("one strided group alone")
        if rank % 2 == 0:
            both(torch.randn(256 * 5120, device=dev, generator=g), mine["strided"], "strided group alone")
        dist.barrier()

        # Then a setup the other ranks do not share: only {0,3} sets up pools (the {1,2} ranks are elsewhere).
        # torch numbers rendezvous per process, so ranks 0 and 3 are now one setup ahead of 1 and 2, and the
        # bf16 pools the block groups set up next pair a rank that is ahead with one that is not. pik must
        # refuse that rendezvous on both ranks (NCCL transport, same bits) instead of waiting forever.
        dog.enter("lone setup, then a shared one")
        anti = {}
        for ranks in ([0, world - 1], list(range(1, world - 1))):
            made = dist.new_group(ranks)
            if rank in ranks:
                anti = made
        if rank in (0, world - 1):
            both(torch.randn(256 * 5120, device=dev, generator=g), anti, "anti group alone")
        dist.barrier()
        x = torch.randn(256 * 5120, device=dev, generator=g).to(torch.bfloat16)
        both(x, mine["block"], "block bf16 after a lone setup")
        if AR._group_key(mine["block"]) not in AR._SKEWED:  # each block group pairs one rank of each kind
            check("block group refused P2P after the skewed setup", torch.zeros(1), torch.ones(1))

    # 3. TP invariance at the row-parallel site: TP = 1, 2 (and world) give identical bits
    dog.enter("tp invariance")
    plan = ReductionPlan(num_leaves=8, leaf_dtype=torch.bfloat16)
    tps = sorted({1, 2, world})
    groups = {1: None}
    for tp in tps:
        if tp == 1:
            continue
        grp = None
        for start in range(0, world, tp):
            ranks = list(range(start, start + tp))
            made = dist.new_group(ranks)
            if rank in ranks:
                grp = made
        groups[tp] = grp
    for M in M_POINTS:
        gs = torch.Generator(device=dev).manual_seed(7 + M)  # same operands on every rank
        x = (torch.randn(M, K, device=dev, generator=gs) * 0.5).to(torch.bfloat16)
        w = (torch.randn(N, K, device=dev, generator=gs) * 0.02).to(torch.bfloat16)
        ref = row_parallel_linear(x, w, None, plan=plan, tp_size=1, k_full=K, out_dtype=torch.bfloat16)
        for tp in tps[1:]:
            r = rank % tp
            k = K // tp
            got = row_parallel_linear(x[:, r * k:(r + 1) * k].contiguous(), w[:, r * k:(r + 1) * k].contiguous(),
                                      None, plan=plan, tp_size=tp, tp_rank=r, k_full=K, group=groups[tp],
                                      out_dtype=torch.bfloat16)
            check(f"row-parallel M={M} tp={tp} == tp=1", got, ref)

    dog.enter("result")
    t = torch.tensor([fails, checks], device=dev)
    dist.all_reduce(t)
    if rank == 0:
        ok = int(t[0]) == 0
        print(f"P2P GROUPS RESULT world={world}: {int(t[1] - t[0])}/{int(t[1])} bitwise-equal checks "
              f"({'PASS' if ok else 'FAIL'}); transport counts {AR.transport_counts()}", flush=True)
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
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()
    mp.spawn(worker, args=(args.world, _port(), args), nprocs=args.world, join=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
