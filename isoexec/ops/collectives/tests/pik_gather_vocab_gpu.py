#!/usr/bin/env python
"""The engine's vocab-parallel logits gather over pik's symmetric memory equals NCCL's all-gather, bitwise.

One GPU per rank. ``pik.allreduce.gather_last_dim`` stages a rank's shard in pik's input pool, takes one
barrier and copies every peer's shard straight into its vocab columns (widened on the copy). It moves
bytes only, so on every rank, for every group, it must give the bytes of an NCCL ``all_gather_into_tensor``
followed by the interleaving copy, for:

  1. the world group: bf16 and widened to fp32, at decode and batch shapes up to Qwen3.6's 248320
     columns at 256 rows, through ``gather_last_dim`` (which must take P2P) and ``gather_vocab``;
  2. at world >= 4, the half-size groups {0,1},{2,3}: two groups of one size in one process, whose pools
     are keyed by the group, so neither may read the other's peers;
  3. CUDA-graph capture with a pool sized eagerly, then three replays on fresh operands;
  4. a capture whose pool would need a setup (a rendezvous cannot run inside a capture) declines on every
     rank and ``gather_vocab`` serves it through NCCL inside the same capture;
  5. with P2P disabled (``ISOEXEC_PIK_P2P=0``: ``p2p_available`` patched off, as the engine transport
     selection does) ``gather_last_dim`` declines and ``gather_vocab`` takes NCCL, same bytes.

A watchdog names the phase and exits non-zero instead of hanging.

    CUDA_VISIBLE_DEVICES=0,1 python isoexec/ops/collectives/tests/pik_gather_vocab_gpu.py --world 2
    CUDA_VISIBLE_DEVICES=0,1,2,3 python isoexec/ops/collectives/tests/pik_gather_vocab_gpu.py --world 4
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

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

VOCAB = 248320  # Qwen3.6
SHAPES = ((1, 7), (37, 1000), (256, None))  # (rows, width); None: VOCAB / group size


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

    from isoexec.runtimes.engine_ops import gather_vocab

    fails = checks = 0
    g = torch.Generator(device=dev).manual_seed(100 + rank)

    def check(tag, got, ref):
        nonlocal fails, checks
        checks += 1
        if got.shape != ref.shape or got.dtype != ref.dtype or not torch.equal(_bits(got), _bits(ref)):
            fails += 1
            print(f"[rank {rank}] FAIL {tag}", flush=True)

    def claim(tag, ok):
        check(tag, torch.ones(1) if ok else torch.zeros(1), torch.ones(1))

    def nccl_gather(local, group, dtype):
        n = dist.get_world_size(group)
        stacked = torch.empty((n, *local.shape), dtype=local.dtype, device=dev)
        dist.all_gather_into_tensor(stacked, local, group=group)
        return stacked.movedim(0, -2).reshape(*local.shape[:-1], -1).to(dtype)

    groups = [("world", dist.group.WORLD)]
    if world >= 4:
        halves = [dist.new_group(list(range(i, i + world // 2))) for i in (0, world // 2)]
        groups.append(("half", halves[rank // (world // 2)]))

    # 1-2. eager, every group, bf16 and widened
    for name, grp in groups:
        dog.enter(f"eager {name}")
        n = dist.get_world_size(grp)
        for rows, width in SHAPES:
            width = width or VOCAB // n
            for dtype in (torch.bfloat16, torch.float32):
                local = torch.randn(rows, width, device=dev, generator=g).to(torch.bfloat16)
                ref = nccl_gather(local, grp, dtype)
                out = torch.empty(rows, n * width, dtype=dtype, device=dev)
                claim(f"{name} rows={rows} {dtype}: P2P taken", AR.gather_last_dim(local, grp, out))
                check(f"{name} gather_last_dim rows={rows} {dtype}", out, ref)
                check(f"{name} gather_vocab rows={rows} {dtype}", gather_vocab(local, grp, out_dtype=dtype), ref)

    # 3. capture with an eagerly sized pool, replayed on fresh operands
    dog.enter("graph")
    local = torch.randn(256, 1024, device=dev, generator=g).to(torch.bfloat16)
    out = torch.empty(256, world * 1024, dtype=torch.float32, device=dev)
    AR.gather_last_dim(local, dist.group.WORLD, out)  # pool sized eagerly
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = AR.gather_last_dim(local, dist.group.WORLD, out)
    claim("capture with a sized pool takes P2P", captured)
    for trial in range(3):
        local.copy_(torch.randn(local.shape, device=dev, generator=g).to(torch.bfloat16))
        graph.replay()
        check(f"graph trial {trial}", out, nccl_gather(local, dist.group.WORLD, torch.float32))

    # 4. a capture that would need a pool setup declines everywhere; gather_vocab captures NCCL instead
    dog.enter("graph decline")
    cap = AR._SYM_IN[(local.device, AR._group_key(dist.group.WORLD), torch.bfloat16)].cap
    big = torch.randn(1, cap + 1, device=dev, generator=g).to(torch.bfloat16)  # one element past the pool
    big_out = torch.empty(1, world * (cap + 1), dtype=torch.float32, device=dev)
    graph2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph2):
        took = AR.gather_last_dim(big, dist.group.WORLD, big_out)
        via_vocab = gather_vocab(big, dist.group.WORLD, out_dtype=torch.float32)
    claim("capture needing a pool setup declines", not took)
    graph2.replay()
    check("graph decline -> NCCL", via_vocab, nccl_gather(big, dist.group.WORLD, torch.float32))

    # 5. P2P disabled: decline, NCCL serves, same bytes
    dog.enter("p2p off")
    available = AR.p2p_available
    AR.p2p_available = lambda group=None: False
    try:
        local = torch.randn(37, 1000, device=dev, generator=g).to(torch.bfloat16)
        out = torch.empty(37, world * 1000, dtype=torch.float32, device=dev)
        claim("P2P off declines", not AR.gather_last_dim(local, dist.group.WORLD, out))
        check("P2P off gather_vocab", gather_vocab(local, dist.group.WORLD, out_dtype=torch.float32),
              nccl_gather(local, dist.group.WORLD, torch.float32))
    finally:
        AR.p2p_available = available

    dog.enter("result")
    t = torch.tensor([fails, checks], device=dev)
    dist.all_reduce(t)
    if rank == 0:
        ok = int(t[0]) == 0
        print(f"GATHER VOCAB RESULT world={world}: {int(t[1] - t[0])}/{int(t[1])} checks "
              f"({'PASS' if ok else 'FAIL'})", flush=True)
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
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()
    if torch.cuda.device_count() < args.world:
        print(f"needs {args.world} visible GPUs, found {torch.cuda.device_count()}", flush=True)
        return 2
    mp.spawn(worker, args=(args.world, _port(), args), nprocs=args.world, join=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
