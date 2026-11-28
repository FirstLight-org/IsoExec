#!/usr/bin/env python
"""The detreduce tree reducer over several process groups, through ``collectives/symm_setup`` (GPU, one process per
rank, NCCL). Every result is compared bitwise with the all-gather form (``DistributedReducer``: rank-ordered
all-gather + the same leaf tree), and every phase runs under a watchdog that names the phase and exits non-zero
instead of hanging.

  1. the world group: TreeReducer == all-gather form, eager and through CUDA-graph capture / replay;
  2. two families of same-size groups in one process ({0,1},{2,3} and {0,2},{1,3} at world 4; at world 2 the world
     group twice, as two group objects of the same ranks): one tree reducer per group, used alternately, then one
     group alone; bitwise, no hang (the workspaces are per group object, never shared by size);
  3. a lone-subgroup setup: a symmetric setup only ranks {0, W-1} take part in (``symm_setup.rendezvous`` on that
     group), then a tree reducer on a group containing rank 0 and a rank that did not take part: every member raises
     ``SymmetricSetupSkew`` at prepare, the reducer (``ActorTPReducer``) keeps the all-gather form, results bitwise
     equal, no hang; a re-created group object of the same ranks stays refused;
  4. the setup log (``ISOEXEC_SYMM_SETUP_LOG``): each group's members logged the same counts in the same order; the
     refused group's members all logged a refusal and no setup.

    CUDA_VISIBLE_DEVICES=0,1 python isoexec/ops/collectives/tests/symm_setup_groups_gpu.py --world 2
    CUDA_VISIBLE_DEVICES=0,1,2,3 python isoexec/ops/collectives/tests/symm_setup_groups_gpu.py --world 4
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

SIZES = (1024, 64 * 5120, 999 * 1024 + 3)


class _Watchdog:
    def __init__(self, rank: int, seconds: float):
        self.rank, self.seconds, self.phase, self.t = rank, seconds, "start", time.time()
        threading.Thread(target=self._run, daemon=True).start()

    def enter(self, phase: str) -> None:
        self.phase, self.t = phase, time.time()

    def _run(self) -> None:
        while True:
            time.sleep(1.0)
            if time.time() - self.t > self.seconds:
                print(f"[rank {self.rank}] WATCHDOG: phase {self.phase!r} exceeded {self.seconds:.0f} s (hang)",
                      flush=True)
                os._exit(3)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


def _actor(group):
    from isoexec.models.deepseek_v4_1_flash.providers import ActorTPReducer
    from isoexec.ops.attention.deepseek_v41_cache_sm100.tp import DistributedReducer
    from isoexec.ops.collectives.detreduce_sm100.ops import TreeReducer
    return ActorTPReducer(TreeReducer(group), DistributedReducer(group))


def _check(reducer, group, rank, dev, tag, checks, seed):
    """Eager reductions through the reducer (tree inside its scope, or the all-gather form) == all-gather form."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100.tp import DistributedReducer
    ref = DistributedReducer(group)
    for n in SIZES:
        g = torch.Generator(device=dev).manual_seed(seed * 7919 + n * 31 + rank)
        root = torch.randn(n, generator=g, device=dev, dtype=torch.float32) * (1 + rank)
        with reducer.scope():
            got = reducer.gather_sum(root)
        want = ref.gather_sum(root)
        torch.cuda.synchronize(dev)
        checks.append((f"{tag} n={n} transport={reducer.transport() if hasattr(reducer, 'transport') else '?'}",
                       _same(got, want)))


def _worker(rank: int, world: int, port: int, out_dir: str, timeout: float) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), ISOEXEC_SYMM_SETUP_LOG=out_dir)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)
    wd = _Watchdog(rank, timeout)
    from isoexec.ops.collectives import symm_setup
    checks: list[tuple[str, bool]] = []

    # 1. world group: eager, then CUDA-graph capture / replay inside the reducer's execution scope
    wd.enter("1 world eager")
    world_red = _actor(dist.group.WORLD).prepare(dev)
    checks.append(("1 world prepared (tree)", world_red.tree_local and world_red.prepared))
    _check(world_red, dist.group.WORLD, rank, dev, "1 world eager", checks, 1)
    wd.enter("1 world graph")
    from isoexec.ops.attention.deepseek_v41_cache_sm100.tp import DistributedReducer
    root = torch.randn(64 * 5120, device=dev, generator=torch.Generator(device=dev).manual_seed(11 + rank))
    graph, side = torch.cuda.CUDAGraph(), torch.cuda.Stream(dev)
    with world_red.tree.execution_scope(capture=True):
        with torch.cuda.stream(side):
            world_red.tree.gather_sum(root)                      # warm the capture stream
            with torch.cuda.graph(graph, stream=side):
                captured = world_red.tree.gather_sum(root)
    for k in range(3):
        root.copy_(torch.randn(root.shape, device=dev, generator=torch.Generator(device=dev).manual_seed(100 + k + rank)))
        with world_red.tree.execution_scope():
            graph.replay()
        torch.cuda.synchronize(dev)
        checks.append((f"1 world replay {k}", _same(captured, DistributedReducer(dist.group.WORLD).gather_sum(root))))

    # 2. two same-size families, alternating, then one group alone
    wd.enter("2 families")
    if world >= 4:
        fam_a = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        fam_b = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        mine_a = fam_a[0] if rank in (0, 1) else fam_a[1]
        mine_b = fam_b[0] if rank in (0, 2) else fam_b[1]
    else:
        mine_a, mine_b = dist.new_group(list(range(world))), dist.new_group(list(range(world)))
    red_a, red_b = _actor(mine_a).prepare(dev), _actor(mine_b).prepare(dev)    # same order on every rank
    checks.append(("2 both prepared (tree)", red_a.tree_local and red_b.tree_local))
    for k in range(3):
        _check(red_a, mine_a, rank, dev, f"2 alternate a{k}", checks, 20 + k)
        _check(red_b, mine_b, rank, dev, f"2 alternate b{k}", checks, 30 + k)
    wd.enter("2 lone group")
    if world >= 4:
        if rank in (0, 1):
            _check(red_a, mine_a, rank, dev, "2 lone {0,1}", checks, 40)
    else:
        _check(red_a, mine_a, rank, dev, "2 lone", checks, 40)
    dist.barrier()

    # 3. a setup only {0, W-1} take part in, then a group of rank 0 with a rank that did not
    wd.enter("3 lone setup")
    lone_ranks, after_ranks = [0, world - 1], [0, 1] if world >= 4 else None
    lone = dist.new_group(lone_ranks)
    if rank in lone_ranks:
        from torch.distributed import _symmetric_memory as symm_mem
        buf = symm_mem.empty(4096, dtype=torch.int32, device=dev)
        symm_setup.rendezvous(buf, lone, site="test.lone")
    if world >= 4:
        wd.enter("3 skewed group")
        after = dist.new_group(after_ranks)
        # a second group object over the same ranks, created by EVERY rank: with a device-bound default group torch
        # builds a subgroup's NCCL communicator eagerly by splitting the world communicator, a collective over all
        # ranks (non-members take a no-color split), so a members-only ``use_local_synchronization=True`` creation
        # here deadlocked against the non-members' barrier at world 4 (ranks 0-1 in ncclCommSplit, 2-3 in barrier)
        again = dist.new_group(after_ranks)
        if rank in after_ranks:
            red = _actor(after).prepare(dev)
            checks.append(("3 skew refused -> all-gather form", not red.tree_local and red.transport() == "all_gather"))
            _check(red, after, rank, dev, "3 skewed group bits", checks, 50)
            checks.append(("3 same ranks stay refused", symm_setup.skewed(again)))
    else:
        # world 2: {0,1} is the lone group itself (both took part): a following tree on a NEW world group object
        wd.enter("3 world-2 follow-up")
        red = _actor(dist.new_group([0, 1])).prepare(dev)
        checks.append(("3 equal counts -> tree", red.tree_local and red.prepared))
        _check(red, None, rank, dev, "3 follow-up bits", checks, 50)
    dist.barrier()
    wd.enter("done")
    ok = sum(c for _, c in checks)
    for name, good in checks:
        if not good:
            print(f"[rank {rank}] FAIL {name}", flush=True)
    print(f"[rank {rank}] {ok}/{len(checks)} checks pass; setups={symm_setup.setup_count()}", flush=True)
    dist.destroy_process_group()
    sys.exit(0 if ok == len(checks) else 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--out", default=os.environ.get("TMPDIR", "/tmp") + "/symm_setup_groups")
    ap.add_argument("--timeout", type=float, default=180)
    a = ap.parse_args()
    if torch.cuda.device_count() < a.world:
        print(f"needs {a.world} GPUs, sees {torch.cuda.device_count()}")
        return 2
    os.makedirs(a.out, exist_ok=True)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.spawn(_worker, args=(a.world, port, a.out, a.timeout), nprocs=a.world, join=False)
    ok = True
    while not ctx.join():
        pass
    for p in ctx.processes:
        ok &= p.exitcode == 0
    # 4. per group, every member logged the same counts in the same order (the refused group logs "skew")
    import collections
    seqs = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in os.listdir(a.out):
        for line in open(os.path.join(a.out, f)):
            kv = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
            seqs[kv["group"]][kv["pid"]].append((kv["count"], line.split()[-1]))
    for g, members in seqs.items():
        verdicts = {v for seq in members.values() for _, v in seq}
        if "skew" in verdicts:
            # the refused group (by rank tuple; at world 4 {0,1} first set up trees in phase 2, then skewed in phase
            # 3): the setups before the first refusal are equal on every member, and from it on every member refuses
            cut = {pid: next(i for i, (_, v) in enumerate(seq) if v == "skew") for pid, seq in members.items()
                   if any(v == "skew" for _, v in seq)}
            refused = (len(cut) == len(members)
                       and len({tuple(seq[:cut[pid]]) for pid, seq in members.items()}) == 1
                       and all(all(v == "skew" for _, v in seq[cut[pid]:]) for pid, seq in members.items()))
            print(f"group {g}: {len(members)} members, equal setups then refused on every member: {refused}")
            ok &= refused
            continue
        same = len({tuple(v) for v in members.values()}) == 1
        print(f"group {g}: {len(members)} members, setup sequences {'EQUAL' if same else 'DIFFER'}")
        ok &= same
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
