"""One rank of ``test_symm_rendezvous_cpu.py``'s two-process check: a real gloo group on CPU, torch's symmetric-memory
rendezvous replaced by a stub (the setup-sequence check, symm_setup over pik's counter, is what runs for real).

Usage: python _symm_rendezvous_child.py RANK WORLD INIT_FILE OUT_JSON SCENARIO"""
from __future__ import annotations

import json
import sys
import types
from datetime import timedelta

import torch
import torch.distributed as dist


def main(rank, world, init_file, out, scenario):
    from torch.distributed import _symmetric_memory as symm_mem

    from isoexec.ops.collectives import symm_setup
    from isoexec.ops.moe import symm_rendezvous as sr

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world,
                            timeout=timedelta(seconds=60))
    symm_mem.rendezvous = lambda buf, g: types.SimpleNamespace(ranks=sr.group_ranks(g), nbytes=buf.numel())
    res = {"rank": rank, "steps": []}

    def step(name, group, numel=4):
        try:
            sr.rendezvous(torch.zeros(numel), group)
            res["steps"].append([name, "ok", symm_setup.setup_count()])
        except sr.RendezvousSkewError:
            res["steps"].append([name, "skew", symm_setup.setup_count()])

    singles = [dist.new_group([r]) for r in range(world)]     # every rank creates every group, in one order
    pair = dist.new_group(list(range(world)))                 # a second group over the same ranks
    if scenario == "lone":
        step("world", None)
        step("pair", pair)                                    # two groups over the same ranks, interleaved
        step("world again", None)
        step("own singleton", singles[rank])                  # every rank alone in its own group: symmetric
        step("world after symmetric singletons", None)
        if rank == 0:
            step("lone singleton", singles[0])                # only rank 0: the sequence is now skewed
        step("world after lone", None)                        # must raise on EVERY rank (no hang)
        step("pair after lone", pair)                         # sticky by rank tuple, no collective
        res["skewed"] = sr.skewed(None)
    elif scenario == "size":
        step("sizes differ", None, numel=4 if rank == 0 else 8)
        res["skewed"] = sr.skewed(pair)
    dist.barrier()                                            # every rank got here: nobody waits in a rendezvous
    res["done"] = True
    with open(out, "w") as f:
        json.dump(res, f)
    dist.destroy_process_group()


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
