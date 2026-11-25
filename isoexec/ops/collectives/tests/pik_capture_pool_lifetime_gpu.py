"""Two-rank CUDA graph -> larger eager collective -> replay lifetime regression.

Run with torch.distributed.run --standalone --nproc_per_node=2. This exercises
all four symmetric data pool registries with their actual CUDA allocations.
Skipped on an NCCL-only arch (sm100), which allocates no symmetric pools.
"""

import os
import torch
import torch.distributed as dist
from isoexec.ops.moe.moe_sm90 import moe_pik_combine_owner as owner


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    from isoexec.ops.collectives.pik_tp_invariant import p2p_arch_refusal

    if (refusal := p2p_arch_refusal()) is not None:
        if rank == 0:
            print(f"SKIPPED: {refusal}", flush=True)
        return
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    owner._ensure_canonical_pik()
    from pik import allreduce as ar

    source = torch.full((32,), float(rank + 1), dtype=torch.float32, device=device)
    outputs = [torch.empty((32,), dtype=torch.bfloat16, device=device) for _ in range(2)]
    getters = [
        lambda n: owner._out_sym_pool(n, device, None),
        lambda n: owner._shared_in_sym_pool(n, device, None, torch.bfloat16),
    ]
    for get in getters:
        pool = get(32)
        assert len(dict(pool.named_buffers())) == 3
        pool.stage()[0].fill_(rank + 1)
    del pool
    # Warm the collective before capture (including its symmetric flags).
    for _ in range(3):
        reduced = ar.tree_all_reduce(source)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        reduced = ar.tree_all_reduce(source)
        for get, output in zip(getters, outputs):
            output.copy_(get(32).stage()[0][:32])
    for get in getters:
        get(2**21).stage()[0].fill_(9)
    large = torch.full((2**21,), float(rank + 1), dtype=torch.float32, device=device)
    ar.tree_all_reduce(large)
    torch.cuda.synchronize()
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(reduced, torch.full_like(reduced, 3, dtype=reduced.dtype, device=device))
        for output in outputs:
            assert torch.equal(output, torch.full_like(output, rank + 1, dtype=output.dtype, device=device))
    assert len(ar._CAPTURED_POOLS) == 4, len(ar._CAPTURED_POOLS)
    print(f"PASS rank={rank} captured pools retained={len(ar._CAPTURED_POOLS)}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
