"""Real Gloo transport: fixed order, subgroup isolation, and replicated EP gradients."""
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from isoexec.ops.collectives.ordered_sum import ordered_sum
from isoexec.ops.moe.ep_fixed_order import DistComm, ExpertMap, reduce_replicated_grads


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=3,
                            timeout=timedelta(seconds=45))
    try:
        # Reassociation changes the answer: ((2**24 + 1) - 2**24) == 0 in FP32.
        values = [2**24, 1, -(2**24)]
        local = torch.full((3, 5), float(values[rank])).t()
        for _ in range(3):
            assert torch.equal(ordered_sum(local, chunk_elements=4), torch.zeros_like(local))
        assert torch.equal(local, torch.full_like(local, float(values[rank])))
        assert ordered_sum(torch.empty(0, 2)).shape == (0, 2)
        for dtype in (torch.bfloat16, torch.float32, torch.float64):
            result = ordered_sum(torch.full((7,), float(rank), dtype=dtype), chunk_elements=3)
            assert result.dtype == dtype and torch.equal(result, torch.full_like(result, 3))
        subgroup = dist.new_group([0, 2])
        if rank != 1:
            assert ordered_sum(torch.tensor([rank + 1.0]), subgroup).item() == 4

        comm = DistComm()
        for shared in ("home", 2):
            emap = ExpertMap.contiguous(6, 3, shared=shared)
            groups = len(emap.local_experts(rank))
            # Rank 1 represents a zero-token rank, but must join every reduction.
            contribution = (1, 0, 3)[rank]
            router = torch.full((2, 3), float(contribution))
            gu = torch.full((groups, 4, 3), float(contribution))
            dn = torch.full((groups, 3, 2), float(contribution))
            result, gu, dn = reduce_replicated_grads(comm, emap, router, gu, dn)
            assert torch.equal(result, torch.full_like(router, 4))
            assert torch.equal(gu[:2], torch.full_like(gu[:2], contribution))
            assert torch.equal(dn[:2], torch.full_like(dn[:2], contribution))
            if shared == "home":
                assert torch.equal(gu[-1], torch.full_like(gu[-1], 4))
                assert torch.equal(dn[-1], torch.full_like(dn[-1], 4))
            elif rank == 2:
                assert torch.equal(gu[-1], torch.full_like(gu[-1], contribution))
    finally:
        dist.destroy_process_group()


def test_rank_order_and_ep_replicated_gradients(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=3, join=True)


def test_local_sum_validation():
    x = torch.ones(3)
    assert ordered_sum(x) is x
    with pytest.raises(ValueError, match="positive"):
        ordered_sum(x, chunk_elements=0)
    with pytest.raises(TypeError, match="floating"):
        ordered_sum(torch.ones(3, dtype=torch.long))
