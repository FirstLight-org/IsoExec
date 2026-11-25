"""EP-held experts gather into one stack in global expert order (two gloo ranks)."""

from __future__ import annotations

import torch

from isoexec.sync.weights import gather_expert_stack


def _stack_rank(rank, rendezvous):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        experts = torch.arange(4 * 6, dtype=torch.float32).reshape(4, 2, 3)
        held = [experts[index].clone() for index in range(2 * rank, 2 * rank + 2)]
        assert torch.equal(gather_expert_stack(held, dist.group.WORLD), experts)
    finally:
        dist.destroy_process_group()


def test_expert_stack_gather_orders_contiguous_ranges_globally(tmp_path):
    torch.multiprocessing.spawn(_stack_rank, args=((tmp_path / "gloo").as_uri(),), nprocs=2, join=True)
