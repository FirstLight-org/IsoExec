"""Engine static-capacity EP dispatch returns the trainer dispatch's bytes (2-4 gloo processes on CPU)."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _inner(rows, routing, gate_up, down, **kw):
    """A per-row expert: scale by (local expert + 1) and route probability; permuted by expert."""
    experts = routing.indices[:, 0].long()
    perm = torch.argsort(experts, stable=True)
    from isoexec.ops.moe.carriers import ExpertOutputs

    out = rows[perm].float() * (experts[perm] + 1).float().unsqueeze(1) * routing.probs[perm].float()
    return ExpertOutputs(rows=out, layout="permuted", routing=routing, extra=perm)


def _combine(routed, shared):
    """Fold each token's k terms in slot order in fp32, then narrow once."""
    rows = routed.rows[routed.combine_rows.long()]  # [T, k, H]
    total = rows[:, 0].clone()
    for j in range(1, rows.shape[1]):
        total = total + rows[:, j]
    return total.to(torch.bfloat16)


def _worker(rank, world, tp, port, tokens, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from isoexec.ops.moe.carriers import RoutingResult
    from isoexec.ops.moe import expert_parallel as ep_ops
    from isoexec.schema import Axis, Placement

    ep_group = dist.new_group(list(range(world)))
    tp_groups = [dist.new_group(list(range(start, start + tp))) for start in range(0, world, tp)]
    tp_group = tp_groups[rank // tp]
    experts, k, hidden = 8, 2, 4
    e_local = experts // world
    # Each data-parallel engine (one TP group) has its own tokens; TP ranks of one engine share them.
    engine = rank // tp
    generator = torch.Generator().manual_seed(100 + engine)
    count = tokens[engine]
    x = torch.randn(count, hidden, generator=generator).to(torch.bfloat16)
    scores = torch.rand(count, experts, generator=generator)
    probs, indices = scores.topk(k, dim=1)
    routing = RoutingResult(indices.to(torch.int32), probs.float(), None)
    gate_up = torch.zeros(e_local, 2, hidden)
    down = torch.zeros(e_local, hidden, 1)
    placement = Placement(tp=Axis(tp, rank % tp, tp_group), ep=Axis(world, rank, ep_group), dp=Axis(1, 0, None),
                          pp=Axis(1, 0, None), etp=Axis(1, 0, None), cp=Axis(1, 0, None), side="engine", leaves=1)
    ep_ops.set_engine_token_bound(lambda n: max(tokens))
    static = ep_ops.engine_expert_dispatch(_inner, ep=world, ep_rank=rank, ep_group=ep_group, combine=_combine,
                                           placement=placement)(x, routing, gate_up, down)
    trainer = ep_ops.expert_dispatch(_inner, ep=world, ep_rank=rank, ep_group=ep_group, combine=_combine,
                                     placement=placement)(x, routing, gate_up, down)
    results[rank] = (static.layout, static.rows.view(torch.int16).tolist(), trainer.rows.view(torch.int16).tolist())
    dist.destroy_process_group()


@pytest.mark.parametrize(("world", "tp", "tokens"), [(2, 1, [5, 3]), (4, 2, [7, 4])])
def test_static_engine_dispatch_matches_trainer_dispatch(world, tp, tokens):
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(_worker, args=(world, tp, _free_port(), tokens, results), nprocs=world, join=True)
    for rank in range(world):
        layout, static, trainer = results[rank]
        assert layout == "combined"
        assert static == trainer
