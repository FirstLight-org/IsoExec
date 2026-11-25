"""Real P2P/NCCL tree-kernel regression for sequence reduce-scatter transport.
On an NCCL-only arch (sm100) the P2P transport is skipped and every check runs over NCCL."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from isoexec.ops.collectives.pik import allreduce as ar
from isoexec.ops.collectives.pik_tp_invariant import p2p_arch_refusal
from isoexec.ops.collectives.tp_bytes import _gather_dim0


def reference(partial, root_dtype):
    world = dist.get_world_size()
    widened = partial.float().contiguous()
    chunk = widened.numel() // world
    received = torch.empty((world, chunk), device=partial.device, dtype=torch.float32)
    dist.all_to_all_single(received.view(-1), widened.view(-1))
    root = torch.empty(chunk, device=partial.device, dtype=torch.float32)
    ar._tree_reduce(received, root)
    return root.to(root_dtype).view(partial.shape[0] // world, *partial.shape[1:])


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    checks = 0
    world = dist.get_world_size()
    assert world == 4, "This qualification probe uses the supported TP4 shape"
    refusal = p2p_arch_refusal()
    if refusal is not None:
        if rank == 0:
            print(f"P2P transport SKIPPED: {refusal}", flush=True)
        ar.p2p_available = lambda group=None: False
    available = ar.p2p_available
    for rows in (256, 4096, 8192, 16384, 20480):
        for wire in (torch.bfloat16, torch.float32):
            torch.manual_seed(1729 + rank)
            x = (torch.randn(rows, 2048, device="cuda") * torch.exp(torch.randn(rows, 2048, device="cuda") * 4)).to(
                wire
            )
            x[:, 0] = -0.0
            x[:, 1] = (1 if rank % 2 == 0 else -1) * 2**20
            for dtype in (torch.bfloat16, torch.float32) if wire == torch.bfloat16 else (torch.float32,):
                want = reference(x, dtype)
                for p2p in (True, False) if refusal is None else (False,):
                    ar.p2p_available = available if p2p else lambda group=None: False
                    got = ar.tree_reduce_scatter(x, root_dtype=dtype)
                    bits = torch.int16 if dtype == torch.bfloat16 else torch.int32
                    ok = torch.tensor(int(torch.equal(got.view(bits), want.view(bits))), device="cuda")
                    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
                    assert ok.item(), (rows, wire, dtype, p2p)
                    checks += 1
    ar.p2p_available = available

    # Noncontiguous inputs exercise the first-dimension ownership after the public entry's copy.
    for wire in (torch.bfloat16, torch.float32):
        x = torch.randn((257, 512), dtype=wire, device="cuda").t()
        want = reference(x, wire)
        got = ar.tree_reduce_scatter(x)
        assert torch.equal(ar._bits(got), ar._bits(want))
        checks += 1

    # Warm every shape before the asynchronous stress, including AR's existing live admissions.
    for rows in (4, 128, 512, 4096):
        for wire in (torch.bfloat16, torch.float32):
            x = torch.full((rows, 32), rank, dtype=wire, device="cuda")
            ar.tree_all_reduce(x)
            ar.tree_reduce_scatter(x)
            _gather_dim0(x, world, dist.group.WORLD)
    torch.cuda.synchronize()
    retained = []
    for index in range(48):
        rows = (4, 128, 512, 4096)[index % 4]
        wire = torch.bfloat16 if index % 2 else torch.float32
        value = index % 16
        x = torch.full((rows, 32), value + rank, dtype=wire, device="cuda")
        ar.tree_all_reduce(x)
        retained.append((ar.tree_reduce_scatter(x), value))
        _gather_dim0(x, world, dist.group.WORLD)
    for got, value in retained:
        want = torch.full_like(got, world * value + world * (world - 1) // 2)
        assert torch.equal(ar._bits(got), ar._bits(want)), value
        checks += 1
    counts = ar.transport_counts()
    if refusal is None:
        assert counts["p2p_calls"] >= 48 and counts["p2p_fallbacks"] == 0, counts
    else:
        assert counts["p2p_calls"] == 0 and counts["nccl_calls"] >= 48, counts
    if rank == 0:
        print(f"PASS: {checks} shapes/dtypes, {dist.get_world_size()} ranks, exact integer views", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
