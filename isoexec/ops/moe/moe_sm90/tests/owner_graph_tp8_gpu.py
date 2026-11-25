"""Eight-rank expert/owner bitwise checks across graph replay and optional saved references.
Launch with torchrun --nproc_per_node=8. Reference tensors stay outside the source tree.
"""

import argparse
import functools
import os

import torch
import torch.distributed as dist

from isoexec.ops.moe.carriers import RoutingResult
from isoexec.ops.moe.moe_sm90.integration import expert_mlp
from isoexec.ops.moe.moe_sm90.moe_combine_backward import fused_combine
from isoexec.ops.moe.moe_sm90.moe_pik_combine_owner import _engine_owner_dispatch
from isoexec.ops.moe.expert_parallel import SEAM_N_LEAVES, expert_reduce


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference")
    p.add_argument("--save-reference")
    p.add_argument("--tokens", default="1,8,32,128,512,2048")
    a = p.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    assert dist.get_world_size() == 8, "this gate requires eight ranks"
    # ETP=8 over 8 fc2 leaves: one leaf per rank, owner-deferred, compact root when the kernel has one.
    etp, n_leaves = 8, 8 // 8
    reduce_fn = expert_reduce(
        functools.partial(expert_mlp, **{SEAM_N_LEAVES: n_leaves}), etp=etp, group=dist.group.WORLD,
        deferred=True, tp_copy=None, compact_root=n_leaves == 1 and getattr(expert_mlp, "supports_compact_root", False),
    )
    torch.manual_seed(177 + rank)
    E, H, I = 256, 2048, 64
    w1 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) * 0.02
    saved = {}
    ref = torch.load(a.reference, weights_only=True) if a.reference else None
    with torch.no_grad():
        for T in map(int, a.tokens.split(",")):
            torch.manual_seed(177 + T)
            x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16) * 0.5
            vals, ids = torch.randn(T, E, device="cuda").topk(8, dim=-1)
            routing = RoutingResult(ids.int().contiguous(), vals.softmax(-1).contiguous(), None)

            def eager(routed, shared, tokens=T):
                assert shared is None
                return fused_combine(routed.rows, routed.extra, tokens, rows=routed.combine_rows)

            def fn(x=x, routing=routing, eager=eager):
                routed = reduce_fn(x, routing, w1, w2)
                return _engine_owner_dispatch(routed, None, dist.group.WORLD, eager)

            y = fn()
            assert torch.isfinite(y).all(), (rank, T, "nonfinite")
            for _ in range(3):
                y = fn()
            y = y.clone()  # independent snapshot if replay reuses an output buffer
            assert torch.isfinite(y).all() and torch.count_nonzero(y) > 0, (rank, T)
            if ref is not None:
                assert torch.equal(y.cpu().view(torch.uint8), ref[str(T)].view(torch.uint8)), (rank, T)
            if rank == 0 and a.save_reference:
                saved[str(T)] = y.cpu()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = fn()
            for _ in range(5):
                graph.replay()
            dist.barrier()
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(y.view(torch.uint8), out.view(torch.uint8)), (rank, T, "graph")
            dist.barrier()
    if rank == 0 and a.save_reference:
        torch.save(saved, a.save_reference)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
