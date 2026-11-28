"""Real multi-GPU tensor parallelism: NCCL all_gather of FP32 subtree roots + the fixed leaf tree.

    python -m isoexec.ops.attention.deepseek_v41_cache_sm100.tests.dist_tp_run N      # N visible GPUs

One process per GPU (rank r owns heads / groups / index heads block r). Every rank runs the full-sequence
program, prefill-into-cache + decode, and paged mixed steps through ``tp.DistributedReducer``; rank 0 also
runs the unsharded TP = 1 program locally and checks every layer output bit for bit. Prints one RESULT line.
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, world, port, out):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import cache as dc
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops, paged, tp
    from isoexec.ops.attention.deepseek_v41_cache_sm100.tests import _harness as h
    from isoexec.ops.attention.deepseek_v41_cache_sm100.tests.test_paged_gpu import SCHEDULE, _pools, _tables

    red = tp.DistributedReducer()
    cfg = h.native_cfg(index_topk=448, candidate_blocks=64)
    dev = torch.device("cuda", rank)
    layers = [h.make_layer(cfg, i, i, device=dev) for i in range(cfg.num_hidden_layers)]
    shards = [tp.shard_layer(m, cfg, rank, world) for m in layers]
    g = torch.Generator(device=dev).manual_seed(83)
    B, T, P0 = 2, 700, 301
    xs = [torch.randn((B, T, cfg.hidden_size), generator=g, device=dev).to(torch.bfloat16) for _ in layers]
    eq = lambda a, b: torch.equal(a.view(torch.int16), b.view(torch.int16))
    checks, bad = 0, []
    kw = dict(linear_fn=h.native_linear, dequant_fn=h.dequant)
    with torch.no_grad():
        # 1) full sequence
        st = {}
        full_tp = [red.run(ops.attention_full_gen(s, x, cfg, i, st, tp=world, **kw)) for i, (s, x) in enumerate(zip(shards, xs))]
        if rank == 0:
            st1 = {}
            full_1 = [ops.attention_full(m, x, cfg, i, st1, **kw) for i, (m, x) in enumerate(zip(layers, xs))]
            for i, (a, b) in enumerate(zip(full_tp, full_1)):
                checks += 1
                if not eq(a, b):
                    bad.append(("full", i))
        # 2) prefill into the decode cache, then decode == the TP-1 full sequence
        cache, st = dc.DecodeCache(cfg, B, 800), {}
        for i, (s, x) in enumerate(zip(shards, xs)):
            red.run(ops.attention_full_gen(s, x[:, :P0].contiguous(), cfg, i, st, cache=cache, tp=world, **kw))
        cache.advance(P0)
        for t in range(P0, P0 + 40):
            for i, (s, x) in enumerate(zip(shards, xs)):
                o = red.run(dc.attention_decode_gen(s, x[:, t:t + 1].contiguous(), cfg, i, cache, tp=world, **kw))
                if rank == 0:
                    checks += 1
                    if not eq(o, full_1[i][:, t:t + 1]):
                        bad.append(("decode", t, i))
            cache.advance()
        # 3) paged mixed prefill + decode steps == the TP-1 contiguous recipe-3 program
        win, comp, P = _tables(3, 9)
        win, comp = win.to(dev), {r: b.to(dev) for r, b in comp.items()}
        pools = _pools(cfg, 3, P)
        if rank == 0:
            pxs = [torch.randn((3, 400, cfg.hidden_size), generator=torch.Generator(device=dev).manual_seed(89 + i),
                               device=dev).to(torch.bfloat16) for i in range(len(layers))]
        else:
            pxs = [torch.empty((3, 400, cfg.hidden_size), dtype=torch.bfloat16, device=dev) for _ in layers]
        for x in pxs:
            dist.broadcast(x, 0)
        lengths = {0: 302, 1: 139, 2: 51}
        if rank == 0:
            ref = {}
            for s, L in lengths.items():
                st1 = {}
                ref[s] = [ops.attention_full(m, x[s:s + 1, :L].contiguous(), cfg, i, st1, recipe=3, **kw)[0]
                          for i, (m, x) in enumerate(zip(layers, pxs))]
        slots = {0: 0, 1: 1, 2: 2}
        for plan in SCHEDULE[:3]:
            sel = torch.tensor([s for s, _, _ in plan], device=dev).long()
            batch = paged.PagedBatch.make([a for _, a, _ in plan], [c for _, _, c in plan], win[sel],
                                          {r: b[sel] for r, b in comp.items()}, [slots[s] for s, _, _ in plan],
                                          ratios=(2,), device=dev)
            for i, (s, x) in enumerate(zip(shards, pxs)):
                xi = torch.cat([x[q, a:a + c] for q, a, c in plan], 0).contiguous()
                o = red.run(paged.attention_paged_gen(s, xi, cfg, i, pools, batch, tp=world, **kw))
                if rank == 0:
                    off = 0
                    for q, a, c in plan:
                        checks += 1
                        if not eq(o[off:off + c], ref[q][i][a:a + c]):
                            bad.append(("paged", plan[0], i, q))
                        off += c
    torch.cuda.synchronize()
    if rank == 0:
        out.put({"world": world, "checks": checks, "mismatches": len(bad), "first": bad[:5],
                 "nccl": torch.cuda.nccl.version(), "devices": [torch.cuda.get_device_name(0)]})
    dist.barrier()
    dist.destroy_process_group()


def main():
    world = int(sys.argv[1])
    assert torch.cuda.device_count() >= world, "needs N visible GPUs"
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29500 + os.getpid() % 1000
    mp.spawn(_worker, args=(world, port, q), nprocs=world, join=True)
    print("RESULT " + json.dumps(q.get()), flush=True)


if __name__ == "__main__":
    main()
