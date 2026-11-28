"""Real multi-GPU tensor parallelism of the CED entry points (``ced.entry_projection``, ``ced.decoder_attention_gen``):
one process per GPU, NCCL ``all_gather`` of the FP32 wo_b subtree roots + the fixed leaf tree (``tp.DistributedReducer``).

    python -m isoexec.ops.attention.deepseek_v41_cache_sm100.tests.dist_ced_tp_run N      # N visible GPUs (2, 4, 8)

Rank r holds head shard r. Every rank runs (1) the trainer decoder segments with window floors and (2) the engine's
bounded replay over tests/test_ced_gpu.py's chunked prefill + decode schedule (entry projection on every token into
its own pools, decoder rows at positions >= s, floors on the layers after b); rank 0 also runs the unsharded TP = 1
program locally and checks every layer output bit for bit; every rank checks its top-k / candidates and its pool
bytes against its own TP = 1 run. Prints one RESULT line (the single-process simulation of the same relations, with
per-head checks, is tests/test_ced_tp_graph_gpu.py).
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
    from types import SimpleNamespace

    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ced, ops, tp
    from isoexec.ops.attention.deepseek_v41_cache_sm100.tests import _harness as h
    from isoexec.ops.attention.deepseek_v41_cache_sm100.tests import test_ced_gpu as c

    red = tp.DistributedReducer()
    W, DEC, KW = c.W, c.DEC, c.KW
    checks, bad = 0, []

    def eq(a, b):
        return c._eq(a, b)

    with torch.no_grad():
        # (1) trainer: decoder segments of two unequal packed sequences, floors at the segment starts
        B, T = 2, 400
        cfg, layers, xs = c._setup(B, T, seed=23)          # same seeds on every rank: identical weights / inputs
        shards = {i: tp.shard_layer(layers[i], cfg, rank, world) for i in DEC}
        starts, lengths = [T - 150, 37], [T, T - 23]
        dkv, cu = c._entry(cfg, layers, xs, lengths)
        seq_all = torch.cat([torch.full((L,), s) for s, L in enumerate(lengths)])
        pos_all = torch.cat([torch.arange(L) for L in lengths])
        s_all = torch.tensor(starts)[seq_all]
        keep = pos_all >= s_all
        rows = c._decoder_rows(seq_all[keep], pos_all[keep], cu, s_all[keep])
        shared1, shared = {}, {}
        for i in DEC:
            x = xs[i][rows.sequence, rows.positions].contiguous()
            ref = ops.drive(ced.decoder_attention_gen(layers[i], x, cfg, i, rows, dkv, shared1, **KW))
            got = red.run(ced.decoder_attention_gen(shards[i], x, cfg, i, rows, dkv, shared, tp=world, **KW))
            checks += 3
            if not eq(got, ref):
                bad.append(("trainer", i))
            if not torch.equal(shared["topk_rows"], shared1["topk_rows"]):
                bad.append(("trainer_topk", i))
            if not torch.equal(shared["candidates_rows"], shared1["candidates_rows"]):
                bad.append(("trainer_candidates", i))

        # (2) engine: bounded replay, per-rank pools
        cfg, layers, xs = c._setup(3, c.MAXLEN, seed=31)
        shards = {i: tp.shard_layer(layers[i], cfg, rank, world) for i in DEC}
        win, comp = c._tables(3, 11)
        pools1, pools = c._pools(cfg, 3), c._pools(cfg, 3)
        start = {s: max(0, P - W) for s, P in c.PROMPTS.items()}
        for plan in c.STEPS:
            xin = c._rows_in(xs, plan)
            b1, bt = c._batch(plan, win, comp), c._batch(plan, win, comp)
            st1, stt = SimpleNamespace(pools=pools1, batch=b1), SimpleNamespace(pools=pools, batch=bt)
            ced.entry_projection(layers[4], xin[4], cfg, 4, b1.tok_pos, linear_fn=h.native_linear, step=st1)
            ced.entry_projection(shards[4], xin[4], cfg, 4, bt.tok_pos, linear_fn=h.native_linear, step=stt)
            seqs = torch.cat([torch.full((n,), s) for s, a, n in plan])
            poss = torch.cat([torch.arange(a, a + n) for s, a, n in plan])
            s_row = torch.tensor([start[int(s)] for s in seqs])
            keep = torch.nonzero(poss >= s_row).reshape(-1)
            if keep.numel():
                r1 = c._engine_rows(b1, keep.cuda(), s_row[keep])
                sh1, sht = {}, {}
                for i in DEC:
                    xi = xin[i].index_select(0, r1.index).contiguous()
                    ref = ops.drive(ced.decoder_attention_gen(layers[i], xi, cfg, i, r1, None, sh1, step=st1, **KW))
                    got = red.run(ced.decoder_attention_gen(shards[i], xi, cfg, i, r1, None, sht, step=stt, tp=world,
                                                            **KW))
                    checks += 1
                    if not eq(got, ref):
                        bad.append(("engine", plan[0], i))
            for a, b in zip(c._pool_bytes(pools), c._pool_bytes(pools1)):
                checks += 1
                if not torch.equal(a, b):
                    bad.append(("engine_pools", plan[0]))
    torch.cuda.synchronize()
    flags = torch.tensor([checks, len(bad)], dtype=torch.int64, device="cuda")
    allf = [torch.empty_like(flags) for _ in range(world)]
    dist.all_gather(allf, flags)
    if rank == 0:
        out.put({"world": world, "checks_per_rank": [int(f[0]) for f in allf],
                 "mismatches_per_rank": [int(f[1]) for f in allf], "first_rank0": bad[:5],
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
