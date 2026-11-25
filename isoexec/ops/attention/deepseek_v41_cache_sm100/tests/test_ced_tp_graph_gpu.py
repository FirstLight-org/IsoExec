"""The CED entry points (``ced.entry_projection``, ``ced.decoder_attention_gen``) at TP > 1 and under CUDA graphs.

* TP: head shards (``tp.shard_layer``) of TP 2 / 4 / 8, simulated in ONE process (shards in lockstep, the wo_b roots
  completed by rank id, ``tp._lockstep`` + ``tp.tree_combine``, as tests/test_tp_gpu.py; the multi-process NCCL form is
  tests/dist_ced_tp_run.py) == TP 1 bit for bit: every layer output; per head the attention output each rank hands to
  ``ops.output_projection`` (rank r's heads == TP 1's heads [r H/t, (r+1) H/t)); the selections (top-k, candidate
  pool) on every rank; the entry projection (replicated); the pool bytes every rank writes. Trainer (decoder segments
  with window floors) and engine (bounded replay over chunked prefill + decode, floors on the layers after b).
* CUDA graph: one engine decode step (the entry projection on every row, then the decoder layers b.. on the decoder
  rows) captured once, FULL_DECODE_ONLY style, at a BUCKET of rows larger than the live batch: the bucket's padded rows
  carry ``valid = False`` (they write no record and attend to nothing), positions, inputs and the per-row window FLOOR
  are device tensors updated in place between replays (the floor enters as data like ``tok_pos``, never as a captured
  constant; model/REQUESTS.md R4). Every replay == the eager, unpadded step on its own pools, bit for bit (outputs of
  the live rows and every pool byte), with this package's window lists and with the engine's own (floored) lists.

Layers of ``_harness.native_cfg``: 4 = the boundary, 5 = Reuse, 6 = Reindex (tests/test_ced_gpu.py)."""
import dataclasses
from types import SimpleNamespace

import pytest
import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100 import ced, ops, paged, tp
from isoexec.ops.attention.deepseek_v41_cache_sm100._extension import extension
from isoexec.ops.attention.deepseek_v41_cache_sm100.tests import _harness as h
from isoexec.ops.attention.deepseek_v41_cache_sm100.tests import test_ced_gpu as c

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEC, KW, W, _eq = c.DEC, c.KW, c.W, c._eq


# ---------------------------------------------------------------------------------------------- per-head capture
@pytest.fixture
def heads_log(monkeypatch):
    """Every ``ops.output_projection`` call: (module, attention output [R, H_rank, D], o_roped)."""
    log = []
    orig = ops.output_projection

    def rec(module, o, *a, **k):
        log.append((module, o.detach().clone(), bool(k.get("o_roped", False))))
        return orig(module, o, *a, **k)
    monkeypatch.setattr(ops, "output_projection", rec)
    return log


def _o_of(log, module):
    hits = [(o, roped) for m, o, roped in log if m is module]
    assert hits, "no output projection recorded for this module"
    return hits[-1]


def _check_heads(log, ref_module, shard_modules, world, H):
    """Rank r's attention output == the TP-1 output's heads [r H/t, (r+1) H/t), bit for bit. Returns the number of
    ranks compared (a rank whose leaf fused the inverse RoPE while TP 1's did not is not comparable here)."""
    o1, roped1 = _o_of(log, ref_module)
    hr = H // world
    n = 0
    for r, m in enumerate(shard_modules):
        o, roped = _o_of(log, m)
        assert o.shape[1] == hr
        if roped != roped1:
            continue
        assert _eq(o, o1[:, r * hr:(r + 1) * hr]), (world, r)
        n += 1
    return n


def _shards(cfg, layers, world):
    return {i: [tp.shard_layer(layers[i], cfg, r, world) for r in range(world)] for i in DEC}


# ---------------------------------------------------------------------------------------------- trainer
@pytest.mark.parametrize("world", [2, 4, 8])
def test_trainer_entry_points_tp_equal_tp1_per_head(world, heads_log):
    B, T = 2, 400
    cfg, layers, xs = c._setup(B, T, seed=23)
    H = cfg.num_attention_heads
    shards = _shards(cfg, layers, world)
    starts, lengths = [T - 150, 37], [T, T - 23]
    with torch.no_grad():
        dkv, cu = c._entry(cfg, layers, xs, lengths)
        hp = torch.cat([xs[4][s, :L] for s, L in enumerate(lengths)])
        pos = torch.cat([torch.arange(L) for L in lengths]).cuda()
        for r in (0, world - 1):            # the entry projection is replicated: every rank's shard gives TP 1's rows
            m, k, wv = ced.entry_projection(shards[4][r], hp, cfg, 4, pos, linear_fn=h.native_linear)
            assert _eq(m, dkv.main_kv) and _eq(wv, dkv.window_kv)
            assert torch.equal(k[0], dkv.index_k[0]) and torch.equal(k[1], dkv.index_k[1])
        seq_all = torch.cat([torch.full((L,), s) for s, L in enumerate(lengths)])
        pos_all = torch.cat([torch.arange(L) for L in lengths])
        s_all = torch.tensor(starts)[seq_all]
        keep = pos_all >= s_all                       # the decoder segments [s_b, L_b), floors s_b
        rows = c._decoder_rows(seq_all[keep], pos_all[keep], cu, s_all[keep])
        x = {i: xs[i][rows.sequence, rows.positions].contiguous() for i in DEC}
        # TP 1: outputs and the decoder chain's selections after each layer
        shared1, ref, sel1 = {}, {}, {}
        heads_log.clear()
        for i in DEC:
            ref[i] = ops.drive(ced.decoder_attention_gen(layers[i], x[i], cfg, i, rows, dkv, shared1, **KW))
            sel1[i] = (shared1["topk_rows"].clone(), shared1["candidates_rows"].clone())
        compared = 0
        for order in (list(range(world)), list(reversed(range(world)))):
            shared = [{} for _ in range(world)]
            for i in DEC:
                gens = [ced.decoder_attention_gen(shards[i][r], x[i], cfg, i, rows, dkv, shared[r], tp=world, **KW)
                        for r in range(world)]
                got = tp.tree_combine(tp._lockstep(gens, order))
                assert got.dtype == torch.bfloat16
                assert _eq(got, ref[i]), (world, i, order[0])
                if order[0] == 0:
                    compared += _check_heads(heads_log, layers[i], shards[i], world, H)
                for r in range(world):
                    assert torch.equal(shared[r]["topk_rows"], sel1[i][0]), (world, i, r)
                    assert torch.equal(shared[r]["candidates_rows"], sel1[i][1]), (world, i, r)
        assert compared == world * len(DEC)           # the trainer leaf never fuses the inverse RoPE


# ---------------------------------------------------------------------------------------------- engine, TP
def _engine_step_tp(cfg, layers_or_shards, world, pools, batches, xin, rows, shareds):
    """Entry projection on every token of the step, then the decoder layers on ``rows``; per-rank pools / batches.
    Returns {layer: reduced output}."""
    steps = [SimpleNamespace(pools=pools[r], batch=batches[r]) for r in range(world)]
    for r in range(world):
        mod = layers_or_shards[4][r] if world > 1 else layers_or_shards[4]
        assert ced.entry_projection(mod, xin[4], cfg, 4, batches[r].tok_pos, linear_fn=h.native_linear,
                                    step=steps[r]) is None
    out = {}
    if rows is None:
        return out
    sel = rows.index
    for i in DEC:
        xi = xin[i].index_select(0, sel).contiguous()
        if world == 1:
            out[i] = ops.drive(ced.decoder_attention_gen(layers_or_shards[i], xi, cfg, i, rows, None, shareds[0],
                                                         step=steps[0], **KW))
        else:
            gens = [ced.decoder_attention_gen(layers_or_shards[i][r], xi, cfg, i, rows, None, shareds[r],
                                              step=steps[r], tp=world, **KW) for r in range(world)]
            out[i] = tp.tree_combine(tp._lockstep(gens, list(range(world))))
    return out


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("engine_lists", [False, True], ids=["pkg_lists", "engine_lists"])
def test_engine_entry_points_tp_equal_tp1_per_head(world, engine_lists, heads_log):
    """Bounded replay (entry projection on every chunk token, decoder rows at positions >= s = max(0, P - W), floors
    s on the layers after b) over test_ced_gpu's chunked prefill + decode schedule, TP t == TP 1."""
    cfg, layers, xs = c._setup(3, c.MAXLEN, seed=31)
    H = cfg.num_attention_heads
    shards = _shards(cfg, layers, world)
    win, comp = c._tables(3, 11)
    pools1 = c._pools(cfg, 3)
    pools_t = [c._pools(cfg, 3) for _ in range(world)]
    start = {s: max(0, P - W) for s, P in c.PROMPTS.items()}
    compared = 0
    with torch.no_grad():
        for plan in c.STEPS:
            xin = c._rows_in(xs, plan)
            b1 = c._batch(plan, win, comp, engine_lists)
            bt = [dataclasses.replace(c._batch(plan, win, comp, engine_lists), step={}) for _ in range(world)]
            seqs = torch.cat([torch.full((n,), s) for s, a, n in plan])
            poss = torch.cat([torch.arange(a, a + n) for s, a, n in plan])
            s_row = torch.tensor([start[int(s)] for s in seqs])
            keep = torch.nonzero(poss >= s_row).reshape(-1)
            rows = None if keep.numel() == 0 else c._engine_rows(b1, keep.cuda(), s_row[keep])
            heads_log.clear()
            ref = _engine_step_tp(cfg, layers, 1, [pools1], [b1], xin, rows, [{}])
            got = _engine_step_tp(cfg, shards, world, pools_t, bt, xin, rows, [{} for _ in range(world)])
            for i in ref:
                assert _eq(got[i], ref[i]), (plan, world, i)
                compared += _check_heads(heads_log, layers[i], shards[i], world, H)
            for r in range(world):
                for a, b in zip(c._pool_bytes(pools_t[r]), c._pool_bytes(pools1)):
                    assert torch.equal(a, b), (plan, world, r)
    assert compared > 0


# ---------------------------------------------------------------------------------------------- engine, CUDA graph
PROMPT = {0: 300, 1: 100, 2: 200}
LIVE = 3


def _copy_pools(dst, src):
    for k in ("window", "comp", "index", "state"):
        for key, t in getattr(src, k).items():
            getattr(dst, k)[key].copy_(t)


def _engine_lists(batch, valid=None):
    """The engine's own window addressing of a batch (test_ced_gpu._batch(engine_lists=True)), padded rows -1."""
    N = batch.tok_seq.numel()
    rows_, lists_ = {}, {}
    for i in DEC:
        rows_[i] = paged._rows(batch.win_bt, batch.tok_seq, batch.tok_pos, valid)
        m = torch.empty((N, W), dtype=torch.int32, device="cuda")
        extension().paged_lists(batch.tok_seq, batch.tok_pos, batch.win_bt, W, None, None, m, None)
        if valid is not None:
            m.masked_fill_(~valid[:, None], -1)
        lists_[i] = m
    return rows_, lists_


def _floors(cur, k):
    """Per-row window floors of replay ``k``: 0 at k == 0 (== no floor), then values that truncate some windows
    (p - f < W - 1), leave others whole and vary per replay."""
    if k == 0:
        return [0] * len(cur)
    return [max(0, p - (37 * k + 53 * s) % 140) for s, p in enumerate(cur)]


@pytest.mark.parametrize("bucket", [4, 8])
@pytest.mark.parametrize("engine_lists", [False, True], ids=["pkg_lists", "engine_lists"])
def test_engine_decode_step_cuda_graph_with_floor_and_bucket_padding_equals_eager(bucket, engine_lists):
    cfg, layers, xs = c._setup(bucket, c.MAXLEN, seed=37)
    win, comp = c._tables(bucket, 13)
    pools_e, pools_g, scratch = c._pools(cfg, bucket), c._pools(cfg, bucket), c._pools(cfg, bucket)
    with torch.no_grad():
        # prefill the live sequences eagerly (entry projection + decoder on all rows), then mirror the pools
        plan = [(s, 0, P) for s, P in PROMPT.items()]
        b0 = c._batch(plan, win, comp)
        xin0 = c._rows_in(xs, plan)
        N0 = xin0[4].shape[0]
        _engine_step_tp(cfg, layers, 1, [pools_e], [b0], xin0,
                        c._engine_rows(b0, torch.arange(N0, device="cuda"), None), [{}])
        _copy_pools(pools_g, pools_e)

        # the captured step: ``bucket`` rows, rows >= LIVE padded (own empty sequence slots, valid False)
        cur0 = [PROMPT[s] for s in range(LIVE)]
        sel = torch.arange(bucket, device="cuda")
        batch = paged.PagedBatch.make(cur0 + [0] * (bucket - LIVE), [1] * bucket, win[sel],
                                      {r: t[sel] for r, t in comp.items()}, list(range(bucket)), ratios=())
        valid = torch.arange(bucket, device="cuda") < LIVE
        batch.valid = valid
        if engine_lists:
            wr, wl = _engine_lists(batch, valid)
            batch.window_rows, batch.window_lists = wr, wl
            batch.win_bt = None
        floor = torch.zeros(bucket, dtype=torch.int32, device="cuda")      # device data, updated in place
        rows = c._engine_rows(batch, sel, floor)
        assert rows.window_floor is floor
        xin = {i: torch.zeros((bucket, cfg.hidden_size), dtype=torch.bfloat16, device="cuda") for i in DEC}

        def run(pools):
            batch.step = {}
            return _engine_step_tp(cfg, layers, 1, [pools], [batch], xin, rows, [{}])

        def load(k):
            cur = [PROMPT[s] + k for s in range(LIVE)]
            for i in DEC:
                xin[i][:LIVE].copy_(torch.stack([xs[i][s, cur[s]] for s in range(LIVE)]))
                xin[i][LIVE:].normal_()                      # padded rows: arbitrary inputs, never read back
            batch.tok_pos.copy_(torch.tensor(cur + [0] * (bucket - LIVE), dtype=torch.int32))
            batch.p0.copy_(batch.tok_pos)
            fl = _floors(cur, k)
            floor.copy_(torch.tensor(fl + [0] * (bucket - LIVE), dtype=torch.int32))
            if engine_lists:
                win_bt = win[sel]
                tmp = dataclasses.replace(batch, win_bt=win_bt)
                wr, wl = _engine_lists(tmp, valid)
                for i in DEC:
                    batch.window_rows[i].copy_(wr[i])
                    batch.window_lists[i].copy_(wl[i])
            return cur, fl

        load(0)
        _copy_pools(scratch, pools_g)          # warm every plan, table and cached weight outside capture
        run(scratch)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gout = run(pools_g)
        for k in range(6):
            cur, fl = load(k)
            graph.replay()
            # eager reference: the LIVE rows alone, unpadded, on their own pools; floor 0 at k == 0 vs no floor
            pl = [(s, cur[s], 1) for s in range(LIVE)]
            be = c._batch(pl, win, comp, engine_lists)
            fe = None if k == 0 else torch.tensor(fl)
            re = c._engine_rows(be, torch.arange(LIVE, device="cuda"), fe)
            eager = _engine_step_tp(cfg, layers, 1, [pools_e], [be], c._rows_in(xs, pl), re, [{}])
            torch.cuda.synchronize()
            for i in DEC:
                assert _eq(gout[i][:LIVE], eager[i]), (bucket, k, i)
            for a, b in zip(c._pool_bytes(pools_g), c._pool_bytes(pools_e)):
                assert torch.equal(a, b), (bucket, k)
        # the floors acted (last replay): the layers after b differ from the unfloored program on the truncated rows;
        # the boundary never applies the floor. Re-running the step on a copy writes the same record bytes.
        trunc = torch.tensor([cur[s] - fl[s] < W - 1 for s in range(LIVE)], device="cuda")
        assert bool(trunc.any()), "the schedule must truncate at least one window"
        _copy_pools(scratch, pools_e)
        be = c._batch(pl, win, comp, engine_lists)
        nofl = _engine_step_tp(cfg, layers, 1, [scratch], [be], c._rows_in(xs, pl),
                               c._engine_rows(be, torch.arange(LIVE, device="cuda"), None), [{}])
        assert _eq(nofl[4], gout[4][:LIVE])
        assert not _eq(nofl[5][trunc], gout[5][:LIVE][trunc])
        assert _eq(nofl[5][~trunc], gout[5][:LIVE][~trunc])
