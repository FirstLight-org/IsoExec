"""Paged caches (vLLM/SGLang-style block tables, V4.1 FP8 528 B window and compressed records, paged FP8 index
keys, FP32 partial-group state slots) with mixed prefill + decode steps: bitwise equal to the contiguous
full-sequence program (recipe 3, fixed-slot lists) at every token, under shuffled page assignment; TP 1/2/4/8
invariant; a CUDA-graph-captured decode step replays bit-identically."""
import dataclasses

import pytest
import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops, paged, tp
from isoexec.ops.attention.deepseek_v41_cache_sm100.tests import _harness as h

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
MAXLEN = 1024


def _eq(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _setup(nseq, seed=61):
    cfg = h.native_cfg(index_topk=448, candidate_blocks=64)
    layers = [h.make_layer(cfg, i, i) for i in range(cfg.num_hidden_layers)]
    g = torch.Generator(device="cuda").manual_seed(seed)
    xs = [torch.randn((nseq, MAXLEN, cfg.hidden_size), generator=g, device="cuda").to(torch.bfloat16)
          for _ in layers]
    return cfg, layers, xs


def _full(cfg, layers, xs, s, L):
    state = {}
    with torch.no_grad():
        return [ops.attention_full(m, x[s:s + 1, :L].contiguous(), cfg, i, state, linear_fn=h.native_linear,
                                   dequant_fn=h.dequant, recipe=3)[0] for i, (m, x) in enumerate(zip(layers, xs))]


def _tables(nseq, seed):
    """Shuffled page ids: sequence s owns pages perm[s*P:(s+1)*P] of every pool (window / ratio 2 / ratio 1)."""
    g = torch.Generator().manual_seed(seed)
    P = MAXLEN // 64
    win = torch.randperm(nseq * P, generator=g).view(nseq, P).int().cuda()
    comp = {2: torch.randperm(nseq * P, generator=g).view(nseq, P).int().cuda(),
            1: torch.randperm(nseq * P, generator=g).view(nseq, P).int().cuda()}
    return win, comp, P


def _step(cfg, layers, xs, pools, plan, win, comp, slots, worlds_shards=None):
    """plan: list of (seq, p0, n). Returns per layer [N, hid] outputs."""
    p0 = [a for _, a, _ in plan]
    n = [c for _, _, c in plan]
    seqs = [s for s, _, _ in plan]
    sel = torch.tensor(seqs, device="cuda").long()
    batch = paged.PagedBatch.make(p0, n, win[sel], {r: b[sel] for r, b in comp.items()}, [slots[s] for s in seqs],
                                  ratios=(2,))
    outs = []
    if worlds_shards is not None:
        batches = [dataclasses.replace(batch, step={}) for _ in worlds_shards[1]]
    with torch.no_grad():
        for i, (m, x) in enumerate(zip(layers, xs)):
            xi = torch.cat([x[s, a:a + c] for s, a, c in plan], 0).contiguous()
            if worlds_shards is None:
                outs.append(paged.attention_paged(m, xi, cfg, i, pools, batch, linear_fn=h.native_linear,
                                                  dequant_fn=h.dequant))
            else:
                shards, rank_pools = worlds_shards
                outs.append(tp.attention_paged_tp(shards[i], xi, cfg, i, rank_pools, batches,
                                                  linear_fn=h.native_linear, dequant_fn=h.dequant))
    return outs


SCHEDULE = [[(0, 0, 300), (1, 0, 100)],                       # two prefills
            [(0, 300, 1), (1, 100, 37), (2, 0, 50)],          # decode + chunk + prefill mixed
            [(2, 50, 1), (0, 301, 1), (1, 137, 1)]] + [[(0, 302 + k, 1), (1, 138 + k, 1), (2, 51 + k, 1)]
                                                       for k in range(12)]


def _pools(cfg, nseq, P):
    return paged.PagedPools(cfg, window_pages=nseq * P, comp_pages={2: nseq * P, 1: nseq * P}, state_slots=nseq,
                            max_len=MAXLEN)


def test_paged_mixed_steps_equal_contiguous_full_sequence():
    cfg, layers, xs = _setup(3)
    win, comp, P = _tables(3, 5)
    pools = _pools(cfg, 3, P)
    slots = {0: 2, 1: 0, 2: 1}
    lengths = {0: 314, 1: 150, 2: 63}
    full = {s: _full(cfg, layers, xs, s, L) for s, L in lengths.items()}
    for plan in SCHEDULE:
        outs = _step(cfg, layers, xs, pools, plan, win, comp, slots)
        for i, o in enumerate(outs):
            off = 0
            for s, a, c in plan:
                assert _eq(o[off:off + c], full[s][i][a:a + c]), (plan, i, s)
                off += c


def test_selected_lengths_are_passed_to_the_leaf_and_change_no_bit(monkeypatch):
    """The schedule port: the engine path tells the leaf each row's occupied prefix of the selected list
    (``paged.selected_lengths``: min(k, visible compressed keys)), so blocks of empty positions are not run. The same
    steps with ``TRIM_SELECTED`` off (every block runs) give the same bits at every token, and the length is an
    upper bound of every row's occupied prefix."""
    from isoexec.ops.attention.sparse_sm100 import dualrail

    runs = {}
    for trim in (False, True):
        monkeypatch.setattr(paged, "TRIM_SELECTED", trim)
        seen = []
        real = dualrail.dualrail_attn

        def spy(q, kv, lists, scale, **kw):
            if kw.get("extra_lists") is not None:
                ln = kw.get("extra_topk_length")
                seen.append(ln is not None)
                if ln is not None:
                    occupied = (kw["extra_lists"] >= 0).sum(1)
                    last = torch.where(kw["extra_lists"] >= 0, torch.arange(kw["extra_lists"].shape[1], device="cuda")[None] + 1, 0).amax(1)
                    assert bool((last <= ln).all()) and bool((occupied <= ln).all()), "a selected key lies past the length"
            return real(q, kv, lists, scale, **kw)
        monkeypatch.setattr(dualrail, "dualrail_attn", spy)
        cfg, layers, xs = _setup(3)
        win, comp, P = _tables(3, 5)
        pools = _pools(cfg, 3, P)
        slots = {0: 2, 1: 0, 2: 1}
        runs[trim] = [_step(cfg, layers, xs, pools, plan, win, comp, slots) for plan in SCHEDULE]
        assert seen and all(x == trim for x in seen), (trim, seen[:8])
        monkeypatch.setattr(dualrail, "dualrail_attn", real)
    for a, b in zip(runs[False], runs[True]):
        for x, y in zip(a, b):
            assert _eq(x, y)


def test_paged_tp_degree_invariance():
    cfg, layers, xs = _setup(3, seed=67)
    win, comp, P = _tables(3, 9)
    slots = {0: 0, 1: 1, 2: 2}
    sched = SCHEDULE[:4]
    ref = None
    for world in (1, 2, 4, 8):
        shards = [[tp.shard_layer(m, cfg, r, world) for r in range(world)] for m in layers]
        rank_pools = [_pools(cfg, 3, P) for _ in range(world)]
        got = [_step(cfg, layers, xs, None, plan, win, comp, slots, (shards, rank_pools)) for plan in sched]
        if ref is None:
            ref = got
            continue
        for a, b in zip(got, ref):
            for x, y in zip(a, b):
                assert _eq(x, y), world


def test_paged_decode_step_cuda_graph_replay():
    cfg, layers, xs = _setup(4, seed=71)
    win, comp, P = _tables(4, 13)
    slots = {s: s for s in range(4)}
    pools = _pools(cfg, 4, P)
    lengths = [77, 130, 1, 64]
    _step(cfg, layers, xs, pools, [(s, 0, L) for s, L in enumerate(lengths)], win, comp, slots)
    eager_pools = _pools(cfg, 4, P)
    _step(cfg, layers, xs, eager_pools, [(s, 0, L) for s, L in enumerate(lengths)], win, comp, slots)
    sel = torch.arange(4, device="cuda")
    batch = paged.PagedBatch.make(lengths, [1] * 4, win[sel], {r: b[sel] for r, b in comp.items()}, [0, 1, 2, 3],
                                  ratios=(2,))
    xin = [torch.empty((4, cfg.hidden_size), dtype=torch.bfloat16, device="cuda") for _ in layers]

    def run():
        batch.step = {}
        return [paged.attention_paged(m, xin[i], cfg, i, pools, batch, linear_fn=h.native_linear,
                                      dequant_fn=h.dequant) for i, m in enumerate(layers)]
    # warm every plan / table outside capture on a scratch copy of the pools
    for i in range(len(layers)):
        xin[i].copy_(torch.stack([xs[i][s, lengths[s]] for s in range(4)]))
    scratch = _pools(cfg, 4, P)
    for k in ("window", "comp", "index", "state"):
        for key, t in getattr(pools, k).items():
            getattr(scratch, k)[key].copy_(t)
    saved = pools
    pools = scratch
    run()
    pools = saved
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gout = run()
    for k in range(6):
        cur = [L + k for L in lengths]
        for i in range(len(layers)):
            xin[i].copy_(torch.stack([xs[i][s, cur[s]] for s in range(4)]))
        batch.tok_pos.copy_(torch.tensor(cur, dtype=torch.int32))
        batch.p0.copy_(torch.tensor(cur, dtype=torch.int32))
        graph.replay()
        eager = _step(cfg, layers, xs, eager_pools, [(s, cur[s], 1) for s in range(4)], win, comp, slots)
        for a, b in zip(gout, eager):
            assert _eq(a, b), k
