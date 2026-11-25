"""Decode cache: incremental == full-sequence (bitwise), batch isolation, stale-weight refusal."""
import pytest
import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100 import cache as dc
from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops
from isoexec.ops.attention.deepseek_v41_cache_sm100.tests import _harness as h

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(autouse=True, params=[1, 3], ids=["sink_v1", "dualrail_sink_v1"])
def sink_recipe(request, monkeypatch):
    """Every relation below holds for each sparse-attention recipe the package serves (ops.DEFAULT_SINK_RECIPE)."""
    monkeypatch.setattr(ops, "DEFAULT_SINK_RECIPE", request.param)
    return request.param


def _inputs(cfg, B, T, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn((B, T, cfg.hidden_size), generator=g, device="cuda").to(torch.bfloat16)
            for _ in range(cfg.num_hidden_layers)]


def _full(cfg, layers, xs):
    state, outs = {}, []
    with torch.no_grad():
        for i, (m, x) in enumerate(zip(layers, xs)):
            outs.append(ops.attention_full(m, x, cfg, i, state, linear_fn=h.native_linear, dequant_fn=h.dequant))
    return outs


def _prefill(cfg, layers, xs, cache, rows):
    state = {}
    with torch.no_grad():
        for i, (m, x) in enumerate(zip(layers, xs)):
            ops.attention_full(m, x, cfg, i, state, linear_fn=h.native_linear, dequant_fn=h.dequant, cache=cache,
                               cache_rows=rows)
    cache.advance(xs[0].shape[1], rows=rows)


def _step(cfg, layers, xs_t, cache):
    outs = [dc.attention_decode(m, x, cfg, i, cache, linear_fn=h.native_linear, dequant_fn=h.dequant)
            for i, (m, x) in enumerate(zip(layers, xs_t))]
    cache.advance()
    return outs


def _eq(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("variant", ["production_selection", "active_candidate_filter"])
@pytest.mark.parametrize("prefill", [1, 301])
def test_incremental_equals_full_sequence(variant, prefill):
    kw = {} if variant == "production_selection" else {"index_topk": 448, "candidate_blocks": 64}
    cfg = h.native_cfg(**kw)
    layers = [h.make_layer(cfg, i, i) for i in range(cfg.num_hidden_layers)]
    B, T = 2, 700 if variant == "active_candidate_filter" else 420
    xs = _inputs(cfg, B, T, 3)
    full = _full(cfg, layers, xs)
    cache = dc.DecodeCache(cfg, B, 1024)
    _prefill(cfg, layers, [x[:, :prefill].contiguous() for x in xs], cache, None)
    bad = []
    for t in range(prefill, T):
        outs = _step(cfg, layers, [x[:, t:t + 1].contiguous() for x in xs], cache)
        for i, (o, f) in enumerate(zip(outs, full)):
            if not _eq(o, f[:, t:t + 1]):
                bad.append((t, i))
    assert not bad, bad[:10]


def test_batch_isolation_with_different_lengths():
    cfg = h.native_cfg(index_topk=448, candidate_blocks=64)
    layers = [h.make_layer(cfg, i, i) for i in range(cfg.num_hidden_layers)]
    lengths, steps, T = (200, 257, 1), 40, 300
    xs = _inputs(cfg, len(lengths), T, 5)
    cache = dc.DecodeCache(cfg, len(lengths), 512)
    for b, n in enumerate(lengths):
        _prefill(cfg, layers, [x[b:b + 1, :n].contiguous() for x in xs], cache, [b])
    outs = []
    for s in range(steps):
        step_x = [torch.stack([x[b, n + s] for b, n in enumerate(lengths)])[:, None].contiguous() for x in xs]
        outs.append(_step(cfg, layers, step_x, cache))
    for b, n in enumerate(lengths):
        full = _full(cfg, layers, [x[b:b + 1, :n + steps].contiguous() for x in xs])
        for s in range(steps):
            for i in range(cfg.num_hidden_layers):
                assert _eq(outs[s][i][b:b + 1], full[i][:, n + s:n + s + 1]), (b, s, i)


def test_weight_reload_invalidates_cache():
    cfg = h.native_cfg()
    layers = [h.make_layer(cfg, i, i) for i in range(cfg.num_hidden_layers)]
    B, P = 2, 64
    xs = _inputs(cfg, B, P + 1, 9)
    cache = dc.DecodeCache(cfg, B, 256)
    _prefill(cfg, layers, [x[:, :P].contiguous() for x in xs], cache, None)
    step = [x[:, P:P + 1].contiguous() for x in xs]
    _step(cfg, layers, step, cache)
    cache.pos -= 1
    # in-place reload of a compressor and an output weight (what a trainer -> rollout sync does)
    with torch.no_grad():
        layers[1].compressor.wgate.weight.mul_(0.5)
        layers[1].wo_a.weight.mul_(2.0)
    with pytest.raises(dc.StaleCacheError):
        _step(cfg, layers, step, cache)
    cache.reset(rows=[0])          # a partial reset cannot rebind changed weights: row 1 holds old state
    with pytest.raises(dc.StaleCacheError):
        _prefill(cfg, layers, [x[:1, :P].contiguous() for x in xs], cache, [0])
    cache.reset()
    _prefill(cfg, layers, [x[:, :P].contiguous() for x in xs], cache, None)
    got = _step(cfg, layers, step, cache)
    fresh = [h.make_layer(cfg, i, i) for i in range(cfg.num_hidden_layers)]
    for a, b in zip(fresh, layers):
        a.load_state_dict(b.state_dict())
    full = _full(cfg, fresh, [x[:, :P + 1].contiguous() for x in xs])
    for o, f in zip(got, full):
        assert _eq(o, f[:, P:P + 1])


def _chunked(cfg, layers, xs, cache, chunks, s0, s1):
    outs = [[] for _ in layers]
    t = cache.pos_host[s0]
    for n in chunks:
        for i, (m, x) in enumerate(zip(layers, xs)):
            outs[i].append(dc.attention_chunk(m, x[s0:s1, t:t + n].contiguous(), cfg, i, cache, rows=(s0, s1),
                                              linear_fn=h.native_linear, dequant_fn=h.dequant))
        cache.advance(n, rows=range(s0, s1))
        t += n
    return [torch.cat(o, 1) for o in outs]


@pytest.mark.parametrize("plan", ["chunks_from_empty", "prefill_then_chunks"])
def test_chunked_prefill_then_decode_equals_full_sequence(plan):
    cfg = h.native_cfg(index_topk=448, candidate_blocks=64)
    layers = [h.make_layer(cfg, i, i) for i in range(cfg.num_hidden_layers)]
    B, T = 2, 760
    xs = _inputs(cfg, B, T, 23)
    full = _full(cfg, layers, xs)
    cache = dc.DecodeCache(cfg, B, 1024, max_chunk=300)
    if plan == "chunks_from_empty":
        chunks, start = [5, 128, 1, 300, 77, 2], 0
    else:
        _prefill(cfg, layers, [x[:, :131].contiguous() for x in xs], cache, None)
        chunks, start = [3, 256, 129, 1, 64], 131
    got = _chunked(cfg, layers, xs, cache, chunks, 0, B)
    end = start + sum(chunks)
    for i in range(cfg.num_hidden_layers):
        assert _eq(got[i], full[i][:, start:end]), i
    for t in range(end, T):
        outs = _step(cfg, layers, [x[:, t:t + 1].contiguous() for x in xs], cache)
        for i, (o, f) in enumerate(zip(outs, full)):
            assert _eq(o, f[:, t:t + 1]), (t, i)


def test_chunked_rows_of_different_lengths():
    cfg = h.native_cfg(index_topk=448, candidate_blocks=64)
    layers = [h.make_layer(cfg, i, i) for i in range(cfg.num_hidden_layers)]
    xs = _inputs(cfg, 3, 400, 29)
    cache = dc.DecodeCache(cfg, 3, 512, max_chunk=200)
    plans = {0: [7, 200, 33], 1: [1, 1, 150], 2: [199]}
    got = {s: _chunked(cfg, layers, xs, cache, c, s, s + 1) for s, c in plans.items()}
    lengths = [sum(c) for _, c in sorted(plans.items())]
    for s, n in enumerate(lengths):
        full = _full(cfg, layers, [x[s:s + 1, :n].contiguous() for x in xs])
        for i in range(cfg.num_hidden_layers):
            assert _eq(got[s][i], full[i]), (s, i)
    for k in range(20):                                 # then decode all rows together at their own positions
        step_x = [torch.stack([x[b, n + k] for b, n in enumerate(lengths)])[:, None].contiguous() for x in xs]
        outs = _step(cfg, layers, step_x, cache)
        if k == 19:
            for s, n in enumerate(lengths):
                full = _full(cfg, layers, [x[s:s + 1, :n + 20].contiguous() for x in xs])
                for i in range(cfg.num_hidden_layers):
                    assert _eq(outs[i][s:s + 1], full[i][:, n + 19:n + 20]), (s, i)
