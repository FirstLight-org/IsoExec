"""TP-degree-invariant head sharding, simulated on one GPU (shards in lockstep, reductions by rank id).

Relations (all bitwise): the leaf-tree kernel == its CPU emulation and == completing the tree from
aligned subtree roots; TP = 1 driver == the unsharded program; TP = 1 == 2 == 4 == 8 for every layer
output and every selected index list (identical on every rank), independent of shard execution order;
batch invariance at TP 8; decode at TP t == the full sequence (computed unsharded).
"""
import pytest
import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100 import cache as dc
from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops, tp
from isoexec.ops.attention.deepseek_v41_cache_sm100.tests import _harness as h

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(autouse=True, params=[1, 3], ids=["sink_v1", "dualrail_sink_v1"])
def sink_recipe(request, monkeypatch):
    """Every relation below holds for each sparse-attention recipe the package serves (ops.DEFAULT_SINK_RECIPE)."""
    monkeypatch.setattr(ops, "DEFAULT_SINK_RECIPE", request.param)
    return request.param
WORLDS = (1, 2, 4, 8)


def _eq(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _setup(B, T, seed=41):
    cfg = h.native_cfg(index_topk=448, candidate_blocks=64)
    layers = [h.make_layer(cfg, i, i) for i in range(cfg.num_hidden_layers)]
    g = torch.Generator(device="cuda").manual_seed(seed)
    xs = [torch.randn((B, T, cfg.hidden_size), generator=g, device="cuda").to(torch.bfloat16) for _ in layers]
    return cfg, layers, xs


def _run_tp(cfg, layers, xs, world, order=None):
    shards = [[tp.shard_layer(m, cfg, r, world) for r in range(world)] for m in layers]
    states = [{} for _ in range(world)]
    outs, topks = [], []
    with torch.no_grad():
        for i, x in enumerate(xs):
            outs.append(tp.attention_full_tp(shards[i], x, cfg, i, states, linear_fn=h.native_linear,
                                             dequant_fn=h.dequant, order=order))
            if i in cfg.index_source_layer_ids and cfg.compress_ratios[i]:
                topks.append([s["topk_native"].clone() for s in states])
    return outs, topks, shards


def _tree_emul(x):
    x = list(x)
    while len(x) > 1:
        x = [x[2 * k] + x[2 * k + 1] for k in range(len(x) // 2)]
    return x[0]


def test_leaf_tree_matches_emulation_and_subtree_completion():
    g = torch.Generator(device="cuda").manual_seed(3)
    leaves = [torch.randn((37, 1001), generator=g, device="cuda") * 10 for _ in range(8)]
    full = ops.leaf_tree(leaves, final_bf16=True)
    want = _tree_emul([leaf.cpu() for leaf in leaves]).to(torch.bfloat16)
    assert torch.equal(full.cpu().view(torch.int16), want.view(torch.int16))
    for world in (2, 4, 8):
        k = 8 // world
        roots = [ops.leaf_tree(leaves[r * k:(r + 1) * k], final_bf16=False) if k > 1 else leaves[r]
                 for r in range(world)]
        assert _eq(tp.tree_combine(roots), full), world


def test_tp1_driver_is_the_unsharded_program():
    cfg, layers, xs = _setup(2, 700)
    state = {}
    with torch.no_grad():
        full = [ops.attention_full(m, x, cfg, i, state, linear_fn=h.native_linear, dequant_fn=h.dequant)
                for i, (m, x) in enumerate(zip(layers, xs))]
    got, _, _ = _run_tp(cfg, layers, xs, 1)
    for a, b in zip(got, full):
        assert _eq(a, b)


def test_tp_degree_invariance():
    cfg, layers, xs = _setup(2, 1100)
    runs = {w: _run_tp(cfg, layers, xs, w) for w in WORLDS}
    base_out, base_topk, _ = runs[1]
    for w in WORLDS[1:]:
        outs, topks, _ = runs[w]
        for i in range(len(xs)):
            assert _eq(outs[i], base_out[i]), (w, i)
        for per_rank, ref in zip(topks, base_topk):
            assert all(torch.equal(t, ref[0]) for t in per_rank), w
    rev, _, _ = _run_tp(cfg, layers, xs, 8, order=list(reversed(range(8))))
    one, _, _ = _run_tp(cfg, layers, [x[1:2] for x in xs], 8)
    for i in range(len(xs)):
        assert _eq(rev[i], base_out[i]) and _eq(one[i], base_out[i][1:2]), i


@pytest.mark.parametrize("world", [2, 8])
def test_tp_decode_equals_full_sequence(world):
    cfg, layers, xs = _setup(2, 520, seed=43)
    state = {}
    with torch.no_grad():
        full = [ops.attention_full(m, x, cfg, i, state, linear_fn=h.native_linear, dequant_fn=h.dequant)
                for i, (m, x) in enumerate(zip(layers, xs))]
    shards = [[tp.shard_layer(m, cfg, r, world) for r in range(world)] for m in layers]
    P = 301
    caches = [dc.DecodeCache(cfg, 2, 600) for _ in range(world)]
    states = [{} for _ in range(world)]
    with torch.no_grad():
        for i, x in enumerate(xs):
            gens = [ops.attention_full_gen(shards[i][r], x[:, :P].contiguous(), cfg, i, states[r],
                                           linear_fn=h.native_linear, dequant_fn=h.dequant, cache=caches[r],
                                           tp=world) for r in range(world)]
            tp._lockstep(gens, list(range(world)))
    for c in caches:
        c.advance(P)
    for t in range(P, 520):
        for i, x in enumerate(xs):
            o = tp.attention_decode_tp(shards[i], x[:, t:t + 1].contiguous(), cfg, i, caches,
                                       linear_fn=h.native_linear, dequant_fn=h.dequant)
            assert _eq(o, full[i][:, t:t + 1]), (t, i)
        for c in caches:
            c.advance()


@pytest.mark.parametrize("M", [1, 256, 1100])
def test_wo_b_single_launch_tree_equals_leaf_form(M):
    g = torch.Generator(device="cuda").manual_seed(M)

    class Mod:
        pass
    for world in (1, 2, 4, 8):
        mod = Mod()
        mod.weight = (torch.randn((1024, 8192 // world), generator=g, device="cuda") / 90).to(torch.bfloat16)
        y = torch.randn((M, 8192 // world), generator=g, device="cuda").to(torch.bfloat16)
        with torch.no_grad():
            a = ops.wo_b_tree(mod, y, world)
            b = ops.wo_b_leaves(mod, y, world)
        assert a.dtype == (torch.bfloat16 if world == 1 else torch.float32)
        assert torch.equal(a.view(torch.int16 if world == 1 else torch.int32),
                           b.view(torch.int16 if world == 1 else torch.int32)), world


def test_tp_chunked_prefill_degree_invariance():
    cfg, layers, xs = _setup(2, 700, seed=47)
    outs = {}
    for world in WORLDS:
        shards = [[tp.shard_layer(m, cfg, r, world) for r in range(world)] for m in layers]
        caches = [dc.DecodeCache(cfg, 2, 1024, max_chunk=256) for _ in range(world)]
        got, t = [[] for _ in layers], 0
        for n in (37, 256, 1, 200):
            for i, x in enumerate(xs):
                got[i].append(tp.attention_chunk_tp(shards[i], x[:, t:t + n].contiguous(), cfg, i, caches, rows=(0, 2),
                                                    linear_fn=h.native_linear, dequant_fn=h.dequant))
            for c in caches:
                c.advance(n, rows=range(2))
            t += n
        for k in range(t, t + 20):                     # then decode
            for i, x in enumerate(xs):
                got[i].append(tp.attention_decode_tp(shards[i], x[:, k:k + 1].contiguous(), cfg, i, caches,
                                                     linear_fn=h.native_linear, dequant_fn=h.dequant))
            for c in caches:
                c.advance()
        outs[world] = [torch.cat(g, 1) for g in got]
    state = {}
    with torch.no_grad():
        full = [ops.attention_full(m, x[:, :t + 20].contiguous(), cfg, i, state, linear_fn=h.native_linear,
                                   dequant_fn=h.dequant) for i, (m, x) in enumerate(zip(layers, xs))]
    for world in WORLDS:
        for i in range(len(xs)):
            assert _eq(outs[world][i], full[i]), (world, i)


def test_leaf_tree_capable_linear_fn_hook():
    cfg, layers, xs = _setup(1, 300, seed=53)
    calls = []

    def provider(m, x, leaves=None, fp32_root=None):
        if leaves is None:
            return h.native_linear(m, x)
        calls.append(leaves)
        return ops.wo_b_tree(m, x, 8 // leaves)
    provider.supports_leaf_tree = True
    s1, s2 = {}, {}
    with torch.no_grad():
        for i, (m, x) in enumerate(zip(layers, xs)):
            a = ops.attention_full(m, x, cfg, i, s1, linear_fn=provider, dequant_fn=h.dequant)
            b = ops.attention_full(m, x, cfg, i, s2, linear_fn=h.native_linear, dequant_fn=h.dequant)
            assert _eq(a, b)
    assert calls == [8] * len(layers)
