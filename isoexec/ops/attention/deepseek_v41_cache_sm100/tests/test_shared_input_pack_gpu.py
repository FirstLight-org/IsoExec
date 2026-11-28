"""The projections that read the attention input as ONE GEMM (dense-lane D1 / D2): ``ops.pack_shared_input_rows`` packs
``wq_a | wkv`` and, on index-source layers, the derived indexer weights (``index_weight_rows``) into one storage; a
provider that runs one projection over adjacent members (``packs`` / ``packed``) then serves wq_a, wkv and the index
weights with one GEMM. Relations: every paged layer output of a mixed prefill / decode schedule equals the unpacked
run bit for bit; the derived rows are re-derived in place by ``refresh_derived_weights``."""
import copy
import dataclasses

import pytest
import torch

from isoexec.core.arch import live_device_arch
from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops
from isoexec.ops.attention.deepseek_v41_cache_sm100.tests import _harness as h
from isoexec.ops.attention.deepseek_v41_cache_sm100.tests import test_paged_gpu as tp_

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="sm100 kernels")


class PackingLinear:
    """``h.native_linear`` plus the provider protocol of the dense lane's ``BF16Linear.packs / packed``."""

    calls = 0

    def __call__(self, module, x):
        return h.native_linear(module, x)

    @staticmethod
    def _adjacent(ws):
        K, expect = ws[0].shape[1], ws[0].data_ptr()
        for w in ws:
            if w.shape[1] != K or not w.is_contiguous() or w.data_ptr() != expect:
                return None
            expect += w.numel() * w.element_size()
        return torch.as_strided(ws[0], (sum(w.shape[0] for w in ws), K), (K, 1))

    def packs(self, holders):
        return self._adjacent([m.weight for m in holders]) is not None

    def packed(self, holders, x):
        w = self._adjacent([m.weight for m in holders])
        PackingLinear.calls += 1
        return h.native_linear(ops._Holder(w), x).split([m.weight.shape[0] for m in holders], dim=-1)


def test_pack_binds_and_refreshes_derived_rows():
    cfg = h.native_cfg()
    i = cfg.index_source_layer_ids[0]
    m = h.make_layer(cfg, i, 3)
    ref = ops.index_weight_rows(copy.deepcopy(m.indexer), cfg).clone()
    wq_a, wkv = m.wq_a.weight.detach().clone(), m.wkv.weight.detach().clone()
    buf = ops.pack_shared_input_rows(m, cfg, i)
    assert buf is not None and buf.shape[0] == wq_a.shape[0] + wkv.shape[0] + 64
    assert torch.equal(m.wq_a.weight, wq_a) and torch.equal(m.wkv.weight, wkv)
    rows = ops.index_weight_rows(m.indexer, cfg)
    assert rows.data_ptr() == buf[wq_a.shape[0] + wkv.shape[0]:].data_ptr()
    assert torch.equal(rows.view(torch.int16), ref.view(torch.int16))
    with torch.no_grad():
        m.indexer.weights_proj.weight.mul_(2)
    ops.refresh_derived_weights(m)
    rows2 = ops.index_weight_rows(m.indexer, cfg)
    assert rows2.data_ptr() == rows.data_ptr()
    assert torch.equal(rows2.float(), ref.float() * 2)
    x = torch.randn((37, cfg.hidden_size), device="cuda").bfloat16()
    w_packed = h.native_linear(ops._Holder(buf), x)[:, -64:]
    assert torch.equal(w_packed.view(torch.int16), ops.mm(x, rows2).view(torch.int16))


def test_packed_layers_equal_unpacked_mixed_schedule():
    cfg, layers, xs = tp_._setup(3)
    packed_layers = copy.deepcopy(layers)
    for i, m in enumerate(packed_layers):
        assert ops.pack_shared_input_rows(m, cfg, i) is not None
    win, comp, P = tp_._tables(3, 5)
    slots = {0: 2, 1: 0, 2: 1}
    pools_a, pools_b = tp_._pools(cfg, 3, P), tp_._pools(cfg, 3, P)
    prov = PackingLinear()
    PackingLinear.calls = 0
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import paged
    for plan in tp_.SCHEDULE[:5]:
        a = tp_._step(cfg, layers, xs, pools_a, plan, win, comp, slots)
        p0 = [s for _, s, _ in plan]
        n = [c for _, _, c in plan]
        sel = torch.tensor([s for s, _, _ in plan], device="cuda").long()
        batch = paged.PagedBatch.make(p0, n, win[sel], {r: bt[sel] for r, bt in comp.items()},
                                      [slots[s] for s, _, _ in plan], ratios=(2,))
        with torch.no_grad():
            b = [paged.attention_paged(m, torch.cat([x[s, q:q + c] for s, q, c in plan], 0).contiguous(), cfg, i,
                                       pools_b, batch, linear_fn=prov, dequant_fn=h.dequant)
                 for i, (m, x) in enumerate(zip(packed_layers, xs))]
        for i, (oa, ob) in enumerate(zip(a, b)):
            assert torch.equal(oa.view(torch.int16), ob.view(torch.int16)), (plan, i)
    assert PackingLinear.calls >= 5 * len(layers)       # one packed GEMM per layer and step


def test_row_kernels_read_strided_token_heads():
    """ROW_INDEX_Q and index_kq_store on a [tokens, heads, 128] column view == on its contiguous copy (addresses only)."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import spec
    from isoexec.ops.attention.deepseek_v41_cache_sm100._extension import extension
    cfg = h.native_cfg()
    cos, sin = ops.rope_table(cfg, 2, 4096, "cuda")
    g = torch.Generator(device="cuda").manual_seed(4)
    for N, hi in ((1, 32), (37, 32), (300, 64)):
        wide = (torch.randn((N, 4096 + hi * 128 + 64), generator=g, device="cuda") * 3).bfloat16()
        v = wide[:, 4096:4096 + hi * 128].reshape(N, hi, 128)
        c = v.reshape(N * hi, 128).contiguous()
        pos = torch.randint(0, 4096, (N,), generator=g, device="cuda", dtype=torch.int32)
        outs = []
        for x in (v, c):
            codes = torch.empty((N, 64, 128), dtype=torch.uint8, device="cuda")
            sf = torch.empty((N, 64), dtype=torch.int32, device="cuda")
            ops.row_op(spec.ROW_INDEX_Q, x, 128, table=(cos, sin), pos=ops.Pos(pos=pos, div=hi),
                       codes=codes.view(N * 64, 128), sf=sf.view(N * 64), dst=(hi, 64, 0), pad_heads=hi if hi < 64 else 0)
            k = (torch.randn((N, 128), generator=torch.Generator(device="cuda").manual_seed(N), device="cuda")).bfloat16()
            pool = torch.zeros((8, 64, 1, 132), dtype=torch.uint8, device="cuda")
            dest = torch.arange(N, device="cuda", dtype=torch.int64) % 512
            codes2 = torch.empty_like(codes)
            sf2 = torch.empty_like(sf)
            extension().index_kq_store(k, torch.ones(128, device="cuda"), 1e-6, cos, sin, pos, dest, pool, x, pos, hi,
                                       codes2.view(N * 64, 128), sf2.view(N * 64), hi, hi if hi < 64 else 0)
            outs.append((codes, sf, codes2, sf2, pool))
        for a, b in zip(*outs):
            assert torch.equal(a, b), (N, hi)


@pytest.mark.parametrize("world", [1, 8])
def test_packed_qr_and_x_layers_equal_unpacked(world):
    """D1 + D2 + D3 packing (``pack_shared_input_rows``, ``pack_qr_rows``) on every rank's layers == unpacked, over a mixed
    prefill / decode schedule; world 8 = one output group per rank (the leaf's fused RoPE reads the strided queries)."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import paged, tp
    cfg, layers, xs = tp_._setup(3, seed=71)
    win, comp, P = tp_._tables(3, 9)
    slots = {0: 0, 1: 1, 2: 2}
    shards = [[tp.shard_layer(m, cfg, r, world) for r in range(world)] for m in layers]
    packed = [[copy.deepcopy(s) for s in row] for row in shards]
    n_qr = 0
    for i, row in enumerate(packed):
        for s in row:
            assert ops.pack_shared_input_rows(s, cfg, i) is not None
            n_qr += ops.pack_qr_rows(s, cfg, i) is not None
    assert n_qr == world * sum(1 for i, r in enumerate(cfg.compress_ratios) if r and i in cfg.index_source_layer_ids)
    prov = PackingLinear()
    PackingLinear.calls = 0
    pools_a = [tp_._pools(cfg, 3, P) for _ in range(world)]
    pools_b = [tp_._pools(cfg, 3, P) for _ in range(world)]
    for plan in tp_.SCHEDULE[:4]:
        p0 = [a for _, a, _ in plan]
        n = [c for _, _, c in plan]
        sel = torch.tensor([s for s, _, _ in plan], device="cuda").long()
        batch = paged.PagedBatch.make(p0, n, win[sel], {r: bt[sel] for r, bt in comp.items()},
                                      [slots[s] for s, _, _ in plan], ratios=(2,))
        ba = [dataclasses.replace(batch, step={}) for _ in range(world)]
        bb = [dataclasses.replace(batch, step={}) for _ in range(world)]
        with torch.no_grad():
            for i, x in enumerate(xs):
                xi = torch.cat([x[s, a:a + c] for s, a, c in plan], 0).contiguous()
                oa = tp.attention_paged_tp(shards[i], xi, cfg, i, pools_a, ba, linear_fn=h.native_linear,
                                           dequant_fn=h.dequant)
                ob = tp.attention_paged_tp(packed[i], xi, cfg, i, pools_b, bb, linear_fn=prov, dequant_fn=h.dequant)
                assert torch.equal(oa.view(torch.int16), ob.view(torch.int16)), (world, plan, i)
    assert PackingLinear.calls > 0
