"""The FP32-tree index fold (``spec.INDEX_FOLD == "fp32_tree"``, ``ops.indexer_tree``; kept for comparison with the
recipe's DeepGEMM fold) == the retired TP-split form of the TP = 4 host, bitwise.

The split form at TP = t: rank r scores its 32/t index heads (8/t leaves) as one FP32 subtree root in the
indexer kernel (``indexer_tree(leaves=8/t, f32=True)``, FTZ adds), the t roots are completed by
``tp.tree_combine`` (the ``leaf_tree`` kernel, IEEE adds: the arithmetic TreeReducer / DistributedReducer
are bitwise-equal to). The canonical local program evaluates all 8 leaves with its top
``spec.INDEX_TREE_IEEE_LEVELS = 2`` levels in IEEE adds, i.e. exactly the t = 4 split.

Relations:
* random inputs (paged decode rows, contiguous prefill rows): canonical == split(4), and the kernel's
  ``ieee_levels = log2 t`` form == split(t) for every t (the level schedule is what each split computed);
* an analytic subnormal edge (two group roots cancelling to 2^-133): canonical == split(4) == split(8) ==
  BF16 0x0001 while split(1) == split(2) == 0 -- the residual TP dependence of the split forms that the
  replicated indexer removes;
* index queries from the full (replicated) weights == the per-rank slices' queries, column for column.
"""
import math

import pytest
import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops, spec, tp
from isoexec.ops.attention.deepseek_v41_cache_sm100.tests import _harness as h

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

ONE = 0x7F7F7F7F        # four ue8m0 2^0 bytes


def _split(codes, sf, w, ke, t, width, **kv):
    """The TP = t split form on one GPU: rank r owns heads [32r/t, 32(r+1)/t) at slots 0.. (zero padded)."""
    hi = 32 // t
    roots = []
    for r in range(t):
        c = torch.zeros_like(codes)
        s = torch.full_like(sf, ONE)
        ww = torch.zeros_like(w)
        c[:, :hi] = codes[:, r * hi:(r + 1) * hi]
        s[:, :hi] = sf[:, r * hi:(r + 1) * hi]
        ww[:, :hi] = w[:, r * hi:(r + 1) * hi]
        # the per-rank call of the split form: the whole local subtree in the kernel's FTZ adds (ieee_levels = 0)
        roots.append(ops.indexer_tree(c, s, ww.contiguous(), ke, width=width, leaves=8 // t, f32=t > 1,
                                      ieee_levels=0, **kv))
    return tp.tree_combine([x.contiguous() for x in roots])


def _in_window(a, b, ke):
    width = a.shape[1]
    mask = torch.arange(width, device=a.device)[None, :] < ke.long()[:, None]
    return torch.equal(a.contiguous().view(torch.int16)[mask], b.contiguous().view(torch.int16)[mask])


def _random_q(rows, g):
    q = torch.randn((rows, 64, 128), generator=g, device="cuda") * 3
    q[:, 32:] = 0
    from isoexec.ops.attention.dsa_sm100.tests._inputs import fp8_row_quantize
    codes, sf = fp8_row_quantize(q)
    sf = sf.view(rows, 64).clone()
    sf[:, 32:] = ONE
    w = torch.zeros((rows, 64), device="cuda")
    w[:, :32] = torch.randn((rows, 32), generator=g, device="cuda") * 0.05
    return codes.view(torch.uint8).contiguous(), sf.contiguous(), w.to(torch.bfloat16).contiguous()


def _paged_keys(rows, width, g):
    from isoexec.ops.attention.dsa_sm100.tests._inputs import fp8_row_quantize
    pages = rows * (width // 64)
    k = torch.randn((pages, 64, 128), generator=g, device="cuda")
    kc, ks = fp8_row_quantize(k)
    cache = torch.cat((kc.view(torch.uint8).reshape(pages, -1), ks.view(torch.uint8).reshape(pages, -1)), 1)
    bt = torch.randperm(pages, generator=g, device="cuda").view(rows, width // 64).to(torch.int32).contiguous()
    return cache.view(pages, 64, 1, 132), bt


@pytest.mark.parametrize("rows,width", [(7, 1024), (130, 4096), (300, 640)])
def test_canonical_equals_split_tp4_paged(rows, width):
    g = torch.Generator(device="cuda").manual_seed(rows)
    codes, sf, w = _random_q(rows, g)
    cache, bt = _paged_keys(rows, width, g)
    ke = torch.randint(0, width + 1, (rows,), generator=g, device="cuda", dtype=torch.int32)
    ke[0] = width
    ke[-1] = 0
    kv = dict(cache=cache, block_table=bt)
    new = ops.indexer_tree(codes, sf, w, ke, width=width, **kv)
    assert _in_window(new, _split(codes, sf, w, ke, 4, width, **kv), ke)
    for t in (1, 2, 4, 8):
        form = ops.indexer_tree(codes, sf, w, ke, width=width, ieee_levels=int(math.log2(t)), **kv)
        assert _in_window(form, _split(codes, sf, w, ke, t, width, **kv), ke), t


@pytest.mark.parametrize("rows,C", [(64, 640), (257, 1024)])
def test_canonical_equals_split_tp4_contiguous(rows, C):
    from isoexec.ops.attention.dsa_sm100.tests._inputs import fp8_row_quantize
    g = torch.Generator(device="cuda").manual_seed(C)
    codes, sf, w = _random_q(rows, g)
    kc, ks = fp8_row_quantize(torch.randn((C, 128), generator=g, device="cuda"))
    ke = (torch.arange(1, rows + 1, device="cuda", dtype=torch.int32) * C // rows).contiguous()
    kv = dict(keys=(kc.view(torch.uint8).contiguous(), ks.contiguous()))
    new = ops.indexer_tree(codes, sf, w, ke, width=C, **kv)
    assert _in_window(new, _split(codes, sf, w, ke, 4, C, **kv), ke)
    for t in (1, 2, 8):
        form = ops.indexer_tree(codes, sf, w, ke, width=C, ieee_levels=int(math.log2(t)), **kv)
        assert _in_window(form, _split(codes, sf, w, ke, t, C, **kv), ke), t


@pytest.mark.parametrize("mode", ["paged", "contiguous"])
def test_subnormal_group_cancellation(mode):
    """d = 129 (head 0) and 130 (head 8) on unit keys; w0 = 2^-126 (1 + 2^-7), w8 = -2^-126: the group roots
    2^-126 * 130.0078125 and -2^-126 * 130 cancel to 2^-133, a subnormal kept only by IEEE adds."""
    rows, width = 2, 128
    f8 = lambda v: torch.tensor(v, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8).item()
    codes = torch.zeros((rows, 64, 128), dtype=torch.uint8, device="cuda")
    codes[:, 0] = f8(1.0)
    codes[:, 0, 0] = f8(2.0)
    codes[:, 8] = f8(1.0)
    codes[:, 8, :2] = f8(2.0)
    sf = torch.full((rows, 64), ONE, dtype=torch.int32, device="cuda")
    w = torch.zeros((rows, 64), dtype=torch.float32, device="cuda")
    w[:, 0] = 2.0 ** -126 * (1 + 2.0 ** -7)
    w[:, 8] = -(2.0 ** -126)
    w = w.to(torch.bfloat16)
    assert w.float()[0, 0].item() == 2.0 ** -126 * (1 + 2.0 ** -7)
    ke = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    if mode == "paged":
        pages = rows * 2
        page = torch.cat((torch.full((64 * 128,), f8(1.0), dtype=torch.uint8, device="cuda"),
                          torch.full((64,), ONE, dtype=torch.int32, device="cuda").view(torch.uint8)))
        kv = dict(cache=page.repeat(pages, 1).view(pages, 64, 1, 132).contiguous(),
                  block_table=torch.arange(pages, dtype=torch.int32, device="cuda").view(rows, 2).contiguous())
    else:
        kv = dict(keys=(torch.full((width, 128), f8(1.0), dtype=torch.uint8, device="cuda"),
                        torch.full((width,), ONE, dtype=torch.int32, device="cuda")))
    got = lambda s: s[:, :width].contiguous().view(torch.int16)
    new = got(ops.indexer_tree(codes, sf, w.contiguous(), ke, width=width, **kv))
    assert (new == 1).all()                                       # BF16 2^-133
    for t, want in ((1, 0), (2, 0), (4, 1), (8, 1)):
        assert (got(_split(codes, sf, w, ke, t, width, **kv)) == want).all(), t
    assert (got(ops.indexer_tree(codes, sf, w.contiguous(), ke, width=width, ieee_levels=0, **kv)) == 0).all()


def test_replicated_queries_are_the_rank_slices():
    """Index Q codes / scales / weights of all heads (replicated weights) == the per-rank sliced weights'
    queries, column for column, at TP 2/4/8 (harness BF16 provider)."""
    cfg = h.native_cfg(hidden=1024)
    layer = h.make_layer(cfg, 1, 5)
    g = torch.Generator(device="cuda").manual_seed(9)
    N = 133
    x = torch.randn((N, cfg.hidden_size), generator=g, device="cuda").to(torch.bfloat16)
    qr = torch.randn((N, cfg.q_lora_rank), generator=g, device="cuda").to(torch.bfloat16)
    pos = torch.randint(0, 4000, (N,), generator=g, device="cuda", dtype=torch.int32)
    table = ops.rope_table(cfg, 2, 4096, x.device)
    Hi = cfg.index_n_heads
    full = ops.index_queries(layer.indexer, x, qr, cfg, table, ops.Pos(pos=pos, div=Hi), h.native_linear,
                             (1, N, cfg.hidden_size), (1, N, cfg.q_lora_rank))
    for t in (2, 4, 8):
        hi = Hi // t
        for r in range(t):
            idx = h.Box()
            idx.wq_b = tp.slice_projection(layer.indexer.wq_b, rows=(r * hi * 128, (r + 1) * hi * 128))
            idx.weights_proj = tp.slice_projection(layer.indexer.weights_proj, rows=(r * hi, (r + 1) * hi))
            c, s, w = ops.index_queries(idx, x, qr, cfg, table, ops.Pos(pos=pos, div=hi), h.native_linear,
                                        (1, N, cfg.hidden_size), (1, N, cfg.q_lora_rank), heads=hi)
            sl = slice(r * hi, (r + 1) * hi)
            assert torch.equal(c[:, :hi], full[0][:, sl]), (t, r)
            assert torch.equal(s[:, :hi], full[1][:, sl]), (t, r)
            assert torch.equal(w[:, :hi].view(torch.int16), full[2][:, sl].view(torch.int16)), (t, r)


@pytest.mark.parametrize("heads", [32, 16, 8, 4])
def test_index_weights_and_queries_through_the_real_mm(heads):
    """index_weights / index_queries on the real mm_sm100 path (no stub) at the full index heads and the TP ranks'
    4..16: the 64-row GEMM's first ``heads`` columns, a 128-byte-row view the index-score kernel reads as is (the
    same bits as a contiguous copy)."""
    cfg = h.native_cfg(hidden=1024)
    layer = h.make_layer(cfg, 1, 5)
    g = torch.Generator(device="cuda").manual_seed(heads)
    N = 37
    x = torch.randn((N, cfg.hidden_size), generator=g, device="cuda").to(torch.bfloat16)
    qr = torch.randn((N, cfg.q_lora_rank), generator=g, device="cuda").to(torch.bfloat16)
    pos = torch.randint(0, 4000, (N,), generator=g, device="cuda", dtype=torch.int32)
    table = ops.rope_table(cfg, 2, 4096, x.device)
    idx = h.Box()
    idx.wq_b = tp.slice_projection(layer.indexer.wq_b, rows=(0, heads * 128))
    idx.weights_proj = tp.slice_projection(layer.indexer.weights_proj, rows=(0, heads))
    c, s, w = ops.index_queries(idx, x, qr, cfg, table, ops.Pos(pos=pos, div=heads), h.native_linear,
                                (1, N, cfg.hidden_size), (1, N, cfg.q_lora_rank), heads=heads)
    hs = spec.index_q_heads(heads)
    assert c.shape == (N, hs, 128) and s.shape == (N, hs) and w.shape == (N, hs)
    wp = torch.cat((idx.weights_proj.weight.to(torch.bfloat16) * spec.index_weight_scale(cfg),
                    torch.zeros((64 - heads, cfg.hidden_size), dtype=torch.bfloat16, device="cuda")), 0).contiguous()
    assert torch.equal(w.view(torch.int16), ops.mm(x, wp)[:, :hs].view(torch.int16))
    if hs in (32, 64):
        from isoexec.ops.attention.dsa_sm100.tests._inputs import fp8_row_quantize
        kc, ks = fp8_row_quantize(torch.randn((640, 128), generator=g, device="cuda"))
        ke = torch.full((N,), 640, dtype=torch.int32, device="cuda")
        kv = dict(keys=(kc.view(torch.uint8).contiguous(), ks.contiguous()))
        a = ops.index_scores(c, s, w, ke, width=640, **kv)
        b = ops.index_scores(c, s, w.contiguous(), ke, width=640, **kv)
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
