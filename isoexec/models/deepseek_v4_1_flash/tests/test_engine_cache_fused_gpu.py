"""The one-launch attention kernels equal the separate launches they replace, byte for byte, on vLLM's own KV
cache allocation (GPU).

The pools, tables and rows are the engine step's (``engine.EngineStep`` / ``bind_window`` over vLLM's planner and
allocator, set up as ``test_engine_cache_cpu`` does, allocated on the device): window rows from vLLM's
sliding-window slots (32-token blocks), compressed tables of record rows (``comp_per`` 1, 64-row page expansion),
index-key pages of 128 (ratio 1) and 64 (ratio 2) entries, the compressor state ring. Each fused form runs on a
copy of the whole backing allocation and must leave every byte of it, and every output, equal to the separate
form's.
"""
import pytest
import torch

pytest.importorskip("vllm")

from isoexec.models.deepseek_v4_1_flash import engine  # noqa: E402
from isoexec.models.deepseek_v4_1_flash.tests import test_engine_cache_cpu as cache_cpu  # noqa: E402
from isoexec.models.deepseek_v4_1_flash.tests.test_engine_cache_cpu import sm100, vllm_config  # noqa: E402,F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

#: (request starts, sequence lengths): mixed prefill chunks + decode rows, and a graph-style decode batch
CASES = {"mixed": ([0, 5, 6, 40, 41], [5, 260, 34, 131]), "decode": ([0, 1, 2, 3], [300, 1, 129])}


def _cuda(ns):
    return type(ns)(**{k: (v.cuda() if isinstance(v, torch.Tensor) and not k.endswith("_cpu") else v)
                       for k, v in vars(ns).items()})


def _step(vllm_config, case, seed=0):
    b = cache_cpu.build(vllm_config, "debug_config.json", 40, device="cuda")
    graph = case == "decode"
    starts, seq_lens = CASES[case]
    R = len(seq_lens)
    tables = {"comp": [[3, 7, 9], [11, 2, 30], [5, 6, 8], [12, 21, 33]][:R], "state": [13, 14, 15, 16][:R]}
    swa_tables = {g: [[(g * 7 + s * 11 + j * 3) % 39 + 1 for j in range(10)] for s in range(R)]
                  for g in range(len(b.groups))}
    md, positions, _, _ = cache_cpu._metadata(b, tables, swa_tables, starts, seq_lens, graph)
    md = {k: _cuda(v) for k, v in md.items()}
    step = engine.EngineStep(b.cfg, b.rows[0], md, positions.cuda(), graph, capturing=graph)
    for i, row in enumerate(b.rows):
        step.bind_window(i, row, md)
    storage = next(iter(b.context.values())).kv_cache.untyped_storage()
    flat = torch.empty(0, dtype=torch.uint8, device="cuda").set_(storage, 0, (storage.nbytes(),), (1,))
    g = torch.Generator(device="cuda").manual_seed(seed)
    # bytes < 0x40: finite FP32 state rows (|x| < 2) and finite E4M3 / E2M1 codes everywhere
    flat.copy_(torch.randint(0, 0x40, flat.shape, generator=g, device="cuda", dtype=torch.int32).to(torch.uint8))
    return b, step, flat


def _rope(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ang = torch.rand((8192, 32), generator=g, device="cuda") * 6.28
    return ang.cos().contiguous(), ang.sin().contiguous()


def _both(flat, separate, fused):
    """Run ``separate`` and ``fused`` from the same allocation bytes; return (allocation, outputs) of each."""
    start = flat.clone()
    ref = separate()
    after_ref = flat.clone()
    flat.copy_(start)
    got = fused()
    torch.cuda.synchronize()
    assert not torch.equal(after_ref, start) or ref is None, "the separate form wrote nothing"
    return after_ref, ref, flat.clone(), got


@pytest.mark.parametrize("case", sorted(CASES))
def test_q_kv_post_on_engine_window_rows(vllm_config, sm100, case):  # noqa: F811
    """q_kv_post (q_norm + window-KV record) == the q_norm row kernel + paged_quant_write kind 0 at the engine's
    window rows (vLLM's 32-token slots in the layer's row view), every window layer."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops, spec
    from isoexec.ops.attention.deepseek_v41_cache_sm100._extension import extension
    b, step, flat = _step(vllm_config, case, 1)
    batch = step.batch
    N = batch.tok_seq.numel()
    cos, sin = _rope(2)
    g = torch.Generator(device="cuda").manual_seed(3)
    qw = torch.rand(1280, generator=g, device="cuda") + 0.5
    kw = torch.rand(512, generator=g, device="cuda") + 0.5
    pos = batch.tok_pos
    for i in range(b.cfg.num_hidden_layers):
        qa = (torch.randn((N, 1280), generator=g, device="cuda") * 2).to(torch.bfloat16)
        kv = (torch.randn((N, 512), generator=g, device="cuda") * 2).to(torch.bfloat16)
        rows, pool = batch.window_rows[i], step.pools.window[i]

        def separate():
            qr = torch.empty((N, 1280), dtype=torch.bfloat16, device="cuda")
            ops.row_op(spec.ROW_Q_NORM, qa, 1280, w=qw, eps=1e-6, out=qr)
            extension().paged_quant_write(0, kv, kw, 1e-6, cos, sin, pos, rows, pool)
            return qr

        def fused():
            qr = torch.empty((N, 1280), dtype=torch.bfloat16, device="cuda")
            extension().q_kv_post(qa, kv, qw, kw, 1e-6, cos, sin, pos, rows, pool, qr)
            return qr

        mem_ref, qr_ref, mem, qr = _both(flat, separate, fused)
        assert torch.equal(qr.view(torch.int16), qr_ref.view(torch.int16)), i
        assert torch.equal(mem, mem_ref), i


@pytest.mark.parametrize("case", sorted(CASES))
def test_comp_post_and_index_k_store_on_engine_tables(vllm_config, sm100, case):  # noqa: F811
    """Ratio 2: comp_post == pool_varlen + state_update + paged_rows (compressed rows through the source's
    record-row table, ``comp_per`` 1; index rows through the 64-row unit table) + paged_quant_write kind 1, on
    the engine's compressed rows and state ring; then index_k_store == row_op ROW_INDEX_K + paged_index_write
    into the 64-entry index pages. Ratio 1: index_k_store into the 128-entry index pages."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops, paged, spec
    from isoexec.ops.attention.deepseek_v41_cache_sm100._extension import extension
    b, step, flat = _step(vllm_config, case, 4)
    cfg, batch, pools = b.cfg, step.batch, step.pools
    N = batch.tok_seq.numel()
    cos, sin = _rope(5)
    g = torch.Generator(device="cuda").manual_seed(6)
    for r in (1, 2):
        src = next(i for i in cfg.kv_source_layer_ids if cfg.compress_ratios[i] == r)
        cbt, ibt = batch.comp_bt_by_source[src], batch.index_bt[r]
        assert pools.index[src].shape[1] == 128 // r
        kw = torch.rand(128, generator=g, device="cuda") + 0.5
        if r == 2:
            w = torch.rand(512, generator=g, device="cuda") + 0.5
            proj = torch.randn((N, 1024), generator=g, device="cuda")
            gseq, gk = batch.groups[r]
            G = gseq.numel()
            st = pools.state[src]

            def separate():
                lat = torch.zeros((G, 512), dtype=torch.bfloat16, device="cuda")
                val = torch.empty(G, dtype=torch.int32, device="cuda")
                cid = torch.empty(G, dtype=torch.int32, device="cuda")
                extension().pool_varlen(proj, batch.cu_q, batch.p0, st, batch.state_slot, gseq, gk, r, w, 1e-6,
                                        lat, val, cid, True)
                extension().state_update(proj, batch.cu_q, batch.p0, st, batch.state_slot, r)
                dest = paged._rows(cbt, gseq, cid, val, batch.comp_per)
                idest = paged._rows(ibt, gseq, cid, val)
                gpos = (cid * r).to(torch.int32).contiguous()
                extension().paged_quant_write(1, lat, None, 0.0, cos, sin, gpos, dest, pools.comp[src])
                return lat, val, cid, gpos, dest, idest

            def fused():
                lat = torch.full((G, 512), 3.0, dtype=torch.bfloat16, device="cuda")
                val = torch.empty(G, dtype=torch.int32, device="cuda")
                cid = torch.empty(G, dtype=torch.int32, device="cuda")
                gpos = torch.empty(G, dtype=torch.int32, device="cuda")
                dest = torch.empty(G, dtype=torch.int64, device="cuda")
                idest = torch.empty(G, dtype=torch.int64, device="cuda")
                extension().comp_post(proj, batch.cu_q, batch.p0, st, batch.state_slot, gseq, gk, r, w, 1e-6, lat,
                                      val, cid, gpos, cbt, batch.comp_per, dest, ibt, idest, cos, sin,
                                      pools.comp[src])
                return lat, val, cid, gpos, dest, idest

            mem_ref, out_ref, mem, out = _both(flat, separate, fused)
            assert torch.equal(out[0].view(torch.int16), out_ref[0].view(torch.int16))
            for a, e in zip(out[1:], out_ref[1:]):
                assert torch.equal(a, e)
            assert torch.equal(mem, mem_ref)
            assert int(out_ref[1].sum()) > 0
            latent, gpos, idest = out_ref[0], out_ref[3], out_ref[5]
        else:
            latent = (torch.randn((N, 512), generator=g, device="cuda")).to(torch.bfloat16)
            gpos = batch.tok_pos
            idest = paged._rows(ibt, batch.tok_seq, gpos, batch.valid)
        k = (torch.randn((latent.shape[0], 128), generator=g, device="cuda") * 2).to(torch.bfloat16)
        R = k.shape[0]

        def separate_k():
            codes = torch.empty((R, 128), dtype=torch.uint8, device="cuda")
            sf = torch.empty((R, 1), dtype=torch.int32, device="cuda")
            ops.row_op(spec.ROW_INDEX_K, k, 128, w=kw, eps=1e-6, table=(cos, sin), pos=ops.Pos(pos=gpos),
                       codes=codes, sf=sf.view(R))
            extension().paged_index_write(codes, sf, idest, pools.index[src])

        def fused_k():
            extension().index_k_store(k, kw, 1e-6, cos, sin, gpos, idest, pools.index[src])

        mem_ref, _, mem, _ = _both(flat, separate_k, fused_k)
        assert torch.equal(mem, mem_ref), r


@pytest.mark.parametrize("case", sorted(CASES))
def test_index_scores_through_tok_seq_on_engine_pages(vllm_config, sm100, case):  # noqa: F811
    """Index scores reading the engine's index table (one entry per vLLM block: 128-entry pages at ratio 1,
    64 at ratio 2) through tok_seq (bt_rows) == the scores on the per-token gathered table; the in-kernel item
    span == the fixed 16-tile span. On the window [0, ke) of every row."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops
    from isoexec.ops.attention.deepseek_v41_cache_sm100._extension import extension
    b, step, flat = _step(vllm_config, case, 7)
    cfg, batch, pools = b.cfg, step.batch, step.pools
    N = batch.tok_seq.numel()
    g = torch.Generator(device="cuda").manual_seed(8)
    for r in (1, 2):
        src = next(i for i in cfg.kv_source_layer_ids if cfg.compress_ratios[i] == r)
        cache = pools.index[src]
        P = cache.shape[1]
        # sensible key scale words (ue8m0 118..131) on every page of the layer
        words = cache.as_strided((cache.shape[0], P * 4), (cache.stride(0), 1), cache.storage_offset() + P * 128)
        words.copy_(torch.randint(118, 132, words.shape, generator=g, device="cuda", dtype=torch.int32).to(torch.uint8))
        rbt = batch.index_read_bt[r]
        width = rbt.shape[1] * P
        qc = torch.randint(0, 0x77, (N, 64, 128), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
        qsf = torch.randint(118, 132, (N, 64, 4), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
        qsf = qsf.view(torch.int32).reshape(N, 64)
        w = torch.randn((N, 64), generator=g, device="cuda") * 0.1
        w[:, cfg.index_n_heads:] = 0
        w = w.to(torch.bfloat16)
        ke = torch.empty(N, dtype=torch.int32, device="cuda")
        extension().index_bounds(batch.tok_pos, batch.valid, r, ke)
        ke = ke.clamp_max(width)
        mask = torch.arange(width, device="cuda").view(1, -1) < ke.view(-1, 1)
        old = ops.INDEX_SPAN_AUTO
        try:
            outs = {}
            for auto in (False, True):
                ops.INDEX_SPAN_AUTO = auto
                outs["select", auto] = ops.index_scores(qc, qsf, w, ke, width=width, cache=cache,
                                                        block_table=rbt.index_select(0, batch.tok_seq))
                outs["rows", auto] = ops.index_scores(qc, qsf, w, ke, width=width, cache=cache, block_table=rbt,
                                                      bt_rows=batch.tok_seq)
        finally:
            ops.INDEX_SPAN_AUTO = old
        ref = torch.where(mask, outs["select", False], 0).view(torch.int16)
        assert bool((ref != 0).any())
        for key, got in outs.items():
            assert torch.equal(torch.where(mask, got, 0).view(torch.int16), ref), (r, key)


@pytest.mark.parametrize("case", sorted(CASES))
def test_topk_extra_list_on_engine_tables(vllm_config, sm100, case):  # noqa: F811
    """The top-k emission's extra list (the selected ids as rows of the source's record-row table, ``comp_per`` 1)
    == paged_lists_masked's extra list as the engine step builds it (zero window table, the engine's own main list),
    on the same selection; the selection itself is unchanged by the second output."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import paged
    from isoexec.ops.attention.deepseek_v41_cache_sm100._extension import extension
    from isoexec.ops.attention.sparse_sm100 import topk as T
    b, step, _ = _step(vllm_config, case, 9)
    cfg, batch, pools = b.cfg, step.batch, step.pools
    N, W, K = batch.tok_seq.numel(), cfg.sliding_window, cfg.index_topk
    g = torch.Generator(device="cuda").manual_seed(10)
    for r in (1, 2):
        src = next(i for i in cfg.kv_source_layer_ids if cfg.compress_ratios[i] == r)
        cbt = batch.comp_bt_by_source[src]
        width = cbt.shape[1] * 64
        ke = torch.empty(N, dtype=torch.int32, device="cuda")
        extension().index_bounds(batch.tok_pos, batch.valid, r, ke)
        ke = ke.clamp_max(width)
        scores = (torch.randn((N, width), generator=g, device="cuda").relu() * 3).to(torch.bfloat16)
        sel_ref = T.dsa_topk(scores, ke, k=K)
        rows_out = torch.full((N, K), 77, dtype=torch.int32, device="cuda")
        sel = T.dsa_topk(scores, ke, k=K, rows_out=rows_out, rows_bt=cbt, rows_seq=batch.tok_seq,
                         rows_per=batch.comp_per)
        assert torch.equal(sel, sel_ref)
        main = torch.empty((N, W), dtype=torch.int32, device="cuda")
        extra = torch.empty((N, K), dtype=torch.int32, device="cuda")
        extension().paged_lists_masked(batch.tok_seq, batch.tok_pos, paged._zero_window_table(pools.max_len, "cuda"),
                                       W, sel_ref, cbt, main, extra, batch.valid, batch.win_per, batch.comp_per)
        assert torch.equal(rows_out, extra), r
        assert bool((extra >= 0).any())


@pytest.mark.parametrize("case", sorted(CASES))
def test_index_kq_store_on_engine_pages(vllm_config, sm100, case):  # noqa: F811
    """index_kq_store (index keys into the engine's index pages + index queries in one launch) == index_k_store +
    row_op ROW_INDEX_Q: every byte of vLLM's allocation and the query codes / scales, at 128-entry (ratio 1) and
    64-entry (ratio 2) pages."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops, paged, spec
    from isoexec.ops.attention.deepseek_v41_cache_sm100._extension import extension
    b, step, flat = _step(vllm_config, case, 11)
    cfg, batch, pools = b.cfg, step.batch, step.pools
    N, H = batch.tok_seq.numel(), cfg.index_n_heads
    cos, sin = _rope(12)
    g = torch.Generator(device="cuda").manual_seed(13)
    for r in (1, 2):
        src = next(i for i in cfg.kv_source_layer_ids if cfg.compress_ratios[i] == r)
        kw = torch.rand(128, generator=g, device="cuda") + 0.5
        k = (torch.randn((N, 128), generator=g, device="cuda") * 2).to(torch.bfloat16)
        iq = (torch.randn((N * H, 128), generator=g, device="cuda") * 2).to(torch.bfloat16)
        gpos = batch.tok_pos
        idest = paged._rows(batch.index_bt[r], batch.tok_seq, gpos, batch.valid)

        def separate():
            extension().index_k_store(k, kw, 1e-6, cos, sin, gpos, idest, pools.index[src])
            codes = torch.full((N, 64, 128), 0x55, dtype=torch.uint8, device="cuda")
            sf = torch.full((N, 64), 7, dtype=torch.int32, device="cuda")
            ops.row_op(spec.ROW_INDEX_Q, iq, 128, table=(cos, sin), pos=ops.Pos(pos=batch.tok_pos, div=H),
                       codes=codes.view(N * 64, 128), sf=sf.view(N * 64), dst=(H, 64, 0), pad_heads=H if H < 64 else 0)
            return codes, sf

        def fused():
            codes = torch.full((N, 64, 128), 0x55, dtype=torch.uint8, device="cuda")
            sf = torch.full((N, 64), 7, dtype=torch.int32, device="cuda")
            extension().index_kq_store(k, kw, 1e-6, cos, sin, gpos, idest, pools.index[src], iq, batch.tok_pos, H,
                                       codes.view(N * 64, 128), sf.view(N * 64), H, H if H < 64 else 0)
            return codes, sf

        mem_ref, (c_ref, s_ref), mem, (c, s) = _both(flat, separate, fused)
        assert torch.equal(c, c_ref) and torch.equal(s, s_ref), r
        assert torch.equal(mem, mem_ref), r
