"""Fused paged index glue: the one-launch integer kernels equal the torch index
glue they replace bit for bit."""
import pytest
import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100 import paged
from isoexec.ops.attention.deepseek_v41_cache_sm100._extension import extension

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _torch_rows(bt, seq, pos):
    seq, pos = seq.long(), pos.long()
    return bt[seq, torch.div(pos, 64, rounding_mode="floor")].long() * 64 + pos % 64


def test_paged_rows_equal_torch_glue_with_masks():
    g = torch.Generator(device="cuda").manual_seed(5)
    B, P, N = 7, 33, 41
    bt = torch.randint(0, 10_000, (B, P), generator=g, device="cuda", dtype=torch.int32)
    seq = torch.randint(0, B, (N,), generator=g, device="cuda", dtype=torch.int32)
    pos = torch.randint(0, P * 64, (N,), generator=g, device="cuda", dtype=torch.int32)
    ref = _torch_rows(bt, seq, pos)
    assert torch.equal(paged._rows(bt, seq, pos), ref)
    valid = torch.rand(N, generator=g, device="cuda") > 0.3
    assert torch.equal(paged._rows(bt, seq, pos, valid), torch.where(valid, ref, -1))
    ivalid = valid.to(torch.int32)
    assert torch.equal(paged._rows(bt, seq, pos, ivalid), torch.where(ivalid.bool(), ref, -1))
    # a strided block table (rows of a wider engine table)
    wide = torch.randint(0, 10_000, (B, 2 * P), generator=g, device="cuda", dtype=torch.int32)[:, :P]
    assert torch.equal(paged._rows(wide, seq, pos), _torch_rows(wide, seq, pos))


def test_paged_index_write_equals_two_scatters():
    g = torch.Generator(device="cuda").manual_seed(6)
    pages, R = 9, 37
    codes = torch.randint(0, 256, (R, 128), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
    sf = torch.randint(-2**31, 2**31 - 1, (R, 1), generator=g, device="cuda", dtype=torch.int64).to(torch.int32)
    dest = torch.randperm(pages * 64, generator=g, device="cuda")[:R].long()
    dest[::5] = -1
    base = torch.randint(0, 256, (pages, 64, 1, 132), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
    ref, new = base.clone(), base.clone()
    page, row = torch.div(dest, 64, rounding_mode="floor"), dest % 64
    ok = dest >= 0
    ci = torch.where(ok, page * (64 * 132 // 128) + row, -1).contiguous()
    si = torch.where(ok, page * (64 * 132 // 4) + (64 * 128) // 4 + row, -1).contiguous()
    extension().scatter_rows(codes, ref, 128, ci.view(1, -1), None)
    extension().scatter_rows(sf, ref, 4, si.view(1, -1), None)
    extension().paged_index_write(codes, sf, dest, new)
    assert torch.equal(new, ref)


@pytest.mark.parametrize("K", [96, 512, 600])
@pytest.mark.parametrize("with_extra", [False, True])
def test_paged_lists_masked_equals_lists_then_masked_fill(with_extra, K):
    g = torch.Generator(device="cuda").manual_seed(7)
    B, Wp, Cp, N, W = 5, 40, 20, 23, 128
    win = torch.randint(0, 5000, (B, Wp), generator=g, device="cuda", dtype=torch.int32)
    comp = torch.randint(0, 5000, (B, Cp), generator=g, device="cuda", dtype=torch.int32)
    seq = torch.randint(0, B, (N,), generator=g, device="cuda", dtype=torch.int32)
    pos = torch.randint(0, Cp * 64, (N,), generator=g, device="cuda", dtype=torch.int32)
    topk = torch.randint(-1, Cp * 64, (N, K), generator=g, device="cuda", dtype=torch.int32) if with_extra else None
    valid = torch.rand(N, generator=g, device="cuda") > 0.4
    m0 = torch.empty((N, W), dtype=torch.int32, device="cuda")
    e0 = torch.empty((N, K), dtype=torch.int32, device="cuda") if with_extra else None
    extension().paged_lists(seq, pos, win, W, topk, comp if with_extra else None, m0, e0)
    m0.masked_fill_(~valid[:, None], -1)
    if with_extra:
        e0.masked_fill_(~valid[:, None], -1)
    m1 = torch.full((N, W), 7, dtype=torch.int32, device="cuda")
    e1 = torch.full((N, K), 7, dtype=torch.int32, device="cuda") if with_extra else None
    extension().paged_lists_masked(seq, pos, win, W, topk, comp if with_extra else None, m1, e1, valid)
    assert torch.equal(m0, m1)
    if with_extra:
        assert torch.equal(e0, e1)
    extension().paged_lists_masked(seq, pos, win, W, topk, comp if with_extra else None, m1, e1, None)
    m2 = torch.empty_like(m1)
    e2 = torch.empty_like(e1) if with_extra else None
    extension().paged_lists(seq, pos, win, W, topk, comp if with_extra else None, m2, e2)
    assert torch.equal(m1, m2) and (not with_extra or torch.equal(e1, e2))


@pytest.mark.parametrize("ratio", [2, 4, 64])
def test_pool_varlen_wide_equals_warp_form(ratio):
    g = torch.Generator(device="cuda").manual_seed(23 + ratio)
    lens = [1, 1, 3, 7, 1, 2 * ratio + 1]
    p0s = [5, 2 * ratio - 1, 0, ratio + 1, 3 * ratio - 1, 1]
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    S = len(lens)
    proj = torch.randn((cu[-1], 1024), generator=g, device="cuda") * 2
    state = torch.randn((S + 2, ratio, 1024), generator=g, device="cuda")
    t = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")
    gs, gk = [], []
    for s_, n in enumerate(lens):
        for k in range((n + ratio - 1) // ratio + 1):
            gs.append(s_)
            gk.append(k)
    G = len(gs)
    w = torch.rand(512, generator=g, device="cuda") + 0.5
    outs = []
    for wide in (False, True):
        out = torch.zeros((G, 512), dtype=torch.bfloat16, device="cuda")
        valid = torch.full((G,), -7, dtype=torch.int32, device="cuda")
        cidx = torch.full((G,), -7, dtype=torch.int32, device="cuda")
        extension().pool_varlen(proj, t(cu), t(p0s), state, t(list(range(1, S + 1))), t(gs), t(gk), ratio, w, 1e-6,
                                out, valid, cidx, wide)
        outs.append((out, valid, cidx))
    assert outs[0][1].tolist() == outs[1][1].tolist() and outs[0][2].tolist() == outs[1][2].tolist()
    assert torch.equal(outs[0][0].view(torch.int16), outs[1][0].view(torch.int16))
    assert int(outs[0][1].sum()) > 0


@pytest.mark.parametrize("ratio", [1, 2, 4])
def test_index_bounds_equals_torch(ratio):
    g = torch.Generator(device="cuda").manual_seed(29)
    pos = torch.randint(0, 100000, (37,), generator=g, device="cuda", dtype=torch.int32)
    valid = torch.rand(37, generator=g, device="cuda") > 0.3
    ref = torch.where(valid, torch.div(pos + 1, ratio, rounding_mode="floor").to(torch.int32), 0)
    for v, r in ((valid, ref), (None, torch.div(pos + 1, ratio, rounding_mode="floor").to(torch.int32))):
        ke = torch.empty(37, dtype=torch.int32, device="cuda")
        extension().index_bounds(pos, v, ratio, ke)
        assert torch.equal(ke, r)


@pytest.mark.parametrize("N", [1, 5, 16, 17, 256, 257, 1024])
@pytest.mark.parametrize("strided", [False, True])
def test_q_kv_post_equals_qnorm_and_window_write(N, strided):
    """BF16 masters: q_kv_post (one launch, two projection outputs) == the q_norm row kernel + paged_quant_write
    kind 0, byte for byte, on both launch shapes (split rows <= 256, 8 rows per CTA above)."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops, spec
    g = torch.Generator(device="cuda").manual_seed(23 + N)
    if strided:                            # rows of wider buffers (column slices)
        qa = (torch.randn((N, 1280 + 64), generator=g, device="cuda") * 2).to(torch.bfloat16)[:, 64:]
        kv = (torch.randn((N, 512 + 128), generator=g, device="cuda") * 2).to(torch.bfloat16)[:, :512]
    else:
        qa = (torch.randn((N, 1280), generator=g, device="cuda") * 2).to(torch.bfloat16)
        kv = (torch.randn((N, 512), generator=g, device="cuda") * 2).to(torch.bfloat16)
    qa[0, :40] = 0
    qw = torch.rand(1280, generator=g, device="cuda") + 0.5
    kw = torch.rand(512, generator=g, device="cuda") + 0.5
    P = 4096
    ang = torch.rand((P, 32), generator=g, device="cuda") * 6.28
    cos, sin = ang.cos().contiguous(), ang.sin().contiguous()
    pos = torch.randint(0, P, (N,), generator=g, device="cuda", dtype=torch.int32)
    pages = (N + 63) // 64 + 2
    dest = torch.randperm(pages * 64, generator=g, device="cuda")[:N].long()
    if N > 2:
        dest[1] = -1
    eps = 1e-6
    qr = torch.empty((N, 1280), dtype=torch.bfloat16, device="cuda")
    ops.row_op(spec.ROW_Q_NORM, qa.contiguous(), 1280, w=qw, eps=eps, out=qr)
    pool_ref = torch.full((pages, 64, 528), 7, dtype=torch.uint8, device="cuda")
    extension().paged_quant_write(0, kv.contiguous(), kw, eps, cos, sin, pos, dest, pool_ref)
    q_out = torch.empty_like(qr)
    pool = torch.full_like(pool_ref, 7)
    extension().q_kv_post(qa, kv, qw, kw, eps, cos, sin, pos, dest, pool, q_out)
    assert torch.equal(q_out.view(torch.int16), qr.view(torch.int16))
    assert torch.equal(pool, pool_ref)
    # repeat: same bytes
    q2 = torch.empty_like(qr)
    pool2 = torch.full_like(pool_ref, 7)
    extension().q_kv_post(qa, kv, qw, kw, eps, cos, sin, pos, dest, pool2, q2)
    assert torch.equal(q2.view(torch.int16), q_out.view(torch.int16)) and torch.equal(pool2, pool)




def _comp_case(ratio, graph_style, seed):
    """A packed varlen batch: prefill chunks and decode rows at every phase of the group, a padded tail of
    n = 0 sequences that still own a k = 0 slot (the graph batch's layout) when graph_style."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    p0 = [0, 5, 7, 8, 1, 3, 130, 64, 2047, 33]
    n = [9, 1, 1, 1, 1, 6, 1, 17, 1, 40]
    if graph_style:
        p0 += [0, 0, 0]
        n += [0, 0, 0]
    B = len(p0)
    cu = [0]
    for v in n:
        cu.append(cu[-1] + v)
    gs, gk = [], []
    for s, v in enumerate(n):
        for k in range(max(1, (v + ratio - 1) // ratio) if graph_style else (v + ratio - 1) // ratio):
            gs.append(s)
            gk.append(k)
    T = cu[-1]
    proj = torch.randn((T, 1024), generator=g).cuda()
    slots = torch.randperm(B, generator=g).to(torch.int32)
    if graph_style:
        slots[-3:] = slots[0]                # padded rows point at a real slot (never touched: n = 0)
    state = torch.randn((B + 2, ratio, 1024), generator=g).cuda()
    # distinct pages per (sequence, entry): no two groups share a record (as in every real table)
    cbt = torch.randperm(B * 40, generator=g).view(B, 40).to(torch.int32).cuda()
    ibt = torch.randperm(B * 40, generator=g).view(B, 40).to(torch.int32).cuda()
    t = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")
    return dict(proj=proj, cu=t(cu), p0=t(p0), state=state, slot=slots.cuda(), gs=t(gs), gk=t(gk), cbt=cbt, ibt=ibt)


@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("graph_style", [False, True])
@pytest.mark.parametrize("per,own_index", [(64, False), (1, True)])
def test_comp_post_equals_pool_state_rows_and_write(ratio, graph_style, per, own_index):
    """comp_post (one launch) == pool_varlen(wide) + state_update + paged_rows (+ index rows) + cidx * ratio +
    paged_quant_write kind 1 into a torch.zeros latent: every output byte, the state and the compressed pool."""
    d = _comp_case(ratio, graph_style, 31 + ratio + 2 * graph_style + per)
    if per == 1:                                          # record-row entries: page id * 64
        d["cbt"] = d["cbt"] * 64
    G = d["gs"].numel()
    g = torch.Generator(device="cuda").manual_seed(3)
    w = torch.rand(512, generator=g, device="cuda") + 0.5
    ang = torch.rand((4096, 32), generator=g, device="cuda") * 6.28
    cos, sin = ang.cos().contiguous(), ang.sin().contiguous()
    eps = 1e-6
    # reference: the separate launches
    st_ref = d["state"].clone()
    lat_ref = torch.zeros((G, 512), dtype=torch.bfloat16, device="cuda")
    val_ref = torch.empty(G, dtype=torch.int32, device="cuda")
    cid_ref = torch.empty(G, dtype=torch.int32, device="cuda")
    extension().pool_varlen(d["proj"], d["cu"], d["p0"], st_ref, d["slot"], d["gs"], d["gk"], ratio, w, eps,
                            lat_ref, val_ref, cid_ref, True)
    extension().state_update(d["proj"], d["cu"], d["p0"], st_ref, d["slot"], ratio)
    dest_ref = paged._rows(d["cbt"], d["gs"], cid_ref, val_ref, per)
    idest_ref = paged._rows(d["ibt"], d["gs"], cid_ref, val_ref) if own_index else None
    gpos_ref = (cid_ref * ratio).to(torch.int32).contiguous()
    pool_ref = torch.full((d["cbt"].numel() * 64 + 64, 528), 5, dtype=torch.uint8, device="cuda")
    extension().paged_quant_write(1, lat_ref, None, 0.0, cos, sin, gpos_ref, dest_ref, pool_ref)
    # fused
    st = d["state"].clone()
    lat = torch.full((G, 512), 3.0, dtype=torch.bfloat16, device="cuda")
    val = torch.empty(G, dtype=torch.int32, device="cuda")
    cid = torch.empty(G, dtype=torch.int32, device="cuda")
    gpos = torch.empty(G, dtype=torch.int32, device="cuda")
    dest = torch.empty(G, dtype=torch.int64, device="cuda")
    idest = torch.empty(G, dtype=torch.int64, device="cuda") if own_index else None
    pool = torch.full_like(pool_ref, 5)
    extension().comp_post(d["proj"], d["cu"], d["p0"], st, d["slot"], d["gs"], d["gk"], ratio, w, eps, lat, val, cid,
                          gpos, d["cbt"], per, dest, d["ibt"] if own_index else None, idest, cos, sin, pool)
    assert torch.equal(lat.view(torch.int16), lat_ref.view(torch.int16))
    assert torch.equal(val, val_ref) and torch.equal(cid, cid_ref) and torch.equal(gpos, gpos_ref)
    assert torch.equal(dest, dest_ref)
    if own_index:
        assert torch.equal(idest, idest_ref)
    assert torch.equal(st, st_ref)
    assert torch.equal(pool, pool_ref)
    assert int(val_ref.sum()) > 0 and int((val_ref == 0).sum()) > 0          # both kinds of slots exercised


@pytest.mark.parametrize("R", [1, 7, 64, 300])
def test_index_k_store_equals_row_op_and_index_write(R):
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops, spec
    g = torch.Generator(device="cuda").manual_seed(41 + R)
    k = (torch.randn((R, 128), generator=g, device="cuda") * 2).to(torch.bfloat16)
    k[0, :32] = 0
    w = torch.rand(128, generator=g, device="cuda") + 0.5
    ang = torch.rand((4096, 32), generator=g, device="cuda") * 6.28
    cos, sin = ang.cos().contiguous(), ang.sin().contiguous()
    pos = torch.randint(0, 4096, (R,), generator=g, device="cuda", dtype=torch.int32)
    pages = (R + 63) // 64 + 2
    dest = torch.randperm(pages * 64, generator=g, device="cuda")[:R].long()
    if R > 2:
        dest[1] = -1
    codes = torch.empty((R, 128), dtype=torch.uint8, device="cuda")
    sf = torch.empty((R, 1), dtype=torch.int32, device="cuda")
    ops.row_op(spec.ROW_INDEX_K, k, 128, w=w, eps=1e-6, table=(cos, sin), pos=ops.Pos(pos=pos), codes=codes, sf=sf.view(R))
    ref = torch.full((pages, 64, 1, 132), 9, dtype=torch.uint8, device="cuda")
    extension().paged_index_write(codes, sf, dest, ref)
    pool = torch.full_like(ref, 9)
    extension().index_k_store(k, w, 1e-6, cos, sin, pos, dest, pool)
    assert torch.equal(pool, ref)


@pytest.mark.parametrize("rows,seqs", [(1, 1), (8, 8), (37, 5), (130, 130)])
def test_index_scores_bt_rows_equals_index_select(rows, seqs):
    """Paged index scores reading the block table through the token -> sequence map (bt_rows) == the scores on the
    gathered per-token table (the per-step index_select it replaces), on the window [0, ke)."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops
    g = torch.Generator(device="cpu").manual_seed(61 + rows)
    P = 40
    pages = seqs * P + 1
    codes = torch.randint(0, 0x77, (pages, 64, 128), generator=g, dtype=torch.int32).to(torch.uint8)
    codes |= torch.randint(0, 2, codes.shape, generator=g, dtype=torch.int32).to(torch.uint8) * 0x80
    sfk = torch.randint(118, 132, (pages, 64, 4), generator=g, dtype=torch.int32).to(torch.uint8)
    pool = torch.cat((codes.reshape(pages, -1), sfk.reshape(pages, -1)), 1).reshape(pages, 64, 1, 132).contiguous().cuda()
    bt = torch.randperm(pages, generator=g)[: seqs * P].view(seqs, P).to(torch.int32).cuda()
    tok_seq = torch.sort(torch.randint(0, seqs, (rows,), generator=g)).values.to(torch.int32).cuda()
    if rows == seqs:
        tok_seq = torch.arange(rows, dtype=torch.int32, device="cuda")
    qc = torch.randint(0, 0x77, (rows, 64, 128), generator=g, dtype=torch.int32).to(torch.uint8).cuda()
    qsf = torch.randint(118, 132, (rows, 64, 4), generator=g, dtype=torch.int32).to(torch.uint8).view(torch.int32).reshape(rows, 64).cuda()
    w = torch.randn((rows, 64), generator=g) * 0.1
    w[:, 32:] = 0
    w = w.to(torch.bfloat16).cuda()
    ke = torch.randint(1, P * 64, (rows,), generator=g, dtype=torch.int32).cuda()
    width = P * 64
    ref = ops.index_scores(qc, qsf, w, ke, width=width, cache=pool, block_table=bt.index_select(0, tok_seq))
    got = ops.index_scores(qc, qsf, w, ke, width=width, cache=pool, block_table=bt, bt_rows=tok_seq)
    mask = torch.arange(width, device="cuda").view(1, -1) < ke.view(-1, 1)
    assert torch.equal(torch.where(mask, got, 0).view(torch.int16), torch.where(mask, ref, 0).view(torch.int16))


@pytest.mark.parametrize("rows,width,nsel", [(1, 4096, 64), (5, 20000, 300), (33, 131072, 2048), (3, 512, 2048)])
def test_candidate_mask_equals_torch(rows, width, nsel):
    """candidate_mask: positions of [0, ke) outside the selected 8-blocks become -inf, others untouched; sel is
    dsa_topk's ascending, -1 padded output (also ids >= the row's block count)."""
    g = torch.Generator(device="cpu").manual_seed(71 + rows)
    blk = 8
    scores = torch.randn((rows, width), generator=g).to(torch.bfloat16).cuda()
    ke = torch.randint(1, width + 1, (rows,), generator=g, dtype=torch.int32)
    ke[0] = width
    sel = torch.full((rows, nsel), -1, dtype=torch.int32)
    for i in range(rows):
        nb = (int(ke[i]) + blk - 1) // blk
        k = int(torch.randint(0, nsel + 1, (1,), generator=g))
        ids = torch.randperm(nb + 7, generator=g)[:k].sort().values      # a few ids past nb
        sel[i, :ids.numel()] = ids.to(torch.int32)
    ke, sel = ke.cuda(), sel.cuda()
    ref = scores.clone()
    pos = torch.arange(width, device="cuda")
    for i in range(rows):
        nb = (int(ke[i]) + blk - 1) // blk
        if nb <= nsel:
            continue
        keep = torch.zeros(nb + 8, dtype=torch.bool, device="cuda")
        s = sel[i][sel[i] >= 0].long()
        keep[s[s < nb]] = True
        drop = (pos < ke[i]) & ~keep[(pos // blk).clamp_max(nb + 7)]
        ref[i][drop] = float("-inf")
    got = scores.clone()
    extension().candidate_mask(got, ke, sel, blk, (width + blk - 1) // blk)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize("band", [4096, 12288, 16384])
def test_topk_two_band_equals_single_launch(band):
    """dsa_topk's two-band execution (rows by live window length, each band its own plan) == the one launch
    planned for the captured width, for windows on both sides of the band edge, ties and NaN padding."""
    from isoexec.ops.attention.indexer_sm100 import topk as T
    g = torch.Generator(device="cpu").manual_seed(band)
    rows, width = 24, 1 << 19
    ke = torch.tensor([0, 1, 511, 512, 513, 2048, band - 1, band, band + 1, 3 * band, 70000, 200000, width]
                      + torch.randint(1, width, (11,), generator=g).tolist(), dtype=torch.int32)
    s = torch.randn((rows, width), generator=g).relu() * 3
    s[:, ::7] = 0
    scores = s.to(torch.bfloat16).cuda()
    ke = ke.cuda()
    ref = T.dsa_topk(scores, ke, k=512, band=0)
    got = T.dsa_topk(scores, ke, k=512, band=band)
    assert torch.equal(got, ref)


@pytest.mark.parametrize("rows,width", [(1, 4096), (8, 131072), (33, 20000)])
def test_block_max_equals_torch(rows, width):
    """block_max: per 8-block of [0, ke) the max score (the newest block +inf), as the candidate select reads it."""
    g = torch.Generator(device="cpu").manual_seed(81 + rows)
    stride = (width + 255) // 256 * 256
    scores = torch.randn((rows, stride), generator=g).to(torch.bfloat16).cuda()[:, :width]
    ke = torch.randint(1, width + 1, (rows,), generator=g, dtype=torch.int32).cuda()
    nbmax = (width + 7) // 8
    bm = torch.full((rows, nbmax), 7.0, dtype=torch.bfloat16, device="cuda")
    nb = torch.empty_like(ke)
    extension().block_max(scores, ke, 8, bm, nb)
    for i in range(rows):
        n = (int(ke[i]) + 7) // 8
        assert int(nb[i]) == n
        ref = scores[i, :(n - 1) * 8].float().view(-1, 8).amax(1).to(torch.bfloat16) if n > 1 else scores.new_empty(0)
        assert torch.equal(bm[i, :n - 1].view(torch.int16), ref.view(torch.int16))
        assert bm[i, n - 1].item() == float("inf")


@pytest.mark.parametrize("N", [1, 8, 256, 300])
@pytest.mark.parametrize("per", [64, 1])
def test_q_kv_post_rows_equals_rows_then_q_kv_post(N, per):
    """q_kv_post_rows (the window record rows computed in the kernel) == paged_rows + q_kv_post: the rows, qr and
    the window pool, byte for byte (masked rows -1, nothing written)."""
    g = torch.Generator(device="cuda").manual_seed(91 + N + per)
    qa = (torch.randn((N, 1280), generator=g, device="cuda") * 2).to(torch.bfloat16)
    kv = (torch.randn((N, 512), generator=g, device="cuda") * 2).to(torch.bfloat16)
    qw = torch.rand(1280, generator=g, device="cuda") + 0.5
    kw = torch.rand(512, generator=g, device="cuda") + 0.5
    ang = torch.rand((4096, 32), generator=g, device="cuda") * 6.28
    cos, sin = ang.cos().contiguous(), ang.sin().contiguous()
    B = max(1, N // 2)
    pages = 4 * B + 8
    bt = torch.randperm(pages, generator=g, device="cuda")[: B * 4].view(B, 4).int()
    seq = torch.randint(0, B, (N,), generator=g, device="cuda", dtype=torch.int32)
    pos = torch.randint(0, 256, (N,), generator=g, device="cuda", dtype=torch.int32)
    valid = torch.rand(N, generator=g, device="cuda") > 0.2
    if per == 1:
        bt = bt * 64                                      # record-row entries (block-chunk placement)
    pos_unique = torch.arange(N, device="cuda", dtype=torch.int32) % 256
    seq = torch.arange(N, device="cuda", dtype=torch.int32) % B
    pos = pos_unique                                       # distinct (seq, pos) pairs: no two rows share a record
    rows = paged._rows(bt, seq, pos, valid, per)
    qr_ref = torch.empty((N, 1280), dtype=torch.bfloat16, device="cuda")
    pool_ref = torch.full((pages * 64 + 64, 528), 3, dtype=torch.uint8, device="cuda")
    extension().q_kv_post(qa, kv, qw, kw, 1e-6, cos, sin, pos, rows, pool_ref, qr_ref)
    dest = torch.full((N,), 123, dtype=torch.int64, device="cuda")
    qr = torch.empty_like(qr_ref)
    pool = torch.full_like(pool_ref, 3)
    extension().q_kv_post_rows(qa, kv, qw, kw, 1e-6, cos, sin, pos, dest, pool, qr, bt, seq, valid, per)
    assert torch.equal(dest, rows)
    assert torch.equal(qr.view(torch.int16), qr_ref.view(torch.int16))
    assert torch.equal(pool, pool_ref)


@pytest.mark.parametrize("rows,P,kemax", [(1, 400, 25000), (8, 300, 19000), (32, 80, 5000), (130, 20, 1200), (3, 40, 1)])
def test_index_scores_auto_span_equals_fixed(rows, P, kemax):
    """The FP32-tree fold's in-kernel item span (INDEX_SPAN_AUTO) == the fixed 16-tile span, on the window [0, ke)."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops
    g = torch.Generator(device="cpu").manual_seed(101 + rows)
    pages = rows * P + 1
    codes = torch.randint(0, 0x77, (pages, 64, 128), generator=g, dtype=torch.int32).to(torch.uint8)
    sfk = torch.randint(118, 132, (pages, 64, 4), generator=g, dtype=torch.int32).to(torch.uint8)
    pool = torch.cat((codes.reshape(pages, -1), sfk.reshape(pages, -1)), 1).reshape(pages, 64, 1, 132).contiguous().cuda()
    bt = torch.randperm(pages, generator=g)[: rows * P].view(rows, P).to(torch.int32).cuda()
    qc = torch.randint(0, 0x77, (rows, 64, 128), generator=g, dtype=torch.int32).to(torch.uint8).cuda()
    qsf = torch.randint(118, 132, (rows, 64, 4), generator=g, dtype=torch.int32).to(torch.uint8).view(torch.int32).reshape(rows, 64).cuda()
    w = torch.randn((rows, 64), generator=g) * 0.1
    w[:, 32:] = 0
    w = w.to(torch.bfloat16).cuda()
    ke = torch.randint(0, kemax + 1, (rows,), generator=g, dtype=torch.int32).clamp_max(P * 64).cuda()
    width = P * 64
    old = ops.INDEX_SPAN_AUTO
    try:
        ops.INDEX_SPAN_AUTO = False
        ref = ops.indexer_tree(qc, qsf, w, ke, width=width, cache=pool, block_table=bt)
        ops.INDEX_SPAN_AUTO = True
        got = ops.indexer_tree(qc, qsf, w, ke, width=width, cache=pool, block_table=bt)
    finally:
        ops.INDEX_SPAN_AUTO = old
    mask = torch.arange(width, device="cuda").view(1, -1) < ke.view(-1, 1)
    assert torch.equal(torch.where(mask, got, 0).view(torch.int16), torch.where(mask, ref, 0).view(torch.int16))




@pytest.mark.parametrize("mode", ["solo", "two_launch"])
def test_topk_solo_rows_and_rows_out(mode):
    """dsa_topk on a cluster-planned width: the one-launch solo form (short rows on rank 0 alone) and the two-launch
    form == the plain cluster launch; the rows output == paged_lists_masked's extra list on the same selection."""
    from isoexec.ops.attention.indexer_sm100 import topk as T
    g = torch.Generator(device="cpu").manual_seed(7)
    rows, width = 24, 1 << 19
    ke = torch.tensor([0, 1, 511, 512, 513, 2048, 32767, 32768, 32769, 70000, 200000, width]
                      + torch.randint(1, width, (12,), generator=g).tolist(), dtype=torch.int32).cuda()
    s = torch.randn((rows, width), generator=g).relu() * 3
    s[:, ::5] = 0
    scores = s.to(torch.bfloat16).cuda()
    ref = T.dsa_topk(scores, ke, k=512, band=0)
    old = T.BAND_MODE
    try:
        T.BAND_MODE = mode
        assert torch.equal(T.dsa_topk(scores, ke, k=512), ref)
    finally:
        T.BAND_MODE = old
    seqs = torch.arange(rows, dtype=torch.int32, device="cuda").flip(0).contiguous()
    pages = width // 64
    cbt = torch.randperm(rows * pages, generator=g).view(rows, pages).to(torch.int32).cuda()
    for per in (64, 1):
        bt = cbt if per == 64 else cbt * 64
        rows_out = torch.empty((rows, 512), dtype=torch.int32, device="cuda")
        sel = T.dsa_topk(scores, ke, k=512, rows_out=rows_out, rows_bt=bt, rows_seq=seqs, rows_per=per)
        assert torch.equal(sel, ref)
        main = torch.empty((rows, 128), dtype=torch.int32, device="cuda")
        extra = torch.empty((rows, 512), dtype=torch.int32, device="cuda")
        pos = torch.full((rows,), 100, dtype=torch.int32, device="cuda")
        wbt = torch.zeros((rows, 4), dtype=torch.int32, device="cuda")
        extension().paged_lists_masked(seqs, pos, wbt, 128, ref, bt, main, extra, None, 64, per)
        assert torch.equal(rows_out, extra)


@pytest.mark.parametrize("N,per", [(1, 64), (8, 64), (300, 1)])
def test_q_kv_post_rows_main_list_equals_paged_lists(N, per):
    """q_kv_post_rows' main list == paged_lists_masked's main list (window slots, masked rows -1)."""
    g = torch.Generator(device="cuda").manual_seed(13 + N)
    qa = (torch.randn((N, 1280), generator=g, device="cuda")).to(torch.bfloat16)
    kv = (torch.randn((N, 512), generator=g, device="cuda")).to(torch.bfloat16)
    qw = torch.rand(1280, generator=g, device="cuda") + 0.5
    kw = torch.rand(512, generator=g, device="cuda") + 0.5
    ang = torch.rand((4096, 32), generator=g, device="cuda") * 6.28
    cos, sin = ang.cos().contiguous(), ang.sin().contiguous()
    B = N
    bt = torch.randperm(B * 8, generator=g, device="cuda").view(B, 8).int()
    if per == 1:
        bt = bt * 64
    seq = torch.arange(N, device="cuda", dtype=torch.int32)
    pos = torch.randint(0, 500, (N,), generator=g, device="cuda", dtype=torch.int32)
    pos[0] = 3                                              # a window shorter than W
    valid = torch.rand(N, generator=g, device="cuda") > 0.2
    valid[0] = True
    main_ref = torch.empty((N, 128), dtype=torch.int32, device="cuda")
    extension().paged_lists_masked(seq, pos, bt, 128, None, None, main_ref, None, valid, per, 64)
    dest = torch.empty(N, dtype=torch.int64, device="cuda")
    pool = torch.zeros((B * 8 * 64 + 64, 528), dtype=torch.uint8, device="cuda")
    qr = torch.empty((N, 1280), dtype=torch.bfloat16, device="cuda")
    main = torch.full((N, 128), 77, dtype=torch.int32, device="cuda")
    extension().q_kv_post_rows(qa, kv, qw, kw, 1e-6, cos, sin, pos, dest, pool, qr, bt, seq, valid, per, main)
    assert torch.equal(main, main_ref)


def test_topk_many_rows_one_cta_plan_equals_cluster():
    """More rows than clusters fit on the SMs at a cluster-planned width: the 1-CTA plan == the cluster plan."""
    from isoexec.ops.attention.indexer_sm100 import topk as T
    g = torch.Generator(device="cpu").manual_seed(9)
    rows, width = 130, 131072
    ke = torch.randint(0, width + 1, (rows,), generator=g, dtype=torch.int32)
    ke[:4] = torch.tensor([width, 70000, 513, 0], dtype=torch.int32)
    s = torch.randn((rows, width), generator=g).relu() * 3
    s[:, ::3] = 0
    scores = s.to(torch.bfloat16).cuda()
    ke = ke.cuda()
    ref = T.dsa_topk(scores, ke, k=512, cluster=2, threads=512)
    assert torch.equal(T.dsa_topk(scores, ke, k=512), ref)




@pytest.mark.parametrize("R,H,Hs", [(1, 32, 64), (8, 32, 64), (130, 32, 64), (5, 64, 64), (1, 32, 32), (130, 32, 32)])
def test_index_kq_store_equals_k_store_and_q_rows(R, H, Hs):
    """index_kq_store (one launch) == index_k_store + row_op ROW_INDEX_Q (codes, scales, padded heads when the rows are
    64 wide, the pool)."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops, spec
    g = torch.Generator(device="cuda").manual_seed(51 + R + H)
    k = (torch.randn((R, 128), generator=g, device="cuda") * 2).to(torch.bfloat16)
    iq = (torch.randn((R * H, 128), generator=g, device="cuda") * 2).to(torch.bfloat16)
    w = torch.rand(128, generator=g, device="cuda") + 0.5
    ang = torch.rand((4096, 32), generator=g, device="cuda") * 6.28
    cos, sin = ang.cos().contiguous(), ang.sin().contiguous()
    kpos = torch.randint(0, 4096, (R,), generator=g, device="cuda", dtype=torch.int32)
    qpos = torch.randint(0, 4096, (R,), generator=g, device="cuda", dtype=torch.int32)
    pages = (R + 63) // 64 + 2
    dest = torch.randperm(pages * 64, generator=g, device="cuda")[:R].long()
    if R > 2:
        dest[1] = -1
    pool_ref = torch.full((pages, 64, 1, 132), 9, dtype=torch.uint8, device="cuda")
    extension().index_k_store(k, w, 1e-6, cos, sin, kpos, dest, pool_ref)
    c_ref = torch.full((R, Hs, 128), 3, dtype=torch.uint8, device="cuda")
    s_ref = torch.full((R, Hs), 3, dtype=torch.int32, device="cuda")
    ops.row_op(spec.ROW_INDEX_Q, iq, 128, table=(cos, sin), pos=ops.Pos(pos=qpos, div=H), codes=c_ref.view(R * Hs, 128),
               sf=s_ref.view(R * Hs), dst=(H, Hs, 0), pad_heads=H if H < Hs else 0)
    pool = torch.full_like(pool_ref, 9)
    c = torch.full_like(c_ref, 3)
    s = torch.full_like(s_ref, 3)
    extension().index_kq_store(k, w, 1e-6, cos, sin, kpos, dest, pool, iq, qpos, H, c.view(R * Hs, 128), s.view(R * Hs),
                               H, H if H < Hs else 0)
    assert torch.equal(pool, pool_ref) and torch.equal(c, c_ref) and torch.equal(s, s_ref)


@pytest.mark.parametrize("rows,width", [(128, 16384), (256, 32768)])
def test_block_max_large_batch_equals_torch(rows, width):
    test_block_max_equals_torch(rows, width)
