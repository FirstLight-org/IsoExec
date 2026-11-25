"""Index scores on DeepGEMM's SM100 MQA logits (``index_logits``, recipe ``sm100_dsa_index_logits_dg_v1``).

Relations (bitwise on the window ``[0, ke)`` of every row unless stated):
* paged (64- and 128-entry pages, rows of one sequence sharing key loads) == contiguous keys (the trainer's form);
* a row's logits do not depend on the batch around it (alone, inside a mixed batch, permuted) nor on the schedule;
* the query scale carried in the MMA scale word == the scale folded into the BF16 weight with unit scale words
  (native vLLM's placement): exact powers of two, the same products;
* a sequence run ending with padded rows (context 0) still scores its other rows (the run-maximum schedule edit);
* CUDA-graph replay == eager == a repeat;
* agreement with an FP64 emulation of the fold (BF16 relu, four BF16 FMA chains, two adds): nearly all bits equal.
"""
import pytest
import torch

from isoexec.ops.attention.dsa_sm100 import index_logits as il
from isoexec.ops.attention.dsa_sm100.tests._inputs import fp8_row_quantize

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _q(g, rows, heads):
    q = torch.randn((rows, heads, 128), generator=g, device="cuda") * 2
    codes, sf = fp8_row_quantize(q)
    w = (torch.randn((rows, heads), generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    return codes.contiguous(), sf.view(rows, heads).contiguous(), w.contiguous()


def _keys(g, seqs, C):
    """codes [seqs, C, 128], scale words [seqs, C] (rows 16-byte aligned, as ``ops.index_keys`` lays them out)."""
    k = torch.randn((seqs, C, 128), generator=g, device="cuda")
    kc, ks = fp8_row_quantize(k)
    Cp = (C + 3) // 4 * 4
    sw = torch.zeros((seqs, Cp), dtype=torch.int32, device="cuda")
    sw[:, :C] = ks.view(seqs, C)
    return kc.view(torch.uint8).contiguous(), sw[:, :C]


def _pool(g, kc, ks, P):
    """The paged pool [pages, P, 1, 132] holding sequence s's key j at (bt[s, j // P], j % P); one spare page."""
    seqs, C, _ = kc.shape
    npg = (C + P - 1) // P
    pages = seqs * npg + 1
    bt = torch.randperm(pages, generator=g, device="cuda")[: seqs * npg].view(seqs, npg).to(torch.int32).contiguous()
    pool = torch.zeros((pages, P * 132), dtype=torch.uint8, device="cuda")
    pad = npg * P - C
    kcp = torch.nn.functional.pad(kc, (0, 0, 0, pad)).view(seqs, npg, P * 128)
    ksp = torch.nn.functional.pad(ks, (0, pad)).view(seqs, npg, P).contiguous().view(torch.uint8).view(seqs, npg, P * 4)
    pool[bt.long().view(-1), : P * 128] = kcp.reshape(-1, P * 128)
    pool[bt.long().view(-1), P * 128:] = ksp.reshape(-1, P * 4)
    return pool.view(pages, P, 1, 132), bt


def _win(a, ke):
    mask = torch.arange(a.shape[1], device=a.device)[None, :] < ke.long()[:, None]
    return a.contiguous().view(torch.int16)[mask]


def _contiguous_rows(codes, sf, w, kc, ks, tok_seq, ke, width):
    """Each row scored over its sequence's contiguous keys, one call per sequence (the trainer's form)."""
    out = torch.zeros((codes.shape[0], width), dtype=torch.bfloat16, device="cuda")
    for s in tok_seq.unique().tolist():
        rows = (tok_seq == s).nonzero().view(-1)
        o = il.index_logits(codes[rows].contiguous(), sf[rows].contiguous(), w[rows].contiguous(), kc[s], ks[s],
                            ke[rows].contiguous(), width=width)
        out[rows] = o
    return out


@pytest.mark.parametrize("seqs,per,C,P,H", [(1, 1, 1000, 64, 32), (7, 1, 5000, 128, 32), (3, 37, 2304, 64, 32),
                                            (2, 5, 777, 64, 64), (5, 9, 4100, 128, 64), (130, 1, 640, 64, 32)])
def test_paged_equals_contiguous(seqs, per, C, P, H):
    g = torch.Generator(device="cuda").manual_seed(seqs * 1000 + per + C + P + H)
    rows = seqs * per
    codes, sf, w = _q(g, rows, H)
    kc, ks = _keys(g, seqs, C)
    pool, bt = _pool(g, kc, ks, P)
    tok_seq = torch.arange(seqs, device="cuda", dtype=torch.int32).repeat_interleave(per).contiguous()
    ke = torch.randint(1, C + 1, (rows,), generator=g, device="cuda", dtype=torch.int32)
    ke[::per] = C                                         # each run's first row sees every key
    width = bt.shape[1] * P
    got = il.index_logits_paged(codes, sf, w, pool, ke, bt, tok_seq, width=width)
    ref = _contiguous_rows(codes, sf, w, kc, ks, tok_seq, ke, width)
    assert torch.equal(_win(got, ke), _win(ref, ke))
    # the per-token table of vLLM's form (one block-table row per query row) == the per-sequence table
    tok = torch.arange(rows, device="cuda", dtype=torch.int32)
    alt = il.index_logits_paged(codes, sf, w, pool, ke, bt.index_select(0, tok_seq.long()).contiguous(), tok,
                                width=width)
    assert torch.equal(_win(alt, ke), _win(got, ke))


@pytest.mark.parametrize("H", [32, 64])
def test_batch_composition_and_schedule_invariance(H):
    g = torch.Generator(device="cuda").manual_seed(7 + H)
    seqs, C, P = 256, 1500, 64
    codes, sf, w = _q(g, seqs, H)
    kc, ks = _keys(g, seqs, C)
    pool, bt = _pool(g, kc, ks, P)
    ke = torch.randint(0, C + 1, (seqs,), generator=g, device="cuda", dtype=torch.int32)
    ke[:4] = torch.tensor([C, 1, 383, 385], device="cuda", dtype=torch.int32)
    width = bt.shape[1] * P
    seq = torch.arange(seqs, device="cuda", dtype=torch.int32)
    full = il.index_logits_paged(codes, sf, w, pool, ke, bt, seq, width=width)
    for r in (0, 1, 2, 3, 77, 255):                      # alone (B = 1: one row, its own schedule)
        one = il.index_logits_paged(codes[r:r + 1], sf[r:r + 1], w[r:r + 1], pool, ke[r:r + 1], bt,
                                    seq[r:r + 1].clone(), width=width)
        assert torch.equal(_win(one, ke[r:r + 1]), _win(full[r:r + 1], ke[r:r + 1])), r
    perm = torch.randperm(seqs, generator=g, device="cuda")
    pg = il.index_logits_paged(codes[perm].contiguous(), sf[perm].contiguous(), w[perm].contiguous(), pool,
                               ke[perm].contiguous(), bt, seq[perm].contiguous(), width=width)
    assert torch.equal(_win(pg, ke[perm]), _win(full[perm], ke[perm]))
    # the first 32 rows under the schedule of a different row set (only the work division differs)
    sub = slice(0, 32)
    meta = il.paged_schedule(ke[sub].contiguous(), seq[sub].contiguous())
    a = il.index_logits_paged(codes[sub], sf[sub], w[sub], pool, ke[sub].contiguous(), bt, seq[sub].contiguous(),
                              width=width, schedule=meta)
    assert torch.equal(_win(a, ke[sub]), _win(full[sub], ke[sub]))


@pytest.mark.parametrize("mode", ["paged", "contiguous"])
def test_query_scale_in_scale_word_equals_scale_in_weight(mode):
    """IsoExec carries the per-(row, head) power of two 2^e in the MMA scale word; native folds it into the BF16
    weight (bf16(w * 2^e), exact) with unit scale words: bit for bit the same logits."""
    g = torch.Generator(device="cuda").manual_seed(11)
    rows, H, C, P = 40, 32, 3000, 64
    q = torch.randn((rows, H, 128), generator=g, device="cuda")
    e = torch.randint(-9, 7, (rows, H), generator=g, device="cuda")
    codes = (q * 64).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    byte = (e + 127).to(torch.int32)
    sf_e = (byte | (byte << 8) | (byte << 16) | (byte << 24)).contiguous()
    sf_1 = torch.full_like(sf_e, 0x7F7F7F7F)
    w = (torch.randn((rows, H), generator=g, device="cuda") * 0.05).to(torch.bfloat16).contiguous()
    w_folded = (w.float() * torch.exp2(e.float())).to(torch.bfloat16).contiguous()
    assert torch.equal(w_folded.float(), w.float() * torch.exp2(e.float()))        # the fold is exact here
    kc, ks = _keys(g, 1, C)
    ke = torch.randint(1, C + 1, (rows,), generator=g, device="cuda", dtype=torch.int32)
    if mode == "paged":
        pool, bt = _pool(g, kc, ks, P)
        seq = torch.zeros(rows, dtype=torch.int32, device="cuda")
        run = lambda s, ww: il.index_logits_paged(codes, s, ww, pool, ke, bt, seq, width=bt.shape[1] * P)
    else:
        run = lambda s, ww: il.index_logits(codes, s, ww, kc[0], ks[0], ke, width=C)
    assert torch.equal(_win(run(sf_e, w), ke), _win(run(sf_1, w_folded), ke))


def test_run_ending_in_padding_rows_is_scored():
    """Rows of one sequence followed by padded rows of the same sequence index with context 0 (CUDA-graph padding):
    every real row is scored (the run's context is the maximum, not the last row's)."""
    g = torch.Generator(device="cuda").manual_seed(5)
    H, C, P = 32, 2000, 64
    kc, ks = _keys(g, 3, C)
    pool, bt = _pool(g, kc, ks, P)
    tok_seq = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.int32, device="cuda")
    ke = torch.tensor([1500, 1600, 0, 2000, 0, 7, 900, 0, 0], dtype=torch.int32, device="cuda")
    rows = tok_seq.numel()
    codes, sf, w = _q(g, rows, H)
    width = bt.shape[1] * P
    out = il.alloc_logits(rows, width, "cuda")
    out.fill_(float("nan"))
    il.index_logits_paged(codes, sf, w, pool, ke, bt, tok_seq, width=width, out=out)
    ref = _contiguous_rows(codes, sf, w, kc, ks, tok_seq, ke, width)
    assert not torch.isnan(out.float()[torch.arange(width, device="cuda")[None] < ke.long()[:, None]]).any()
    assert torch.equal(_win(out, ke), _win(ref, ke))


def test_graph_replay_and_repeat():
    g = torch.Generator(device="cuda").manual_seed(3)
    H, C, P, rows = 32, 4096, 64, 128
    codes, sf, w = _q(g, rows, H)
    kc, ks = _keys(g, rows, C)
    pool, bt = _pool(g, kc, ks, P)
    seq = torch.arange(rows, dtype=torch.int32, device="cuda")
    ke = torch.randint(1, C + 1, (rows,), generator=g, device="cuda", dtype=torch.int32)
    width = bt.shape[1] * P
    eager = il.index_logits_paged(codes, sf, w, pool, ke, bt, seq, width=width)
    for _ in range(2):
        assert torch.equal(_win(il.index_logits_paged(codes, sf, w, pool, ke, bt, seq, width=width), ke), _win(eager, ke))
    out = il.alloc_logits(rows, width, "cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        il.index_logits_paged(codes, sf, w, pool, ke, bt, seq, width=width, out=out)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        il.index_logits_paged(codes, sf, w, pool, ke, bt, seq, width=width, out=out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_win(out, ke), _win(eager, ke))


def _bf16(x64):
    return x64.to(torch.float32).to(torch.bfloat16).to(torch.float64)


@pytest.mark.parametrize("H", [32, 64])
def test_agrees_with_fp64_emulation_of_the_fold(H):
    """FP64 emulation: d = fl32(exact dot), r = bf16(relu d), chains c = bf16(r_h w_h + c) over h = 4i + c, then
    bf16(c0 + c2), bf16(c1 + c3), bf16(low + high). The MMA's FP32 accumulation is the hardware's, so a few
    logits may differ by an ulp or two; nearly all are equal."""
    g = torch.Generator(device="cuda").manual_seed(H)
    rows, C = 24, 1000
    codes, sf, w = _q(g, rows, H)
    kc, ks = _keys(g, 1, C)
    ke = torch.full((rows,), C, dtype=torch.int32, device="cuda")
    got = il.index_logits(codes, sf, w, kc[0], ks[0], ke, width=C).double()
    deq = lambda c, s: c.view(torch.float8_e4m3fn).double() * torch.exp2(((s & 255) - 127).double()).unsqueeze(-1)
    qd = deq(codes, sf)                                                   # [rows, H, 128]
    kd = deq(kc[0], ks[0])                                                # [C, 128]
    d = torch.einsum("rhd,cd->rch", qd, kd).float().double()             # [rows, C, H]
    r = _bf16(d.clamp_min(0))
    wd = w.double()[:, None, :]
    ch = [torch.zeros((rows, C), dtype=torch.float64, device="cuda") for _ in range(4)]
    for h in range(H):
        ch[h % 4] = _bf16(r[..., h] * wd[..., h] + ch[h % 4])
    ref = _bf16(_bf16(ch[0] + ch[2]) + _bf16(ch[1] + ch[3]))
    exact = (got == ref).double().mean().item()
    a = got.float().to(torch.bfloat16).view(torch.int16).int()
    b = ref.float().to(torch.bfloat16).view(torch.int16).int()
    assert exact > 0.995, exact
    assert (a - b).abs().max().item() <= 8


def test_refusals():
    g = torch.Generator(device="cuda").manual_seed(1)
    codes, sf, w = _q(g, 4, 16)
    kc, ks = _keys(g, 1, 256)
    with pytest.raises(RuntimeError, match="index heads"):
        il.index_logits(codes, sf, w, kc[0], ks[0], torch.full((4,), 256, dtype=torch.int32, device="cuda"), width=256)
    n = il._ext().MAX_PAGED_ROWS + 1
    with pytest.raises(RuntimeError, match="rows per schedule"):
        il.paged_schedule(torch.ones(n, dtype=torch.int32, device="cuda"), torch.zeros(n, dtype=torch.int32, device="cuda"))
