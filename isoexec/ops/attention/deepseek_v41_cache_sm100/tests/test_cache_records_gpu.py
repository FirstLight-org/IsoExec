"""The V4.1 cache program (port/CACHE-SPEC.md): the FP8 528 B window / compressed records and the FP8 index keys the
engine stores == the quantize-then-dequantize values the trainer computes, bit for bit; the records equal the CPU
statement of the quantizer; a record's bytes depend on its own row only (batch composition, chunking, repeat, CUDA
graph); the sparse-attention leaf reading the records == the leaf reading the trainer's BF16 rows."""
import pytest
import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops, reference as ref, spec
from isoexec.ops.attention.deepseek_v41_cache_sm100._extension import extension

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _cfg():
    from ._config import published_config
    return published_config()


def _rows(n, seed, scale):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((n, 512), generator=g, device="cuda") * scale
    x[0] = 0                                                   # the 1e-4 floor: exponent byte 105
    if n > 2:
        x[1, :32] *= 1e4                                       # one block far above the rest of its row
        x[2, 480:] *= 1e-6                                     # tiny RoPE-tail blocks
    return x.to(torch.bfloat16)


def decode_records(pool, dest):
    """uint8 [.., 528] records at rows ``dest`` -> BF16 [n, 512]: e4m3(code) * 2^(byte - 127), exact."""
    rec = pool.reshape(-1, spec.KV_RECORD_BYTES)[dest.long()]
    vals = rec[:, :512].contiguous().view(torch.float8_e4m3fn).float().view(-1, 16, 32)
    scale = torch.exp2(rec[:, 512:].float() - 127.0)
    return (vals * scale[..., None]).view(-1, 512).to(torch.bfloat16)


def _write(kind, x, w, eps, cos, sin, pos, dest, pages, fill=7):
    pool = torch.full((pages, 64, spec.KV_RECORD_BYTES), fill, dtype=torch.uint8, device="cuda")
    extension().paged_quant_write(kind, x, w, eps, cos, sin, pos, dest, pool)
    return pool


def _case(n, seed, ratio):
    cfg = _cfg()
    cos, sin = ops.rope_table(cfg, ratio, 8192, "cuda")
    g = torch.Generator(device="cuda").manual_seed(seed)
    pos = torch.randint(0, 8192, (n,), generator=g, device="cuda", dtype=torch.int32)
    pages = (n + 63) // 64 + 3
    dest = torch.randperm(pages * 64, generator=g, device="cuda")[:n].long().contiguous()
    w = torch.rand(512, generator=g, device="cuda") + 0.5
    return cfg, cos, sin, pos, pages, dest, w


@pytest.mark.parametrize("n", [1, 7, 300])
def test_window_record_equals_qdq(n):
    cfg, cos, sin, pos, pages, dest, w = _case(n, 11 + n, 0)
    x = _rows(n, n, 3.0)
    pool = _write(0, x, w, cfg.rms_norm_eps, cos, sin, pos, dest, pages)
    qdq = ops.window_kv_qdq(x, w, cfg.rms_norm_eps, (cos, sin), ops.Pos(pos=pos))
    assert torch.equal(decode_records(pool, dest).view(torch.int16), qdq.view(torch.int16))
    emul = ref.window_kv_emul(x, w, cfg.rms_norm_eps, cos, sin, pos)
    assert torch.equal(qdq.cpu().view(torch.int16), emul.view(torch.int16))
    assert int(pool.reshape(-1, 528)[dest[0], 512:].unique().item()) == 105        # zero row: 2^-22 scales
    # untouched rows keep their bytes
    mask = torch.ones(pages * 64, dtype=torch.bool, device="cuda")
    mask[dest] = False
    assert bool((pool.reshape(-1, 528)[mask] == 7).all())


@pytest.mark.parametrize("n", [1, 7, 300])
@pytest.mark.parametrize("ratio", [1, 2])
def test_compressed_record_equals_qdq(n, ratio):
    cfg, cos, sin, pos, pages, dest, _ = _case(n, 23 + n, ratio)
    gpos = (pos // ratio * ratio).to(torch.int32).contiguous()
    x = _rows(n, 3 * n + ratio, 2.0)
    pool = _write(1, x, None, 0.0, cos, sin, gpos, dest, pages)
    qdq = ops.compressed_kv_qdq(x, (cos, sin), ops.Pos(pos=gpos))
    assert torch.equal(decode_records(pool, dest).view(torch.int16), qdq.view(torch.int16))
    emul = ref.compressed_kv_emul(x, cos, sin, gpos)
    assert torch.equal(qdq.cpu().view(torch.int16), emul.view(torch.int16))


def test_compressed_record_quantizes_the_fp32_rotation():
    """The compressed record is not the window quantizer on BF16-rounded RoPE output (CACHE-SPEC D3/D11)."""
    n = 4096
    cfg, cos, sin, pos, pages, dest, _ = _case(n, 5, 2)
    x = _rows(n, 77, 2.0)
    qdq = ops.compressed_kv_qdq(x, (cos, sin), ops.Pos(pos=pos))
    via_bf16 = ref.quant_emul(ref.rope_emul(x, cos, sin, pos), 32)
    assert not torch.equal(qdq.cpu().view(torch.int16), via_bf16.view(torch.int16))


def test_skipped_rows_and_negative_dest():
    cfg, cos, sin, pos, pages, dest, w = _case(9, 3, 0)
    x = _rows(9, 4, 1.0)
    dest = dest.clone()
    dest[3] = -1
    pool = _write(0, x, w, cfg.rms_norm_eps, cos, sin, pos, dest, pages)
    keep = dest >= 0
    full = _write(0, x[keep].contiguous(), w, cfg.rms_norm_eps, cos, sin, pos[keep].contiguous(),
                  dest[keep].contiguous(), pages)
    assert torch.equal(pool, full)


@pytest.mark.parametrize("kind", [0, 1])
def test_batch_composition_chunking_and_repeat(kind):
    """Each record is a function of its own row: a permuted batch, two chunks and a second run write the same pool."""
    n = 257
    cfg, cos, sin, pos, pages, dest, w = _case(n, 41 + kind, 2 if kind else 0)
    x = _rows(n, 9 + kind, 2.5)
    ww = w if kind == 0 else None
    eps = cfg.rms_norm_eps if kind == 0 else 0.0
    full = _write(kind, x, ww, eps, cos, sin, pos, dest, pages)
    perm = torch.randperm(n, generator=torch.Generator(device="cuda").manual_seed(1), device="cuda")
    permuted = _write(kind, x[perm].contiguous(), ww, eps, cos, sin, pos[perm].contiguous(), dest[perm].contiguous(),
                      pages)
    assert torch.equal(full, permuted)
    chunked = torch.full_like(full, 7)
    for a, b in ((0, 100), (100, n)):
        extension().paged_quant_write(kind, x[a:b].contiguous(), ww, eps, cos, sin, pos[a:b].contiguous(),
                                      dest[a:b].contiguous(), chunked)
    assert torch.equal(full, chunked)
    again = _write(kind, x, ww, eps, cos, sin, pos, dest, pages, fill=0xAB)
    mask = torch.zeros(pages * 64, dtype=torch.bool, device="cuda")
    mask[dest] = True
    assert torch.equal(full.reshape(-1, 528)[mask], again.reshape(-1, 528)[mask])


def test_record_writers_in_a_cuda_graph():
    n = 64
    cfg, cos, sin, pos, pages, dest, w = _case(n, 61, 2)
    x = _rows(n, 62, 2.0)
    k = (torch.randn((n, 128), device="cuda") * 2).to(torch.bfloat16)
    kw = torch.rand(128, device="cuda") + 0.5
    eager_w = _write(0, x, w, cfg.rms_norm_eps, cos, sin, pos, dest, pages)
    eager_c = _write(1, x, None, 0.0, cos, sin, pos, dest, pages)
    eager_i = torch.full((pages, 64, 1, 132), 3, dtype=torch.uint8, device="cuda")
    extension().index_k_store(k, kw, 1e-6, cos, sin, pos, dest, eager_i)
    pw, pc = torch.full_like(eager_w, 7), torch.full_like(eager_c, 7)
    pi = torch.full_like(eager_i, 3)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            extension().paged_quant_write(0, x, w, cfg.rms_norm_eps, cos, sin, pos, dest, pw)
            extension().paged_quant_write(1, x, None, 0.0, cos, sin, pos, dest, pc)
            extension().index_k_store(k, kw, 1e-6, cos, sin, pos, dest, pi)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(3):
        pw.fill_(7), pc.fill_(7), pi.fill_(3)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(pw, eager_w) and torch.equal(pc, eager_c) and torch.equal(pi, eager_i)


def test_index_key_record_is_one_exponent_per_key():
    n = 300
    cfg, cos, sin, pos, pages, dest, _ = _case(n, 71, 2)
    k = (torch.randn((n, 128), device="cuda", generator=torch.Generator(device="cuda").manual_seed(2)) * 3)
    k[0] = 0
    k = k.to(torch.bfloat16)
    w = torch.rand(128, device="cuda") + 0.5
    pool = torch.full((pages, 64, 1, 132), 3, dtype=torch.uint8, device="cuda")
    extension().index_k_store(k, w, cfg.rms_norm_eps, cos, sin, pos, dest, pool)
    flat = pool.view(pages, 64 * 132)
    page, row = dest // 64, dest % 64
    codes = torch.stack([flat[p, r * 128:(r + 1) * 128] for p, r in zip(page.tolist(), row.tolist())]).cpu()
    sfb = torch.stack([flat[p, 64 * 128 + r * 4:64 * 128 + r * 4 + 4] for p, r in zip(page.tolist(), row.tolist())])
    sfb = sfb.cpu()
    assert bool((sfb == sfb[:, :1]).all()) and int(sfb[0, 0]) == 105
    ec, es = ref.index_k_emul(k, w, cfg.rms_norm_eps, cos, sin, pos)
    assert torch.equal(codes, ec) and torch.equal(sfb.view(torch.int32).view(-1), es)


@pytest.mark.parametrize("H", [64, 8])
def test_leaf_on_records_equals_leaf_on_qdq_rows(H):
    """Dual-rail sink leaf: window records (main list) + compressed records (extra list) == the same leaf on the
    trainer's BF16 quantize/dequantize rows (one arena, extra rows offset), bit for bit."""
    from isoexec.ops.attention.sparse_sm100.dualrail import dualrail_attn
    cfg = _cfg()
    g = torch.Generator(device="cuda").manual_seed(H)
    S, Se, R = 1024, 512, 9
    cw, sw = ops.rope_table(cfg, 0, S, "cuda")
    cc, sc = ops.rope_table(cfg, 2, 2 * Se, "cuda")
    xw = (torch.randn((S, 512), generator=g, device="cuda") * 2).to(torch.bfloat16)
    xc = (torch.randn((Se, 512), generator=g, device="cuda") * 3).to(torch.bfloat16)
    w = torch.rand(512, generator=g, device="cuda") + 0.5
    pw = torch.arange(S, device="cuda", dtype=torch.int32)
    pc = (torch.arange(Se, device="cuda", dtype=torch.int32) * 2).contiguous()
    win = _write(0, xw, w, cfg.rms_norm_eps, cw, sw, pw, torch.arange(S, device="cuda"), S // 64)
    comp = _write(1, xc, None, 0.0, cc, sc, pc, torch.arange(Se, device="cuda"), Se // 64)
    dw = ops.window_kv_qdq(xw, w, cfg.rms_norm_eps, (cw, sw), ops.Pos(pos=pw))
    dc = ops.compressed_kv_qdq(xc, (cc, sc), ops.Pos(pos=pc))
    q = (torch.randn((R, H, 512), generator=g, device="cuda") * 2).to(torch.bfloat16)
    lo = torch.rand((R, S), generator=g, device="cuda").topk(128, -1).indices.int().contiguous()
    le = torch.rand((R, Se), generator=g, device="cuda").topk(448, -1).indices.int()
    le = torch.cat([le, torch.full((R, 64), -1, dtype=torch.int32, device="cuda")], 1).contiguous()
    sink = torch.randn(H, generator=g, device="cuda")
    o, l = dualrail_attn(q, win, lo, 512 ** -0.5, sink=sink, extra_kv=comp, extra_lists=le)
    arena = torch.cat([dw, dc]).contiguous()
    lb = torch.cat([lo, torch.where(le >= 0, le + S, le)], 1).contiguous()
    ob, lbb = dualrail_attn(q, arena, lb, 512 ** -0.5, sink=sink)
    assert torch.equal(o.view(torch.int16), ob.view(torch.int16))
    assert torch.equal(l.view(torch.int32), lbb.view(torch.int32))


def _log2f_edge_rows(cos, sin, limit=8):
    """Latent rows whose FP32 rotation puts the block maximum just above 448 * 2^k, where libdevice log2f returns
    exactly k (CACHE-SPEC section 2): one pair (a, 0) at frequency j of position p, every other element 0, so the
    block maximum is fl(a * cos) exactly. Searched on the CPU over BF16 a in [2, 16) and every (p, j)."""
    from isoexec.ops.quant.checkpoint_sm100 import reference as qref
    a_all = torch.arange(0x4000, 0x4180, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()   # [2, 16)
    c, s = cos.cpu(), sin.cpu()
    wider = (c.abs() > s.abs()).unsqueeze(0)
    R = torch.tensor(1.0 / 448.0, dtype=torch.float32)
    found = []
    for a0 in range(0, a_all.numel(), 32):
        a = a_all[a0:a0 + 32]
        t = a.view(-1, 1, 1) * c.unsqueeze(0)                                   # [A, P, 32], one fp32 rounding
        bits = (t.abs() * R).view(torch.int32)
        k = ((bits >> 23) & 255) - 127
        mant = bits & 0x7fffff
        m = torch.zeros_like(k)
        for kk, mm in qref._LOG2F_EXACT_K.items():
            m = torch.where(k == kk, mm, m)
        hit = (mant >= 1) & (mant <= m) & wider
        for ai, p, j in hit.nonzero().tolist():
            found.append((float(a[ai]), p, j, int(k[ai, p, j])))
        if len(found) >= limit:
            break
    found = found[:limit]
    rows = torch.zeros((len(found), 512), dtype=torch.bfloat16)
    pos = torch.empty(len(found), dtype=torch.int32)
    blocks = torch.empty(len(found), dtype=torch.int64)
    ks = torch.empty(len(found), dtype=torch.int64)
    for i, (av, p, j, kk) in enumerate(found):
        rows[i, 448 + 2 * j] = av
        pos[i], blocks[i], ks[i] = p, (448 + 2 * j) // 32, kk
    return rows, pos, blocks, ks


def test_compressed_record_log2f_exponent_edge():
    cfg = _cfg()
    cos, sin = ops.rope_table(cfg, 2, 8192, "cuda")
    rows, pos, blocks, k = _log2f_edge_rows(cos, sin)
    assert rows.shape[0] > 0, "no edge case in the search range"
    n = rows.shape[0]
    dest = torch.arange(n, device="cuda")
    pool = _write(1, rows.cuda(), None, 0.0, cos, sin, pos.cuda(), dest, 1)
    rec = pool.reshape(-1, 528)[:n].cpu()
    got = rec[torch.arange(n), 512 + blocks]
    assert torch.equal(got.long(), (k + 127).long())                 # log2f's k, not the exact ceiling k + 1
    emul = ref.compressed_kv_emul(rows, cos, sin, pos)
    assert torch.equal(decode_records(pool, dest).cpu().view(torch.int16), emul.view(torch.int16))
