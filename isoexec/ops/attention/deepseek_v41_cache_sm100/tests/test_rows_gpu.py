"""Row kernels: bit-exact against the CPU emulation and against the sub-recipe kernels they fuse.
"""
import pytest
import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops, reference as ref, spec

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _cfg():
    from ._config import published_config
    return published_config()


def _rand(shape, seed, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, device="cuda", generator=g) * scale).to(torch.bfloat16)


def _bits_equal(a, b):
    a, b = a.cpu(), b.cpu()
    assert a.shape == b.shape and a.dtype == b.dtype
    if a.dtype in (torch.bfloat16, torch.float16):
        a, b = a.view(torch.int16), b.view(torch.int16)
    elif a.dtype == torch.float32:
        a, b = a.view(torch.int32), b.view(torch.int32)
    diff = (a != b).sum().item()
    assert diff == 0, f"{diff} mismatching elements"


@pytest.mark.parametrize("ratio", [0, 2])
@pytest.mark.parametrize("rows", [1, 7, 300])
def test_window_kv_bitexact(ratio, rows):
    cfg = _cfg()
    cos, sin = ops.rope_table(cfg, ratio, 4096, "cuda")
    x = _rand((rows, 512), 1 + rows, 3.0)
    w = (torch.rand(512, device="cuda") + 0.5).to(torch.bfloat16)
    pos = torch.randint(0, 4096, (rows,), device="cuda", dtype=torch.int32)
    out = torch.empty_like(x)
    ops.row_op(spec.ROW_WINDOW_KV, x, 512, w=w.float(), eps=cfg.rms_norm_eps, table=(cos, sin),
               pos=ops.Pos(pos=pos), out=out)
    _bits_equal(out, ref.window_kv_emul(x, w, cfg.rms_norm_eps, cos, sin, pos))


@pytest.mark.parametrize("rows", [1, 33, 513])
def test_q_norm_and_ropes_bitexact(rows):
    cfg = _cfg()
    cos, sin = ops.rope_table(cfg, 2, 2048, "cuda")
    x = _rand((rows, 1280), 5, 2.0)
    w = torch.rand(1280, device="cuda", generator=torch.Generator(device="cuda").manual_seed(rows)) + 0.5
    out = torch.empty_like(x)
    ops.row_op(spec.ROW_Q_NORM, x, 1280, w=w, eps=cfg.rms_norm_eps, out=out)
    _bits_equal(out, ref.rms_emul(x, w, cfg.rms_norm_eps))
    q = _rand((rows * 4, 512), 6, 4.0)
    pos = torch.arange(rows, device="cuda", dtype=torch.int32).repeat_interleave(4) * 3
    for kind, inverse in ((spec.ROW_Q_ROPE, False), (spec.ROW_OUT_INV_ROPE, True)):
        out = torch.empty_like(q)
        ops.row_op(kind, q, 512, table=(cos, sin), pos=ops.Pos(div=4, mul=3), out=out)
        _bits_equal(out, ref.rope_emul(q, cos, sin, pos, inverse=inverse))


def test_group_major_mapping():
    cfg = _cfg()
    cos, sin = ops.rope_table(cfg, 0, 64, "cuda")
    tokens, H, G = 5, 16, 8
    q = _rand((tokens * H, 512), 9)
    out = torch.empty_like(q)
    ops.row_op(spec.ROW_OUT_INV_ROPE, q, 512, table=(cos, sin), pos=ops.Pos(div=H), out=out,
               group_major=(tokens, H, H // G))
    pos = torch.arange(tokens).repeat_interleave(H)
    expect = ref.rope_emul(q, cos, sin, pos, inverse=True).view(tokens, G, H // G, 512).transpose(0, 1)
    _bits_equal(out.view(G, tokens, H // G, 512), expect.contiguous())


@pytest.mark.parametrize("rows", [1, 77])
def test_index_codes_bitexact(rows):
    cfg = _cfg()
    cos, sin = ops.rope_table(cfg, 1, 4096, "cuda")
    k = _rand((rows, 128), 21, 2.0)
    w = torch.rand(128, device="cuda") + 0.5
    codes = torch.empty((rows, 128), dtype=torch.uint8, device="cuda")
    sf = torch.empty((rows,), dtype=torch.int32, device="cuda")
    ops.row_op(spec.ROW_INDEX_K, k, 128, w=w, eps=cfg.rms_norm_eps, table=(cos, sin), pos=ops.Pos(mul=1),
               codes=codes, sf=sf)
    ec, es = ref.index_k_emul(k, w, cfg.rms_norm_eps, cos, sin, torch.arange(rows))
    assert torch.equal(codes.cpu(), ec) and torch.equal(sf.cpu(), es)
    # index q with heads padded 32 -> 64
    tokens, H = rows, 32
    q = _rand((tokens * H, 128), 22, 3.0)
    qc = torch.full((tokens, 64, 128), 0x55, dtype=torch.uint8, device="cuda")       # garbage: padding is written
    qs = torch.full((tokens, 64), 7, dtype=torch.int32, device="cuda")
    ops.row_op(spec.ROW_INDEX_Q, q, 128, table=(cos, sin), pos=ops.Pos(div=H), codes=qc.view(-1, 128),
               sf=qs.view(-1), dst=(H, 64, 0), pad_heads=H)
    ec, es = ref.index_q_emul(q, cos, sin, torch.arange(tokens).repeat_interleave(H))
    assert torch.equal(qc[:, :H].reshape(-1, 128).cpu(), ec) and torch.equal(qs[:, :H].reshape(-1).cpu(), es)
    assert int(qc[:, H:].count_nonzero()) == 0 and bool((qs[:, H:] == spec.UE8M0_ONE_PACKED).all())


def _pool_statement(v, s, ratio):
    """The compressor's channel pool (numerics.cuh ``pool_channel``) stated in torch ops on the device, in the
    kernel's order: v, s fp32 [G, ratio, 512] -> fp32 [G, 512]. max chain over j, then the sum of
    exp(s_j - max) in j order, then result += (exp(s_j - max) / sum) * v_j in j order; every product, sum and
    quotient one binary32 rounding (the extension builds with --fmad=false --ftz=false; torch's CUDA exp and the
    kernel's expf are the same device expf)."""
    maximum = torch.full_like(s[:, 0], float("-inf"))
    for j in range(ratio):
        maximum = torch.fmax(maximum, s[:, j])
    denominator = torch.zeros_like(maximum)
    for j in range(ratio):
        denominator = denominator + torch.exp(s[:, j] - maximum)
    result = torch.zeros_like(maximum)
    for j in range(ratio):
        p = torch.exp(s[:, j] - maximum) / denominator
        result = result + p * v[:, j]
    return result


def _pool_case(ratio, tokens):
    B = 3
    g = torch.Generator(device="cuda").manual_seed(31)
    proj = torch.randn((B * tokens, 1024), device="cuda", generator=g) * 2
    w = torch.rand(512, device="cuda", generator=g) + 0.5
    C = tokens // ratio
    return B, C, proj, w


@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("tokens", [1, 2, 9, 64])
def test_pool_norm_matches_pool_statement(ratio, tokens):
    """pool_norm (prefill groups) == the torch statement of the channel pool, BF16-rounded, then the exact
    RMSNorm emulation (reference.rms_emul), byte for byte."""
    cfg = _cfg()
    B, C, proj, w = _pool_case(ratio, tokens)
    out = torch.empty((B * C, 512), dtype=torch.bfloat16, device="cuda")
    ops.extension().pool_norm(proj, ratio, w, cfg.rms_norm_eps, out, C, tokens, None, None, None)
    if C == 0:
        return
    groups = proj.reshape(B, tokens, 1024)[:, :C * ratio].reshape(B * C, ratio, 1024)
    pooled = _pool_statement(groups[..., :512], groups[..., 512:], ratio).to(torch.bfloat16)
    _bits_equal(out, ref.rms_emul(pooled, w, cfg.rms_norm_eps))


@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("tokens", [1, 2, 9, 64])
@pytest.mark.parametrize("wide", [False, True])
def test_pool_norm_equals_pool_varlen(ratio, tokens, wide):
    """pool_norm (prefill groups) == pool_varlen (the paged compressor's kernel, both launch forms) on the same
    complete groups: the same declared pool + BF16 + RMSNorm program in two independent kernels."""
    cfg = _cfg()
    B, C, proj, w = _pool_case(ratio, tokens)
    out = torch.empty((B * C, 512), dtype=torch.bfloat16, device="cuda")
    ops.extension().pool_norm(proj, ratio, w, cfg.rms_norm_eps, out, C, tokens, None, None, None)
    if C == 0:
        return
    t = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")
    cu_q, p0 = t([b * tokens for b in range(B + 1)]), t([0] * B)
    state = torch.zeros((B, ratio, 1024), device="cuda")
    gs, gk = t([b for b in range(B) for _ in range(C)]), t([k for _ in range(B) for k in range(C)])
    lat = torch.empty((B * C, 512), dtype=torch.bfloat16, device="cuda")
    valid = torch.empty(B * C, dtype=torch.int32, device="cuda")
    cidx = torch.empty(B * C, dtype=torch.int32, device="cuda")
    ops.extension().pool_varlen(proj, cu_q, p0, state, t(list(range(B))), gs, gk, ratio, w, cfg.rms_norm_eps, lat,
                                valid, cidx, wide)
    assert bool((valid == 1).all())
    _bits_equal(out, lat)


def test_decode_pool_equals_prefill_pool():
    cfg = _cfg()
    ratio, B, T = 4, 2, 12
    g = torch.Generator(device="cuda").manual_seed(41)
    proj = torch.randn((B, T, 1024), device="cuda", generator=g)
    w = torch.rand(512, device="cuda") + 0.5
    full = torch.empty((B * (T // ratio), 512), dtype=torch.bfloat16, device="cuda")
    ops.extension().pool_norm(proj.view(B * T, 1024), ratio, w, cfg.rms_norm_eps, full, T // ratio, T, None, None, None)
    state = torch.zeros((B, ratio, 1024), device="cuda")
    got = []
    for t in range(T):
        out = torch.zeros((B, 512), dtype=torch.bfloat16, device="cuda")
        flag = torch.empty(B, dtype=torch.int32, device="cuda")
        pos = torch.full((B,), t, dtype=torch.int32, device="cuda")
        ops.extension().pool_norm(proj[:, t].contiguous(), ratio, w, cfg.rms_norm_eps, out, 0, 0, state, pos, flag)
        assert flag.tolist() == [int((t + 1) % ratio == 0)] * B
        if (t + 1) % ratio == 0:
            got.append(out)
    _bits_equal(torch.stack(got, 1).reshape(-1, 512), full.view(B, T // ratio, 512).reshape(-1, 512))


def test_row_backward_accuracy():
    cfg = _cfg()
    cos, sin = ops.rope_table(cfg, 2, 256, "cuda")
    rows = 64
    x = _rand((rows, 512), 51, 2.0)
    w = torch.rand(512, device="cuda") + 0.5
    dy = _rand((rows, 512), 52)
    pos = torch.arange(rows, device="cuda")
    dw_rows = torch.empty((rows, 512), device="cuda")
    dx = ops.row_bwd(spec.ROW_WINDOW_KV, x, dy, 512, w=w, eps=cfg.rms_norm_eps, table=(cos, sin),
                     pos=ops.Pos(), dw_rows=dw_rows)
    dw = ops.segment_sum(dw_rows)
    xd = x.double().requires_grad_(True)
    wd = w.double().requires_grad_(True)
    n = xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + cfg.rms_norm_eps) * wd
    c, s = cos.double()[pos], sin.double()[pos]
    a, b = n[:, 448::2], n[:, 449::2]
    y = torch.cat((n[:, :448], torch.stack((a * c - b * s, a * s + b * c), -1).flatten(1)), 1)
    gx, gw = torch.autograd.grad(y, (xd, wd), dy.double())
    for got, want in ((dx.double(), gx), (dw.double(), gw)):
        err = ((got - want).norm() / want.norm()).item()
        assert err < 4e-3, err


@pytest.mark.parametrize("rows", [1, 64 * 3 + 5])
def test_inplace_tail_rope_equals_row_kernel(rows):
    cfg = _cfg()
    cos, sin = ops.rope_table(cfg, 0, 512, "cuda")
    q = _rand((rows, 512), 61, 2.0)
    want = torch.empty_like(q)
    ops.row_op(spec.ROW_Q_ROPE, q, 512, table=(cos, sin), pos=ops.Pos(div=4, mod=97), out=want)
    got = ops.rope_tail_(q.clone(), 512, (cos, sin), ops.Pos(div=4, mod=97))
    _bits_equal(got, want)
    back = ops.rope_tail_(got.clone(), 512, (cos, sin), ops.Pos(div=4, mod=97), inverse=True)
    want2 = torch.empty_like(q)
    ops.row_op(spec.ROW_OUT_INV_ROPE, got, 512, table=(cos, sin), pos=ops.Pos(div=4, mod=97), out=want2)
    _bits_equal(back, want2)


def test_permute_rows_round_trip():
    G, M, R = 8, 37, 1024
    x = _rand((G, M, R), 71)
    y = torch.empty((M, G * R), dtype=torch.bfloat16, device="cuda")
    ops.extension().permute_rows(x, y, G, M, True)
    _bits_equal(y, x.transpose(0, 1).reshape(M, G * R))
    z = torch.empty_like(x)
    ops.extension().permute_rows(y, z, G, M, False)
    _bits_equal(z, x)


def test_decode_prep_indices():
    pos = torch.tensor([0, 1, 5, 127, 128, 999], dtype=torch.int32, device="cuda")
    B, A, off, W, comp_off, r, pages = 6, 3000, 256, 128, 1536, 2, 8
    i64 = torch.empty((5, B), dtype=torch.int64, device="cuda")
    i32 = torch.empty((2, B), dtype=torch.int32, device="cuda")
    ops.extension().decode_prep(pos, A, off, W, comp_off, r, pages, i64, i32)
    p = pos.long().cpu()
    b = torch.arange(B)
    ring = b * A + off + p % W
    c = p // r
    page = b * pages + c // 64
    assert torch.equal(i64.cpu(), torch.stack((ring, ring + W, b * A + comp_off + c, page * 66 + c % 64,
                                               page * 2112 + 2048 + c % 64)))
    assert torch.equal(i32.cpu(), torch.stack(((p - (r - 1)).clamp_min(0), (p + 1) // r)).int())


def test_rms_emulation_sqrt_regression_and_many_weights():
    """Regression: rows=513, w seed 13, row 97 col 138 sits 1 ulp above a BF16
    tie; torch's CPU float32 sqrt is 1 ulp low there. The emulation now rounds sqrt/div correctly, and
    the kernel (sqrt.rn) must match it for many weight draws (the previously unseeded case)."""
    cfg = _cfg()
    x = _rand((513, 1280), 5, 2.0)
    for seed in [13] + list(range(100, 140)):
        w = torch.rand(1280, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed)) + 0.5
        out = torch.empty_like(x)
        ops.row_op(spec.ROW_Q_NORM, x, 1280, w=w, eps=cfg.rms_norm_eps, out=out)
        _bits_equal(out, ref.rms_emul(x, w, cfg.rms_norm_eps))


@pytest.mark.parametrize("kind,D", [(spec.ROW_Q_NORM, 1280), (spec.ROW_WINDOW_KV, 512), (spec.ROW_Q_ROPE, 512),
                                    (spec.ROW_COMPRESSED_KV, 512), (spec.ROW_INDEX_K, 128)])
def test_row_kernels_repeat_bitwise_on_dirty_memory(kind, D):
    """Same inputs, 50 launches into NaN/garbage-prefilled outputs after dirtying the allocator:
    every launch is bit-identical (no uninitialized read reaches an output)."""
    cfg = _cfg()
    cos, sin = ops.rope_table(cfg, 2, 4096, "cuda")
    rows = 513
    x = _rand((rows, D), 77, 2.0)
    w = torch.rand(D, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7)) + 0.5
    first = None
    for k in range(50):
        junk = torch.full((1 << 22,), float("nan"), device="cuda")
        del junk
        kw = dict(table=(cos, sin), pos=ops.Pos(mul=2))
        if kind in (spec.ROW_Q_NORM, spec.ROW_WINDOW_KV, spec.ROW_INDEX_K):
            kw.update(w=w, eps=cfg.rms_norm_eps)
        if kind == spec.ROW_INDEX_K:
            codes = torch.full((rows, 128), 0xFF, dtype=torch.uint8, device="cuda")
            sf = torch.full((rows,), -1, dtype=torch.int32, device="cuda")
            ops.row_op(kind, x, D, codes=codes, sf=sf, **kw)
            got = torch.cat((codes.view(torch.int32).view(-1), sf))
        else:
            out = torch.full((rows, D), float("nan"), dtype=torch.bfloat16, device="cuda")
            ops.row_op(kind, x, D, out=out, **kw)
            got = out.view(torch.int16)
        if first is None:
            first = got.clone()
        assert torch.equal(got, first), k
