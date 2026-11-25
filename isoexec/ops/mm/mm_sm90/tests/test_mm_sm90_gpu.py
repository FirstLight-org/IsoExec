"""Bitwise battery of the sm90 segment-tree GEMM: the program is the same in every configuration, at every
M, at every row position, under column sharding and packing, split vs onchip, captured vs eager, and on
adversarial values; and it is the reference program (an independent Triton chain + fp32 tree + one round)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from isoexec.core.arch import live_device_arch  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or live_device_arch() != "sm90",
    reason="the sm90 mm kernel runs on a Hopper device only",
)


def _bits(t):
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


@pytest.fixture(scope="module")
def mm():
    from isoexec.ops.mm.mm_sm90 import ops

    ops.require_sm90()
    ops.config_ids()
    return ops


def operands(M, N, K, seed=7, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(M, K, generator=g, device="cuda") * scale).bfloat16().contiguous()
    # wide exponent spread so that any change of accumulation order shows
    w = (torch.randn(N, K, generator=g, device="cuda")
         * torch.exp(torch.randn(N, K, generator=g, device="cuda") * 2) * 0.02).bfloat16().contiguous()
    return x, w


def _single_chain_configs():
    from isoexec.ops.mm.mm_sm90 import segments as seg

    return seg.configs(out_fp32=0, exact_levels=0)


# Qwen3.8-27B census at TP4 (column) and TP1 (trainer), plus odd shapes
SHAPES = [(5120, 3584), (5120, 4120), (5120, 8704), (1536, 5120), (4352, 5120), (512, 136), (5120, 2056), (5120, 2060),
          (1024, 1028),
          # K not a whole number of 64-wide blocks: the zero-padded tail (MiMo fc2 at TP4 is K = 2752)
          (2752, 1024), (520, 264), (8, 64), (24, 72)]
MS = (1, 2, 7, 8, 31, 32, 63, 64, 65, 127, 128, 129, 255, 320, 511, 777, 1024, 2048)


@pytest.mark.parametrize("K,N", SHAPES)
def test_batch_invariance_every_config_every_M(mm, K, N):
    """Row r's bits: identical for any M, any position of the row in the batch, any configuration."""
    big_m = 2048
    x, w = operands(big_m, N, K, seed=K + N)
    ref = mm.mm_sm90(x, w.t())
    ids = mm.config_ids()
    ext = mm.extension()
    for M in MS:
        for off in sorted({0, 3, big_m - M}):
            if off + M > big_m:
                continue
            xs = x[off:off + M].contiguous()
            got = mm.mm_sm90(xs, w.t())
            assert torch.equal(_bits(got), _bits(ref[off:off + M])), (M, off)
    pitch = -(-N // 8) * 8  # an unaligned N is written through a 16-byte output pitch
    for name in _single_chain_configs():
        for M in (1, 33, 200, big_m):
            out = torch.empty(M, pitch, dtype=torch.bfloat16, device="cuda")
            ext.run(ids[name], x[:M], w, out, M, N, K, 1, pitch, K, K, M * pitch, 0, False, 0)
            assert torch.equal(_bits(out[:, :N]), _bits(ref[:M])), (name, M)


@pytest.mark.parametrize("K,N", [(5120, 3584), (1536, 5120)])
def test_equals_the_reference_program(mm, K, N):
    """S = 1: one independent Triton chain (64-row tl.dot, wgmma), rounded once."""
    from isoexec.ops.mm.mm_sm90 import reference as R

    x, w = operands(300, N, K)
    want = R.segtree(x, w, 1, leaf=R.chain_leaf_triton)
    assert torch.equal(_bits(mm.mm_sm90(x, w.t())), _bits(want))


@pytest.mark.parametrize("K,N", [(5120, 3584), (5120, 8704), (1536, 5120)])
def test_s1_equals_the_existing_sm90_providers(mm, K, N):
    """Evidence (not a twin claim): S = 1 is the canonical Hopper chain the vendored Triton kernel and
    the pinned cuBLASLt leaf also compute."""
    from isoexec.ops.mm import mm_batch_invariant as V

    stock = getattr(V, "_isoexec_stock_matmul_persistent", V.matmul_persistent)
    for M in (1, 32, 320, 2048):
        x, w = operands(M, N, K, seed=M)
        assert torch.equal(_bits(mm.mm_sm90(x, w.t())), _bits(stock(x, w.t()))), M


@pytest.mark.parametrize("K,N,S", [(1536, 5120, 2), (4352, 5120, 2), (6144, 1024, 8), (2048, 520, 4),
                                   (2752, 1024, 2), (64, 256, 8)])
@pytest.mark.parametrize("round_leaf", [False, True])
def test_execution_splits_split_vs_onchip_vs_reference(mm, K, N, S, round_leaf):
    from isoexec.ops.mm.mm_sm90 import reference as R
    from isoexec.ops.mm.mm_sm90 import segments as seg

    for M in (1, 40, 333):
        x, w = operands(M, N, K, seed=S * M)
        want_f32 = R.segtree(x, w, S, leaf=R.chain_leaf_triton, round_leaf=round_leaf, out_dtype=torch.float32)
        want = want_f32.to(torch.bfloat16)
        plans = seg.plans_at_split(M, K, N, S, out_fp32=0) + [
            p for p in seg.plans_at_split(M, K, N, S, out_fp32=1) if p.mode == seg.ONCHIP]
        assert plans
        for p in plans:
            got = mm.run_plan(x, w, p, round_leaf=round_leaf)
            ref = want_f32 if got.dtype is torch.float32 else want
            assert torch.equal(_bits(got), _bits(ref)), (M, p)


def test_column_shard_and_packing_invariance(mm):
    """The same logical weight's columns: TP1 whole vs every TP4 shard vs packed with others."""
    K = 5120
    x, _ = operands(257, 8, K, seed=11)
    _, gate = operands(1, 2048, K, seed=12)
    _, up = operands(1, 2048, K, seed=13)
    _, q = operands(1, 1536, K, seed=14)
    _, kv = operands(1, 512, K, seed=15)
    whole_gate = mm.mm_sm90(x, gate.t())
    whole_up = mm.mm_sm90(x, up.t())
    packed = mm.mm_sm90(x, torch.cat([gate, up]).t())
    assert torch.equal(_bits(packed[:, :2048]), _bits(whole_gate))
    assert torch.equal(_bits(packed[:, 2048:]), _bits(whole_up))
    qkv = mm.mm_sm90(x, torch.cat([q, kv, gate]).t())
    assert torch.equal(_bits(qkv[:, 2048:]), _bits(whole_gate))
    for tp in (2, 4, 8):
        n = 2048 // tp
        for r in range(tp):
            shard = gate[r * n:(r + 1) * n].contiguous()
            got = mm.mm_sm90(x, shard.t())
            assert torch.equal(_bits(got), _bits(whole_gate[:, r * n:(r + 1) * n])), (tp, r)
    # an unaligned shard width (TP8 of N = 16480 is 2060) through the padded pitch
    odd = torch.cat([gate, up])[:2060].contiguous()
    got = mm.mm_sm90(x, odd.t())
    assert torch.equal(_bits(got[:, :2048]), _bits(whole_gate))


def test_graph_capture_replay_and_determinism(mm):
    K, N = 4352, 5120
    x, w = operands(64, N, K)
    eager = mm.mm_sm90(x, w.t()).clone()
    lk = K // 2
    lv = torch.empty(2, 64, N, dtype=torch.bfloat16, device="cuda")
    nd = torch.empty(64, N, dtype=torch.float32, device="cuda")
    mm.leaves(x, w, lk, lv)
    mm.leaf_node(x, w, lk, nd)
    lv_e, nd_e = lv.clone(), nd.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            out = mm.mm_sm90(x, w.t())
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = mm.mm_sm90(x, w.t())
        mm.leaves(x, w, lk, lv)
        mm.leaf_node(x, w, lk, nd)
    for _ in range(3):
        lv.zero_()
        nd.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(out), _bits(eager))
        assert torch.equal(_bits(lv), _bits(lv_e)) and torch.equal(_bits(nd), _bits(nd_e))
    for _ in range(3):
        assert torch.equal(_bits(mm.mm_sm90(x, w.t())), _bits(eager))


def _adversarial(M, N, K, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, K, generator=g, device="cuda")
    w = torch.randn(N, K, generator=g, device="cuda")
    # signed zeros, bf16 subnormals, near-overflow magnitudes, exact cancellations
    x[0] = -0.0
    w[0] = 0.0
    x[1, ::2] = 1e-39
    x[2, :] = 3e38
    w[1, :] = 3e38
    x[3, : K // 2] = 1.0
    x[3, K // 2:] = -1.0
    w[2, :] = 1.0
    x[4, 5] = float("inf")
    x[5, 7] = float("nan")
    w[3, 9] = float("-inf")
    x[6, 9] = 0.0  # 0 * -inf = nan
    x[7, :] = -1e-40
    w[4, :] = 1e-38
    return x.bfloat16(), w.bfloat16()


@pytest.mark.parametrize("K,N", [(512, 64), (1536, 5120), (520, 64)])
def test_adversarial_values_match_the_reference_program(mm, K, N):
    from isoexec.ops.mm.mm_sm90 import reference as R

    x, w = _adversarial(64, N, K, seed=K)
    want = R.segtree(x, w, 1, leaf=R.chain_leaf_triton)
    for name in _single_chain_configs():
        out = torch.empty(64, N, dtype=torch.bfloat16, device="cuda")
        mm.extension().run(mm.config_ids()[name], x, w, out, 64, N, K, 1, N, K, K, 64 * N, 0, False, 0)
        assert R.same_bits_mod_nan(out, want), name
    # the tree under adversarial leaves (inf - inf, -0 + -0, overflow in the fp32 adds)
    for S in (2, 4, 8):
        if K % (64 * S):
            continue
        want_s = R.segtree(x, w, S, leaf=R.chain_leaf_triton, round_leaf=True, out_dtype=torch.float32)
        from isoexec.ops.mm.mm_sm90 import segments as seg

        for p in seg.plans_at_split(64, K, N, S, out_fp32=1):
            if p.mode != seg.ONCHIP:
                continue
            got = mm.run_plan(x, w, p, round_leaf=True)
            assert R.same_bits_mod_nan(got, want_s), (S, p.config)


def test_signed_zero_chain(mm):
    """An all-(-0) product chain: scale-d = 0 on the first step keeps -0, as the reference does."""
    from isoexec.ops.mm.mm_sm90 import reference as R

    x = torch.full((8, 256), -0.0, device="cuda").bfloat16()
    x[1] = 1.0
    w = torch.full((64, 256), 1.0, device="cuda").bfloat16()
    w[1] = -0.0
    got = mm.mm_sm90(x, w.t())
    want = R.segtree(x, w, 1, leaf=R.chain_leaf_triton)
    assert torch.equal(_bits(got), _bits(want))


def test_declines_rather_than_serves_what_it_cannot(mm):
    x, w = operands(8, 1152, 2048)
    assert mm.supported_shape(x, w.t())
    assert not mm.supported_shape(x.half(), w.half().t())
    assert not mm.supported_shape(x[:, :2001], w[:, :2001].t())  # K % 8: no 16-byte row pitch
    with pytest.raises(mm.UnsupportedShape):
        mm.mm(x, w.t(), bias=torch.zeros(1152, device="cuda", dtype=torch.bfloat16))
