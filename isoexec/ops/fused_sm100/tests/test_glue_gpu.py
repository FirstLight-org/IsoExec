"""GPU battery for the fused glue kernels.

Evidence kinds: bitwise equality with the torch composition each kernel replaces, run-to-run
bitwise equality, batch invariance (row i alone == row i in a batch), CUDA-graph replay, and the edge cases
(empty batch, ties, NaN, signed zeros, widths below one block, repeated streams)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from isoexec.core.arch import live_device_arch
from isoexec.ops.fused_sm100 import ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="the sm100 fused glue kernels run on a Blackwell device only")
DEV = "cuda"


def _gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


# ------------------------------------------------------------------------------------ scatter_rows
@pytest.mark.parametrize("N,B,L", [(1, 1, 8), (4, 4, 64), (37, 6, 300), (256, 8, 1024)])
@pytest.mark.parametrize("offset", [0, 3])
def test_scatter_rows_matches_index_put(N, B, L, offset):
    g = _gen(N)
    pairs = torch.randperm(B * (L - offset), device=DEV, generator=g)[:N]
    rows, cols = (pairs // (L - offset)).long(), (pairs % (L - offset)).long()
    widths = [(512, torch.bfloat16), (128, torch.bfloat16), (528, torch.uint8), (64, torch.float32)]
    srcs = [torch.randn(N, w, device=DEV, generator=g).to(torch.float32).to(dt) if dt != torch.uint8
            else torch.randint(0, 255, (N, w), device=DEV, dtype=torch.uint8, generator=g) for w, dt in widths]
    ref = [torch.randint(0, 3, (B, L, w), device=DEV).to(dt) if dt != torch.uint8
           else torch.randint(0, 255, (B, L, w), device=DEV, dtype=torch.uint8) for w, dt in widths]
    got = [r.clone() for r in ref]
    for r, s in zip(ref, srcs):
        r[rows, cols + offset] = s
    ops.scatter_rows(srcs, got, rows, cols, offset)
    for a, b in zip(ref, got):
        assert a.view(torch.uint8).equal(b.view(torch.uint8))


def test_scatter_rows_empty_and_run_to_run():
    dst = torch.zeros(2, 8, 64, device=DEV, dtype=torch.bfloat16)
    e = torch.empty(0, dtype=torch.long, device=DEV)
    ops.scatter_rows([torch.empty(0, 64, device=DEV, dtype=torch.bfloat16)], [dst], e, e)
    assert not dst.any()
    src = torch.randn(2, 64, device=DEV).bfloat16()
    r, c = torch.tensor([0, 1], device=DEV), torch.tensor([5, 7], device=DEV)
    outs = []
    for _ in range(3):
        d = torch.zeros_like(dst)
        ops.scatter_rows([src], [d], r, c)
        outs.append(d)
    assert outs[0].equal(outs[1]) and outs[1].equal(outs[2])


def test_scatter_rows_graph_replay_device_columns():
    B, L = 4, 32
    dst = torch.zeros(B, L, 256, device=DEV, dtype=torch.bfloat16)
    src = torch.randn(B, 256, device=DEV).bfloat16()
    rows = torch.arange(B, device=DEV)
    col = torch.zeros(B, dtype=torch.long, device=DEV)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        ops.scatter_rows([src], [dst], rows, col)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ops.scatter_rows([src], [dst], rows, col)
    for c in (3, 17):
        col.fill_(c)
        dst.zero_()
        g.replay()
        ref = torch.zeros_like(dst)
        ref[rows, col] = src
        assert dst.equal(ref)


# ------------------------------------------------------------------------------------ embed_expand
@pytest.mark.parametrize("shape", [(1,), (5,), (3, 7), (2, 1, 4)])
@pytest.mark.parametrize("S,H,V", [(4, 4096, 300), (4, 5120, 129), (2, 64, 10)])
def test_embed_expand_matches_embedding(shape, S, H, V):
    g = _gen(1)
    w = torch.randn(V, H, device=DEV, generator=g).bfloat16()
    ids = torch.randint(0, V, shape, device=DEV, generator=g)
    ref = F.embedding(ids, w).unsqueeze(-2).expand(*shape, S, H).contiguous()
    out, pre = ops.embed_expand(w, ids, S, one_hot_pre=True)
    assert out.equal(ref)
    ref_pre = torch.zeros(*shape, S, dtype=torch.float32, device=DEV)
    ref_pre[..., 0] = 1
    assert pre.equal(ref_pre)
    assert ops.embed_expand(w, ids, S).equal(ref)


def test_embed_expand_batch_invariance_and_empty():
    w = torch.randn(50, 128, device=DEV).bfloat16()
    ids = torch.randint(0, 50, (9,), device=DEV)
    full = ops.embed_expand(w, ids, 4)
    for i in range(9):
        assert ops.embed_expand(w, ids[i:i + 1], 4).equal(full[i:i + 1])
    assert ops.embed_expand(w, ids[:0], 4).shape == (0, 4, 128)


# ------------------------------------------------------------------------------------ argmax_rows
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.parametrize("R,V", [(1, 1), (3, 7), (4, 1023), (5, 1024), (4, 154880), (2, 129280), (64, 8200), (300, 4099), (1, 262144)])
def test_argmax_matches_torch(dtype, R, V):
    x = torch.randn(R, V, device=DEV, generator=_gen(V)).to(dtype)
    assert ops.argmax_rows(x).equal(x.argmax(-1))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_argmax_ties_nan_signed_zero(dtype):
    V = 5000
    x = torch.zeros(8, V, device=DEV, dtype=dtype)
    x[0, 100] = x[0, 3000] = 2.0                      # tie: lowest index
    x[1, 4999] = 1.0                                  # last
    x[2, :] = -0.0
    x[2, 17] = 0.0                                    # -0 == +0: lowest index wins
    x[3, 7] = float("nan")
    x[3, 9] = 5.0                                     # NaN beats everything
    x[4, 3000] = x[4, 2000] = float("nan")            # lowest NaN
    x[5, :] = float("-inf")
    x[6, 4000] = float("inf")
    x[6, 10] = float("inf")
    x[7] = torch.arange(V, device=DEV).to(dtype)      # monotone (bf16 saturates: ties)
    assert ops.argmax_rows(x).equal(x.argmax(-1))


@pytest.mark.parametrize("V", [8200, 20001, 154880])
def test_argmax_ties_across_slice_boundaries(V):
    """Equal maxima planted on both sides of every candidate slice boundary: the lowest index always wins, whatever S."""
    for R in (1, 7, 200):
        x = torch.zeros(R, V, device=DEV, dtype=torch.bfloat16)
        for k in range(1, 40):
            x[:, min(V - 1, k * (V // 40))] = 1.0
            x[:, min(V - 1, k * (V // 40) + 1)] = 1.0
        assert ops.argmax_rows(x).equal(x.argmax(-1))
        x[R // 2, V - 1] = float("nan")
        assert ops.argmax_rows(x).equal(x.argmax(-1))


def test_argmax_batch_invariance_and_run_to_run():
    x = torch.randn(6, 3001, device=DEV).bfloat16()
    full = ops.argmax_rows(x)
    assert full.equal(ops.argmax_rows(x))
    for i in range(6):
        assert ops.argmax_rows(x[i:i + 1].contiguous()).equal(full[i:i + 1])


# ------------------------------------------------------------------------------------ mul_rows
@pytest.mark.parametrize("rows,H", [(1, 8), (4, 4096), (54, 4096), (1024, 5120), (7, 1536), (3, 512)])
def test_mul_rows_matches_torch(rows, H):
    g = _gen(rows)
    y = torch.randn(rows, H, device=DEV, generator=g).bfloat16()
    w = torch.randn(H, device=DEV, generator=g).bfloat16()
    assert ops.mul_rows(y, w).equal(w * y)
    y3 = y.view(rows, 1, H)
    assert ops.mul_rows(y3, w).equal(w * y3)


def test_mul_rows_special_values():
    H = 64
    y = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1e-38, 3e38, -1e-45] * 8, device=DEV).bfloat16().view(1, H)
    w = torch.tensor([1.0, -1.0, 0.0, 2.0, 1e30, 3e38, 1e-38, -0.0] * 8, device=DEV).bfloat16()
    r, o = (w * y), ops.mul_rows(y, w)
    assert r.view(torch.int16).equal(o.view(torch.int16)) or torch.equal(torch.nan_to_num(r), torch.nan_to_num(o))
    assert torch.equal(torch.isnan(r), torch.isnan(o))
