"""Fused decode operands (``engine.fuse_decode_operands``) are bit for bit the unfused decode path.

The fused path runs the input projections as one GEMM, fb/gb as one batched GEMM, the convolution over
the GEMM's qkv columns in place, and the gate/beta stages inside the decode row kernel. Every output,
record byte and convolution window must equal the unfused path's, across a chunk close, with NULL lanes,
for every TP head count, eagerly and replayed from a CUDA graph.
"""
import copy

import pytest
import torch

from isoexec.ops.kda.glm_sm100 import engine as E

HIDDEN = 4096


def _gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("needs an SM100 GPU")


def _shard(H, seed, world):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s, sc=0.02: (torch.randn(*s, generator=g, device="cuda") * sc).to(torch.bfloat16)
    HK = H * 128
    return E.KdaShard(rank=0, world=world, H=H, hidden=HIDDEN, w_qkv=r(3 * HK, HIDDEN),
                      conv_w=(torch.randn(3 * HK, 4, generator=g, device="cuda") * 0.5).contiguous(),
                      w_fa=r(128, HIDDEN), w_fb=r(HK, 128, sc=0.1), w_b=r(H, HIDDEN, sc=0.05),
                      w_ga=r(128, HIDDEN), w_gb=r(HK, 128, sc=0.1), w_o=r(HIDDEN, HK),
                      dt_bias=torch.randn(HK, generator=g, device="cuda") * 0.5,
                      A_log=torch.randn(H, generator=g, device="cuda") * 0.5,
                      norm_w=r(128, sc=1.0), eps=1e-6, lower_bound=-5.0)


def _fused(shard):
    s = copy.copy(shard)
    return E.fuse_decode_operands(s)


@pytest.mark.parametrize("H,world", [(64, 1), (32, 2), (16, 4)])
def test_fused_decode_bitwise_eager(H, world):
    _gpu()
    torch.manual_seed(0)
    base = _shard(H, 11 + H, world)
    fused = _fused(base)
    assert fused.w_in is not None and base.w_in is None
    N, slots_n = 17, 24
    pools = [E.KdaStatePool(slots_n, H, device="cuda") for _ in range(2)]
    slots = torch.randperm(slots_n - 1, device="cuda")[:N].to(torch.int32) + 1
    slots[5] = 0                           # a NULL lane (null block 0)
    batch = E.EngineBatch(None, slots, None, True, 0)
    with torch.no_grad():
        for step in range(70):             # crosses the chunk close at row 63
            x = (torch.randn(N, HIDDEN, device="cuda") * 2).to(torch.bfloat16)
            a = E.forward(base, x, batch, pools[0])
            b = E.forward(fused, x, batch, pools[1])
            assert a.dtype == b.dtype and torch.equal(a.view(torch.uint8), b.view(torch.uint8)), step
    assert torch.equal(pools[0].rec.data, pools[1].rec.data)
    assert torch.equal(pools[0].conv.view(torch.uint8), pools[1].conv.view(torch.uint8))


@pytest.mark.parametrize("reducer", ["chunk", "token"])
@pytest.mark.parametrize("H,world", [(4, 8), (12, 4), (1, 8)])
def test_fused_decode_with_local_heads_not_a_multiple_of_8_is_the_unfused_path(reducer, H, world):
    """32 heads at TP 8 (4 local heads), 96 at TP 8 split 4 ways (12), 8 at TP 8 (1): the fused operand's rows are
    padded to a multiple of 8 so the decode GEMM's output pitch is 16 bytes; every output, record byte and
    convolution window equals the unfused path's, across a chunk close, with a NULL lane."""
    _gpu()
    from isoexec.ops.kda import schedule
    eng = schedule.select(schedule.CHUNK if reducer == "chunk" else schedule.TOKEN)
    torch.manual_seed(0)
    base = _shard(H, 31 + H, world)
    fused = eng.fuse_decode_operands(copy.copy(base))
    N, slots_n = 5, 8
    pools = [eng.KdaStatePool(slots_n, H, device="cuda") for _ in range(2)]
    slots = torch.tensor([3, 1, 0, 7, 5], dtype=torch.int32, device="cuda")   # lane 2: the NULL block 0
    batch = eng.EngineBatch(None, slots, None, True, 0)
    with torch.no_grad():
        for step in range(66):             # crosses the chunk close at row 63
            x = (torch.randn(N, HIDDEN, device="cuda") * 2).to(torch.bfloat16)
            a = eng.forward(base, x, batch, pools[0])
            b = eng.forward(fused, x, batch, pools[1])
            assert a.dtype == b.dtype and torch.equal(a.view(torch.uint8), b.view(torch.uint8)), step
    state = (lambda p: p.rec.data) if reducer == "chunk" else (lambda p: p.state.view(torch.uint8))
    assert torch.equal(state(pools[0]), state(pools[1]))
    assert torch.equal(pools[0].conv.view(torch.uint8), pools[1].conv.view(torch.uint8))


def test_fused_decode_graph_replay_equals_eager():
    _gpu()
    H, N = 32, 16
    base = _shard(H, 5, 2)
    fused = _fused(base)
    pools = [E.KdaStatePool(32, H, device="cuda") for _ in range(2)]
    slots = (torch.arange(N, device="cuda", dtype=torch.int32) * 2 + 1)
    batch = E.EngineBatch(None, slots, None, True, 0)
    xs = torch.empty(N, HIDDEN, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        xs.normal_()
        E.forward(fused, xs, batch, pools[1])          # warm (also advances pool 1 once)
        E.forward(base, xs, batch, pools[0])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = E.forward(fused, xs, batch, pools[1])
        for step in range(66):
            xs.copy_((torch.randn(N, HIDDEN, device="cuda") * 2).to(torch.bfloat16))
            ref = E.forward(base, xs, batch, pools[0])
            graph.replay()
            assert torch.equal(ref.view(torch.uint8), out.view(torch.uint8)), step
    assert torch.equal(pools[0].rec.data, pools[1].rec.data)
