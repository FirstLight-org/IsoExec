"""Single-GPU bitwise gate of the BF16 actor's canonical MegaMoE forward (EP = 1) against the grouped-kernel
recipe (the trainer's and rollout's current bits), at a reduced geometry (E = 256, H = 512, I = 256, top-6) over
M = 1..4096, skewed routing and CUDA graph replays. The real geometry and EP = W run in ``_actor_mega_ep_child.py``
(torchrun)."""
from __future__ import annotations

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

from isoexec.ops.moe.deepseek_v41_sm100 import ops  # noqa: E402

DEV = "cuda"
E, K, H, I, L = 256, 6, 512, 256, 10.0
KW = dict(top_k=K, routed_scale=1.5, limit=L)
MS = [1, 2, 3, 5, 8, 16, 17, 31, 64, 127, 128, 255, 256, 511, 512, 1000, 1024, 2047, 2048, 4096]


def _bits(t):
    return t.contiguous().view(torch.int16)


@pytest.fixture(scope="module")
def weights():
    g = torch.Generator(device=DEV).manual_seed(5)
    gu = (torch.randn(E + 1, 2 * I, H, generator=g, device=DEV) * 0.08).to(torch.bfloat16)
    dn = (torch.randn(E + 1, H, I, generator=g, device=DEV) * 0.08).to(torch.bfloat16)
    rw = (torch.randn(E, H, generator=g, device=DEV) * 0.05).to(torch.bfloat16)
    rb = torch.randn(E, generator=g, device=DEV) * 0.01
    return gu, dn, rw, rb


def _x(M, seed):
    return (torch.randn(M, H, generator=torch.Generator(device=DEV).manual_seed(seed), device=DEV) * 2).to(torch.bfloat16)


def test_mega_forward_is_grouped(weights):
    gu, dn, rw, rb = weights
    for M in range(1, 4097):
        x = _x(M, M)
        assert torch.equal(_bits(ops.moe_forward_mega(x, rw, rb, gu, dn, **KW)),
                           _bits(ops.moe_forward(x, rw, rb, gu, dn, **KW))), M
    b = rb.clone()
    b[:K] += 50.0                      # every route on experts 0..5
    for M in (300, 2048):
        x = _x(M, 7 * M)
        assert torch.equal(_bits(ops.moe_forward_mega(x, rw, b, gu, dn, **KW)),
                           _bits(ops.moe_forward(x, rw, b, gu, dn, **KW))), M


@pytest.mark.parametrize("chain", [[2048, 64, 1024, 8], [1, 3, 4096, 17]])
def test_mega_forward_graph_replays(weights, chain):
    gu, dn, rw, rb = weights
    xs = [_x(M, M) for M in chain]
    fn = lambda xb: ops.moe_forward_mega(xb, rw, rb, gu, dn, **KW)  # noqa: E731
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for xb in xs:
            fn(xb)
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        ys = [fn(xb) for xb in xs]
    for rep in range(3):
        for j, (M, xb) in enumerate(zip(chain, xs)):
            xb.copy_(_x(M, 1000 * rep + 17 * j + M))
        gr.replay()
        torch.cuda.synchronize()
        for M, xb, y in zip(chain, xs, ys):
            assert torch.equal(_bits(y), _bits(ops.moe_forward(xb, rw, rb, gu, dn, **KW))), (rep, M)


def test_grad_enabled_call_is_refused(weights):
    gu, dn, rw, rb = weights
    x = _x(64, 11)
    with pytest.raises(RuntimeError, match="no-grad forward"):
        ops.moe_forward_mega(x.clone().requires_grad_(), rw, rb, gu, dn, **KW)
    with torch.no_grad():
        assert torch.equal(_bits(ops.moe_forward_mega(x, rw, rb, gu, dn, **KW)),
                           _bits(ops.moe_forward(x, rw, rb, gu, dn, **KW)))
