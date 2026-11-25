"""mm_sm100/general: the layouts and shapes the segtree leaf declines (dgrad, wgrad, K < 256, N < 64, bmm)."""

from __future__ import annotations

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or live_device_arch() != "sm100", reason="needs an sm100 device"
)


def _bits(t):
    return t.contiguous().view(torch.int16 if t.dtype is torch.bfloat16 else torch.int32)


def _eq(x, y):
    return torch.equal(_bits(x), _bits(y))


def _r(*s, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*s, device="cuda", dtype=torch.bfloat16, generator=g)


def _variants(G, a, b, **kw):
    f = G.mm if a.dim() == 2 else G.bmm
    base = f(a, b, **kw)
    a_mn, b_mn = G._layouts(a, b)
    outs = [f(a, b, grid=5, **kw), f(a, b, **kw)]
    for cl, widths in ((1, (64, 128, 256)), (2, (128, 256))):
        for w in widths:
            if b_mn and (w // cl) % 64:
                continue
            outs.append(f(a, b, bn=w, cl=cl, **kw))
    return base, outs


CASES = {
    "fwd K=128": lambda: (_r(1024, 128), _r(8192, 128, seed=1).t()),
    "fwd N=32": lambda: (_r(54, 4096), _r(32, 4096, seed=1).t()),
    "dgrad": lambda: (_r(1000, 8192), _r(8192, 640, seed=1)),
    "dgrad split": lambda: (_r(256, 32768), _r(32768, 512, seed=1)),
    "wgrad": lambda: (_r(1024, 1536).t(), _r(1024, 1024, seed=1)),
    "wgrad split": lambda: (_r(16384, 384).t(), _r(16384, 256, seed=1)),
    "bmm": lambda: (_r(64, 64, 512), _r(64, 640, 512, seed=1).transpose(1, 2)),
    "bmm A^T": lambda: (_r(64, 128, 640).transpose(1, 2), _r(64, 128, 512, seed=1)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_accuracy_and_ownership_invariance(name):
    from isoexec.ops.mm.mm_sm100.general import ops as G

    a, b = CASES[name]()
    assert G.supported(a, b)
    base, outs = _variants(G, a, b)
    for o in outs:
        assert _eq(base, o), name
    ref = a.double() @ b.double()
    cub = (a @ b).double()
    scale = ref.abs().max().item()
    err = (base.double() - ref).abs().max().item() / scale
    cerr = (cub - ref).abs().max().item() / scale
    assert err <= max(2 * cerr, 1e-6), (name, err, cerr)


@pytest.mark.parametrize("name", list(CASES))
def test_the_k_reduction_is_one_whole_k_chain_in_every_layout(name):
    from isoexec.ops.mm.mm_sm100.general import ops as G

    a, b = CASES[name]()
    a_mn, b_mn = G._layouts(a, b)
    K = a.shape[-1]
    S = G.split_for(K, a_mn, b_mn)
    y = G.mm(a, b) if a.dim() == 2 else G.bmm(a, b)
    assert S == 1, "the schedule policy gives every tensor-core K reduction one segment"
    # The declared program: ONE ascending chain over the whole K, whatever the operand layout. The K-major
    # copy runs the chain kernel's own layout; an MN-major operand must give the same FP32 root.
    assert _eq(y, _chain_f32(G, a, b).to(torch.bfloat16)), "one segment: the whole-K chain"
    if a.dim() == 2 and a.shape[0] >= 64 and K < 16384:
        # cuBLAS runs an unsplit chain too at these K. At K >= 16384 (the two "split" cases, which the
        # removed K-split rule cut into 2 segments) cuBLAS's own program is not one chain and is not compared.
        assert _eq(y, a @ b), "an unsplit chain is cuBLAS's unsplit chain"


def _chain_f32(G, a, b):
    """FP32 root of one chain over this K slice (bmm path returns it unrounded)."""
    a3 = a if a.dim() == 3 else a.unsqueeze(0)
    b3 = b if b.dim() == 3 else b.unsqueeze(0)
    # a K-slice of an MN-major operand stays MN-major; force a single chain by a K-major copy
    out = G.bmm(a3.contiguous(), b3.transpose(-1, -2).contiguous().transpose(-1, -2), out_dtype=torch.float32)
    return out if a.dim() == 3 else out[0]


def test_k_major_calls_are_row_invariant():
    from isoexec.ops.mm.mm_sm100.general import ops as G

    w = _r(8192, 128, seed=3).t()
    x = _r(512, 128, seed=4)
    full = G.mm(x, w)
    for rows in (1, 7, 64, 129):
        assert _eq(G.mm(x[:rows].contiguous(), w), full[:rows])


def test_bmm_fp32_root_is_unrounded_and_cublas_equal():
    from isoexec.ops.mm.mm_sm100.general import ops as G

    a, b = _r(32, 64, 512), _r(32, 640, 512, seed=1).transpose(1, 2)
    y = G.bmm(a, b, out_dtype=torch.float32)
    assert y.dtype is torch.float32
    assert _eq(y, torch.bmm(a, b, out_dtype=torch.float32))
    assert _eq(y.to(torch.bfloat16), G.bmm(a, b))


def test_declines_what_it_cannot_express():
    from isoexec.ops.mm.mm_sm100.general import ops as G

    a = _r(64, 96)                     # K % 64 != 0
    assert not G.supported(a, _r(96, 64))
    a = _r(64, 300)[:, :256]           # 600-byte row pitch: not 16-byte aligned
    assert not G.supported(a, _r(256, 64))
    assert not G.supported(_r(64, 128).float(), _r(128, 64).float())
