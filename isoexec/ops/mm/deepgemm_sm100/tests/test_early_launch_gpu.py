"""Early launch of the DeepGEMM BF16 GEMM (weight tiles before the dependency wait, early dependent trigger) changes
no bit: every mode equals the kernel without it, eagerly and in a CUDA graph whose kernels each read what the kernel
before them wrote (so an activation read before the wait would show as a stale value)."""
import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

from isoexec.ops.mm.deepgemm_sm100 import bf16, ops, segtree  # noqa: E402

MS = [1, 7, 16, 64, 128, 256, 512, 2048]
MODES = [1, 3, 9, 33, 65, 193]


def _bits(t):
    return t.contiguous().view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


def _w(n, k, g, sc=0.05):
    return (torch.randn(n, k, generator=g, device="cuda") * sc).to(torch.bfloat16)


def _calls(g):
    """(name, M -> callable) for every early-launch entry point: dense GEMM, grouped einsum, leaf tree, FP32 root."""
    wqb, wob = _w(4096, 1280, g), _w(5120, 1024, g)
    woa =(torch.randn(4, 1024, 4096, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    wob8 = _w(5120, 8192, g, 0.01)
    xs = {}

    def x(m, k):
        if (m, k) not in xs:
            xs[(m, k)] = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
        return xs[(m, k)]

    def grouped(m):
        og = x(m, 4 * 4096).view(m, 4, 4096)
        out = torch.empty((m, 4, 1024), dtype=torch.bfloat16, device="cuda")
        ops.bf16_grouped_bhr_hdr_bhd(og, woa, out)
        return out

    return [
        ("dense 4096x1280", lambda m: bf16._gemm(x(m, 1280), wqb)),
        ("fp32 root 5120x1024", lambda m: segtree.linear(x(m, 1024), wob, 1, out_dtype=torch.float32)),
        ("grouped einsum", grouped),
        ("leaf tree L8 5120x8192", lambda m: segtree.leaf_linear(x(m, 8192), wob8, 8)),
    ]


@pytest.mark.parametrize("mode", MODES)
def test_early_launch_bits_eager(mode, monkeypatch):
    seen = []
    dg = ops._dg()
    real = dg.set_early_launch
    monkeypatch.setattr(dg, "set_early_launch", lambda v: (seen.append(v), real(v))[1], raising=False)
    g = torch.Generator(device="cuda").manual_seed(5)
    calls = _calls(g)
    for m in MS:
        for name, fn in calls:
            monkeypatch.setattr(ops, "EARLY", 0)
            with torch.inference_mode():
                want = fn(m)
            monkeypatch.setattr(ops, "EARLY", mode)
            with torch.inference_mode():
                got = fn(m)
            assert torch.equal(_bits(got), _bits(want)), (name, m)
    assert mode in seen
    assert dg.get_early_launch() == 0


@pytest.mark.parametrize("mode", MODES)
def test_early_launch_bits_graph_chain(mode, monkeypatch):
    """A CUDA graph of dependent kernels: a copy writes the activation, then GEMM -> GEMM -> elementwise -> GEMM,
    each GEMM's A written by the kernel right before it; replays with fresh inputs equal the eager no-early chain."""
    g = torch.Generator(device="cuda").manual_seed(9)
    w1, w2, w3 = _w(1280, 1280, g), _w(4096, 1280, g), _w(5120, 1024, g)
    for m in (1, 16, 128, 512):
        src = torch.empty(m, 1280, dtype=torch.bfloat16, device="cuda")

        def chain():
            h = bf16._gemm(src, w1)
            y = bf16._gemm(h, w2)
            z = (y[:, :1024] * 2).contiguous()
            return segtree.linear(z, w3, 1, out_dtype=torch.float32)

        monkeypatch.setattr(ops, "EARLY", mode)
        with torch.inference_mode():
            src.copy_(torch.randn(m, 1280, generator=g, device="cuda").to(torch.bfloat16))
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                chain()
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            inp = torch.empty_like(src)
            with torch.cuda.graph(graph):
                src.copy_(inp)
                out = chain()
            for rep in range(12):
                inp.copy_(torch.randn(m, 1280, generator=g, device="cuda").to(torch.bfloat16))
                graph.replay()
                torch.cuda.synchronize()
                monkeypatch.setattr(ops, "EARLY", 0)
                src.copy_(inp)
                want = chain()
                monkeypatch.setattr(ops, "EARLY", mode)
                assert torch.equal(_bits(out), _bits(want)), (m, rep)
