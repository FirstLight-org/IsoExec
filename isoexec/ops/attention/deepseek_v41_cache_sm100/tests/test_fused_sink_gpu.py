"""The learned-sink root fused into the G = 1 sparse_sm100 leaf epilogue (an execution change) produces
exactly the bits of the published leaf -> fp32 partials -> sparse_sink_combine_kernel program."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _ext():
    from isoexec.ops.attention.sparse_sm100._extension import extension
    return extension("attn")


def _lists(T, K, S, gen, empty_every=0):
    ids = torch.rand((T, S), generator=gen, device="cuda").topk(min(K, S), -1).indices.sort(-1).values.int()
    out = torch.full((T, K), -1, dtype=torch.int32, device="cuda")
    n = torch.randint(0, min(K, S) + 1, (T,), generator=gen, device="cuda")
    keep = torch.arange(min(K, S), device="cuda")[None] < n[:, None]
    out[:, :min(K, S)] = torch.where(keep, ids, -1)
    if empty_every:
        out[::empty_every] = -1
    return out.contiguous()


@pytest.mark.parametrize("H", [16, 64, 128])
@pytest.mark.parametrize("K", [128, 640])
@pytest.mark.parametrize("T", [1, 37, 300])
def test_fused_sink_root_is_bit_identical(H, K, T):
    ext = _ext()
    gen = torch.Generator(device="cuda").manual_seed(H * 1000 + K + T)
    S = 2000
    q = (torch.randn((T, H, 512), generator=gen, device="cuda") * 2).to(torch.bfloat16)
    kv = torch.randn((S, 512), generator=gen, device="cuda").to(torch.bfloat16)
    sink = torch.randn(H, generator=gen, device="cuda") * 3
    lists = _lists(T, K, S, gen, empty_every=5)
    for recipe in (1, 2):
        out0 = torch.empty((T, H, 512), dtype=torch.bfloat16, device="cuda")
        lse0 = torch.empty((T, H), dtype=torch.float32, device="cuda")
        partials = torch.empty((T, 1, H, 520), dtype=torch.float32, device="cuda")
        dummy = torch.empty((1, 1), dtype=torch.int32, device="cuda")
        ext.sparse_sink_fwd(q, kv, dummy, None, lists, None, 512 ** -0.5, K, out0, lse0, partials, recipe, sink)
        out1 = torch.empty_like(out0)
        lse1 = torch.empty_like(lse0)
        ext.sparse_sink_fused_fwd(q, kv, lists, 512 ** -0.5, out1, lse1, recipe, sink)
        assert torch.equal(out0.view(torch.int16), out1.view(torch.int16)), recipe
        assert torch.equal(lse0.view(torch.int32), lse1.view(torch.int32)), recipe
