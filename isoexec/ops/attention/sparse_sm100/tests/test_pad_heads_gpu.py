"""Ranks with fewer than 64 heads (TP 2 / 4 / 8 of a 64-head layer): the leaf pads the CTA's 64 rows inside
it (zero query rows, predicated stores; no pad copies), and every head equals the 64-head run's bit for bit
-- bf16 kv (2048 / 2112 slots), the fp8 V4.1 cache (2048 / 2112 slots), and the fused learned-sink root."""
import pytest
import torch

from isoexec.core.arch import live_device_arch  # noqa: E402

from isoexec.ops.attention.sparse_sm100.tests import _inputs as inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="the DSA kernels run on a Blackwell device only")


def _ext():
    from isoexec.ops.attention.sparse_sm100._extension import extension
    return extension("attn")


def _lists(T, K, S, gen):
    n = min(K - 61, S)
    ids = torch.rand((T, S), generator=gen, device="cuda").topk(n, -1).indices.sort(-1).values.int()
    out = torch.full((T, K), -1, dtype=torch.int32, device="cuda")
    out[:, :n] = ids
    out[::7, n // 2:] = -1
    return out.contiguous()


def _run(ext, mode, q, kv, cache, bt, lists, sink, recipe):
    T, H = q.shape[:2]
    out = torch.empty((T, H, 512), dtype=torch.bfloat16, device="cuda")
    lse = torch.empty((T, H), dtype=torch.float32, device="cuda")
    K = lists.shape[1]
    if mode == "bf16":
        dummy = torch.empty((1, 1), dtype=torch.int32, device="cuda")
        ext.sparse_fwd(q, kv, dummy, None, lists, None, 512 ** -0.5, K, out, lse, None, recipe)
    elif mode == "fp8":
        bt_row = torch.zeros(T, dtype=torch.int32, device="cuda")
        ext.sparse_fwd(q, cache, bt, bt_row, lists, None, 512 ** -0.5, K, out, lse, None, recipe)
    else:
        ext.sparse_sink_fused_fwd(q, kv, lists, 512 ** -0.5, out, lse, recipe, sink[:H].contiguous())
    return out, lse


@pytest.mark.parametrize("mode,K", [("bf16", 640), ("bf16", 2112), ("fp8", 2048), ("fp8", 2112), ("sink", 640)])
@pytest.mark.parametrize("recipe", [1, 2])
def test_heads_below_64_equal_the_64_head_run(mode, K, recipe):
    ext = _ext()
    gen = torch.Generator(device="cuda").manual_seed(K + recipe)
    T, S = 37, 4096
    q = (torch.randn((T, 64, 512), generator=gen, device="cuda") * 2).to(torch.bfloat16)
    kv = torch.randn((S, 512), generator=gen, device="cuda").to(torch.bfloat16)
    cache = inputs.quantize_v41(kv.view(-1, 64, 512))
    bt = torch.arange(S // 64, dtype=torch.int32, device="cuda").view(1, -1)
    sink = torch.randn(64, generator=gen, device="cuda") * 3
    lists = _lists(T, K, S, gen)
    o64, l64 = _run(ext, mode, q, kv, cache, bt, lists, sink, recipe)
    for hr in (32, 16, 8):
        for r in range(64 // hr):
            hs = slice(r * hr, (r + 1) * hr)
            sk = torch.cat([sink[hs], sink[hs]])            # _run slices sink[:H]
            o, lse = _run(ext, mode, q[:, hs].contiguous(), kv, cache, bt, lists, sk, recipe)
            assert torch.equal(o.view(torch.int16), o64[:, hs].view(torch.int16)), (hr, r)
            assert torch.equal(lse.view(torch.int32), l64[:, hs].view(torch.int32)), (hr, r)
