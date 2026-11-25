"""The paged indexer on 128-entry pages (vLLM's ratio-1 DeepseekV4IndexerCache: ``[pages, 128, 1, 132]``,
128 x 128 e4m3 then 128 ue8m0 words per page, the vLLM block stride between pages).

``page_rows`` is address math only: the same logical keys stored as 64-entry pages and as 128-entry pages
(scattered by a permuted block table inside a buffer with a larger page stride, junk between pages) give
BITWISE the scores of each other and of the contiguous (non-paged) path, for the plain indexer and the
``sm100_dsa_indexer_tree_v1`` form.

    python -m pytest isoexec/ops/attention/dsa_sm100/tests/test_indexer_page128_gpu.py -q
"""

from __future__ import annotations

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs an SM100 device")

from isoexec.ops.attention.dsa_sm100.tests import _inputs as inputs  # noqa: E402
from isoexec.ops.attention.dsa_sm100.indexer import dsa_indexer_decode, dsa_indexer_prefill  # noqa: E402

H, D = 64, 128
STRIDE = 110880          # the vLLM block stride of the probe (a multiple of 16, > 128 * 132)
KES = [1, 63, 64, 65, 127, 128, 129, 1000, 2049, 2111, 4097, 5000]


def _paged(k_fp, k_sf, P, seed, stride=STRIDE):
    """Store the logical keys as P-entry pages at a permuted physical order inside a junk-filled buffer."""
    n = k_fp.shape[0]
    npg = (n + P - 1) // P
    phys = npg + 5
    g = torch.Generator(device="cuda").manual_seed(seed)
    buf = torch.randint(0, 256, (phys * stride,), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
    order = torch.randperm(phys, generator=g, device="cuda")[:npg]
    codes = k_fp.view(torch.uint8)
    sfb = k_sf.contiguous().view(torch.uint8)
    for lp in range(npg):
        base = int(order[lp]) * stride
        rows = min(P, n - lp * P)
        buf[base:base + rows * D] = codes[lp * P:lp * P + rows].reshape(-1)
        buf[base + P * D:base + P * D + rows * 4] = sfb[lp * P * 4:(lp * P + rows) * 4]
    cache = buf.as_strided((phys, P, 1, D + 4), (stride, D + 4, D + 4, 1))
    return cache, order.to(torch.int32)


def _inputs(seed=11, n=5000):
    g = torch.Generator(device="cuda").manual_seed(seed)
    B = len(KES)
    q_fp, q_sf = inputs.fp8_row_quantize(torch.randn((B, H, D), generator=g, device="cuda"))
    w = (torch.randn((B, H), generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    k_fp, k_sf = inputs.fp8_row_quantize(torch.randn((n, D), generator=g, device="cuda"))
    ke = torch.tensor(KES, dtype=torch.int32, device="cuda")
    layouts = {}
    for P in (64, 128):
        cache, order = _paged(k_fp, k_sf, P, seed + P)
        layouts[P] = (cache, order.view(1, -1).repeat(B, 1).contiguous())   # every row: the same logical keys
    return q_fp, q_sf, w, k_fp, k_sf, ke, layouts, n


def _bits_in_window(out, ke):
    return [out[b, :int(ke[b])].view(torch.int16) for b in range(out.shape[0])]


def _assert_rows_equal(a, b, what):
    for i, (x, y) in enumerate(zip(a, b)):
        assert torch.equal(x, y), f"{what}: row {i} (ke {KES[i]}) differs in {(x != y).sum().item()} entries"


@pytest.mark.parametrize("span", [16, 3, 1])
def test_decode_page128_equals_page64_and_contiguous(span):
    q_fp, q_sf, w, k_fp, k_sf, ke, layouts, n = _inputs()
    B = q_fp.shape[0]
    q = (q_fp.view(B, 1, H, D), q_sf.view(B, 1, H))
    got = {P: _bits_in_window(dsa_indexer_decode(q, cache, w, ke, bt, max_len=n, span=span), ke)
           for P, (cache, bt) in layouts.items()}
    flat = dsa_indexer_prefill((q_fp, q_sf), (k_fp, k_sf), w, torch.zeros_like(ke), ke, width=n)
    _assert_rows_equal(got[128], got[64], "128-entry pages vs 64-entry pages")
    _assert_rows_equal(got[128], _bits_in_window(flat, ke), "128-entry pages vs contiguous keys")


def test_decode_page128_default_width():
    """Without max_len the width is block-table entries x entries per page."""
    q_fp, q_sf, w, k_fp, k_sf, ke, layouts, n = _inputs(seed=12, n=4097)
    ke = ke.clamp(max=n)
    B = q_fp.shape[0]
    q = (q_fp.view(B, 1, H, D), q_sf.view(B, 1, H))
    out = {}
    for P, (cache, bt) in layouts.items():
        o = dsa_indexer_decode(q, cache, w, ke, bt)
        assert o.shape[1] == bt.shape[1] * P
        out[P] = _bits_in_window(o, ke)
    _assert_rows_equal(out[128], out[64], "default width: 128-entry vs 64-entry pages")


def test_indexer_tree_page128_equals_page64_and_contiguous():
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops

    q_fp, q_sf, w, k_fp, k_sf, ke, layouts, n = _inputs(seed=13)
    codes = q_fp.view(torch.uint8)
    got = {P: _bits_in_window(ops.indexer_tree(codes, q_sf, w, ke, width=n, cache=cache, block_table=bt), ke)
           for P, (cache, bt) in layouts.items()}
    flat = ops.indexer_tree(codes, q_sf, w, ke, width=n, keys=(k_fp.view(torch.uint8), k_sf))
    _assert_rows_equal(got[128], got[64], "tree: 128-entry pages vs 64-entry pages")
    _assert_rows_equal(got[128], _bits_in_window(flat, ke), "tree: 128-entry pages vs contiguous keys")
