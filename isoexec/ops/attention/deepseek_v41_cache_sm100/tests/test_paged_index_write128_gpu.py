"""paged_index_write on P-entry index-key pages ([pages, P, 1, 132], P = 64 or 128 -- vLLM's ratio-1
DeepseekV4IndexerCache holds 128): record dest = page * P + row lands at page * stride + row * 128 (codes)
and page * stride + P * 128 + row * 4 (the ue8m0 word), at any 16-byte multiple page stride, and no
other byte of the buffer changes. P = 64 is the unchanged default."""
import pytest
import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100._extension import extension

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

STRIDE = 110880   # the vLLM block stride of the probe


def _reference(buf, codes, sf, dest, P, stride):
    ref = buf.clone()
    sfb = sf.contiguous().view(torch.uint8).view(-1, 4)
    for r, d in enumerate(dest.tolist()):
        if d < 0:
            continue
        page, row = divmod(d, P)
        base = page * stride
        ref[base + row * 128:base + row * 128 + 128] = codes[r]
        ref[base + P * 128 + row * 4:base + P * 128 + row * 4 + 4] = sfb[r]
    return ref


@pytest.mark.parametrize("P,stride", [(128, STRIDE), (64, STRIDE), (64, 64 * 132), (128, 128 * 132)])
def test_paged_index_write_page_layout(P, stride):
    g = torch.Generator(device="cuda").manual_seed(17 + P)
    pages, R = 9, 300
    codes = torch.randint(0, 256, (R, 128), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
    sf = torch.randint(-2**31, 2**31 - 1, (R,), generator=g, device="cuda", dtype=torch.int64).to(torch.int32)
    dest = torch.randperm(pages * P, generator=g, device="cuda")[:R].long()
    dest[::7] = -1
    buf = torch.full((pages * stride,), 0xA5, dtype=torch.uint8, device="cuda")   # sentinel
    ref = _reference(buf, codes, sf, dest, P, stride)
    pool = buf.as_strided((pages, P, 1, 132), (stride, 132, 132, 1))
    extension().paged_index_write(codes, sf, dest, pool)
    torch.cuda.synchronize()
    bad = (buf != ref).nonzero().flatten()
    assert bad.numel() == 0, f"{bad.numel()} bytes differ, first at {bad[:8].tolist()}"


def test_paged_index_write_page128_is_the_page64_pair():
    """A 128-entry page is the two 64-entry halves' bytes rearranged: codes rows 0..127 then 128 words."""
    g = torch.Generator(device="cuda").manual_seed(29)
    pages, R = 4, 512
    codes = torch.randint(0, 256, (R, 128), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
    sf = torch.randint(-2**31, 2**31 - 1, (R,), generator=g, device="cuda", dtype=torch.int64).to(torch.int32)
    dest = torch.arange(R, device="cuda", dtype=torch.long)
    buf = torch.zeros((pages * STRIDE,), dtype=torch.uint8, device="cuda")
    extension().paged_index_write(codes, sf, dest, buf.as_strided((pages, 128, 1, 132), (STRIDE, 132, 132, 1)))
    pg = buf.view(pages, STRIDE)
    assert torch.equal(pg[:, :128 * 128].reshape(R, 128), codes)
    assert torch.equal(pg[:, 128 * 128:128 * 132].contiguous().view(torch.int32).reshape(R), sf)
    assert (pg[:, 128 * 132:] == 0).all()
