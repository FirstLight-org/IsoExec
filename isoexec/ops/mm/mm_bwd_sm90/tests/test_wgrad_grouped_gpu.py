"""Grouped (MoE) wgrad accumulate: each expert/member block is the dense wgrad chain, bit for bit."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires H100/SM90"
)


@pytest.mark.parametrize("counts", [(300, 0, 17, 1, 640, 64), (0, 0, 5), (1000,), tuple(range(0, 64, 3))])
@pytest.mark.parametrize("ends_dtype", [torch.int32, torch.int64])
def test_grouped_matches_dense_per_expert(counts, ends_dtype):
    from isoexec.ops.mm.mm_bwd_sm90 import ops as W

    gen = torch.Generator(device="cuda").manual_seed(len(counts))
    E, I, H = len(counts), 136, 256
    T = sum(counts)
    di = torch.randn(T, 2 * I, device="cuda", generator=gen).to(torch.bfloat16)
    x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
    ends = torch.tensor(counts, device="cuda").cumsum(0).to(ends_dtype)
    mgs = [(torch.randn(I, H, device="cuda", generator=gen), torch.randn(I, H, device="cuda", generator=gen))
           for _ in range(E)]
    want = [(g.clone(), u.clone()) for g, u in mgs]
    start = 0
    for e, c in enumerate(counts):
        if c:
            W.accumulate(di[start:start + c, :I], x[start:start + c], want[e][0], config="co_128x128")
            W.accumulate(di[start:start + c, I:], x[start:start + c], want[e][1], config="co_128x128")
        start += c
    groups = [g for e, (a, b) in enumerate(mgs) for g in ((e, 0, a), (e, I, b))]
    W.grouped_accumulate(di, x, ends, groups)
    for (a, b), (wa, wb) in zip(mgs, want):
        assert bitwise_equal(a, wa) and bitwise_equal(b, wb)
    again = [(g.clone(), u.clone()) for g, u in want]
    W.grouped_accumulate(di, x, ends, [g for e, (a, b) in enumerate(again) for g in ((e, 0, a), (e, I, b))])
    W.grouped_accumulate(di, x, ends, groups)  # deterministic: same increments
    for (a, b), (ra, rb) in zip(mgs, again):
        assert bitwise_equal(a, ra) and bitwise_equal(b, rb)
