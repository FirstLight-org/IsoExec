"""Run-to-run stress of the dual-rail leaf (one-CTA and split plans): many back-to-back launches of the same inputs must
give identical bits.

The softmax warpgroup's two threads of a row exchange values through one shared buffer: the row max every tile, then
the row sum after the last tile. Without a barrier between the last tile's max read and the sum write (FlashMLA #241),
a thread can read its partner's sum as the max, which changes that row's P and output bits nondeterministically. The
inputs give every row many tiles and keep the last tile's maxima far from any plausible sum, so a torn exchange shows as
a bit difference. Launches are queued back to back (no host sync between them) with the CTAs of one wave racing each
other; a repeat that differs anywhere fails."""
import os

import pytest
import torch

from isoexec.core.arch import live_device_arch  # noqa: E402

from isoexec.ops.attention.sparse_sm100.dualrail import dualrail_attn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="the dual-rail leaf runs on a Blackwell device only")
SCALE = 512 ** -0.5
REPEATS = int(os.environ.get("ISOEXEC_DUALRAIL_STRESS_REPEATS", "64"))


def _case(B, H, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n_main, n_comp = B * 192, 8192
    main = torch.empty((n_main // 64, 64, 528), dtype=torch.uint8, device="cuda")
    main[..., :512] = (torch.randn(main[..., :512].shape, generator=g, device="cuda") * 50).to(
        torch.float8_e4m3fn).view(torch.uint8)
    main[..., 512:] = torch.randint(118, 126, main[..., 512:].shape, generator=g, device="cuda", dtype=torch.uint8)
    extra = torch.empty((n_comp // 64, 64, 528), dtype=torch.uint8, device="cuda")
    extra[..., :512] = (torch.randn(extra[..., :512].shape, generator=g, device="cuda") * 50).to(
        torch.float8_e4m3fn).view(torch.uint8)
    extra[..., 512:] = torch.randint(114, 122, extra[..., 512:].shape, generator=g, device="cuda", dtype=torch.uint8)
    lists = torch.randperm(n_main, generator=g, device="cuda")[: B * 128].view(B, 128).int().contiguous()
    extra_lists = torch.stack([torch.randperm(n_comp, generator=g, device="cuda")[:512].sort().values
                               for _ in range(B)]).int().contiguous()
    q = (torch.randn((B, H, 512), generator=g, device="cuda") * 4.0).bfloat16()
    sink = torch.randn(H, generator=g, device="cuda")
    return q, main, lists, extra, extra_lists, sink


@pytest.mark.parametrize("plan", ["one", "auto"])
@pytest.mark.parametrize("B,H", [(148, 64), (296, 8), (8, 64), (32, 8)])
def test_repeated_launches_identical_bits(plan, B, H):
    q, main, lists, extra, extra_lists, sink = _case(B, H, 7 + B + H)
    if plan == "one":
        os.environ["ISOEXEC_DUALRAIL_PLAN"] = "one"
    try:
        outs = []
        for _ in range(REPEATS):          # queued back to back, no host sync
            o, l = dualrail_attn(q, main, lists, SCALE, sink=sink, extra_kv=extra, extra_lists=extra_lists)
            outs.append((o, l))
        torch.cuda.synchronize()
    finally:
        os.environ.pop("ISOEXEC_DUALRAIL_PLAN", None)
    o0, l0 = outs[0]
    assert torch.isfinite(o0.float()).all()
    for k, (o, l) in enumerate(outs[1:], 1):
        assert torch.equal(o.view(torch.int16), o0.view(torch.int16)), f"repeat {k}: output bits differ"
        assert torch.equal(l.view(torch.int32), l0.view(torch.int32)), f"repeat {k}: lse bits differ"
