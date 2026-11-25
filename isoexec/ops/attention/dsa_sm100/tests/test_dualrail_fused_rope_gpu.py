"""The dual-rail leaf's fused RoPE steps (csrc/dualrail/fused_rope.cuh) == the separate row launches, bit for bit.

rope_q: ``ops.q_rope`` (``rope_tail_inplace``) on the query, then the leaf  ==  the leaf with the Q RoPE prologue.
rope_o: the leaf, then the row op ``ROW_OUT_INV_ROPE``                        ==  the leaf with the inverse-RoPE epilogue.
Both plans (one CTA per row, split), 8 / 16 / 64 heads, with and without the compressed (extra) cache, positions over
the whole table, rows with subnormal-range query and output tails (the fused arithmetic is the rows.cu non-FTZ graph),
lse unchanged, the query tensor itself untouched."""
import os

import pytest
import torch

from isoexec.core.arch import live_device_arch  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="the dual-rail leaf runs on a Blackwell device only")
SCALE = 512 ** -0.5


def _tables(ratio):
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops
    from isoexec.ops.attention.deepseek_v41_cache_sm100.tests import _config
    return ops.rope_table(_config.published_config(), ratio, 8192, "cuda")


def _case(B, H, extra, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n_main, n_comp = B * 192 + 64, 8192
    main = torch.empty((n_main // 64, 64, 528), dtype=torch.uint8, device="cuda")
    main[..., :512] = (torch.randn(main[..., :512].shape, generator=g, device="cuda") * 50).to(
        torch.float8_e4m3fn).view(torch.uint8)
    main[..., 512:] = torch.randint(118, 126, main[..., 512:].shape, generator=g, device="cuda", dtype=torch.uint8)
    lists = torch.randperm(n_main, generator=g, device="cuda")[: B * 128].view(B, 128).int().contiguous()
    kw = {}
    if extra:
        ex = torch.empty((n_comp // 64, 64, 528), dtype=torch.uint8, device="cuda")
        ex[..., :512] = (torch.randn(ex[..., :512].shape, generator=g, device="cuda") * 50).to(
            torch.float8_e4m3fn).view(torch.uint8)
        ex[..., 512:] = torch.randint(114, 122, ex[..., 512:].shape, generator=g, device="cuda", dtype=torch.uint8)
        el = torch.stack([torch.randperm(n_comp, generator=g, device="cuda")[:512].sort().values
                          for _ in range(B)]).int().contiguous()
        kw = dict(extra_kv=ex, extra_lists=el)
    q = torch.randn((B, H, 512), generator=g, device="cuda") * 4.0
    if B > 3:
        q[1, :, 448:] *= 1e-38          # subnormal-range tails: products and sums below FLT_MIN
        lists[2, 64:] = -1
    q = q.bfloat16()
    sink = torch.randn(H, generator=g, device="cuda")
    pos = torch.randint(0, 8192, (B,), generator=g, device="cuda", dtype=torch.int32)
    pos[0] = 8191
    return q, main, lists, kw, sink, pos


def _plan(plan):
    if plan == "one":
        os.environ["ISOEXEC_DUALRAIL_PLAN"] = "one"
    else:
        os.environ.pop("ISOEXEC_DUALRAIL_PLAN", None)


@pytest.mark.parametrize("plan", ["one", "auto"])
@pytest.mark.parametrize("B,H", [(1, 8), (8, 64), (33, 16), (148, 8), (150, 64)])
@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize("ratio", [0, 2])
def test_fused_rope_equals_row_launches(plan, B, H, extra, ratio):
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops, spec
    from isoexec.ops.attention.dsa_sm100.dualrail import dualrail_attn
    cos, sin = _tables(ratio)
    q, main, lists, kw, sink, pos = _case(B, H, extra, 1000 * B + H + 10 * extra + ratio)
    q0 = q.clone()
    _plan(plan)
    try:
        # the separate launches (paged.py / ops.output_projection before the fusion)
        qr = q.clone()
        ops.q_rope(qr.view(B * H, 512), 512, (cos, sin), ops.Pos(pos=pos, div=H))
        o_ref, l_ref = dualrail_attn(qr, main, lists, SCALE, sink=sink, **kw)
        o_inv = torch.empty_like(o_ref)
        ops.row_op(spec.ROW_OUT_INV_ROPE, o_ref.view(B * H, 512), 512, table=(cos, sin), pos=ops.Pos(pos=pos, div=H),
                   out=o_inv.view(B * H, 512))
        # fused, each step alone and both
        o_q, l_q = dualrail_attn(q, main, lists, SCALE, sink=sink, rope=(cos, sin, pos), rope_q=True, **kw)
        o_o, l_o = dualrail_attn(qr, main, lists, SCALE, sink=sink, rope=(cos, sin, pos), rope_o=True, **kw)
        o_b, l_b = dualrail_attn(q, main, lists, SCALE, sink=sink, rope=(cos, sin, pos), rope_q=True, rope_o=True, **kw)
        torch.cuda.synchronize()
    finally:
        os.environ.pop("ISOEXEC_DUALRAIL_PLAN", None)
    eq = lambda a, b: torch.equal(a.view(torch.int16), b.view(torch.int16))   # noqa: E731
    eqf = lambda a, b: torch.equal(a.view(torch.int32), b.view(torch.int32))  # noqa: E731
    assert eq(q, q0), "the fused Q RoPE must not write the query tensor"
    assert eq(o_q, o_ref) and eqf(l_q, l_ref), "Q RoPE prologue != q_rope + leaf"
    assert eq(o_o, o_inv) and eqf(l_o, l_ref), "inverse-RoPE epilogue != leaf + ROW_OUT_INV_ROPE"
    assert eq(o_b, o_inv) and eqf(l_b, l_ref), "both fused != the three launches"
    assert not eq(o_ref, o_inv), "the inverse rotation must change the tail (test sanity)"
