"""Production SM100 RoPE dispatch across shapes, views, MRoPE, and in-place paths."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.attention.rope_sm100 import build_table, qk_norm_rope, rope
from isoexec.ops.attention.rope_sm100 import reference as R

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="requires SM100"
)
DEV = "cuda"
DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def rnd(*shape, dtype=torch.bfloat16, gen=None):
    return torch.randn(*shape, device=DEV, generator=gen).to(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hq,hk,d,rot", [(32, 8, 128, 128), (20, 1, 64, 64), (2, 1, 256, 64),
                                         (5, 3, 256, 256), (4, 1, 128, 32), (1, 0, 64, 64)])
@pytest.mark.parametrize("style,out_style", [("neox", "neox"), ("interleaved", "neox")])
def test_rope_production_plan_matches_recipe(dtype, hq, hk, d, rot, style, out_style):
    g = torch.Generator(device=DEV).manual_seed(hq * 1000 + d + rot)
    tab = build_table(1e6, rot, 70000, DEV)
    q = rnd(67, hq, d, dtype=dtype, gen=g)
    k = rnd(67, hk, d, dtype=dtype, gen=g) if hk else None
    pos = torch.randint(0, 70000, (67,), device=DEV, generator=g)
    qo, ko = rope(q, k, pos, tab, style=style, out_style=out_style)
    kw = dict(in_interleaved=style == "interleaved", out_interleaved=out_style == "interleaved")
    assert bitwise_equal(qo, R.rope_recipe(q, pos, tab.table, **kw))
    if hk:
        assert bitwise_equal(ko, R.rope_recipe(k, pos, tab.table, **kw))
    else:
        assert ko is None


def test_mrope_partial_rotary_and_inplace_agree():
    g = torch.Generator(device=DEV).manual_seed(72)
    tab = build_table(1e7, 64, 262144, DEV, mrope_section=(11, 11, 10), mrope_interleaved=True)
    q, k = rnd(300, 6, 256, gen=g), rnd(300, 2, 256, gen=g)
    pos = torch.randint(0, 262144, (3, 300), device=DEV, generator=g)
    qo, ko = rope(q, k, pos, tab)
    assert bitwise_equal(qo, R.rope_recipe(q, pos, tab.table, tab.m1, tab.m2))
    assert bitwise_equal(ko, R.rope_recipe(k, pos, tab.table, tab.m1, tab.m2))
    work = q.clone()
    got, _ = rope(work, None, pos, tab, inplace=True)
    assert got.data_ptr() == work.data_ptr() and bitwise_equal(got, qo)


def test_strided_and_misaligned_views_match_contiguous():
    g = torch.Generator(device=DEV).manual_seed(7)
    tab = build_table(1e6, 64, 70000, DEV)
    pos = torch.randint(0, 70000, (41,), device=DEV, generator=g)
    odd = rnd(41 * 4 * 64 + 1, gen=g)[1:].view(41, 4, 64)
    packed = rnd(41, 2 * 256 + 98, gen=g)
    col = packed[:, 98:610].view(41, 2, 256)
    perm = rnd(4, 41, 64, gen=g).transpose(0, 1)
    for x in (odd, col, perm):
        got, _ = rope(x, None, pos, tab, style="interleaved", out_style="neox")
        ref, _ = rope(x.contiguous(), None, pos, tab, style="interleaved", out_style="neox")
        assert bitwise_equal(got, ref), tuple(x.shape)


@pytest.mark.parametrize("d,rot", [(256, 64), (256, 0), (128, 128), (64, 16)])
@pytest.mark.parametrize("zero_centered", [True, False])
def test_qk_production_plan_matches_recipe_and_inplace(d, rot, zero_centered):
    g = torch.Generator(device=DEV).manual_seed(d + rot + zero_centered)
    tab = build_table(1e7, rot, 70000, DEV) if rot else None
    for hq, hk in ((4, 1), (16, 4), (3, 3)):
        packed = rnd(45, (hq + hk) * d, gen=g)
        q, k = packed[:, :hq * d].view(45, hq, d), packed[:, hq * d:].view(45, hk, d)
        wq, wk = rnd(d, gen=g), rnd(d, gen=g)
        pos = torch.randint(0, 70000, (45,), device=DEV, generator=g)
        yq, yk = qk_norm_rope(q, k, wq, wk, pos, tab, 1e-6, zero_centered=zero_centered)
        table = tab.table if tab is not None else None
        rq, _ = R.qk_norm_rope_recipe(q, wq, pos, table, 1e-6, zero_centered=zero_centered)
        rk, _ = R.qk_norm_rope_recipe(k, wk, pos, table, 1e-6, zero_centered=zero_centered)
        assert bitwise_equal(yq, rq) and bitwise_equal(yk, rk)
        work = packed.clone()
        iq, ik = work[:, :hq * d].view(45, hq, d), work[:, hq * d:].view(45, hk, d)
        oq, ok = qk_norm_rope(iq, ik, wq, wk, pos, tab, 1e-6, zero_centered=zero_centered, inplace=True)
        assert oq.data_ptr() == iq.data_ptr() and ok.data_ptr() == ik.data_ptr()
        assert bitwise_equal(oq, yq) and bitwise_equal(ok, yk)


def test_qk_mrope_views_and_hazard_values_match_recipe():
    g = torch.Generator(device=DEV).manual_seed(11)
    tab = build_table(1e7, 64, 262144, DEV, mrope_section=(11, 11, 10), mrope_interleaved=True)
    pos = torch.randint(0, 262144, (3, 130), device=DEV, generator=g)
    wq, wk = rnd(256, gen=g), rnd(256, gen=g)
    packed = rnd(130, 4 * 256 + 256 + 100, gen=g)
    qv = packed[:, 100:612].view(130, 2, 256)
    kv = packed[:, 612:868].view(130, 1, 256)
    hz = rnd(130, 2, 256, gen=g)
    hz[0, 0, :4] = torch.tensor([-0.0, 0.0, 1e-40, -1e-40], device=DEV, dtype=hz.dtype)
    hz[1, 0, :] = 3.0e38
    hz[2, 0, :] = 0.0
    for q, k in ((qv, kv), (qv.contiguous(), kv.contiguous()), (hz, rnd(130, 1, 256, gen=g))):
        got = qk_norm_rope(q, k, wq, wk, pos, tab, 1e-6)
        rq, _ = R.qk_norm_rope_recipe(q, wq, pos, tab.table, 1e-6, tab.m1, tab.m2)
        rk, _ = R.qk_norm_rope_recipe(k, wk, pos, tab.table, 1e-6, tab.m1, tab.m2)
        for actual, expected in ((got[0], rq), (got[1], rk)):
            finite = torch.isfinite(expected)
            assert bitwise_equal(torch.where(finite, actual, 0), torch.where(finite, expected, 0))
            assert torch.equal(torch.isnan(actual), torch.isnan(expected))


def test_qk_inplace_refuses_autograd_and_misaligned_operands():
    g = torch.Generator(device=DEV).manual_seed(5)
    tab = build_table(1e7, 64, 70000, DEV)
    q, k = rnd(12, 2, 256, gen=g), rnd(12, 1, 256, gen=g)
    wq, wk = rnd(256, gen=g), rnd(256, gen=g)
    pos = torch.randint(0, 70000, (12,), device=DEV, generator=g)
    with pytest.raises(ValueError, match="records no autograd node"):
        qk_norm_rope(q.detach().clone().requires_grad_(True), k, wq, wk, pos, tab, 1e-6, inplace=True)
    odd = rnd(12 * 2 * 256 + 1, gen=g)[1:].view(12, 2, 256)
    with pytest.raises(ValueError, match="declined the operands"):
        qk_norm_rope(odd, k, wq, wk, pos, tab, 1e-6, inplace=True)
