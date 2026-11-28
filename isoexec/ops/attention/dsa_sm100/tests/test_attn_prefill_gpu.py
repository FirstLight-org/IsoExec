"""GPU battery of the DSA sparse attention, prefill (relations R-A6, R-A7, and R-A1/R-A2/
R-A5/R-A8 on the bf16-kv entry).

    python -m pytest isoexec/ops/attention/dsa_sm100/tests/test_attn_prefill_gpu.py -q

Prefill is the decode leaf over token rows (one CTA per (token, 64-head block), no query stacking).
Two kv entries: the fp8 V4.1 cache through a block table (the chunked-prefill path) and a contiguous
bf16 ``[S, 512]`` kv (conversion = identity).  The bridge between them: the bf16 kv holding exactly
the cache's dequantised values gives the fp8 path's bits (the conversion point is the only difference).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs an SM100 device")

from isoexec.ops.attention.dsa_sm100 import reference as ref  # noqa: E402
from isoexec.ops.attention.dsa_sm100.tests import _inputs as inputs  # noqa: E402
from isoexec.ops.attention.dsa_sm100.fwd import (dsa_sparse_attn_decode, dsa_sparse_attn_prefill)  # noqa: E402
from isoexec.ops.attention.dsa_sm100.tests._prims import DevicePrims  # noqa: E402

import functools as _ft
import os as _os
#: DSA_TEST_RECIPE=2 runs this battery on recipe v2 (sm100_dsa_sparse_v2, lazy rescale) against the
#: lazy emulation; counts go to the *_v2.json file.
RECIPE = int(_os.environ.get("DSA_TEST_RECIPE", "1"))
dsa_sparse_attn_decode = _ft.partial(dsa_sparse_attn_decode, recipe=RECIPE)
dsa_sparse_attn_prefill = _ft.partial(dsa_sparse_attn_prefill, recipe=RECIPE)
_emul = _ft.partial(ref.sparse_attn_emul, lazy=RECIPE == 2)

_TALLY: dict = {}
_OUT = Path(__file__).resolve().parent / ("battery_attn_prefill" + ("_v2" if RECIPE == 2 else "") + ".json")
D, PAGE = 512, 64
SCALE = 1 / math.sqrt(512)


def _count(rel, checks, mism):
    t = _TALLY.setdefault(rel, [0, 0])
    t[0] += checks
    t[1] += mism
    _OUT.write_text(json.dumps({"relations": len(_TALLY), "checks": sum(v[0] for v in _TALLY.values()),
                                "mismatches": sum(v[1] for v in _TALLY.values()), "per_relation": _TALLY},
                               indent=1))


def _ne(a, b):
    if a.dtype == torch.bfloat16:
        return int((a.view(torch.int16) != b.view(torch.int16)).sum())
    return int((a.view(torch.int32) != b.view(torch.int32)).sum())


class Seq:
    """One sequence of ``ctx`` tokens: a bf16 kv, its V4.1 fp8 cache under a permuted page table, the
    bf16 kv of the cache's dequantised values, and causal ascending lists for the last T tokens."""

    def __init__(self, ctx, T, K, h_q=64, seed=0, mag=0.3):
        g = torch.Generator(device="cuda").manual_seed(seed)
        npg = (ctx + PAGE - 1) // PAGE
        kvb = (torch.randn((npg * PAGE, D), generator=g, device="cuda") * mag).to(torch.bfloat16)
        perm = torch.randperm(npg, generator=g, device="cuda")
        self.cache = torch.empty((npg, PAGE, 1, 528), dtype=torch.uint8, device="cuda")
        self.cache[perm] = inputs.quantize_v41(kvb.view(npg, PAGE, D)).view(npg, PAGE, 1, 528)
        self.bt = perm.to(torch.int32).view(1, npg).contiguous()
        self.kv_deq = inputs.dequantize_v41(self.cache[perm].view(npg, PAGE, 528)).view(npg * PAGE, D).contiguous()
        self.kv = kvb.contiguous()
        self.q = (torch.randn((T, h_q, D), generator=g, device="cuda") * mag).to(torch.bfloat16)
        lists = torch.full((T, K), -1, dtype=torch.int32, device="cuda")
        prefix = ctx - T
        for t in range(T):
            lim = prefix + t + 1
            n = min(K, lim)
            lists[t, :n] = torch.randperm(lim, generator=g, device="cuda")[:n].sort().values.to(torch.int32)
        self.lists, self.T, self.K, self.ctx = lists, T, K, ctx


def test_ra7_prefill_row_equals_decode_row():
    s = Seq(9000, 24, 2048, seed=1)
    po, pl = dsa_sparse_attn_prefill(s.q, s.cache, s.lists, SCALE, block_table=s.bt)
    bad = 0
    for t in range(s.T):
        do, dl = dsa_sparse_attn_decode(s.q[t:t + 1], s.cache, s.bt, s.lists[t:t + 1], SCALE)
        bad += _ne(do, po[t:t + 1]) + _ne(dl, pl[t:t + 1])
    _count("R-A7 prefill row == decode row", po.numel() + pl.numel(), bad)
    assert bad == 0


def test_ra6_chunked_prefill():
    s = Seq(6000, 40, 2048, seed=2)
    fo, fl = dsa_sparse_attn_prefill(s.q, s.cache, s.lists, SCALE, block_table=s.bt)
    bad = 0
    for cut in (1, 17, 39):
        a = dsa_sparse_attn_prefill(s.q[:cut], s.cache, s.lists[:cut], SCALE, block_table=s.bt)
        b = dsa_sparse_attn_prefill(s.q[cut:], s.cache, s.lists[cut:], SCALE, block_table=s.bt)
        bad += _ne(torch.cat([a[0], b[0]]), fo) + _ne(torch.cat([a[1], b[1]]), fl)
    _count("R-A6 chunked prefill vs whole prefill (cuts 1, 17, 39)", 3 * (fo.numel() + fl.numel()), bad)
    assert bad == 0


def test_bf16_kv_equals_fp8_cache_at_the_conversion_point():
    s = Seq(5000, 16, 2048, seed=3)
    fo, fl = dsa_sparse_attn_prefill(s.q, s.cache, s.lists, SCALE, block_table=s.bt)
    bo, bl = dsa_sparse_attn_prefill(s.q, s.kv_deq, s.lists, SCALE)
    bad = _ne(fo, bo) + _ne(fl, bl)
    _count("R-A6b bf16 kv (dequantised values) == fp8 cache", fo.numel() + fl.numel(), bad)
    assert bad == 0


@pytest.mark.parametrize("K,h_q", [(2048, 64), (640, 64), (2048, 128), (640, 40)])
def test_ra1_ra2_bf16_kv(K, h_q):
    s = Seq(4096 + K, 6, K, h_q=h_q, seed=4 + K + h_q)
    tl = None
    out, lse = dsa_sparse_attn_prefill(s.q, s.kv, s.lists, SCALE)
    kv = s.kv.float().cpu().numpy()
    mism = n = 0
    worst = 0.0
    for t in range(s.T):
        lst = s.lists[t].cpu().numpy()
        valid = lst >= 0
        rows = np.full((K, D), np.nan, np.float32)
        rows[valid] = kv[lst[valid]]
        q = s.q[t].float().cpu().numpy()
        eo, el, _ = _emul(q, rows, valid, SCALE, groups=1, prims=DevicePrims)
        go = out[t].contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
        gl = lse[t].cpu().numpy()
        mism += int((go != eo).sum()) + int((gl.view(np.uint32) != el.view(np.uint32)).sum())
        n += go.size + gl.size
        o64, _ = ref.sparse_attn_fp64(q, rows, valid, SCALE)
        worst = max(worst, float(np.abs(ref.bf16_bits_to_f32(go) - o64).max()))
    _count("R-A1 fp64 reference (bf16 kv)", n, int(worst > 2e-2))
    _count("R-A2 fp32 emulation, bf16 kv (bit-exact)", n, mism)
    assert worst < 2e-2 and mism == 0, (worst, mism)


def test_ra5_ra8_bf16_kv_pads_refusal_repeat():
    s = Seq(3000, 8, 2048, seed=7)
    lists = s.lists.clone()
    lists[0, :] = -1
    lists[1, 1:] = -1
    o, l = dsa_sparse_attn_prefill(s.q, s.kv, lists, SCALE)
    assert (o[0].view(torch.int16) == 0).all()
    bad = 0
    for _ in range(20):
        o2, l2 = dsa_sparse_attn_prefill(s.q, s.kv, lists, SCALE)
        bad += _ne(o2, o) + _ne(l2, l)
    # a list entry past the kv is refused (NaN row), the other rows untouched
    past = lists.clone()
    past[3, 5] = s.kv.shape[0] + 7
    past[3, 6:] = -1
    o3, l3 = dsa_sparse_attn_prefill(s.q, s.kv, past, SCALE)
    refused = bool(torch.isnan(o3[3].float()).all()) and bool(torch.isnan(l3[3]).all())
    bad += int(not refused) + _ne(o3[4:], o[4:])
    _count("R-A5/R-A8 bf16 kv: pads, repeat x20, out-of-range entry refused", 21 * o.numel(), bad)
    assert bad == 0


@pytest.mark.parametrize("h_q,K", [(64, 2048), (128, 640)])
def test_ra1_ra2_bf16_kv_576(h_q, K):
    """The training geometry (SPEC 6): D_QK = 576 bf16 kv (latent 0..511 then rope 512..575, V = the
    first 512), the QK chain over 576 dims ascending -- bit-exact against the emulation."""
    g = torch.Generator(device="cuda").manual_seed(40 + h_q)
    S, T = 3000, 5
    kv = (torch.randn((S, 576), generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    q = (torch.randn((T, h_q, 576), generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    lists = torch.full((T, K), -1, dtype=torch.int32, device="cuda")
    for t in range(T):
        n = K - 7 * t
        lists[t, :n] = torch.randperm(S, generator=g, device="cuda")[:n].sort().values.to(torch.int32)
    scale = 576 ** -0.5
    out, lse = dsa_sparse_attn_prefill(q, kv, lists, scale)
    kvn = kv.float().cpu().numpy()
    mism = n = 0
    worst = 0.0
    for t in range(T):
        lst = lists[t].cpu().numpy()
        valid = lst >= 0
        rows = np.full((K, 576), np.nan, np.float32)
        rows[valid] = kvn[lst[valid]]
        qq = q[t].float().cpu().numpy()
        eo, el, _ = _emul(qq, rows, valid, scale, groups=1, prims=DevicePrims)
        go = out[t].contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
        gl = lse[t].cpu().numpy()
        mism += int((go != eo).sum()) + int((gl.view(np.uint32) != el.view(np.uint32)).sum())
        n += go.size + gl.size
        o64, _ = ref.sparse_attn_fp64(qq, rows, valid, scale)
        worst = max(worst, float(np.abs(ref.bf16_bits_to_f32(go) - o64).max()))
    _count("R-A1 fp64 reference (576 bf16 kv)", n, int(worst > 2e-2))
    _count("R-A2 fp32 emulation, 576 bf16 kv (bit-exact)", n, mism)
    assert worst < 2e-2 and mism == 0, (worst, mism)
