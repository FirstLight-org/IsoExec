"""GPU battery of the DSA sparse attention, decode leaf (relations R-A1..R-A12).

    python -m pytest isoexec/ops/attention/sparse_sm100/tests/test_attn_gpu.py -q

R-A2 is bit-exact against ``reference.sparse_attn_emul`` with ``tests/_prims.DevicePrims``: numpy
float32 for every scalar step of SPEC section 4, the device's exp2f / division / lse expression under
this package's profile, and the tensor-core model (each K16 step's 16-term bf16 block dot formed
exactly and added to the fp32 accumulator with one round-to-nearest).  R-A6 / R-A7 (prefill) and
R-A9 / R-A10 (G > 1) are in their own files.  Counts go to
``tests/battery_attn.json``.
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

from isoexec.ops.attention.sparse_sm100 import reference as ref  # noqa: E402
from isoexec.ops.attention.sparse_sm100.tests import _inputs as inputs  # noqa: E402
from isoexec.ops.attention.sparse_sm100.fwd import sparse_attn, check_ascending  # noqa: E402
from isoexec.ops.attention.sparse_sm100.tests._prims import DevicePrims  # noqa: E402

import functools as _ft
import os as _os
#: SPARSE_TEST_RECIPE=2 runs this battery on recipe v2 (sm100_sparse_v2, lazy rescale) against the
#: lazy emulation; counts go to the *_v2.json file.
RECIPE = int(_os.environ.get("SPARSE_TEST_RECIPE", "1"))
sparse_attn = _ft.partial(sparse_attn, recipe=RECIPE)
_emul = _ft.partial(ref.sparse_attn_emul, lazy=RECIPE == 2)

_TALLY: dict = {}
_OUT = Path(__file__).resolve().parent / ("battery_attn" + ("_v2" if RECIPE == 2 else "") + ".json")
D, PAGE = 512, 64
SCALE = 1 / math.sqrt(512)


def _count(rel, checks, mism):
    t = _TALLY.setdefault(rel, [0, 0])
    t[0] += checks
    t[1] += mism
    _OUT.write_text(json.dumps({"relations": len(_TALLY), "checks": sum(v[0] for v in _TALLY.values()),
                                "mismatches": sum(v[1] for v in _TALLY.values()), "per_relation": _TALLY},
                               indent=1))


class Case:
    """B requests with their own contexts, a permuted page table and a V4.1 fp8 cache."""

    def __init__(self, ctxs, K, h_q=64, seed=0, perm=True, nan_unused=False, mag=0.3):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.B, self.K, self.h_q, self.ctxs = len(ctxs), K, h_q, list(ctxs)
        ppr = max((c + PAGE - 1) // PAGE for c in ctxs)
        npages = self.B * ppr + 2
        kvb = (torch.randn((npages, PAGE, D), generator=g, device="cuda") * mag).to(torch.bfloat16)
        self.cache = inputs.quantize_v41(kvb)
        if nan_unused:
            self.cache.view(npages, PAGE, 528)[..., :512] = 0x7F          # every e4m3 a NaN ...
        gp = torch.Generator(device="cuda").manual_seed(seed + 1000)
        order = torch.randperm(npages, generator=gp, device="cuda") if perm else torch.arange(npages, device="cuda")
        self.bt = order[:self.B * ppr].view(self.B, ppr).to(torch.int32).contiguous()
        self.q = (torch.randn((self.B, h_q, D), generator=g, device="cuda") * mag).to(torch.bfloat16)
        lists = torch.full((self.B, K), -1, dtype=torch.int32, device="cuda")
        for b, c in enumerate(ctxs):
            n = min(K, c)
            pick = torch.randperm(c, generator=g, device="cuda")[:n].sort().values
            lists[b, :n] = pick.to(torch.int32)
        self.lists = lists
        if nan_unused:
            # ... except the rows the lists select, re-quantised to real values
            for b in range(self.B):
                for p in lists[b][lists[b] >= 0].tolist():
                    pg, off = int(self.bt[b, p // PAGE]), p % PAGE
                    self.cache[pg, off, 0] = inputs.quantize_v41(kvb[pg:pg + 1, off:off + 1])[0, 0, 0]

    def run(self, **kw):
        return sparse_attn(self.q, self.cache, self.bt, self.lists, SCALE, groups=1, **kw)

    def rows_of(self, b, topk_length=None):
        deq = inputs.dequantize_v41(self.cache).float().cpu().numpy().reshape(-1, D)
        lst = self.lists[b].cpu().numpy()
        valid = lst >= 0
        if topk_length is not None:
            valid &= np.arange(self.K) < topk_length
        rows = np.full((self.K, D), np.nan, np.float32)
        bt = self.bt[b].cpu().numpy()
        for p in np.nonzero(valid)[0]:
            rows[p] = deq[bt[lst[p] // PAGE] * PAGE + lst[p] % PAGE]
        return rows, valid


def _bits_out(o):
    return o.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def _emul_check(rel, case, out, lse, rows_b=None, topk_length=None):
    mism = n = 0
    worst = 0.0
    for b in range(case.B if rows_b is None else len(rows_b)):
        bb = b if rows_b is None else rows_b[b]
        rows, valid = case.rows_of(bb, None if topk_length is None else int(topk_length[bb]))
        q = case.q[bb].float().cpu().numpy()
        eo, el, _ = _emul(q, rows, valid, SCALE, groups=1, prims=DevicePrims)
        go = _bits_out(out[bb])
        gl = lse[bb].cpu().numpy()
        mism += int((go != eo).sum()) + int((gl.view(np.uint32) != el.view(np.uint32)).sum())
        n += go.size + gl.size
        o64, l64 = ref.sparse_attn_fp64(q, rows, valid, SCALE)
        worst = max(worst, float(np.abs(ref.bf16_bits_to_f32(go) - o64).max()))
    return mism, n, worst


def test_ra1_ra2_reference_and_emulation():
    case = Case([5000, 3000, 70000, 2048], K=2048, seed=1)
    out, lse = case.run()
    mism, n, worst = _emul_check("R-A2", case, out, lse)
    _count("R-A1 fp64 reference", n, int(worst > 2e-2))
    assert worst < 2e-2, worst
    _count("R-A2 fp32 emulation (bit-exact)", n, mism)
    assert mism == 0, f"{mism} of {n} differ from the emulation"


def test_ra3_batch_composition():
    case = Case([4000, 9000, 2100, 6000, 300], K=2048, seed=2)
    out, lse = case.run()
    bad = 0
    for b in range(case.B):
        sub = Case.__new__(Case)
        sub.__dict__.update(case.__dict__)
        sub.q, sub.bt, sub.lists, sub.B = case.q[b:b + 1], case.bt[b:b + 1], case.lists[b:b + 1], 1
        o1, l1 = sub.run()
        bad += int((o1.view(torch.int16) != out[b:b + 1].view(torch.int16)).sum())
        bad += int((l1.view(torch.int32) != lse[b:b + 1].view(torch.int32)).sum())
    _count("R-A3 batch composition", case.B * case.h_q * (D + 1), bad)
    assert bad == 0


def test_ra4_list_and_page_permutation():
    a = Case([7000, 3000], K=2048, seed=3, perm=True)
    # the same logical cache under another physical page order: cache_b[pi[p]] = cache_a[p],
    # block_table_b = pi[block_table_a]
    b = Case.__new__(Case)
    b.__dict__.update(a.__dict__)
    npages = a.cache.shape[0]
    pi = torch.randperm(npages, generator=torch.Generator(device="cuda").manual_seed(77), device="cuda")
    b.cache = torch.empty_like(a.cache)
    b.cache[pi] = a.cache
    b.bt = pi[a.bt.long()].to(torch.int32).contiguous()
    oa, la = a.run()
    ob, lb = b.run()
    bad = int((oa.view(torch.int16) != ob.view(torch.int16)).sum()) + int((la.view(torch.int32) != lb.view(torch.int32)).sum())
    _count("R-A4 page permutation", oa.numel() + la.numel(), bad)
    assert bad == 0
    shuffled = a.lists.clone()
    shuffled[0, :8] = shuffled[0, :8].flip(0)
    with pytest.raises(ValueError):
        check_ascending(shuffled)
    with pytest.raises(ValueError):
        sparse_attn(a.q, a.cache, a.bt, shuffled, SCALE, groups=1, check=True)
    # the device check (graph capture runs with check=False): a refused list poisons its row with NaN
    bad_lists = [shuffled]
    gap = a.lists.clone(); gap[0, 100] = -1; bad_lists.append(gap)          # a pad before a valid entry
    dup = a.lists.clone(); dup[0, 700] = dup[0, 699]; bad_lists.append(dup)  # a repeated position
    edge = a.lists.clone(); edge[0, 63], edge[0, 64] = edge[0, 64].item(), edge[0, 63].item(); bad_lists.append(edge)
    refused = 0
    for bl in bad_lists:
        o, l = sparse_attn(a.q, a.cache, a.bt, bl, SCALE, groups=1, check=False)
        refused += int(bool(torch.isnan(o[0].float()).all()) and bool(torch.isnan(l[0]).all()))
        refused += int(bool((o[1].view(torch.int16) == oa[1].view(torch.int16)).all()))   # the other row is untouched
    # a selected position whose block-table entry is negative is refused too (not silently a pad)
    bt_bad = a.bt.clone(); bt_bad[0, int(a.lists[0, 5]) // PAGE] = -1
    o, l = sparse_attn(a.q, a.cache, bt_bad, a.lists, SCALE, groups=1)
    refused += int(bool(torch.isnan(o[0].float()).all())) + int(bool((o[1].view(torch.int16) == oa[1].view(torch.int16)).all()))
    _count("R-A4 non-ascending list refused", 4 + 2 * len(bad_lists), 2 + 2 * len(bad_lists) - refused)
    assert refused == 2 + 2 * len(bad_lists)


def test_ra5_pad_hazards_and_empty_row():
    case = Case([4096, 4096, 4096, 4096, 4096], K=2048, seed=4, nan_unused=True)
    lists = case.lists
    lists[0, :] = -1                                   # 0 valid: the declared empty output
    lists[1, 1:] = -1                                  # 1 valid
    lists[2, 2047:] = -1                               # K - 1 valid
    out, lse = case.run()
    assert (out[0].view(torch.int16) == 0).all(), "empty row: out = bf16(0 / 1) = +0"
    want = np.float32(math.log(1.0)) + np.float32(-1e30) / np.float32(math.log2(math.e))
    assert np.allclose(lse[0].cpu().numpy(), want, rtol=1e-6)
    mism, n, _ = _emul_check("R-A5", case, out, lse)
    _count("R-A5 pad hazards (0, 1, K-1, K valid; NaN pages)", n, mism)
    assert torch.isfinite(out.float()).all()
    assert mism == 0


def test_ra8_repeat_and_graph():
    case = Case([30000, 12000, 60000], K=2048, seed=5)
    first_o, first_l = case.run()
    bad = 0
    for _ in range(20):
        o, l = case.run()
        bad += int((o.view(torch.int16) != first_o.view(torch.int16)).sum()) + int((l.view(torch.int32) != first_l.view(torch.int32)).sum())
    out = torch.empty_like(first_o)
    lse = torch.empty_like(first_l)
    case.run(out=out, lse=lse)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case.run(out=out, lse=lse, check=False)
    for _ in range(5):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        bad += int((out.view(torch.int16) != first_o.view(torch.int16)).sum())
    _count("R-A8 repeat x20 + graph", 25 * first_o.numel(), bad)
    assert bad == 0


def test_ra11_degenerate_shapes():
    # B = 1; h_q not a multiple of 64 (rows past h_q masked); K = 64 (one tile); topk_length < K
    for ctxs, K, h_q in (([5000], 2048, 64), ([100, 2000], 64, 64), ([3000, 3000], 2048, 40), ([3000], 128, 128)):
        case = Case(ctxs, K=K, h_q=h_q, seed=6 + K + h_q)
        out, lse = case.run()
        mism, n, _ = _emul_check("R-A11", case, out, lse)
        _count("R-A11 degenerate shapes", n, mism)
        assert mism == 0, (ctxs, K, h_q)
    case = Case([5000, 5000, 5000], K=2048, seed=9)
    tl = torch.tensor([2048, 1000, 37], dtype=torch.int32, device="cuda")
    out, lse = case.run(topk_length=tl)
    mism, n, _ = _emul_check("R-A11", case, out, lse, topk_length=tl.cpu().numpy())
    _count("R-A11 degenerate shapes (topk_length)", n, mism)
    assert mism == 0


def test_ra12_nonfinite_unused():
    case = Case([8000, 8000], K=2048, seed=10, nan_unused=True)
    out, lse = case.run()
    ok = bool(torch.isfinite(out.float()).all()) and bool(torch.isfinite(lse).all())
    clean = Case([8000, 8000], K=2048, seed=10, nan_unused=False)
    oc, lc = clean.run()
    bad = int((out.view(torch.int16) != oc.view(torch.int16)).sum())
    _count("R-A12 nonfinite unused rows", out.numel(), bad + int(not ok))
    assert ok and bad == 0


def test_ra13_h128_head_equals_h64_head():
    # h_q = 128 is two CTAs along heads; head h of the 128-head call == head h of a 64-head call over
    # the same 64-head block (same list, same cache)
    case = Case([6000, 3000, 9000], K=2048, h_q=128, seed=11)
    o128, l128 = case.run()
    bad = 0
    for blk in (0, 1):
        sub = Case.__new__(Case)
        sub.__dict__.update(case.__dict__)
        sub.q = case.q[:, 64 * blk:64 * (blk + 1)].contiguous()
        sub.h_q = 64
        o64, l64 = sub.run()
        bad += int((o64.view(torch.int16) != o128[:, 64 * blk:64 * (blk + 1)].view(torch.int16)).sum())
        bad += int((l64.view(torch.int32) != l128[:, 64 * blk:64 * (blk + 1)].contiguous().view(torch.int32)).sum())
    _count("R-A13 h_q=128 head == h_q=64 head", o128.numel() + l128.numel(), bad)
    assert bad == 0
