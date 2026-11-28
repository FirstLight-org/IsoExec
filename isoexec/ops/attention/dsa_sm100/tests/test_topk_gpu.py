"""GPU battery of the DSA top-k (relations R-T1..R-T9), bf16 and fp32 scores.

    python -m pytest isoexec/ops/attention/dsa_sm100/tests/test_topk_gpu.py -q

Every test names its relation.  ``_battery`` counts relations / checks / mismatches into
``tests/battery_topk.json`` (README "Battery").  The reference for large rows is an exact torch
selection on the composite integer key ``kappa * 2^20 + (2^20 - 1 - j)`` (a total order that IS
SPEC section 3's), cross-checked against ``reference.topk_ref`` (numpy lexsort) on smaller rows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs an SM100 device")

from isoexec.ops.attention.dsa_sm100 import reference as ref  # noqa: E402
from isoexec.ops.attention.dsa_sm100.topk import dsa_topk  # noqa: E402

K = 2048
_TALLY: dict = {}
_OUT = Path(__file__).resolve().parent / "battery_topk.json"


def _count(rel: str, checks: int, mismatches: int):
    t = _TALLY.setdefault(rel, [0, 0])
    t[0] += checks
    t[1] += mismatches
    _OUT.write_text(json.dumps({"relations": len(_TALLY),
                                "checks": sum(v[0] for v in _TALLY.values()),
                                "mismatches": sum(v[1] for v in _TALLY.values()),
                                "per_relation": _TALLY}, indent=1))


def _keys(scores: torch.Tensor) -> torch.Tensor:
    """kappa as int64, torch (SPEC section 3 item 1)."""
    if scores.dtype == torch.bfloat16:
        x = scores.view(torch.int16).to(torch.int64) & 0xFFFF
        k = torch.where((x & 0x8000) != 0, (~x) & 0xFFFF, x ^ 0x8000)
        nan = (x & 0x7FFF) > 0x7F80
    else:
        x = scores.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        k = torch.where((x & 0x80000000) != 0, (~x) & 0xFFFFFFFF, x ^ 0x80000000)
        nan = (x & 0x7FFFFFFF) > 0x7F800000
    return torch.where(nan, torch.zeros_like(k), k)


def ref_topk(scores: torch.Tensor, ke: torch.Tensor, ks: torch.Tensor | None, k: int = K) -> torch.Tensor:
    rows, width = scores.shape
    ks = torch.zeros_like(ke) if ks is None else ks
    j = torch.arange(width, device=scores.device, dtype=torch.int64)
    comp = _keys(scores) * (1 << 20) + ((1 << 20) - 1 - j)
    inwin = (j[None] >= ks[:, None]) & (j[None] < ke[:, None])
    comp = torch.where(inwin, comp, torch.full_like(comp, -1))
    kk = min(k, width)
    vals, idx = torch.topk(comp, kk, dim=1)
    idx = torch.where(vals >= 0, idx, torch.full_like(idx, 1 << 40))
    idx = torch.sort(idx, dim=1).values
    idx = torch.where(idx >= (1 << 40), torch.full_like(idx, -1), idx)
    out = torch.full((rows, k), -1, dtype=torch.int32, device=scores.device)
    out[:, :kk] = idx.to(torch.int32)
    return out


def _check(rel, got, want):
    mism = int((got != want).any(dim=1).sum().item())
    _count(rel, got.shape[0], mism)
    assert mism == 0, f"{rel}: {mism} rows differ"


def _rows(dtype, rows, width, seed, kind="normal"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((rows, width), generator=g, device="cuda", dtype=torch.float32)
    if kind == "coarse":           # heavy ties
        x = torch.round(x * 4) / 4
    return x.to(dtype)


DTYPES = [torch.bfloat16, torch.float32]


@pytest.mark.parametrize("dtype", DTYPES)
def test_rt1_rt2_reference_and_order(dtype):
    for width, rows in ((5000, 64), (32768, 32), (65536, 8), (131072, 8)):
        s = _rows(dtype, rows, width, seed=width)
        g = torch.Generator(device="cuda").manual_seed(7)
        ke = torch.randint(width // 2, width + 1, (rows,), generator=g, device="cuda", dtype=torch.int32)
        ks = torch.randint(0, 100, (rows,), generator=g, device="cuda", dtype=torch.int32)
        for kso in (None, ks):
            got = dsa_topk(s, ke, kso)
            want = ref_topk(s, ke, kso)
            _check("R-T1 reference", got, want)
            valid = got >= 0
            asc = (got[:, 1:] > got[:, :-1]) | ~valid[:, 1:]
            pads_trail = (valid[:, 1:] <= valid[:, :-1])
            _count("R-T2 ascending, pads trailing", rows, int((~(asc & pads_trail)).any(1).sum()))
            assert asc.all() and pads_trail.all()
    # the numpy lexsort reference agrees with the torch composite reference
    s = _rows(dtype, 6, 9000, seed=3, kind="coarse")
    ke = torch.tensor([9000, 8000, 4000, 2049, 2048, 17], device="cuda", dtype=torch.int32)
    ks = torch.tensor([0, 5, 1000, 0, 0, 3], device="cuda", dtype=torch.int32)
    np_scores = s.view(torch.int16).cpu().numpy().view(np.uint16) if dtype == torch.bfloat16 else s.cpu().numpy()
    want_np = ref.topk_ref(np_scores, ks.cpu().numpy(), ke.cpu().numpy(), K,
                           "bf16" if dtype == torch.bfloat16 else "f32")
    got = dsa_topk(s, ke, ks).cpu().numpy()
    _count("R-T1 reference", 6, int((got != want_np).any(1).sum()))
    assert np.array_equal(got, want_np)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rt3_tie_classes(dtype):
    width = 70000
    s = torch.full((6, width), 1.0, device="cuda", dtype=dtype)        # all equal
    s[1, 100:4196] = 5.0                                               # 4096 equal maxima
    s[2] = _rows(dtype, 1, width, 11, "coarse")[0]
    s[3, ::2] = 2.0                                                    # 35000 equal maxima
    s[4, -4096:] = 3.0                                                 # maxima at the end
    s[5, :] = 0.0
    s[5, 1::2] = -0.0                                                  # -0 ranks just below +0
    ke = torch.full((6,), width, device="cuda", dtype=torch.int32)
    got = dsa_topk(s, ke)
    want = ref_topk(s, ke, None)
    _check("R-T3 tie classes", got, want)
    assert got[0, :K].tolist() == list(range(K))
    assert got[1, :K].tolist() == list(range(100, 100 + K))
    assert got[5, :K].tolist() == list(range(0, 2 * K, 2))


@pytest.mark.parametrize("dtype", DTYPES)
def test_rt4_rt5_nan_and_inf(dtype):
    width = 9000
    s = _rows(dtype, 5, width, 21)
    s[0, ::3] = float("nan")                                           # NaN never beats a real key
    s[1, :] = float("nan")
    s[1, 5000:5010] = float("-inf")                                    # -inf above NaN
    s[2, :] = float("-inf")
    s[2, 17] = -1e30
    s[3, :] = -1e30
    s[3, ::7] = float("-inf")
    s[4, : width - 3000] = float("nan")
    ke = torch.full((5,), width, device="cuda", dtype=torch.int32)
    got = dsa_topk(s, ke)
    want = ref_topk(s, ke, None)
    _check("R-T4 NaN policy", got[:2], want[:2])
    _check("R-T5 -inf / -1e30 hazards", got[2:], want[2:])
    nan_sel = torch.isnan(s[0].float())[got[0].long()].sum().item()
    assert nan_sel == 0, "a NaN was selected while real keys existed"
    assert set(range(5000, 5010)).issubset(set(got[1].tolist()))
    # SCORE_PAD outside the window is never a key: a window of 10 inside a row of +inf
    t = torch.full((1, 4096), float("inf"), device="cuda", dtype=dtype)
    t[0, 100:110] = -5.0
    got = dsa_topk(t, torch.tensor([110], device="cuda", dtype=torch.int32),
                   torch.tensor([100], device="cuda", dtype=torch.int32))
    assert got[0, :10].tolist() == list(range(100, 110)) and (got[0, 10:] == -1).all()
    _count("R-T5 -inf / -1e30 hazards", 1, 0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rt6_short_rows(dtype):
    width = 4096
    s = _rows(dtype, 7, width, 31)
    ke = torch.tensor([0, 1, 5, K - 1, K, K + 1, width], device="cuda", dtype=torch.int32)
    ks = torch.tensor([0, 0, 2, 0, 0, 0, 0], device="cuda", dtype=torch.int32)
    got = dsa_topk(s, ke, ks)
    want = ref_topk(s, ke, ks)
    _check("R-T6 rows < K, = K, = 1, = 0", got, want)
    assert (got[0] == -1).all() and got[1, 0] == 0 and (got[1, 1:] == -1).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_rt7_batch_composition(dtype):
    width = 40000
    s = _rows(dtype, 33, width, 41, "coarse")
    g = torch.Generator(device="cuda").manual_seed(3)
    ke = torch.randint(1, width + 1, (33,), generator=g, device="cuda", dtype=torch.int32)
    full = dsa_topk(s, ke)
    bad = 0
    for r in (0, 5, 17, 32):
        alone = dsa_topk(s[r:r + 1].contiguous(), ke[r:r + 1])
        bad += int((alone[0] != full[r]).any())
        sub = dsa_topk(s[r:].contiguous(), ke[r:])
        bad += int((sub[0] != full[r]).any())
    _count("R-T7 batch composition", 8, bad)
    assert bad == 0


@pytest.mark.parametrize("dtype", DTYPES)
def test_rt8_single_vs_multi_cta(dtype):
    for width in (9000, 70000, 131072):
        s = _rows(dtype, 12, width, 51 + width, "coarse" if width == 9000 else "normal")
        s[0, :] = 1.0
        s[1, ::5] = float("nan")
        g = torch.Generator(device="cuda").manual_seed(5)
        ke = torch.randint(K + 1, width + 1, (12,), generator=g, device="cuda", dtype=torch.int32)
        ks = torch.randint(0, 9, (12,), generator=g, device="cuda", dtype=torch.int32)
        base = dsa_topk(s, ke, ks, cluster=1, threads=1024, stash_keys=8, algo=1)
        want = ref_topk(s, ke, ks)
        _check("R-T1 reference", base, want)
        algos = (0, 1) if dtype == torch.bfloat16 else (1,)
        for algo in algos:
            for cl in (1, 2, 4):
                for th in ((256, 512, 1024) if (algo == 0 and cl == 1) else (512, 1024)):
                    for st in (0, 8, 4096, 49152 if dtype == torch.bfloat16 else 32768):
                        got = dsa_topk(s, ke, ks, cluster=cl, threads=th, stash_keys=st, algo=algo)
                        _check("R-T8 single- vs multi-CTA / plan", got, base)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rt9_repeat_and_graph(dtype):
    width = 65536
    s = _rows(dtype, 64, width, 61, "coarse")
    ke = torch.full((64,), width, device="cuda", dtype=torch.int32)
    first = dsa_topk(s, ke)
    bad = 0
    for _ in range(20):
        bad += int((dsa_topk(s, ke) != first).any(1).sum())
    _count("R-T9 repeat x20", 20 * 64, bad)
    out = torch.empty_like(first)
    dsa_topk(s, ke, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dsa_topk(s, ke, out=out)
    for _ in range(5):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        bad += int((out != first).any(1).sum())
    _count("R-T9 CUDA graph replay", 5 * 64, int(bad))
    assert bad == 0
