"""GPU battery of the DSA lightning indexer (relations R-I1..R-I8).

    python -m pytest isoexec/ops/attention/indexer_sm100/tests/test_indexer_gpu.py -q

R-I2 compares against ``reference.indexer_emul`` bit for bit: its only hardware hypothesis is the
tensor-core accumulation model (one k-step's 32-term block-scaled dot product formed exactly and added
to the fp32 accumulator with one round-to-nearest); ``test_ri2`` is what validates it.  Counts go to
``tests/battery_indexer.json``.
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
from isoexec.ops.attention.indexer_sm100.indexer import dsa_indexer_decode, dsa_indexer_prefill  # noqa: E402

_TALLY: dict = {}
_OUT = Path(__file__).resolve().parent / "battery_indexer.json"
H, D, PAGE = 64, 128, 64


def _count(rel, checks, mism):
    t = _TALLY.setdefault(rel, [0, 0])
    t[0] += checks
    t[1] += mism
    _OUT.write_text(json.dumps({"relations": len(_TALLY), "checks": sum(v[0] for v in _TALLY.values()),
                                "mismatches": sum(v[1] for v in _TALLY.values()), "per_relation": _TALLY},
                               indent=1))


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16)


class Case:
    """One request batch: logical K per request, a permuted page table, and the paged cache."""

    def __init__(self, ctxs, seed=0, perm=True, nan_tail=False):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.B = len(ctxs)
        self.ctxs = list(ctxs)
        self.maxp = max((c + PAGE - 1) // PAGE for c in ctxs) + 1
        q = torch.randn((self.B, H, D), generator=g, device="cuda")
        self.q_fp, self.q_sf = inputs.fp8_row_quantize(q)
        w = torch.zeros((self.B, H), dtype=torch.bfloat16, device="cuda")
        w[:] = (torch.randn((self.B, H), generator=g, device="cuda") * 0.3).to(torch.bfloat16)
        self.w = w
        self.k = []
        pages_total = sum((c + PAGE - 1) // PAGE for c in ctxs) + 3
        gp = torch.Generator(device="cuda").manual_seed(seed + 1000)   # the page order has its own stream
        order = torch.randperm(pages_total, generator=gp, device="cuda") if perm else torch.arange(pages_total, device="cuda")
        self.bt = torch.zeros((self.B, self.maxp), dtype=torch.int32, device="cuda")
        cache = torch.zeros((pages_total, PAGE * (D + 4)), dtype=torch.uint8, device="cuda")
        nxt = 0
        for b, c in enumerate(ctxs):
            npg = (c + PAGE - 1) // PAGE
            kb = torch.randn((max(npg, 1) * PAGE, D), generator=g, device="cuda")
            k_fp, k_sf = inputs.fp8_row_quantize(kb)
            if nan_tail and c % PAGE:
                k_fp.view(torch.uint8)[c:] = 0x7F                      # NaN rows past the window
            self.k.append((k_fp[:max(c, 0)], k_sf[:max(c, 0)]))
            for pg in range(npg):
                phys = int(order[nxt]); nxt += 1
                self.bt[b, pg] = phys
                cache[phys, :PAGE * D] = k_fp.view(torch.uint8)[pg * PAGE:(pg + 1) * PAGE].reshape(-1)
                cache[phys, PAGE * D:] = k_sf[pg * PAGE:(pg + 1) * PAGE].contiguous().view(torch.uint8)
        self.cache = cache.view(pages_total, PAGE, 1, D + 4)
        self.ke = torch.tensor(ctxs, dtype=torch.int32, device="cuda")
        self.width = max(max(ctxs), 1)

    def decode(self, **kw):
        return dsa_indexer_decode((self.q_fp.view(self.B, 1, H, D), self.q_sf.view(self.B, 1, H)), self.cache,
                                  self.w, self.ke, self.bt, max_len=self.width, **kw)

    def emul_row(self, b):
        c = self.ctxs[b]
        k_fp, k_sf = self.k[b]
        qb = self.q_fp[b:b + 1].view(torch.uint8).cpu().numpy()
        qs = self.q_sf[b:b + 1].cpu().numpy()
        kb = k_fp.view(torch.uint8).cpu().numpy()
        ksf = k_sf.cpu().numpy()
        wb = self.w[b:b + 1].view(torch.int16).cpu().numpy().view(np.uint16)
        e = ref.indexer_emul(qb, qs, kb, ksf, wb, np.array([0]), np.array([c]))
        f = ref.indexer_fp64(qb, qs, kb, ksf, wb, np.array([0]), np.array([c]))
        return e[0], f[0]


def _cmp(rel, got_rows, want_rows):
    """got/want: lists of int16 arrays (bf16 bits) over the windows."""
    mism = sum(int((np.asarray(g) != np.asarray(w)).sum()) for g, w in zip(got_rows, want_rows))
    n = sum(len(np.asarray(g)) for g in got_rows)
    _count(rel, n, mism)
    return mism


def test_ri1_ri2_reference_and_emulation():
    case = Case([2048, 1000, 64, 1, 4097, 700], seed=1)
    out = case.decode()
    got, emu, far = [], [], []
    for b, c in enumerate(case.ctxs):
        e, f = case.emul_row(b)
        got.append(out[b, :c].view(torch.int16).cpu().numpy().view(np.uint16))
        emu.append(e[:c])
        far.append(f[:c])
    # R-I1: the fp64 value within the bf16 rounding (relative 2^-8) plus the fp32 fold's error
    worst = 0.0
    for g, f in zip(got, far):
        gv = ref.bf16_bits_to_f32(g).astype(np.float64)
        err = np.abs(gv - f) / np.maximum(np.abs(f), 1e-2)
        worst = max(worst, float(err.max()) if err.size else 0.0)
    _count("R-I1 fp64 reference", sum(len(g) for g in got), int(worst > 2 ** -7))
    assert worst <= 2 ** -7, worst
    # R-I2: bit-exact against the fp32 emulation
    mism = _cmp("R-I2 fp32 emulation (bit-exact)", got, emu)
    assert mism == 0, f"{mism} scores differ from the emulation"


def test_ri3_batch_composition_and_ri6_splits():
    case = Case([3000, 5000, 128, 2049, 9000, 64, 777], seed=2)
    full = case.decode()
    bad = 0
    for b in range(case.B):
        c = case.ctxs[b]
        sub = Case.__new__(Case)
        sub.__dict__.update(case.__dict__)
        sub.B = 1
        sub.q_fp, sub.q_sf, sub.w = case.q_fp[b:b + 1], case.q_sf[b:b + 1], case.w[b:b + 1]
        sub.bt, sub.ke, sub.ctxs = case.bt[b:b + 1], case.ke[b:b + 1], [c]
        alone = sub.decode()
        bad += int((alone[0, :c].view(torch.int16) != full[b, :c].view(torch.int16)).sum())
    _count("R-I3 batch composition", sum(case.ctxs), bad)
    assert bad == 0
    bad = 0
    for span in (1, 2, 3, 16):
        for grid in (1, 5, 148):
            o = case.decode(span=span, grid=grid)
            for b, c in enumerate(case.ctxs):
                bad += int((o[b, :c].view(torch.int16) != full[b, :c].view(torch.int16)).sum())
    _count("R-I6 key-split / worker count", 12 * sum(case.ctxs), bad)
    assert bad == 0


def test_ri4_page_permutation():
    a = Case([3333, 128, 6000], seed=3, perm=True)
    b = Case([3333, 128, 6000], seed=3, perm=False)
    oa, ob = a.decode(), b.decode()
    bad = sum(int((oa[i, :c].view(torch.int16) != ob[i, :c].view(torch.int16)).sum()) for i, c in enumerate(a.ctxs))
    _count("R-I4 page permutation", sum(a.ctxs), bad)
    assert bad == 0


def test_ri5_decode_row_equals_prefill_row():
    case = Case([4000, 1500], seed=4)
    dec = case.decode()
    bad = 0
    for b, c in enumerate(case.ctxs):
        k_fp, k_sf = case.k[b]
        # prefill: the same query as 3 rows (qb = 2 blocks), windows [0, c) for the copy of row b
        qf = case.q_fp[b:b + 1].expand(3, H, D).contiguous()
        qs = case.q_sf[b:b + 1].expand(3, H).contiguous()
        w = case.w[b:b + 1].expand(3, H).contiguous()
        ks = torch.tensor([0, 0, 17], dtype=torch.int32, device="cuda")
        ke = torch.tensor([c, c, c], dtype=torch.int32, device="cuda")
        for qb in (1, 2):
            pre = dsa_indexer_prefill((qf, qs), (k_fp, k_sf), w, ks, ke, qb=qb)
            for r in range(3):
                lo = int(ks[r])
                bad += int((pre[r, lo:c].view(torch.int16) != dec[b, lo:c].view(torch.int16)).sum())
    _count("R-I5 decode row == prefill row", 6 * sum(case.ctxs), bad)
    assert bad == 0


def test_ri7_window_hazards():
    case = Case([65, 64, 1, 129, 4096 + 1], seed=5, nan_tail=True)
    out = case.decode()
    got, emu = [], []
    for b, c in enumerate(case.ctxs):
        e, _ = case.emul_row(b)
        got.append(out[b, :c].view(torch.int16).cpu().numpy().view(np.uint16))
        emu.append(e[:c])
        assert np.isfinite(ref.bf16_bits_to_f32(got[-1])).all(), "a NaN key past the window leaked"
    mism = _cmp("R-I7 window hazards", got, emu)
    assert mism == 0
    # an empty window writes nothing: a sentinel survives
    z = Case([0, 300], seed=6)
    o = torch.full((2, 300), 7.0, dtype=torch.bfloat16, device="cuda")
    z.decode(out=o)
    assert (o[0] == 7.0).all()
    _count("R-I7 window hazards", 300, 0)


def test_ri8_repeat_and_graph():
    case = Case([8192, 4096, 7777], seed=7)
    first = case.decode()
    bad = 0
    for _ in range(20):
        o = case.decode()
        for b, c in enumerate(case.ctxs):
            bad += int((o[b, :c].view(torch.int16) != first[b, :c].view(torch.int16)).sum())
    out = torch.empty_like(first)
    case.decode(out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case.decode(out=out)
    for _ in range(5):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        for b, c in enumerate(case.ctxs):
            bad += int((out[b, :c].view(torch.int16) != first[b, :c].view(torch.int16)).sum())
    _count("R-I8 repeat x20 + graph", 25 * sum(case.ctxs), bad)
    assert bad == 0
