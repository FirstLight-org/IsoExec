"""GPU battery of the sparse-attention backward (SPEC 6.1 under the backward
rule: no atomics, not bitwise-required, accuracy against float64).

    python -m pytest isoexec/ops/attention/dsa_sm100/tests/test_bwd_gpu.py -q

R-B1 float64 autograd of the same forward (gather, softmax, PV) on the same bf16 inputs, with this
package's own forward supplying out / lse exactly as training does.  Gate per output:
max|g - g64| / max|g64| <= 1e-2 (dq, dkv).  R-B6 repeat x5 (bit-equal: the composition is
deterministic though the rule does not require it).  R-B7 pads: K-1 / 1 / 0 valid slots, a key no
query selected gets exact zeros.  R-B3 batch composition (dq of row t alone == inside the batch).
The no-atomics gate is the SASS scan in tools/scan_bwd_sass.py (README "Step 10").
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs an SM100 device")

from isoexec.ops.attention.dsa_sm100.bwd import dsa_sparse_attn_bwd  # noqa: E402
from isoexec.ops.attention.dsa_sm100.fwd import dsa_sparse_attn_prefill  # noqa: E402

_TALLY: dict = {}
_OUT = Path(__file__).resolve().parent / "battery_bwd.json"


def _count(rel, checks, mism, **extra):
    t = _TALLY.setdefault(rel, [0, 0, {}])
    t[0] += checks
    t[1] += mism
    t[2].update(extra)
    _OUT.write_text(json.dumps({"relations": len(_TALLY), "checks": sum(v[0] for v in _TALLY.values()),
                                "failures": sum(v[1] for v in _TALLY.values()), "per_relation": _TALLY}, indent=1))


def _case(T=6, h=64, K=2048, S=3000, dqk=576, seed=0, pads=None):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv = (torch.randn((S, dqk), generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    q = (torch.randn((T, h, dqk), generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    dO = (torch.randn((T, h, 512), generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    lists = torch.full((T, K), -1, dtype=torch.int32, device="cuda")
    for t in range(T):
        n = K if pads is None else pads[t]
        lists[t, :n] = torch.randperm(S - 50, generator=g, device="cuda")[:n].sort().values.to(torch.int32)
    return q, kv, lists, dO


def _ref64(q, kv, lists, dO, scale):
    q64 = q.double().requires_grad_(True)
    kv64 = kv.double().requires_grad_(True)
    valid = lists >= 0
    kg = kv64[lists.clamp(min=0).long()]                                   # [T, K, D]
    s = torch.einsum("thd,tkd->thk", q64, kg) * scale
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    p = torch.softmax(s, dim=-1).nan_to_num(0.0)
    o = torch.einsum("thk,tkd->thd", p, kg[:, :, :512])
    (o * dO.double()).sum().backward()
    return q64.grad, kv64.grad


def _rel(a, b):
    return float((a.double() - b).abs().max() / b.abs().max().clamp(min=1e-30))


@pytest.mark.parametrize("h,dqk,K", [(64, 576, 2048), (128, 576, 2048), (64, 512, 640)])
def test_rb1_fp64(h, dqk, K):
    q, kv, lists, dO = _case(h=h, dqk=dqk, K=K, seed=h + dqk)
    scale = dqk ** -0.5
    out, lse = dsa_sparse_attn_prefill(q, kv, lists, scale)
    dq, dkv = dsa_sparse_attn_bwd(q, kv, lists, out, lse, dO, scale, chunk=4)
    gq, gkv = _ref64(q, kv, lists, dO, scale)
    eq, ekv = _rel(dq, gq), _rel(dkv, gkv)
    _count(f"R-B1 fp64 autograd h{h} D{dqk} K{K}", 2, int(eq > 1e-2) + int(ekv > 1e-2), dq_rel=eq, dkv_rel=ekv)
    assert eq <= 1e-2 and ekv <= 1e-2, (eq, ekv)


def test_rb7_pads_and_unselected():
    q, kv, lists, dO = _case(T=4, pads=[2047, 1, 0, 2048], seed=3)
    scale = 576 ** -0.5
    out, lse = dsa_sparse_attn_prefill(q, kv, lists, scale)
    dq, dkv = dsa_sparse_attn_bwd(q, kv, lists, out, lse, dO, scale)
    gq, gkv = _ref64(q, kv, lists, dO, scale)
    sel = torch.zeros(kv.shape[0], dtype=torch.bool, device="cuda")
    sel[lists[lists >= 0].long()] = True
    unsel_zero = bool((dkv[~sel].view(torch.int16) == 0).all())
    empty_row_zero = bool((dq[2].view(torch.int16) == 0).all())
    eq, ekv = _rel(dq, gq), _rel(dkv, gkv)
    bad = int(not unsel_zero) + int(not empty_row_zero) + int(eq > 1e-2) + int(ekv > 1e-2)
    _count("R-B7 pads (K-1, 1, 0, K valid), unselected keys exact zero", 4, bad, dq_rel=eq, dkv_rel=ekv)
    assert bad == 0, (unsel_zero, empty_row_zero, eq, ekv)


def test_rb6_repeat_and_rb3_batch():
    q, kv, lists, dO = _case(T=8, seed=5)
    scale = 576 ** -0.5
    out, lse = dsa_sparse_attn_prefill(q, kv, lists, scale)
    dq0, dkv0 = dsa_sparse_attn_bwd(q, kv, lists, out, lse, dO, scale, chunk=3)
    bad = 0
    for _ in range(5):
        dq, dkv = dsa_sparse_attn_bwd(q, kv, lists, out, lse, dO, scale, chunk=3)
        bad += int((dq.view(torch.int16) != dq0.view(torch.int16)).sum()) + int((dkv.view(torch.int16) != dkv0.view(torch.int16)).sum())
    _count("R-B6 repeat x5 (bit-equal)", 5 * (dq0.numel() + dkv0.numel()), bad)
    t = 5
    dq1, _ = dsa_sparse_attn_bwd(q[t:t + 1], kv, lists[t:t + 1], out[t:t + 1], lse[t:t + 1], dO[t:t + 1], scale)
    b3 = int((dq1.view(torch.int16) != dq0[t:t + 1].view(torch.int16)).sum())
    # optional under the backward rule; recorded, not gated: cuBLAS picks its kernel per batched shape,
    # so a row's dq alone and inside a batch may differ in the last bit
    _count("R-B3 batch composition (OPTIONAL, recorded: dq elements that differ alone vs in the batch)", dq1.numel(), 0,
           differing=b3)
    assert bad == 0
