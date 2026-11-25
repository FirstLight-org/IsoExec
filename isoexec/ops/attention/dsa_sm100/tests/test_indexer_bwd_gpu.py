"""(lowest priority; the indexer is frozen in training): the indexer backward vs float64."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs an SM100 device")

from isoexec.ops.attention.dsa_sm100.indexer_bwd import dsa_indexer_bwd  # noqa: E402

_OUT = Path(__file__).resolve().parent / "battery_indexer_bwd.json"


def test_indexer_bwd_fp64():
    g = torch.Generator(device="cuda").manual_seed(1)
    T, H, D, S, K = 8, 64, 128, 4000, 2048
    q = (torch.randn((T, H, D), generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    k = (torch.randn((S, D), generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    w = torch.randn((T, H), generator=g, device="cuda")
    lists = torch.full((T, K), -1, dtype=torch.int32, device="cuda")
    for t in range(T):
        n = K - 100 * t
        lists[t, :n] = torch.randperm(S - 10, generator=g, device="cuda")[:n].sort().values.to(torch.int32)
    dsc = torch.randn((T, K), generator=g, device="cuda")
    dq, dw, dk = dsa_indexer_bwd(q, w, k, lists, dsc, chunk=3)
    q64 = q.double().requires_grad_(True); k64 = k.double().requires_grad_(True); w64 = w.double().requires_grad_(True)
    valid = lists >= 0
    kg = k64[lists.clamp(min=0).long()]
    s = (w64[:, :, None] * torch.relu(torch.einsum("thd,tkd->thk", q64, kg))).sum(1)
    (s * torch.where(valid, dsc.double(), 0.0)).sum().backward()
    rel = lambda a, b: float((a.double() - b).abs().max() / b.abs().max())
    e = {"dq": rel(dq, q64.grad), "dw": rel(dw, w64.grad), "dk": rel(dk, k64.grad)}
    sel = torch.zeros(S, dtype=torch.bool, device="cuda"); sel[lists[valid].long()] = True
    unsel_zero = bool((dk[~sel] == 0).all())
    _OUT.write_text(json.dumps({"relations": 2, "rel_err": e, "unselected_dk_zero": unsel_zero,
                                "failures": sum(v > 1e-2 for v in e.values()) + int(not unsel_zero)}, indent=1))
    assert all(v <= 1e-2 for v in e.values()) and unsel_zero, (e, unsel_zero)
