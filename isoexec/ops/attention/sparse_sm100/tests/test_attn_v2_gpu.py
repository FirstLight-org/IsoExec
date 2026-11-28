"""Recipe v2 (``sm100_sparse_v2``: v1 + the lazy rescale at 8 log2 units) -- the relations that
only v2 needs.  The full R-A battery runs on v2 with ``SPARSE_TEST_RECIPE=2`` (test_attn_gpu.py,
test_attn_groups_gpu.py, test_attn_prefill_gpu.py, against the lazy emulation).

* ``lazy_rescale_branches`` (mla_sm100 round 5's witness): rows whose per-tile maximum is controlled
  (q = e_0, so a score is key dim 0 exactly, bf16 kv so the conversion is the identity).  Every tile's
  max climbing by 12 log2 units -> v2 commits every tile and must be bitwise equal to v1; a falling
  max -> both keep, equal; climbing by 4 -> v2 keeps and must DIFFER (the lazy branch ran).
* fp64 accuracy of v2 next to v1's on the same rows.
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
from isoexec.ops.attention.sparse_sm100.fwd import sparse_attn_prefill  # noqa: E402
from isoexec.ops.attention.sparse_sm100.tests._prims import DevicePrims  # noqa: E402

_TALLY: dict = {}
_OUT = Path(__file__).resolve().parent / "battery_attn_v2_only.json"
D = 512


def _count(rel, checks, mism):
    t = _TALLY.setdefault(rel, [0, 0])
    t[0] += checks
    t[1] += mism
    _OUT.write_text(json.dumps({"relations": len(_TALLY), "checks": sum(v[0] for v in _TALLY.values()),
                                "mismatches": sum(v[1] for v in _TALLY.values()), "per_relation": _TALLY},
                               indent=1))


def _controlled(step_log2, T=4, K=2048, h=64, scale=1.0):
    """kv rows whose dim 0 = tile * step / scale_log2 - small spread; q = e_0 for every head."""
    sl = scale * math.log2(math.e)
    tiles = K // 64
    base = torch.arange(tiles, dtype=torch.float32).repeat_interleave(64) * (step_log2 / sl)
    spread = -torch.arange(64, dtype=torch.float32).repeat(tiles) * 0.03125
    kv = torch.zeros((K, D), dtype=torch.float32)
    kv[:, 0] = base + spread + (0 if step_log2 >= 0 else -float(base.min()))
    kv[:, 1:] = torch.randn((K, D - 1), generator=torch.Generator().manual_seed(1)) * 0.1
    kv = kv.to(torch.bfloat16).cuda()
    q = torch.zeros((T, h, D), dtype=torch.bfloat16, device="cuda")
    q[:, :, 0] = 1.0
    lists = torch.arange(K, dtype=torch.int32, device="cuda").repeat(T, 1).contiguous()
    return q, kv, lists, scale


@pytest.mark.parametrize("step,expect_equal", [(12.0, True), (-12.0, True), (4.0, False)])
def test_lazy_rescale_branches(step, expect_equal):
    q, kv, lists, scale = _controlled(step)
    o1, l1 = sparse_attn_prefill(q, kv, lists, scale, recipe=1)
    o2, l2 = sparse_attn_prefill(q, kv, lists, scale, recipe=2)
    diff = int((o1.view(torch.int16) != o2.view(torch.int16)).sum()) + int((l1.view(torch.int32) != l2.view(torch.int32)).sum())
    # and v2 is bit-exact against the lazy emulation on one row either way
    rows = kv.float().cpu().numpy()
    valid = np.ones(rows.shape[0], bool)
    eo, el, _ = ref.sparse_attn_emul(q[0].float().cpu().numpy(), rows, valid, scale, groups=1,
                                     prims=DevicePrims, lazy=True)
    go = o2[0].contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
    emis = int((go != eo).sum()) + int((l2[0].cpu().numpy().view(np.uint32) != el.view(np.uint32)).sum())
    ok = (diff == 0) == expect_equal
    _count(f"lazy_rescale_branches step {step:+g} ({'equal' if expect_equal else 'differ'})", 1, int(not ok))
    _count("R-A2 v2 bit-exact vs the lazy emulation (controlled rows)", go.size + 64, emis)
    assert ok, (step, diff)
    assert emis == 0


def test_v2_fp64_accuracy():
    g = torch.Generator(device="cuda").manual_seed(5)
    T, K, S = 6, 2048, 9000
    kv = (torch.randn((S, D), generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    q = (torch.randn((T, 64, D), generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    lists = torch.stack([torch.randperm(S, generator=g, device="cuda")[:K].sort().values for _ in range(T)]).to(torch.int32)
    scale = 1 / math.sqrt(512)
    worst = {}
    for rec in (1, 2):
        out, lse = sparse_attn_prefill(q, kv, lists, scale, recipe=rec)
        w = 0.0
        for t in range(T):
            rows = kv[lists[t].long()].float().cpu().numpy()
            o64, l64 = ref.sparse_attn_fp64(q[t].float().cpu().numpy(), rows, np.ones(K, bool), scale)
            w = max(w, float(np.abs(out[t].float().cpu().numpy() - o64).max()))
        worst[rec] = w
    _count(f"v2 fp64 accuracy (worst |out - fp64| v1 {worst[1]:.2e}, v2 {worst[2]:.2e})", 1, int(worst[2] > 2e-2))
    assert worst[2] < 2e-2, worst
