"""Decode reads nothing it must not: before every step, every record region outside the open chunk's
meaningful rows (k^ / X rows >= j, v / delta columns >= j, P rows and columns outside the open block,
G / beta >= j) is overwritten with NaN, and the pool starts NaN-filled. A step that consumed any of
those bytes as data (instead of zero-filling what it uses) would leak NaN. Decode must still equal
prefill bit for bit, under every execution plan.

Claim map: tests/README.md.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.gdn.gdn_cpr_sm90 import ops as G
from isoexec.ops.gdn.gdn_cpr_sm90.tests._fixtures import (
    DECODE_PLANS,
    DEV,
    SM90,
    fresh,
    ids,
    make,
    prefill,
    rows,
    state_equal,
)

pytestmark = SM90

LANES, HK, HV = [61, 130, 7, 64, 65], 2, 4
NAN16, NAN32 = 0x7FC1, 0x7FC00001


@pytest.fixture(scope="module")
def ref():
    d = make(LANES, HK, HV, seed=3)
    st = fresh(len(LANES) + 1, HV)
    o = prefill(d, state=st, slot_in=ids(range(1, len(LANES) + 1)))
    return d, o, st


def _masks(j):
    """int index tensors of the record elements a step at row j must not read as data."""
    kb = G._kb_index().view(64, 128)[j:].reshape(-1)
    cols = G._cols_index().view(128, 64)[:, j:].reshape(-1)
    return kb.to(DEV), cols.to(DEV)


def _poison(pool, slot, j):
    L = G.record_layout()
    rec = pool.data[slot]   # [hv, bytes]
    kb, cols = _masks(j)
    rec[:, L["kb"]:L["kb"] + 64 * 128 * 2].view(torch.int16)[:, kb] = NAN16
    for nm in ("vb", "db"):
        rec[:, L[nm]:L[nm] + 64 * 128 * 2].view(torch.int16)[:, cols] = NAN16
    rec[:, L["xb"]:L["xb"] + 64 * 64 * 4].view(torch.int32).view(-1, 64, 64)[:, j:] = NAN32
    pb = rec[:, L["pb"]:L["pb"] + 16 * 64 * 4].view(torch.int32).view(-1, 16, 64)
    pb[:, (j % 16 if j >= 16 else 0):] = NAN32
    pb[:, :, 16 * (j // 16):] = NAN32
    for nm in ("gb", "bb"):
        rec[:, L[nm]:L[nm] + 256].view(torch.int32)[:, j:] = NAN32


@pytest.mark.parametrize("split", DECODE_PLANS)
def test_decode_reads_only_meaningful_rows(ref, split):
    d, o, st_p = ref
    n = len(LANES)
    st = G.CprState.allocate(n + 1, HV, DEV, zero=False)
    st.data.view(torch.int16)[:] = NAN16   # a poisoned pool
    st.reset(torch.arange(n + 1, device=DEV))
    od = torch.zeros_like(o)
    for t in range(max(LANES)):
        idx = [i for i in range(n) if t < LANES[i]]
        for i in idx:
            _poison(st, i + 1, t % 64)
        r = torch.tensor([d["cu"][i] + t for i in idx], device=DEV)
        od[r] = G.decode(*rows(d, r), d["A_log"], d["dt_bias"], st, ids(i + 1 for i in idx), split=split)
    assert not torch.isnan(od.float()).any()
    assert bitwise_equal(od, o)
    assert state_equal(st, st_p, range(1, n + 1))
