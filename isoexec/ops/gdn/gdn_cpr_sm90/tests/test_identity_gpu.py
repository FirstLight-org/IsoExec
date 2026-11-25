"""One identity across sites: trainer forward == one-shot prefill == prefill + decode steps, under every
decode execution plan; and the identity is the gated delta rule (float64 oracle, with tolerances).

Every equality claim is compared on integer views (``ops._testing.bitwise_equal``). Claim map:
tests/README.md.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.gdn.gdn_cpr_sm90 import ops as G
from isoexec.ops.gdn.gdn_cpr_sm90.integration import gdn_cpr_sm90_core
from isoexec.ops.gdn.gdn_cpr_sm90.reference import gdn_reference, materialize
from isoexec.ops.gdn.gdn_cpr_sm90.tests._fixtures import (
    DECODE_PLANS,
    DEV,
    SM90,
    decode_stream,
    fresh,
    ids,
    make,
    prefill,
    rel_err,
    state_equal,
)

pytestmark = SM90

LANES, HK, HV = [70, 130, 5, 64], 2, 4   # crosses boundaries, ends mid-chunk, ends exactly on one


@pytest.fixture(scope="module")
def pack():
    d = make(LANES, HK, HV, seed=1)
    st = fresh(len(LANES) + 1, HV)
    o = prefill(d, state=st, slot_in=ids(range(1, len(LANES) + 1)))
    return d, o, st


def test_trainer_forward_is_one_shot_prefill(pack):
    d, o_state, _ = pack
    o = prefill(d)   # stateless
    assert bitwise_equal(o, o_state), "stateless prefill != stateful prefill from fresh slots"
    names = ("q", "k", "v", "a", "b", "A_log", "dt_bias")
    with torch.no_grad():
        o_ng = gdn_cpr_sm90_core(*(d[n] for n in names), d["cut"])
    leaves = [d[n].detach().clone().requires_grad_(True) for n in names]
    o_g = gdn_cpr_sm90_core(*leaves, d["cut"])
    assert bitwise_equal(o_ng, o) and bitwise_equal(o_g.detach(), o)


@pytest.mark.parametrize("split", DECODE_PLANS)
def test_decode_stream_is_prefill(pack, split):
    """Every token by decode (every boundary resync inside the launch) == one prefill, output and state."""
    d, o, st_p = pack
    st = fresh(len(LANES) + 1, HV)
    od = decode_stream(d, LANES, st, plan=split)
    assert bitwise_equal(od, o)
    assert state_equal(st, st_p, range(1, len(LANES) + 1))
    assert bitwise_equal(st.sh, st.S.bfloat16()), "S^ must be rn(S)"


@pytest.mark.parametrize("head", [1, 37, 63, 64, 65])
def test_prefill_then_decode_is_prefill(pack, head):
    """A prefill of the first ``head`` tokens, then decode steps (plans changing every step) == one prefill."""
    d, o, st_p = pack
    lanes = LANES
    n = len(lanes)
    heads = [min(head, L) for L in lanes]
    st = fresh(n + 1, HV)
    parts = torch.zeros_like(o)
    cu = [0]
    sel = []
    for i, h in enumerate(heads):
        cu.append(cu[-1] + h)
        sel += list(range(d["cu"][i], d["cu"][i] + h))
    sel = torch.tensor(sel, device=DEV)
    parts[sel] = prefill(d, sel, cu=cu, state=st, slot_in=ids(range(1, n + 1)))
    plans = DECODE_PLANS
    od = decode_stream(d, lanes, st, start=heads, plan=lambda t: plans[(t * 5 + head) % len(plans)])
    rest = torch.ones(sum(lanes), dtype=torch.bool, device=DEV)
    rest[sel] = False
    parts[rest] = od[rest]
    assert bitwise_equal(parts, o)
    assert state_equal(st, st_p, range(1, n + 1))


def test_load_initial_prefill_decode_and_oracle():
    lanes = [70, 9]
    d = make(lanes, HK, HV, seed=5)
    S0 = torch.randn(len(lanes), HV, 128, 128, device=DEV) * 0.05
    sl = ids(range(1, len(lanes) + 1))
    st_p, st_d = fresh(3, HV), fresh(3, HV)
    st_p.load_initial(sl, S0)
    st_d.load_initial(sl, S0)
    o = prefill(d, state=st_p, slot_in=sl)
    od = decode_stream(d, lanes, st_d)
    assert bitwise_equal(od, o) and state_equal(st_d, st_p, [1, 2])
    ro, rs = gdn_reference(d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"], d["cu"],
                           initial_state=S0)
    assert rel_err(o, ro)[0] <= 1.5e-2
    assert rel_err(materialize(st_p, sl), rs)[0] <= 1.5e-2


@pytest.mark.parametrize("hk,hv", [(2, 4), (3, 6), (1, 5)])
def test_matches_float64_oracle(hk, hv):
    """Tolerance, not bits: the oracle is the function, not the rounding schedule (FLA lands at ~6e-3)."""
    lanes = [61, 128, 7]
    d = make(lanes, hk, hv, seed=6)
    st = fresh(len(lanes) + 1, hv)
    sl = ids(range(1, len(lanes) + 1))
    o = prefill(d, state=st, slot_in=sl)
    ro, rs = gdn_reference(d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"], d["cu"])
    mx, l2 = rel_err(o, ro)
    assert mx <= 1.5e-2 and l2 <= 1e-2, (mx, l2)
    smx, sl2 = rel_err(materialize(st, sl), rs)
    assert smx <= 1.5e-2 and sl2 <= 1e-2, (smx, sl2)
    assert torch.isfinite(o).all()


@pytest.mark.two_pass_only
@pytest.mark.parametrize("hv", [4, 6])
def test_sweep_width_is_execution_only(hv):
    """The prefill sweep's value rows per CTA (16 or 32) and who runs the sweep roles (an SM's first CTA,
    or only the CTAs whose outputs need them -- the claim fallback) are execution choices: outputs and
    records are bitwise identical, for fresh and continuing sequences (a mid-chunk start, slot moves and
    an empty sequence that only moves its slot included)."""
    lanes = [61, 130, 7, 200, 64, 0]
    d = make(lanes, 2, hv, seed=41)
    n = len(lanes)
    one, two = ids(range(1, n + 1)), ids(range(n + 1, 2 * n + 1))
    results = []
    for rows, on_demand in ((16, False), (32, False), (16, True), (32, True)):
        st = fresh(2 * n + 1, hv)
        first = [L // 3 for L in lanes]
        cu1 = [0]
        for L in first:
            cu1.append(cu1[-1] + L)
        sl1 = torch.cat([torch.arange(d["cu"][i], d["cu"][i] + first[i]) for i in range(n)]).to(DEV)
        sl2 = torch.cat([torch.arange(d["cu"][i] + first[i], d["cu"][i + 1]) for i in range(n)]).to(DEV)
        cu2 = [0]
        for i, L in enumerate(lanes):
            cu2.append(cu2[-1] + L - first[i])
        with G._sweep_schedule(on_demand=on_demand):
            o1 = prefill(d, sl1, cu=cu1, state=st, slot_in=one, sweep_rows=rows)
            o2 = prefill(d, sl2, cu=cu2, state=st, slot_in=one, slot_out=two, sweep_rows=rows)
        results.append((o1, o2, st))
    a1, a2, sa = results[0]
    for b1, b2, sb in results[1:]:
        assert bitwise_equal(a1, b1) and bitwise_equal(a2, b2)
        assert state_equal(sa, sb, list(range(n + 1, 2 * n + 1)))


@pytest.mark.parametrize("rows", [16, 32])
def test_prefill_is_deterministic_at_scale(rows):
    """A trainer-sized pack (hundreds of pass-1 CTAs, every sweep width) gives the same bits run after run,
    each sequence equals its own single-sequence prefill (an independent packing and schedule), and the first
    sequence equals the decode stream over its first 200 tokens (catches races that small packs hide)."""
    lanes, hv = [1100, 700, 900, 300], 8
    d = make(lanes, 4, hv, seed=43)
    outs = [prefill(d, sweep_rows=rows) for _ in range(4)]
    for o in outs[1:]:
        assert bitwise_equal(o, outs[0])
    for n, L in enumerate(lanes):
        lo = d["cu"][n]
        alone = prefill(d, torch.arange(lo, lo + L, device=DEV), cu=[0, L])
        assert bitwise_equal(alone, outs[0][lo:lo + L]), f"sequence {n}"
    st = fresh(2, hv)
    L0 = 200
    od = decode_stream(d, [L0] + [0] * (len(lanes) - 1), st)
    assert bitwise_equal(od[:L0], outs[0][:L0])
