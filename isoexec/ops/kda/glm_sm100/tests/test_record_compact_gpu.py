"""The compact engine record (state planes SH / SL / SC, csrc/record_state.cuh): the kernels read and write the
planes exactly as record_layout.py says, and the fp32 state they hold is the one the recipe produced -- bit for
bit, across chunk closes, from prefill and from decode, with rn ties in the state."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops.kda.glm_sm100 import ops, record_layout as R
from isoexec.ops.kda.glm_sm100.tests._fixtures import cu_of, make_inputs, require_sm100


@pytest.fixture(autouse=True)
def _gpu():
    require_sm100()


def planes_state(st, slot):
    """The fp32 boundary state S_c the slot's planes hold, [H, 128, 128]."""
    d, L = st.data[slot], R.REC
    return R.unpack_planes(d[:, L["SH"]:L["SH"] + R.SH_BYTES], d[:, L["SL"]:L["SL"] + R.SL_BYTES],
                           d[:, L["SC"]:L["SC"] + R.SC_BYTES])


def boundary_state(x, n, H):
    """forward's final state of the first 64 * (n // 64) tokens: S_c of a sequence at position n."""
    q, k, v, g, b = (t[:64 * (n // 64)] for t in x)
    m = q.shape[0]
    if m == 0:
        return torch.zeros(1, H, 128, 128, device=q.device)
    return ops.forward(q, k, v, g, b, cu_of([m]), None, output_final_state=True)[1]


def test_record_size_matches_the_layout_mirror():
    assert ops.extension().rec_bytes() == R.REC_BYTES


@pytest.mark.parametrize("n", [0, 1, 63, 64, 65, 127, 128, 190, 256])
def test_prefill_leaves_the_boundary_state_in_the_planes(n):
    H = 4
    x = make_inputs([max(n, 1)], H=H, state=False, seed=n)[:5]
    st = ops.KdaState(3, H)
    if n:
        ops.prefill(*(t[:n] for t in x), cu_of([n]), st, [1])
    assert torch.equal(planes_state(st, 1) if n else planes_state(st, 1), boundary_state(x, n, H)[0])
    if n % 64 == 0:   # at a boundary the record IS the forward's final state
        assert torch.equal(ops.record_state(st, [1])[0], planes_state(st, 1))


@pytest.mark.parametrize("start", [1, 31, 60, 63, 64, 100])
def test_decode_closes_write_the_planes(start):
    H, steps = 4, 140
    x = make_inputs([start + steps], H=H, state=False, seed=7)[:5]
    st = ops.KdaState(3, H)
    ops.prefill(*(t[:start] for t in x), cu_of([start]), st, [1])
    for i in range(start, start + steps):
        ops.decode_step(*(t[i:i + 1] for t in x), st, [1])
        if (i + 1) % 64 == 0 or i + 1 in (start + 1, start + 70):
            assert torch.equal(planes_state(st, 1), boundary_state(x, i + 1, H)[0]), i + 1
            if (i + 1) % 64 == 0:
                assert torch.equal(ops.record_state(st, [1])[0], planes_state(st, 1))


def _tied(h0):
    """h0 with every fourth element's low half forced to the rn tie 0x8000 (both parities of the top half occur)."""
    bits = h0.view(torch.int32).clone()
    mask = torch.zeros_like(bits, dtype=torch.bool)
    mask.view(-1)[::4] = True
    bits = torch.where(mask, (bits & ~0xFFFF) | 0x8000, bits)
    return bits.view(torch.float32)


def test_initial_state_with_rn_ties_roundtrips_and_rolls_out():
    H, lens = 4, [150, 70]
    q, k, v, g, b, cu, h0 = make_inputs(lens, H=H, seed=13, state_scale=0.3)
    h0 = _tied(h0)
    o_ref, ht_ref = ops.forward(q, k, v, g, b, cu, h0, output_final_state=True)
    st = ops.KdaState(5, H)
    st.load_state(0, h0[0])
    st.load_state(1, h0[1])
    assert torch.equal(ops.record_state(st, [0, 1]), h0)                      # kernel read of host-written planes
    assert torch.equal(planes_state(st, 0), h0[0])
    o = [ops.prefill(q[:50], k[:50], v[:50], g[:50], b[:50], cu_of([50]), st, [0])]
    for i in range(50, 150):                                                    # decode through two closes
        o.append(ops.decode_step(q[i:i + 1], k[i:i + 1], v[i:i + 1], g[i:i + 1], b[i:i + 1], st, [0]))
    o1 = ops.prefill(q[150:], k[150:], v[150:], g[150:], b[150:], cu_of([70]), st, [1])
    assert torch.equal(torch.cat(o + [o1]), o_ref)
    assert torch.equal(ops.record_state(st, [0, 1]), ht_ref)


@pytest.mark.parametrize("n", [37, 64, 100])
def test_zero_token_move_keeps_the_position(n):
    """A record moved without tokens continues exactly as if it had not moved (POS is part of the copy)."""
    H, total = 4, n + 30
    x = make_inputs([total], H=H, state=False, seed=17)[:5]
    o_ref, ht_ref = ops.forward(*x, cu_of([total]), None, output_final_state=True)
    st = ops.KdaState(5, H)
    o = [ops.prefill(*(t[:n] for t in x), cu_of([n]), st, [1])]
    ops.prefill(*(t[:0] for t in x), cu_of([0]), st, [1], [3])
    assert st.positions()[3].item() == n
    o.append(ops.prefill(*(t[n:] for t in x), cu_of([total - n]), st, [3]))
    assert torch.equal(torch.cat(o), o_ref)
    assert torch.equal(ops.record_state(st, [3]), ht_ref)


def test_fresh_slot_is_the_zero_state():
    st = ops.KdaState(2, 4)
    assert not ops.record_state(st, [0, 1]).view(torch.int32).any()
