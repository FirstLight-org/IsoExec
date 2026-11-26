"""Engine records (chunks anchored at absolute position 0): what rollout produces -- chunked prefill split
anywhere, one-token decode steps, any batch composition -- is BIT FOR BIT the trainer's single
whole-sequence forward, for every output row and for the final / continuation state."""

from __future__ import annotations

import random

import pytest
import torch

from isoexec.ops.kda.glm_sm100 import ops
from isoexec.ops.kda.glm_sm100.tests._fixtures import cu_of, make_inputs, require_sm100


@pytest.fixture(autouse=True)
def _gpu():
    require_sm100()


def _rollout(lens, H, seed, p_prefill, max_piece=80, state=None, slot_of=None, data_seed=None):
    """Drive every sequence to its end with random prefill pieces and decode steps over random subsets."""
    rng = random.Random(seed)
    q, k, v, g, b, cu, _ = make_inputs(lens, H=H, state=False, seed=seed if data_seed is None else data_seed)
    st = state or ops.KdaState(len(lens) + 3, H)
    slot_of = slot_of or {s: s + 1 for s in range(len(lens))}
    done, o = [0] * len(lens), torch.zeros_like(v)
    tok = lambda s, a, z, t: t[int(cu[s]) + a:int(cu[s]) + z]
    while any(done[s] < lens[s] for s in range(len(lens))):
        live = [s for s in range(len(lens)) if done[s] < lens[s]]
        rng.shuffle(live)
        live = live[:rng.randint(1, len(live))]
        if rng.random() < p_prefill:
            parts = []
            for s in live:
                n = rng.randint(1, min(max_piece, lens[s] - done[s]))
                parts.append((s, done[s], done[s] + n))
            cat = lambda t: torch.cat([tok(s, a, z, t) for s, a, z in parts])
            out = ops.prefill(cat(q), cat(k), cat(v), cat(g), cat(b), cu_of([z - a for _, a, z in parts]), st,
                              [slot_of[s] for s, _, _ in parts])
            off = 0
            for s, a, z in parts:
                o[int(cu[s]) + a:int(cu[s]) + z] = out[off:off + z - a]
                off += z - a
                done[s] = z
        else:
            cat = lambda t: torch.cat([tok(s, done[s], done[s] + 1, t) for s in live])
            out = ops.decode_step(cat(q), cat(k), cat(v), cat(g), cat(b), st, [slot_of[s] for s in live])
            for i, s in enumerate(live):
                o[int(cu[s]) + done[s]] = out[i]
                done[s] += 1
    return (q, k, v, g, b, cu), o, st, slot_of


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("p_prefill", [0.0, 0.3, 1.0])
def test_rollout_equals_whole_sequence_forward_bitwise(seed, p_prefill):
    lens = [1, 63, 64, 65, 130, 200][: 3 + seed % 4] if p_prefill > 0 else [1, 66, 130]
    inputs, o, st, slot_of = _rollout(lens, H=4, seed=seed, p_prefill=p_prefill)
    o_ref, ht_ref = ops.forward(*inputs, None, output_final_state=True)
    assert torch.equal(o, o_ref)
    hs = ops.record_state(st, [slot_of[s] for s in range(len(lens))])
    assert torch.equal(hs, ht_ref)


def test_records_left_by_different_splits_continue_identically():
    """Two rollouts of the same tokens, split differently, leave records that every continuation (decode
    steps and a prefill) treats identically, bit for bit."""
    lens, H = [150, 77], 4
    (q, k, v, g, b, cu), _, st1, slots = _rollout(lens, H=H, seed=3, p_prefill=1.0, data_seed=0)
    _, _, st2, _ = _rollout(lens, H=H, seed=4, p_prefill=0.5, data_seed=0)
    assert torch.equal(st1.positions(), st2.positions())
    assert torch.equal(ops.record_state(st1, [1, 2]), ops.record_state(st2, [1, 2]))
    x = make_inputs([40, 40], H=H, seed=21, state=False)
    for st in (st1, st2):
        outs = []
        for t in range(5):
            sl = lambda z: torch.stack([z[t], z[40 + t]])
            outs.append(ops.decode_step(*(sl(z) for z in x[:5]), st, [1, 2]))
        rest = lambda z: torch.cat([z[5:40], z[45:80]])
        outs.append(ops.prefill(*(rest(z) for z in x[:5]), cu_of([35, 35]), st, [1, 2]))
        st.result = (torch.cat(outs), ops.record_state(st, [1, 2]))
    assert torch.equal(st1.result[0], st2.result[0]) and torch.equal(st1.result[1], st2.result[1])


def test_initial_state_and_slot_moves():
    H, lens = 4, [100, 40]
    q, k, v, g, b, cu, h0 = make_inputs(lens, H=H, seed=9)
    o_ref, ht_ref = ops.forward(q, k, v, g, b, cu, h0, output_final_state=True)
    st = ops.KdaState(6, H)
    st.load_state(0, h0[0])
    st.load_state(1, h0[1])
    # sequence 0: prefill 37 tokens 0 -> 2 (a move), decode 5 steps in 2, prefill the rest 2 -> 3
    s0 = lambda a, z, t: t[a:z]
    o0 = [ops.prefill(*(s0(0, 37, t) for t in (q, k, v, g, b)), cu_of([37]), st, [0], [2])]
    for i in range(37, 42):
        o0.append(ops.decode_step(*(s0(i, i + 1, t) for t in (q, k, v, g, b)), st, [2]))
    o0.append(ops.prefill(*(s0(42, 100, t) for t in (q, k, v, g, b)), cu_of([58]), st, [2], [3]))
    # sequence 1: a zero-length move 1 -> 4, then the whole sequence from 4
    ops.prefill(q[:0], k[:0], v[:0], g[:0], b[:0], cu_of([0]), st, [1], [4])
    s1 = lambda t: t[100:140]
    o1 = ops.prefill(*(s1(t) for t in (q, k, v, g, b)), cu_of([40]), st, [4])
    assert torch.equal(torch.cat(o0 + [o1]), o_ref)
    assert torch.equal(ops.record_state(st, [3, 4]), ht_ref)


def test_null_decode_lanes_and_run_to_run():
    H, lens = 4, [70, 70]
    inputs, o1, st1, _ = _rollout(lens, H=H, seed=5, p_prefill=0.3)
    _, o2, st2, _ = _rollout(lens, H=H, seed=5, p_prefill=0.3)
    assert torch.equal(o1, o2) and torch.equal(st1.data, st2.data)
    q, k, v, g, b, _, _ = make_inputs([3], H=H, seed=11, state=False)
    before = st1.data.clone()
    out = ops.decode_step(q, k, v, g, b, st1, [-1, 1, -1])
    assert torch.equal(out[0], torch.zeros_like(out[0])) and torch.equal(out[2], torch.zeros_like(out[2]))
    changed = (st1.data != before).flatten(1).any(1).tolist()
    assert changed == [False, True, False, False, False]
