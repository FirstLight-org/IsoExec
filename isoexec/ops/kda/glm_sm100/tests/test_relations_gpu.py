"""Determinism and numerical relations of recipe sm100_kda_v1 (README.md, "Relations").

Bitwise: run-to-run (forward, decode, backward); batch invariance (a sequence's outputs and final state do
not depend on what else is in the call, where, or how many CTAs run); segmented == whole when every split
is a multiple of the 64-token chunk from the sequence start; decode == forward of one-token sequences.
Within tolerance (a different program by declaration): splits that are not chunk-aligned, and N decode
steps against one N-token prefill."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops.kda.glm_sm100 import ops, reference as R
from isoexec.ops.kda.tests._fixtures import assert_close, cu_of, make_inputs, require_sm100


@pytest.fixture(autouse=True)
def _gpu():
    require_sm100()


def _rows(t, cu, s):
    return t[int(cu[s]):int(cu[s + 1])]


def test_forward_and_backward_are_run_to_run_bitwise():
    q, k, v, g, b, cu, h0 = make_inputs([300, 64, 17], H=4)
    first = ops.forward(q, k, v, g, b, cu, h0, output_final_state=True)
    gen = torch.Generator(device="cuda").manual_seed(6)
    do = torch.randn(q.shape, device="cuda", generator=gen).bfloat16()
    dht = torch.randn(h0.shape, device="cuda", generator=gen)
    grads = ops.backward(q, k, v, g, b, cu, h0, do, dht)
    for _ in range(3):
        again = ops.forward(q, k, v, g, b, cu, h0, output_final_state=True)
        assert all(torch.equal(x, y) for x, y in zip(first, again))
        assert all(torch.equal(x, y) for x, y in zip(grads, ops.backward(q, k, v, g, b, cu, h0, do, dht)))


@pytest.mark.parametrize("H", [4, 64])
def test_a_sequence_is_bitwise_independent_of_its_batch(H):
    lens = [150, 64, 1, 77]
    q, k, v, g, b, cu, h0 = make_inputs(lens, H=H)
    o, ht = ops.forward(q, k, v, g, b, cu, h0, output_final_state=True)
    for s in range(len(lens)):
        cs = cu_of([lens[s]])
        sl = lambda t: _rows(t, cu, s)
        oa, ha = ops.forward(sl(q), sl(k), sl(v), sl(g), sl(b), cs, h0[s:s + 1], output_final_state=True)
        assert torch.equal(oa, sl(o)) and torch.equal(ha[0], ht[s]), s
    # a different composition and order: sequence 0 last, behind 33 copies of sequence 2
    order = [2] * 33 + [3, 1, 0]
    cat = lambda t: torch.cat([_rows(t, cu, s) for s in order])
    cu2 = cu_of([lens[s] for s in order])
    o2, h2 = ops.forward(cat(q), cat(k), cat(v), cat(g), cat(b), cu2, h0[order], output_final_state=True)
    for pos, s in enumerate(order):
        assert torch.equal(_rows(o2, cu2, pos), _rows(o, cu, s)) and torch.equal(h2[pos], ht[s]), (pos, s)


@pytest.mark.parametrize("splits", [[128, 72], [64, 64, 64, 8], [192, 8]])
def test_chunk_aligned_continuation_equals_whole_bitwise(splits):
    q, k, v, g, b, cu, h0 = make_inputs([sum(splits)], H=4)
    o, ht = ops.forward(q, k, v, g, b, cu, h0, output_final_state=True)
    state, t0, outs = h0, 0, []
    for n in splits:
        sl = lambda t: t[t0:t0 + n]
        oo, state = ops.forward(sl(q), sl(k), sl(v), sl(g), sl(b), cu_of([n]), state, output_final_state=True)
        outs.append(oo)
        t0 += n
    assert torch.equal(torch.cat(outs), o) and torch.equal(state, ht)


def test_unaligned_continuation_is_within_tolerance():
    splits = [100, 37, 63]
    q, k, v, g, b, cu, h0 = make_inputs([sum(splits)], H=4)
    o, ht = ops.forward(q, k, v, g, b, cu, h0, output_final_state=True)
    ro, rh = R.recurrence(q, k, v, g, b, cu, h0)
    state, t0, outs = h0, 0, []
    for n in splits:
        sl = lambda t: t[t0:t0 + n]
        oo, state = ops.forward(sl(q), sl(k), sl(v), sl(g), sl(b), cu_of([n]), state, output_final_state=True)
        outs.append(oo)
        t0 += n
    seg = torch.cat(outs)
    assert_close(seg, ro, 1e-2, 2.5e-2, "segmented vs oracle")
    assert_close(state, rh, 1e-2, 2.5e-2, "segmented state vs oracle")
    assert_close(seg, o, 1e-2, 2.5e-2, "segmented vs whole")


@pytest.mark.parametrize("with_state", [True, False])
@pytest.mark.parametrize("N,H", [(5, 4), (300, 64)])
def test_decode_equals_one_token_forward_bitwise(N, H, with_state):
    q, k, v, g, b, cu, h0 = make_inputs([1] * N, H=H, state=with_state, state_scale=0.3)
    of, hf = ops.forward(q, k, v, g, b, cu, h0, output_final_state=True)
    od, hd = ops.decode(q, k, v, g, b, h0)
    assert torch.equal(of, od) and torch.equal(hf, hd)
    if h0 is not None:   # in place
        h2 = h0.clone()
        od2, _ = ops.decode(q, k, v, g, b, h2, out_state=h2)
        assert torch.equal(od2, od) and torch.equal(h2, hd)
    # a row does not depend on the decode batch
    o1, h1 = ops.decode(q[2:3], k[2:3], v[2:3], g[2:3], b[2:3], h0[2:3] if h0 is not None else None)
    assert torch.equal(o1[0], od[2]) and torch.equal(h1[0], hd[2])


def test_decode_is_run_to_run_bitwise_and_a_stream_tracks_prefill():
    steps, H = 40, 4
    q, k, v, g, b, cu, h0 = make_inputs([steps], H=H)
    o, ht = ops.forward(q, k, v, g, b, cu, h0, output_final_state=True)
    state, outs = h0.clone(), []
    for t in range(steps):
        sl = lambda x: x[t:t + 1]
        od, state = ops.decode(sl(q), sl(k), sl(v), sl(g), sl(b), state)
        outs.append(od)
    stream = torch.cat(outs)
    state2 = h0.clone()
    for t in range(steps):
        sl = lambda x: x[t:t + 1]
        _, state2 = ops.decode(sl(q), sl(k), sl(v), sl(g), sl(b), state2)
    assert torch.equal(state, state2)
    ro, rh = R.recurrence(q, k, v, g, b, cu, h0)
    assert_close(stream, ro, 1e-2, 2.5e-2, "decode stream vs oracle")
    assert_close(state, rh, 1e-2, 2.5e-2, "decode state vs oracle")
    assert_close(stream, o, 1e-2, 2.5e-2, "decode stream vs prefill")
