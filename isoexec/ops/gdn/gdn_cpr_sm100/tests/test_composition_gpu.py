"""Graph capture, replay and repeated launches: the execution plan is not in the bits."""

from __future__ import annotations

import torch

from isoexec.ops.gdn.gdn_cpr_sm100 import ops as G

from ._fixtures import SM100, biteq, fresh, ids, make, prefill, rows, state_equal

pytestmark = SM100


def test_decode_graph_replay_equals_eager():
    d = make([1] * 4, 4, 8, seed=37)
    slots = ids([1, 2, 3, 4])
    q, k, v, a, b = rows(d, slice(None))
    eager_state = fresh(6, 8)
    eager_out = torch.empty(4, 8, 128, device="cuda", dtype=torch.bfloat16)
    G.decode(q, k, v, a, b, d["A_log"], d["dt_bias"], eager_state, slots, out=eager_out)

    graph_state = fresh(6, 8)
    buf = torch.empty_like(eager_out)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            G.decode(q, k, v, a, b, d["A_log"], d["dt_bias"], graph_state, slots, out=buf)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    graph_state.reset(slots)
    with torch.cuda.graph(graph):
        G.decode(q, k, v, a, b, d["A_log"], d["dt_bias"], graph_state, slots, out=buf)
    graph_state.reset(slots)
    graph.replay()
    torch.cuda.synchronize()
    assert biteq(eager_out, buf)
    assert state_equal(eager_state, graph_state, slots)


def test_graph_replayed_repeatedly_advances_like_eager_steps():
    """Three captured decode steps over a chunk boundary, against three eager steps."""
    d = make([3], 4, 8, seed=53)
    slots = ids([1])
    st_e = fresh(2, 8)
    st_e.pos[slots.long()] = 62   # the second captured step crosses row 63
    st_g = fresh(2, 8)
    st_g.pos[slots.long()] = 62
    q, k, v, a, b = rows(d, slice(None))
    outs = []
    for t in range(3):
        outs.append(G.decode(q[t:t + 1], k[t:t + 1], v[t:t + 1], a[t:t + 1], b[t:t + 1], d["A_log"],
                             d["dt_bias"], st_e, slots).clone())
    qb, kb, vb, ab, bb = (x[:1].clone() for x in (q, k, v, a, b))
    buf = torch.empty(1, 8, 128, device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            G.decode(qb, kb, vb, ab, bb, d["A_log"], d["dt_bias"], fresh(2, 8), slots, out=buf)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        G.decode(qb, kb, vb, ab, bb, d["A_log"], d["dt_bias"], st_g, slots, out=buf)
    st_g.pos[slots.long()] = 62
    st_g.S[slots.long()] = 0.0
    st_g.set_sh(slots.long(), torch.zeros(1, 8, 128, 128, device="cuda", dtype=torch.bfloat16))
    for t in range(3):
        qb.copy_(q[t:t + 1]); kb.copy_(k[t:t + 1]); vb.copy_(v[t:t + 1])
        ab.copy_(a[t:t + 1]); bb.copy_(b[t:t + 1])
        graph.replay()
        torch.cuda.synchronize()
        assert biteq(outs[t], buf), t
    assert state_equal(st_e, st_g, slots)


def test_prefill_launched_beside_another_prefill_is_unchanged():
    """Another launch competing for SMs cannot change the bits: no CTA waits on another."""
    d = make([200, 64, 13], 4, 8, seed=59)
    want = prefill(d)
    noise = make([4096], 16, 32, seed=61)
    for _ in range(3):
        prefill(noise)
        got = prefill(d)
        assert biteq(want, got)


def test_the_recurrent_decode_kernel_is_three_ctas_per_sm():
    """The shipped decode step's occupancy, from the driver rather than from the arithmetic."""
    lay = G.record_layout()
    assert lay["decode_rec_smem"] <= 74 * 1024, lay["decode_rec_smem"]
    assert lay["decode_rec_blocks_per_sm"] >= 3, lay["decode_rec_blocks_per_sm"]
    assert G._decode_blocks_per_sm() == lay["decode_rec_blocks_per_sm"]


def test_launch_a_of_the_prepared_mode_is_two_ctas_per_sm():
    """The occupancy this pass went after, asserted against the hardware and not against arithmetic."""
    lay = G.record_layout()
    assert lay["prep_a_smem"] == 112896, lay["prep_a_smem"]
    assert lay["prep_a_smem"] * 2 <= torch.cuda.get_device_properties(0).shared_memory_per_multiprocessor
    assert lay["prep_a_blocks_per_sm"] == 2, lay["prep_a_blocks_per_sm"]
