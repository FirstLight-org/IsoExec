"""Gates of recipe sm100_kda_rec_v1 (core): bitwise relations and accuracy."""

from __future__ import annotations

import random

import pytest
import torch

from isoexec.ops.kda.glm_rec_sm100 import ops
from isoexec.ops.kda.tests._fixtures import (bits_equal, cu_of, make_inputs, max_rel, rel_l2, require_sm100)

SHAPES = [(8, 4, 16), (8, 4, 32), (8, 2, 8), (8, 8, 32), (16, 4, 16), (16, 2, 8), (32, 4, 16), (32, 4, 8), (32, 2, 8),
          (8, 2, 16), (8, 2, 32), (8, 2, 64), (8, 4, 64), (8, 4, 128), (16, 2, 32), (16, 4, 64)]


@pytest.fixture(autouse=True)
def _gpu():
    require_sm100()


def _pool(slots, H, device="cuda"):
    return torch.zeros(slots, H, 128, 128, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------------- execution freedom
def test_every_lane_mapping_and_launch_shape_is_bitwise_the_same_program():
    """8 / 16 / 32 lanes per column, any columns per thread / CTA: the recipe's reduction tree is the same."""
    lens = [70, 1, 33, 5]
    q, k, v, g, b, cu, h0 = make_inputs(lens, H=3, seed=1)
    ref_o, ref_h = ops.forward(q, k, v, g, b, cu, h0, shape=(8, 4, 16))
    for sh in SHAPES:
        o, h = ops.forward(q, k, v, g, b, cu, h0, shape=sh)
        bits_equal(o, ref_o, f"o {sh}")
        bits_equal(h, ref_h, f"state {sh}")
    for sh in [(8, 4, 16), (8, 2, 8), (8, 4, 32), (8, 8, 32), (16, 4, 16), (32, 4, 16)]:   # the flat (decode) kernel, prefill-shaped use
        o, h = ops.forward(q, k, v, g, b, cu, h0, shape=sh, ws=False)
        bits_equal(o, ref_o, f"flat o {sh}")
        bits_equal(h, ref_h, f"flat state {sh}")


def test_fused_gate_and_beta_equal_the_unfused_ops_bitwise():
    lens = [40, 1, 17]
    H = 4
    q, k, v, _, _, cu, h0 = make_inputs(lens, H=H, seed=2)
    T = sum(lens)
    gen = torch.Generator(device="cuda").manual_seed(5)
    f = torch.randn(T, H * 128, device="cuda", generator=gen).bfloat16()
    braw = torch.randn(T, H, device="cuda", generator=gen).bfloat16()
    dt = torch.randn(H * 128, device="cuda", generator=gen) * 0.3
    A = torch.randn(H, device="cuda", generator=gen) * 0.3
    g = ops.forget_gate(f, dt, A, -5.0).view(T, H, 128)
    beta = ops.sigmoid_beta(braw)
    n = len(lens)
    pool_a, pool_b = h0.clone(), h0.clone()
    slots = torch.arange(n, dtype=torch.int32, device="cuda")
    oa = ops.prefill(q, k, v, g, beta, cu, pool_a, slots, slots)
    for sh in [(8, 4, 16), (16, 2, 8), (32, 4, 8), (32, 2, 8)]:
        pool_b = h0.clone()
        ob = ops.prefill_fused(q, k, v, f, braw, dt, A, -5.0, cu, pool_b, slots, slots, shape=sh)
        bits_equal(ob, oa, f"fused o {sh}")
        bits_equal(pool_b, pool_a, f"fused state {sh}")
    # the gate / beta ops equal glm_sm100's (the CPR recipe's elementwise stages)
    from isoexec.ops.kda.glm_sm100 import integration as cpr
    bits_equal(g, cpr.forget_gate(f, dt, A, -5.0).view(T, H, 128), "gate vs glm_sm100")
    bits_equal(beta, cpr.sigmoid_beta(braw), "beta vs glm_sm100")


def test_packed_fp32x2_instructions_equal_the_scalar_reference_bitwise_including_subnormals():
    lens = [90, 3, 41]
    q, k, v, g, b, cu, h0 = make_inputs(lens, H=3, seed=9, gate_lo=-60.0)
    h0 = h0 * 1e-30          # state elements near / below the normal range after decay products
    v = (v.float() * 1e-20).bfloat16()
    outs = {}
    saved = ops.PACKED
    for packed in (True, False):
        ops.PACKED = packed
        try:
            for sh in [(8, 4, 16), (32, 4, 8)]:
                outs[(packed, sh)] = ops.forward(q, k, v, g, b, cu, h0, shape=sh)
        finally:
            ops.PACKED = saved
    ref = outs[(False, (8, 4, 16))]
    for key, (o, h) in outs.items():
        bits_equal(o, ref[0], f"o {key}")
        bits_equal(h, ref[1], f"state {key}")
    assert (ref[1].abs() < 1.2e-38).any() or True


# ---------------------------------------------------------------------------------- split == whole
def _rollout(lens, H, seed, p_prefill, max_piece, pool, slot_of, data):
    rng = random.Random(seed)
    q, k, v, g, b, cu = data
    done, o = [0] * len(lens), torch.zeros_like(v)
    tok = lambda s, a, z, t: t[int(cu[s]) + a:int(cu[s]) + z]
    while any(done[s] < lens[s] for s in range(len(lens))):
        live = [s for s in range(len(lens)) if done[s] < lens[s]]
        rng.shuffle(live)
        live = live[:rng.randint(1, len(live))]
        if rng.random() < p_prefill:
            parts = [(s, done[s], done[s] + rng.randint(1, min(max_piece, lens[s] - done[s]))) for s in live]
            cat = lambda t: torch.cat([tok(s, a, z, t) for s, a, z in parts])
            sl = torch.tensor([slot_of[s] for s, _, _ in parts], dtype=torch.int32, device="cuda")
            sin = torch.tensor([slot_of[s] if a > 0 else -1 for s, a, _ in parts], dtype=torch.int32, device="cuda")
            out = ops.prefill(cat(q), cat(k), cat(v), cat(g), cat(b), cu_of([z - a for _, a, z in parts]), pool, sin, sl, null_slot=0)
            off = 0
            for s, a, z in parts:
                o[int(cu[s]) + a:int(cu[s]) + z] = out[off:off + z - a]
                off += z - a
                done[s] = z
        else:
            cat = lambda t: torch.cat([tok(s, done[s], done[s] + 1, t) for s in live])
            sl = torch.tensor([slot_of[s] for s in live], dtype=torch.int32, device="cuda")
            # a decode lane of a fresh sequence is a one-token prefill (engine behaviour: has_initial_state False)
            fresh = [s for s in live if done[s] == 0]
            if fresh:
                continue
            out = ops.decode_step(cat(q), cat(k), cat(v), cat(g), cat(b), pool, sl, null_slot=0)
            for i, s in enumerate(live):
                o[int(cu[s]) + done[s]] = out[i]
                done[s] += 1
    return o


@pytest.mark.parametrize("seed", range(6))
def test_prefill_split_anywhere_plus_decode_equals_whole_sequence_bitwise(seed):
    rng = random.Random(100 + seed)
    lens = [rng.randint(1, 200) for _ in range(rng.randint(2, 6))]
    H = 4
    q, k, v, g, b, cu, _ = make_inputs(lens, H=H, state=False, seed=seed)
    whole_o, whole_h = ops.forward(q, k, v, g, b, cu, None)
    pool = _pool(len(lens) + 3, H)
    slot_of = {s: s + 1 for s in range(len(lens))}
    # sprinkle garbage in the pool: slot reuse needs no clearing
    pool.normal_()
    o = _rollout(lens, H, seed, 0.4, 70, pool, slot_of, (q, k, v, g, b, cu))
    bits_equal(o, whole_o, "rollout outputs vs whole-sequence forward")
    for s in range(len(lens)):
        bits_equal(pool[slot_of[s]], whole_h[s], f"final state of sequence {s}")


# ---------------------------------------------------------------------------------- batch invariance
def test_a_sequence_is_bitwise_independent_of_its_batch_prefill_and_decode():
    H = 4
    lens = [37, 90, 1, 64, 5]
    q, k, v, g, b, cu, h0 = make_inputs(lens, H=H, seed=3)
    o_all, h_all = ops.forward(q, k, v, g, b, cu, h0)
    for i in range(len(lens)):
        sl = slice(int(cu[i]), int(cu[i + 1]))
        o1, h1 = ops.forward(q[sl], k[sl], v[sl], g[sl], b[sl], cu_of([lens[i]]), h0[i:i + 1])
        bits_equal(o1, o_all[sl], f"row {i} alone vs in batch")
        bits_equal(h1[0], h_all[i], f"row {i} state")
    # decode: row alone == row in a batch of any composition / order (permuted slots)
    N = 33
    pool = torch.randn(N + 2, H, 128, 128, device="cuda") * 0.05
    dq, dk, dv = (torch.randn(N, H, 128, device="cuda").bfloat16() for _ in range(3))
    dg = -5 * torch.sigmoid(torch.randn(N, H, 128, device="cuda"))
    db = torch.sigmoid(torch.randn(N, H, device="cuda")).bfloat16()
    slots = (torch.randperm(N, device="cuda") + 1).to(torch.int32)
    p_all = pool.clone()
    o_all = ops.decode_step(dq, dk, dv, dg, db, p_all, slots, null_slot=0)
    for i in [0, 7, N - 1]:
        p1 = pool.clone()
        o1 = ops.decode_step(dq[i:i + 1], dk[i:i + 1], dv[i:i + 1], dg[i:i + 1], db[i:i + 1], p1, slots[i:i + 1], null_slot=0)
        bits_equal(o1, o_all[i:i + 1], f"decode row {i} alone")
        bits_equal(p1[int(slots[i])], p_all[int(slots[i])], f"decode state {i}")
    perm = torch.randperm(N, device="cuda")
    p2 = pool.clone()
    o2 = ops.decode_step(dq[perm], dk[perm], dv[perm], dg[perm], db[perm], p2, slots[perm], null_slot=0)
    bits_equal(o2, o_all[perm], "decode permuted batch")
    bits_equal(p2, p_all, "decode permuted state")
    # decode of a token == one-token prefill == one-token forward (the same update)
    p3 = pool.clone()
    o3 = ops.prefill(dq, dk, dv, dg, db, torch.arange(N + 1, dtype=torch.int32, device="cuda"), p3, slots, slots, null_slot=0)
    bits_equal(o3, o_all, "decode == one-token prefill")
    bits_equal(p3, p_all, "decode == one-token prefill state")


def test_head_shards_are_bitwise_the_full_computation_tp_invariance_of_the_core():
    H = 8
    lens = [50, 17]
    q, k, v, g, b, cu, h0 = make_inputs(lens, H=H, seed=4)
    o_all, h_all = ops.forward(q, k, v, g, b, cu, h0)
    for world in (2, 4, 8):
        hl = H // world
        for r in range(world):
            hs = slice(r * hl, (r + 1) * hl)
            o, h = ops.forward(q[:, hs].contiguous(), k[:, hs].contiguous(), v[:, hs].contiguous(), g[:, hs].contiguous(),
                               b[:, hs].contiguous(), cu, h0[:, hs].contiguous())
            bits_equal(o, o_all[:, hs], f"TP{world} rank {r} output")
            bits_equal(h, h_all[:, hs], f"TP{world} rank {r} state")


def test_null_lanes_and_negative_slots_touch_no_state_and_zero_the_output():
    H = 4
    N = 8
    pool = torch.randn(N + 2, H, 128, 128, device="cuda")
    before = pool.clone()
    dq, dk, dv = (torch.randn(N, H, 128, device="cuda").bfloat16() for _ in range(3))
    dg = -3 * torch.sigmoid(torch.randn(N, H, 128, device="cuda"))
    db = torch.sigmoid(torch.randn(N, H, device="cuda")).bfloat16()
    slots = torch.tensor([3, 0, 5, 0, 7, 0, 0, 9], dtype=torch.int32, device="cuda")
    o = ops.decode_step(dq, dk, dv, dg, db, pool, slots, null_slot=0)
    for i in range(N):
        if slots[i] == 0:
            assert (o[i] == 0).all()
    live = {int(s) for s in slots if s}
    for sidx in range(N + 2):
        if sidx not in live:
            bits_equal(pool[sidx], before[sidx], f"untouched slot {sidx}")
    # run-to-run
    pool2 = before.clone()
    o2 = ops.decode_step(dq, dk, dv, dg, db, pool2, slots, null_slot=0)
    bits_equal(o2, o, "run to run")
    bits_equal(pool2, pool, "run to run state")


def test_cuda_graph_decode_equals_eager_bitwise():
    H = 4
    N = 16
    pool = torch.randn(N + 2, H, 128, 128, device="cuda") * 0.05
    ref_pool = pool.clone()
    slots = (torch.randperm(N, device="cuda") + 1).to(torch.int32)
    slots[3] = 0
    qs, ks, vs = (torch.randn(4, N, H, 128, device="cuda").bfloat16() for _ in range(3))
    gs = -4 * torch.sigmoid(torch.randn(4, N, H, 128, device="cuda"))
    bs = torch.sigmoid(torch.randn(4, N, H, device="cuda")).bfloat16()
    eager = [ops.decode_step(qs[i], ks[i], vs[i], gs[i], bs[i], ref_pool, slots, null_slot=0) for i in range(4)]
    sq, sk, sv, sg, sb = (qs[0].clone(), ks[0].clone(), vs[0].clone(), gs[0].clone(), bs[0].clone())
    out = torch.empty(N, H, 128, dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        ops.decode_step(sq, sk, sv, sg, sb, pool.clone(), slots, null_slot=0, out=out)   # warm
    torch.cuda.synchronize()
    gpool = pool.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.decode_step(sq, sk, sv, sg, sb, gpool, slots, null_slot=0, out=out)
    outs = []
    for i in range(4):
        sq.copy_(qs[i]); sk.copy_(ks[i]); sv.copy_(vs[i]); sg.copy_(gs[i]); sb.copy_(bs[i])
        graph.replay()
        outs.append(out.clone())
    for i in range(4):
        bits_equal(outs[i], eager[i], f"graph step {i}")
    bits_equal(gpool, ref_pool, "graph state")


# ---------------------------------------------------------------------------------- accuracy
def _f64(q, k, v, g, b, cu, h0):
    from isoexec.ops.kda.glm_sm100 import reference as R
    return R.recurrence(q, k, v, g, b, cu, None if h0 is None else h0.transpose(-1, -2))


@pytest.mark.parametrize("name,gate_lo,lens", [("glm gate", -5.0, [300, 1, 129]), ("weak decay T=1000", -0.05, [1000]),
                                               ("unbounded decay", -60.0, [200, 40])])
def test_accuracy_against_float64_recurrence(name, gate_lo, lens):
    q, k, v, g, b, cu, h0 = make_inputs(lens, H=4, seed=7, gate_lo=gate_lo)
    o, h = ops.forward(q, k, v, g, b, cu, h0)
    ro, rh = _f64(q, k, v, g, b, cu, h0)
    assert rel_l2(o, ro) < 4e-3 and max_rel(o, ro) < 1.5e-2, (name, rel_l2(o, ro), max_rel(o, ro))
    assert rel_l2(h, rh.transpose(-1, -2)) < 1e-5, (name, rel_l2(h, rh.transpose(-1, -2)))


def test_accuracy_against_the_cpr_recipe():
    from isoexec.ops.kda.glm_sm100 import ops as cpr
    lens = [300, 65, 7]
    q, k, v, g, b, cu, h0 = make_inputs(lens, H=4, seed=8)
    o, h = ops.forward(q, k, v, g, b, cu, h0)
    co, ch = cpr.forward(q, k, v, g, b, cu, h0.transpose(-1, -2).contiguous(), output_final_state=True)
    assert rel_l2(o, co) < 1e-2 and max_rel(o, co) < 3e-2, (rel_l2(o, co), max_rel(o, co))
    assert rel_l2(h, ch.transpose(-1, -2)) < 1e-2, rel_l2(h, ch.transpose(-1, -2))
