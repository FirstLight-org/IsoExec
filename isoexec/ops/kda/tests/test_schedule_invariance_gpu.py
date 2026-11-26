"""Bitwise invariance of the KDA schedule, each reducer against itself (SM100).

The schedule is rule 1 of ``isoexec.ops.schedule_policy``: segments of the reducer's block anchored at absolute
position 0 of the sequence, one continued chain. A sequence's bits may therefore depend on nothing but its own
tokens: not on a repeat, the batch, its position in the batch, its neighbours, how calls cut it (inside a block, at
a block boundary, one before or after), whether the engine or the trainer forward ran it, which slot held its
state, the head shard, or padding lanes. Every comparison is ``torch.equal`` on the raw bits (bf16 outputs as
int16, fp32 states as int32).

Both reducers run the same cases: ``chunk`` (``glm_sm100``, block 64, the default) and ``token``
(``glm_rec_sm100``, block 1). They are never compared with each other here: they are different programs.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from isoexec.ops.kda import schedule

H = 2
LENGTHS = (1, 63, 64, 65, 127, 128, 129)
LONG = 1000


def _require_sm100():
    from isoexec.core.arch import live_device_arch

    if not torch.cuda.is_available() or live_device_arch() != "sm100":
        pytest.skip("requires an SM100 device")


@pytest.fixture(autouse=True)
def _gpu():
    _require_sm100()


def bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32) if t.dtype == torch.float32 else t


def same(a, b) -> bool:
    return a.shape == b.shape and torch.equal(bits(a), bits(b))


def cu_of(lens):
    out = [0]
    for n in lens:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=torch.int32, device="cuda")


def seq(n, seed, heads=H):
    """One sequence's operands: bf16 q, k, v ``[n, heads, 128]``, fp32 log decay in GLM's bounded range, bf16 beta."""
    gen = torch.Generator(device="cuda").manual_seed(1000 + seed)
    rnd = lambda *s: torch.randn(*s, device="cuda", generator=gen)
    q, k, v = (rnd(n, heads, 128).bfloat16() for _ in range(3))
    g = -5.0 * torch.sigmoid(rnd(n, heads, 128))
    b = torch.sigmoid(rnd(n, heads)).bfloat16()
    return q, k, v, g, b


def pack(seqs):
    return tuple(torch.cat([s[i] for s in seqs], 0).contiguous() for i in range(5))


def cut(s, a, b):
    return tuple(t[a:b].contiguous() for t in s)


class Chunk:
    """The chunkwise reducer (block 64): trainer forward = ``ops.forward``; engine state = records."""

    reducer = schedule.CHUNK

    def __init__(self):
        from isoexec.ops.kda.glm_sm100 import ops

        self.ops = ops

    def whole(self, s, cu):
        return self.ops.forward(*s, cu, None, output_final_state=True)

    def pool(self, slots, heads=H):
        return self.ops.KdaState(slots, heads)

    def prefill(self, pool, s, cu, slot_in, slot_out):
        return self.ops.prefill(*s, cu, pool, slot_in, slot_out)

    def decode(self, pool, s, slots):
        return self.ops.decode_step(*s, pool, slots)

    def state(self, pool, slots):
        return self.ops.record_state(pool, slots)

    def raw(self, pool):
        return pool.data


class Token:
    """The per-token reducer (block 1): trainer forward = ``ops.forward``; engine state = the fp32 state itself."""

    reducer = schedule.TOKEN

    def __init__(self):
        from isoexec.ops.kda.glm_rec_sm100 import ops

        self.ops = ops

    def whole(self, s, cu):
        return self.ops.forward(*s, cu, None)

    def pool(self, slots, heads=H):
        return torch.zeros(slots, heads, 128, 128, dtype=torch.float32, device="cuda")

    def prefill(self, pool, s, cu, slot_in, slot_out):
        return self.ops.prefill(*s, cu, pool, slot_in, slot_out)

    def decode(self, pool, s, slots):
        return self.ops.decode_step(*s, pool, slots)

    def state(self, pool, slots):
        return pool[torch.as_tensor(slots, device="cuda").long()].clone()

    def raw(self, pool):
        return pool


@pytest.fixture(scope="module", params=["chunk", "token"])
def R(request):
    _require_sm100()
    return Chunk() if request.param == "chunk" else Token()


def alone(R, s):
    """The reference bits of one sequence: the trainer forward of that sequence by itself."""
    o, st = R.whole(s, cu_of([s[0].shape[0]]))
    return o, st[0]


def engine_calls(R, s, sizes):
    """One sequence through the engine in calls of the given sizes (size 1 = a decode step), from a fresh slot."""
    pool = R.pool(3)
    outs, pos = [], 0
    for i, n in enumerate(sizes):
        part = cut(s, pos, pos + n)
        if n == 1 and i > 0:
            outs.append(R.decode(pool, part, [1]))
        else:
            outs.append(R.prefill(pool, part, cu_of([n]), [-1 if i == 0 else 1], [1]))
        pos += n
    assert pos == s[0].shape[0]
    return torch.cat(outs, 0), R.state(pool, [1])[0]


# --------------------------------------------------------------------------------------------- the build
def test_loaded_kernel_was_compiled_with_the_policy_segment(R):
    if R.reducer is schedule.CHUNK:
        from isoexec.ops.kda.glm_sm100._extension import extension

        assert extension().chunk_len() == schedule.segments(schedule.CHUNK).S == 64
    else:
        assert schedule.segments(schedule.TOKEN).S == 1


# ------------------------------------------------------------------------------------------ repeated runs
@pytest.mark.parametrize("n", LENGTHS + (LONG,))
def test_repeated_runs_are_bitwise(R, n):
    s = seq(n, n)
    o0, st0 = alone(R, s)
    for _ in range(2):
        o, st = alone(R, s)
        assert same(o, o0) and same(st, st0)
    e0 = engine_calls(R, s, [n])
    e1 = engine_calls(R, s, [n])
    assert same(e0[0], e1[0]) and same(e0[1], e1[1])


# ------------------------------------------------------- batch size, row position, neighbours (trainer forward)
@pytest.mark.parametrize("n", LENGTHS + (LONG,))
@pytest.mark.parametrize("batch", [2, 3, 8, 17, 64])
def test_a_sequence_alone_equals_itself_inside_every_batch_and_position(R, n, batch):
    s = seq(n, n)
    o0, st0 = alone(R, s)
    other_lens = [1, 200, 63, 64, 65, 7, 129, 128]
    for position in sorted({0, batch // 2, batch - 1}):
        seqs, lens = [], []
        for i in range(batch):
            if i == position:
                seqs.append(s), lens.append(n)
            else:
                m = other_lens[(i + position) % len(other_lens)]
                seqs.append(seq(m, 7919 * batch + i)), lens.append(m)
        o, st = R.whole(pack(seqs), cu_of(lens))
        a = sum(lens[:position])
        assert same(o[a:a + n], o0), (batch, position)
        assert same(st[position], st0), (batch, position)


@pytest.mark.parametrize("n", [1, 64, 65, 129])
@pytest.mark.parametrize("kind", ["nan", "inf", "huge", "zero", "same_tokens"])
def test_neighbour_contents_do_not_reach_a_sequence(R, n, kind):
    s = seq(n, n)
    o0, st0 = alone(R, s)
    nb = [list(seq(m, 31 + m)) for m in (70, 64, 3)]
    for x in nb:
        if kind == "nan":
            x[0][0, 0, 0] = float("nan"); x[2][-1, 0, 5] = float("nan")
        elif kind == "inf":
            x[1][0, 0, 0] = float("inf"); x[2][-1, 1, 5] = float("-inf")
        elif kind == "huge":
            x[2].mul_(1e30)
        elif kind == "zero":
            for t in x:
                t.zero_()
    if kind == "same_tokens":
        nb = [list(cut(s, 0, min(n, m))) for m in (70, 64, 3)]
    seqs = [tuple(nb[0]), s, tuple(nb[1]), tuple(nb[2])]
    lens = [t[0].shape[0] for t in seqs]
    o, st = R.whole(pack(seqs), cu_of(lens))
    assert same(o[lens[0]:lens[0] + n], o0) and same(st[1], st0)


# ------------------------------------------------------------- engine path == trainer forward, on the same rows
@pytest.mark.parametrize("n", LENGTHS + (LONG,))
def test_engine_prefill_of_a_fresh_slot_equals_the_trainer_forward(R, n):
    s = seq(n, n)
    o0, st0 = alone(R, s)
    o, st = engine_calls(R, s, [n])
    assert same(o, o0) and same(st, st0)


def test_engine_batch_equals_the_trainer_forward_rows():
    for R_ in (Chunk(), Token()):
        lens = [129, 1, 64, 300, 63, 65]
        seqs = [seq(m, 500 + i) for i, m in enumerate(lens)]
        ow, stw = R_.whole(pack(seqs), cu_of(lens))
        pool = R_.pool(8)
        slots = [5, 0, 7, 2, 3, 1]
        oe = R_.prefill(pool, pack(seqs), cu_of(lens), [-1] * 6, slots)
        assert same(oe, ow) and same(R_.state(pool, slots), stw)


# ------------------------------------------------------------------------------------------ call boundaries
def _splits(n):
    """Two-call cuts of a sequence of n tokens: inside a block, at each block boundary and one either side."""
    pts = {1, 2, 15, 16, 17, 31, 32, 33, 47, 48, 49, 62, 63, 64, 65, 66, 100, 127, 128, 129, 191, 192, 193, n - 1}
    return sorted(p for p in pts if 0 < p < n)


@pytest.mark.parametrize("n", [2, 63, 64, 65, 127, 128, 129, 200])
def test_one_call_equals_two_calls_at_every_kind_of_boundary(R, n):
    s = seq(n, n)
    o0, st0 = alone(R, s)
    for p in _splits(n):
        o, st = engine_calls(R, s, [p, n - p])
        assert same(o, o0) and same(st, st0), p


@pytest.mark.parametrize("n", [65, 129, 200])
@pytest.mark.parametrize("size", [1, 7, 16, 63, 64, 65])
def test_one_call_equals_calls_of_a_fixed_size(R, n, size):
    s = seq(n, n)
    o0, st0 = alone(R, s)
    sizes = [size] * (n // size) + ([n % size] if n % size else [])
    o, st = engine_calls(R, s, sizes)
    assert same(o, o0) and same(st, st0)


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("prefill", [1, 2, 62, 63, 64, 65, 127, 128])
def test_prefill_then_decode_steps_equals_the_trainer_forward(R, n, prefill):
    if prefill >= n:
        pytest.skip("the prefill covers the whole sequence (covered by the fresh-slot test)")
    s = seq(n, n)
    o0, st0 = alone(R, s)
    o, st = engine_calls(R, s, [prefill] + [1] * (n - prefill))
    assert same(o, o0) and same(st, st0)


@pytest.mark.parametrize("n", LENGTHS)
def test_decode_steps_only_equal_the_trainer_forward(R, n):
    s = seq(n, n)
    o0, st0 = alone(R, s)
    pool = R.pool(3)
    outs = [R.decode(pool, cut(s, t, t + 1), [2]) for t in range(n)]
    assert same(torch.cat(outs, 0), o0) and same(R.state(pool, [2])[0], st0)


def test_long_sequence_in_mixed_calls_equals_one_call(R):
    n = 4097
    s = seq(n, 77)
    o0, st0 = alone(R, s)
    sizes, left, i = [], n, 0
    pattern = [1000, 1, 1, 63, 64, 65, 1, 500, 129, 1, 1, 1, 777]
    while left:
        m = min(left, pattern[i % len(pattern)])
        sizes.append(m)
        left -= m
        i += 1
    o, st = engine_calls(R, s, sizes)
    assert same(o, o0) and same(st, st0)


def test_decode_batch_with_other_sequences_at_other_positions(R):
    """Decode lanes whose sequences sit at different positions of their blocks (one closing a block, one opening
    one, one mid-block) do not reach each other, at every batch size."""
    lens = [200, 130, 80]
    pre = [63, 64, 5]
    seqs = [seq(m, 900 + i) for i, m in enumerate(lens)]
    refs = [alone(R, s) for s in seqs]
    for width in (1, 2, 3):
        pool = R.pool(4)
        live = list(range(width))
        for i in live:
            o = R.prefill(pool, cut(seqs[i], 0, pre[i]), cu_of([pre[i]]), [-1], [i + 1])
            assert same(o, refs[i][0][:pre[i]])
        for step in range(66):
            part = pack([cut(seqs[i], pre[i] + step, pre[i] + step + 1) for i in live])
            o = R.decode(pool, part, [i + 1 for i in live])
            for j, i in enumerate(live):
                assert same(o[j], refs[i][0][pre[i] + step]), (width, step, i)


# ------------------------------------------------------------------------- slots: reuse, moves, continuation
def test_slot_reuse_move_and_preemption(R):
    s, t = seq(150, 1), seq(90, 2)
    os_, sts = alone(R, s)
    ot, stt = alone(R, t)
    pool = R.pool(4)
    R.prefill(pool, cut(t, 0, 90), cu_of([90]), [-1], [1])                      # another request used slot 1
    o1 = R.prefill(pool, cut(s, 0, 70), cu_of([70]), [-1], [1])                 # reuse: fresh, no clearing
    o2 = R.prefill(pool, cut(s, 70, 100), cu_of([30]), [1], [3])                # continue slot 1, leave it in slot 3
    o3 = torch.cat([R.decode(pool, cut(s, p, p + 1), [3]) for p in range(100, 150)], 0)
    assert same(torch.cat([o1, o2, o3], 0), os_) and same(R.state(pool, [3])[0], sts)
    o4 = R.prefill(pool, t, cu_of([90]), [-1], [3])                             # preemption: recompute from 0
    assert same(o4, ot) and same(R.state(pool, [3])[0], stt)


def test_states_left_by_different_call_histories_continue_identically(R):
    s = seq(260, 5)
    o0, st0 = alone(R, s)
    tails = []
    for sizes in ([100], [64, 36], [63, 1, 36], [1] * 100, [99, 1], [65, 35]):
        pool = R.pool(3)
        pos = 0
        for i, n in enumerate(sizes):
            part = cut(s, pos, pos + n)
            if n == 1 and i:
                R.decode(pool, part, [1])
            else:
                R.prefill(pool, part, cu_of([n]), [-1 if i == 0 else 1], [1])
            pos += n
        tail = R.prefill(pool, cut(s, 100, 260), cu_of([160]), [1], [1])
        tails.append((tail, R.state(pool, [1])[0]))
    for tail, st in tails:
        assert same(tail, o0[100:]) and same(st, st0)


def test_reset_kernel_makes_a_used_slot_fresh():
    """``slot_reset_sm100``: a slot zeroed by the reset kernel is a fresh sequence at position 0, for both reducers'
    state layouts (the chunk reducer's records; the token reducer's fp32 state as bytes)."""
    from isoexec.ops.kda.slot_reset_sm100 import reset_fresh_slots

    for R_ in (Chunk(), Token()):
        s, t = seq(130, 11), seq(77, 12)
        o0, st0 = alone(R_, s)
        pool = R_.pool(4)
        conv = torch.ones(4, 3 * H * 128, 3, dtype=torch.bfloat16, device="cuda")
        R_.prefill(pool, t, cu_of([77]), [-1], [2])
        raw = R_.raw(pool)
        byte_view = raw if raw.dtype == torch.uint8 else raw.view(torch.uint8).view(raw.shape[0], raw.shape[1], -1)
        slots = torch.tensor([2, 1], dtype=torch.int32, device="cuda")
        before = byte_view[1].clone()
        reset_fresh_slots(conv, byte_view, slots, torch.tensor([0, 9], dtype=torch.int32, device="cuda"), null_slot=-1)
        assert int(byte_view[2].count_nonzero()) == 0 and int(conv[2].count_nonzero()) == 0
        assert torch.equal(byte_view[1], before) and bool((conv[1] == 1).all())      # position 9: a continuation, kept
        outs = [R_.decode(pool, cut(s, p, p + 1), [2]) for p in range(130)]
        assert same(torch.cat(outs, 0), o0) and same(R_.state(pool, [2])[0], st0)


# ---------------------------------------------------------------------------------------------- padding
def test_padding_lanes_touch_nothing_and_do_not_reach_live_lanes(R):
    s, t = seq(140, 21), seq(140, 22)
    refs = [alone(R, s), alone(R, t)]
    pool = R.pool(6)
    R.prefill(pool, pack([cut(s, 0, 60), cut(t, 0, 64)]), cu_of([60, 64]), [-1, -1], [1, 4])
    junk = seq(8, 99)
    for step in range(70):
        untouched = R.raw(pool)[[0, 2, 3, 5]].clone()
        a, b = cut(s, 60 + step, 61 + step), cut(t, 64 + step, 65 + step)
        lanes = [-1, 1, -1, -1, 4, -1, -1, -1]
        part = pack([cut(junk, 0, 1), a, cut(junk, 1, 3), b, cut(junk, 3, 6)])
        o = R.decode(pool, part, lanes)
        assert same(o[1], refs[0][0][60 + step]) and same(o[4], refs[1][0][64 + step]), step
        assert int(bits(o[[0, 2, 3, 5, 6, 7]]).count_nonzero()) == 0                  # padded lanes: +0
        assert torch.equal(R.raw(pool)[[0, 2, 3, 5]], untouched)
    full_s, full_t = alone(R, cut(s, 0, 130)), alone(R, cut(t, 0, 134))
    assert same(R.state(pool, [1])[0], full_s[1]) and same(R.state(pool, [4])[0], full_t[1])


def test_empty_and_one_token_sequences_between_others():
    """Zero-length sequences inside a packed trainer batch (chunk reducer; the token reducer's forward is run on
    the same batch without them)."""
    R_ = Chunk()
    a, b = seq(70, 41), seq(129, 42)
    empty = cut(a, 0, 0)
    o, st = R_.whole(pack([a, empty, b, empty]), cu_of([70, 0, 129, 0]))
    assert same(o[:70], alone(R_, a)[0]) and same(o[70:], alone(R_, b)[0])
    assert same(st[0], alone(R_, a)[1]) and same(st[2], alone(R_, b)[1])
    assert int(bits(st[1]).count_nonzero()) == 0 and int(bits(st[3]).count_nonzero()) == 0


# ------------------------------------------------------------------------------------- head shards (core)
@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_head_shards_of_the_core_are_the_full_computation(R, world):
    heads = 8
    lens = [65, 1, 129]
    seqs = [seq(m, 600 + i, heads) for i, m in enumerate(lens)]
    full = pack(seqs)
    o, st = R.whole(full, cu_of(lens))
    hl = heads // world
    for r in range(world):
        part = tuple(t[:, r * hl:(r + 1) * hl].contiguous() for t in full)
        orr, strr = R.whole(part, cu_of(lens))
        assert same(orr, o[:, r * hl:(r + 1) * hl]) and same(strr, st[:, r * hl:(r + 1) * hl]), (world, r)


# ------------------------------------------------------------------------------------ the composed layer
LH, HIDDEN = 16, 512


def _layer():
    """Random weights with the attributes ``engine.shard_layer`` reads. This is test scaffolding for the engine
    composition (projections, convolutions, gates, recurrence, gated norm, o_proj tree); it is not a model."""
    gen = torch.Generator(device="cuda").manual_seed(3)
    rnd = lambda *s, sc=1.0: (torch.randn(*s, device="cuda", generator=gen) * sc)
    lin = lambda o, i: SimpleNamespace(weight=rnd(o, i, sc=i ** -0.5).bfloat16())
    HK = LH * 128
    return SimpleNamespace(
        cfg=SimpleNamespace(linear_num_heads=LH, linear_head_dim=128, hidden=HIDDEN, rms_norm_eps=1e-6,
                            linear_gate_lower_bound=-5.0),
        q_proj=lin(HK, HIDDEN), k_proj=lin(HK, HIDDEN), v_proj=lin(HK, HIDDEN),
        q_conv1d=SimpleNamespace(weight=rnd(HK, 1, 4, sc=0.3)), k_conv1d=SimpleNamespace(weight=rnd(HK, 1, 4, sc=0.3)),
        v_conv1d=SimpleNamespace(weight=rnd(HK, 1, 4, sc=0.3)),
        f_a_proj=lin(128, HIDDEN), f_b_proj=lin(HK, 128), g_a_proj=lin(128, HIDDEN), g_b_proj=lin(HK, 128),
        b_proj=lin(LH, HIDDEN), o_proj=lin(HIDDEN, HK), dt_bias=rnd(HK), A_log=rnd(LH, sc=0.5),
        o_norm=SimpleNamespace(weight=(1 + 0.1 * rnd(128)).bfloat16()))


def _tokens(n, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(n, HIDDEN, device="cuda", generator=gen).bfloat16()


@pytest.fixture(scope="module", params=["chunk", "token"])
def E(request):
    _require_sm100()
    return schedule.select(schedule.CHUNK if request.param == "chunk" else schedule.TOKEN)


def test_layer_trainer_forward_is_tp_degree_invariant(E):
    mod = _layer()
    lens = [37, 64, 129, 1, 70]
    x, cu = _tokens(sum(lens), 1), cu_of(lens)
    ref = E.forward_tp([E.shard_layer(mod, 0, 1)], x, cu_seqlens=cu)
    assert same(ref, E.forward_tp([E.shard_layer(mod, 0, 1)], x, cu_seqlens=cu))
    for world in (2, 4, 8):
        out = E.forward_tp([E.shard_layer(mod, r, world) for r in range(world)], x, cu_seqlens=cu)
        assert same(out, ref), world


@pytest.mark.parametrize("world", [1, 2, 4])
def test_layer_engine_calls_equal_the_layer_trainer_forward(E, world):
    """The composed layer through the engine (a prefill call, a continuing prefill call, decode steps with padded
    lanes, across block boundaries) gives the trainer forward's rows, for every head-shard degree."""
    mod = _layer()
    lens = [150, 66, 1, 129]
    xs = [_tokens(n, 10 + i) for i, n in enumerate(lens)]
    whole = E.forward_tp([E.shard_layer(mod, 0, 1)], torch.cat(xs, 0), cu_seqlens=cu_of(lens))
    shards = [E.shard_layer(mod, r, world) for r in range(world)]
    pools = [E.KdaStatePool(6, LH // world) for _ in shards]
    slots = [4, 1, 5, 2]
    idx = torch.tensor(slots, dtype=torch.int32, device="cuda")
    first = [60, 63, 1, 64]
    got = [[] for _ in lens]
    done = list(first)
    y = E.forward_tp(shards, torch.cat([x[:n] for x, n in zip(xs, first)], 0),
                     E.EngineBatch(cu_of(first), idx, torch.zeros(4, dtype=torch.bool, device="cuda")), pools)
    o = 0
    for i, n in enumerate(first):
        got[i].append(y[o:o + n]); o += n
    live = [i for i in range(4) if done[i] < lens[i]]
    second = {0: 5, 1: 1, 3: 64}                                       # a continuing prefill call (mixed sizes)
    y = E.forward_tp(shards, torch.cat([xs[i][done[i]:done[i] + second[i]] for i in live], 0),
                     E.EngineBatch(cu_of([second[i] for i in live]), idx[live], torch.ones(len(live), dtype=torch.bool, device="cuda")), pools)
    o = 0
    for i in live:
        got[i].append(y[o:o + second[i]]); o += second[i]; done[i] += second[i]
    while any(done[i] < lens[i] for i in range(4)):                   # decode steps, 4 lanes, finished lanes padded
        lanes = [slots[i] if done[i] < lens[i] else -1 for i in range(4)]
        x = torch.cat([xs[i][done[i]:done[i] + 1] if done[i] < lens[i] else _tokens(1, 99) for i in range(4)], 0)
        y = E.forward_tp(shards, x, E.EngineBatch(cu_of([1] * 4), torch.tensor(lanes, dtype=torch.int32, device="cuda"),
                                                   decode=True, null_block_id=-1), pools)
        for i in range(4):
            if done[i] < lens[i]:
                got[i].append(y[i:i + 1]); done[i] += 1
    o = 0
    for i, n in enumerate(lens):
        assert same(torch.cat(got[i], 0), whole[o:o + n]), (world, i)
        o += n


def test_layer_fused_decode_equals_unfused(E):
    mod = _layer()
    base = E.shard_layer(mod, 0, 1)
    fused = E.fuse_decode_operands(copy.copy(E.shard_layer(mod, 0, 1)))
    pa, pb = E.KdaStatePool(5, LH), E.KdaStatePool(5, LH)
    x, cu = _tokens(100, 7), cu_of([60, 40])
    idx = torch.tensor([2, 4], dtype=torch.int32, device="cuda")
    batch = E.EngineBatch(cu, idx, torch.zeros(2, dtype=torch.bool, device="cuda"))
    assert same(E.forward(base, x, batch, pa), E.forward(base, x, batch, pb))
    lanes = torch.tensor([4, -1, 2], dtype=torch.int32, device="cuda")
    dec = E.EngineBatch(cu_of([1] * 3), lanes, decode=True, null_block_id=-1)
    for step in range(30):
        y = _tokens(3, 70 + step)
        assert same(E.forward(base, y, dec, pa), E.forward(fused, y, dec, pb)), step


def test_an_engine_refuses_the_other_reducers_pool():
    cpr, rec = schedule.select(schedule.CHUNK), schedule.select(schedule.TOKEN)
    mod = _layer()
    x = _tokens(4, 1)
    idx = torch.tensor([1], dtype=torch.int32, device="cuda")
    for mine, other in ((cpr, rec), (rec, cpr)):
        batch = mine.EngineBatch(cu_of([4]), idx, torch.zeros(1, dtype=torch.bool, device="cuda"))
        with pytest.raises(ValueError, match="not mixed inside one deployment"):
            mine.forward(mine.shard_layer(mod, 0, 1), x, batch, other.KdaStatePool(2, LH))
