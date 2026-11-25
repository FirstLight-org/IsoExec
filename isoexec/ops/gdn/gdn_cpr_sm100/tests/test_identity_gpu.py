"""The one-identity claims of the SM100 CPR core, all bitwise between kernels, plus the accuracy gates against a float64 reference of the same mathematical function."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops.gdn.gdn_cpr_sm100 import ops as G
from isoexec.ops.gdn.gdn_cpr_sm100.reference import gdn_reference, materialize

from ._fixtures import SM100, biteq, decode_step, fresh, ids, make, prefill, rel_err, rows, state_equal

pytestmark = SM100

#: (label, key heads, value heads). Qwen3.5-9B and Qwen3.5-35B-A3B both declare 16 key / 32 value heads
#: at head_dim 128; the other rows are the tensor-parallel shards the fleet runs.
GEOMETRIES = [("tp8", 2, 4), ("tp4", 4, 8), ("tp1", 16, 32), ("hv==hk", 16, 16)]


@pytest.mark.parametrize("lanes", [[1], [17], [64], [65], [128], [200], [255], [64, 64], [13, 64, 129, 1]])
@pytest.mark.parametrize("slow", [True, False], ids=["slow_decay", "fast_decay"])
def test_prefill_matches_the_float64_reference(lanes, slow):
    d = make(lanes, 4, 8, seed=sum(lanes) + len(lanes), slow=slow)
    o = prefill(d)
    ref, _ = gdn_reference(d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"], d["cu"])
    mx, l2 = rel_err(o, ref)
    assert torch.isfinite(o).all()
    assert mx < 1.5e-2 and l2 < 1e-2, f"max-rel {mx:.3e} rel-L2 {l2:.3e}"


@pytest.mark.parametrize("label,hk,hv", GEOMETRIES)
def test_every_geometry_matches_the_reference(label, hk, hv):
    d = make([70, 64, 1], hk, hv, seed=29)
    o = prefill(d)
    ref, _ = gdn_reference(d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"], d["cu"])
    mx, l2 = rel_err(o, ref)
    assert mx < 1.5e-2 and l2 < 1e-2, f"{label}: max-rel {mx:.3e} rel-L2 {l2:.3e}"


@pytest.mark.parametrize("lanes", [[70], [200], [200, 64]])
def test_boundary_state_matches_the_reference(lanes):
    d = make(lanes, 4, 8, seed=11)
    n = len(lanes)
    st = fresh(n + 1, 8)
    slots = ids(range(1, n + 1))
    prefill(d, state=st, slot_in=slots)
    _, refS = gdn_reference(d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"], d["cu"])
    mx, l2 = rel_err(materialize(st, slots), refS)
    assert mx < 1.5e-2 and l2 < 1e-2, f"max-rel {mx:.3e} rel-L2 {l2:.3e}"


def test_continuation_from_a_loaded_initial_state():
    d = make([200], 4, 8, seed=41)
    st = fresh(2, 8)
    slots = ids([1])
    st.load_initial(slots, torch.randn(1, 8, 128, 128, device="cuda") * 0.05)
    # snapshot it: the call replaces the record's S with the sequence's FINAL state
    S0 = st.S[slots.long()].clone()
    o = prefill(d, state=st, slot_in=slots)
    ref, refS = gdn_reference(d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"], d["cu"],
                              initial_state=S0)
    mx, l2 = rel_err(o, ref)
    assert mx < 1.5e-2 and l2 < 1e-2
    mx, l2 = rel_err(materialize(st, slots), refS)
    assert mx < 1.5e-2 and l2 < 1e-2


@pytest.mark.parametrize("lanes", [[1024], [4096], [2048, 2048]])
def test_long_sequences_are_finite_and_repeatable(lanes):
    """Test lengths beyond the float64 reference's practical range."""
    d = make(lanes, 4, 8, seed=sum(lanes))
    first = prefill(d)
    assert torch.isfinite(first).all()
    assert biteq(first, prefill(d))
    n, T = len(lanes), sum(lanes)
    slots = ids(range(1, n + 1))
    split_state = fresh(n + 1, 8)
    pieces = torch.zeros_like(first)
    for lo, hi in ((0, 640), (640, T)):
        cu2 = [0]
        for i in range(n):
            s, e = d["cu"][i], d["cu"][i + 1]
            cu2.append(cu2[-1] + max(0, min(e, hi) - max(s, lo)))
        pieces[lo:hi] = prefill(d, torch.arange(lo, hi, device="cuda"), cu=cu2, state=split_state,
                                slot_in=slots)
    assert biteq(first, pieces)


def test_repeat_launches_are_bit_identical():
    d = make([200, 64, 13], 4, 8, seed=3)
    first = prefill(d)
    for _ in range(4):
        assert biteq(first, prefill(d))


def test_stateful_prefill_equals_the_stateless_one():
    for lanes in ([64], [130], [200, 64]):
        d = make(lanes, 4, 8, seed=11)
        st = fresh(len(lanes) + 1, 8)
        assert biteq(prefill(d, state=st, slot_in=ids(range(1, len(lanes) + 1))), prefill(d))


@pytest.mark.parametrize("lanes,cuts", [([200], [64, 128]), ([200], [1, 17, 65, 130]),
                                        ([64, 130], [64, 100, 194]), ([300], [64, 65, 66, 128, 192, 256])])
def test_prefill_split_anywhere_equals_one_prefill(lanes, cuts):
    """Claim (b): a prefill split mid-chunk, on boundaries or into one-token pieces is the same bits --
    outputs and every meaningful record byte."""
    d = make(lanes, 4, 8, seed=5)
    n, T = len(lanes), sum(lanes)
    slots = ids(range(1, n + 1))
    whole_state = fresh(n + 1, 8)
    whole = prefill(d, state=whole_state, slot_in=slots)
    split_state = fresh(n + 1, 8)
    pieces = torch.zeros_like(whole)
    bounds = [0] + sorted(set(cuts)) + [T]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if lo >= hi:
            continue
        cu2 = [0]
        for i in range(n):
            s, e = d["cu"][i], d["cu"][i + 1]
            cu2.append(cu2[-1] + max(0, min(e, hi) - max(s, lo)))
        pieces[lo:hi] = prefill(d, torch.arange(lo, hi, device="cuda"), cu=cu2, state=split_state, slot_in=slots)
    assert biteq(whole, pieces)
    assert state_equal(whole_state, split_state, slots)


@pytest.mark.parametrize("pre_len,steps", [(0, 70), (60, 8), (64, 5), (127, 3), (1, 130)])
def test_prefill_then_decode_equals_one_prefill(pre_len, steps):
    """Claim (c): prefill plus N decode steps is one prefill, across chunk boundaries -- the in-launch
    resync. Decode is the same kernel asked for one row."""
    d = make([pre_len + steps], 4, 8, seed=9)
    slots = ids([1])
    whole_state = fresh(2, 8)
    whole = prefill(d, state=whole_state, slot_in=slots)
    step_state = fresh(2, 8)
    got = torch.zeros_like(whole)
    if pre_len:
        got[:pre_len] = prefill(d, slice(0, pre_len), cu=[0, pre_len], state=step_state, slot_in=slots)
    for t in range(steps):
        got[pre_len + t] = decode_step(d, [pre_len + t], step_state, slots)[0]
    assert biteq(whole, got)
    assert state_equal(whole_state, step_state, slots)


def test_trainer_forward_is_the_stateless_prefill():
    """Claim (a), for both save modes."""
    d = make([200, 64], 4, 8, seed=31)
    want = prefill(d)
    nchunks = sum(-(-L // 64) for L in d["lanes"])
    for save_states in (False, True):
        o, saved = G.trainer_forward(d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"], d["cut"],
                                     save_states=save_states)
        assert biteq(want, o)
        for name, t in saved.items():
            # `h` is allocated for the worst case T // 64 + N chunks; only the pack's own
            # `sum(ceil(len / 64))` rows are written, and FLA's backward reads only those.
            used = t[:nchunks] if name == "h" else t
            assert used.numel() == 0 or torch.isfinite(used).all(), name


def test_decode_only_stream_equals_one_prefill():
    """A sequence fed entirely one token at a time, across two chunk boundaries."""
    d = make([130], 4, 8, seed=77)
    slots = ids([1])
    whole_state = fresh(2, 8)
    whole = prefill(d, state=whole_state, slot_in=slots)
    step_state = fresh(2, 8)
    got = torch.zeros_like(whole)
    for t in range(130):
        got[t] = decode_step(d, [t], step_state, slots)[0]
    assert biteq(whole, got)
    assert state_equal(whole_state, step_state, slots)


# --------------------------------------------------------------------------- the pruning relations
#: Every row count a call can ask for inside one 64-row chunk. The k-step prune is a function of the
#: TILE and not of the call, which is what makes every call shape agree -- and therefore what makes the
#: cross-shape tests above STRUCTURALLY BLIND to a wrong prune: if a skipped step were not free, every
#: shape would be wrong by the same amount and they would still agree with each other. The gate that
#: sees it has to be against the float64 reference, AT EVERY PRUNED BOUNDARY, per row.
_ROWS_IN_A_CHUNK = list(range(1, 65))

#: Per-row bound for the relation below, and it is LOOSER than the 1.5e-2 whole-tensor bound on purpose.
#: A per-row scale removes the averaging the whole-tensor number gets for free: over all 2,080 (nrow, row)
#: comparisons the worst row is 1.64e-2 max-rel while the whole tensor of the same call is 5.2e-3, and the
#: worst rows are ROW 0 OF A FRESH SEQUENCE, whose output is one 128-long bf16 dot product
#: (o_0 = (k^_0 . q^_0) beta_0 v_0 off a zero state) and whose own scale is 5.3e-3 against the tensor's
#: 1.07e-1 -- cancellation inside a single inner product, twenty times below the tensor's range.
#: The bound is set at 2.4x the measured worst row, and what it has to catch is not a last-bit change: a
#: k16 step wrongly dropped from a four- or eight-step chain removes an eighth to a quarter of the sum, an
#: O(1) relative error in the rows at and above the boundary. The bitwise gate on the prune's PREMISE is
#: `test_the_rows_the_prune_calls_zero_are_exactly_zero` below.
_ROW_MAX_REL, _ROW_REL_L2 = 4e-2, 2e-2


@pytest.mark.parametrize("nrow", _ROWS_IN_A_CHUNK)
def test_every_row_count_in_a_chunk_matches_the_reference_per_row(nrow):
    """A call of `nrow` rows agrees with float64 ROW BY ROW, for every nrow from 1 to 64."""
    d = make([nrow], 4, 8, seed=400 + nrow)
    o = prefill(d)
    ref, _ = gdn_reference(d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"], d["cu"])
    assert torch.isfinite(o).all()
    for j in range(nrow):
        mx, l2 = rel_err(o[j], ref[j])
        assert mx < _ROW_MAX_REL and l2 < _ROW_REL_L2, f"row {j} of {nrow}: max-rel {mx:.3e} rel-L2 {l2:.3e}"


@pytest.mark.parametrize("j0", [0, 1, 15, 16, 31, 32, 47, 48, 63])
def test_every_open_chunk_offset_matches_the_reference_per_row(j0):
    """The same relation from the other side: a chunk OPENED at row j0 by a first call and finished by a second, compared per row against float64."""
    d = make([64], 4, 8, seed=500 + j0)
    pool = fresh(1, 8)
    slots = ids([0])
    o1 = prefill(d, slice(0, j0), cu=[0, j0], state=pool, slot_in=slots) if j0 else None
    o2 = prefill(d, slice(j0, 64), cu=[0, 64 - j0], state=pool, slot_in=slots)
    o = torch.cat([o1, o2]) if j0 else o2
    ref, _ = gdn_reference(d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"], d["cu"])
    assert torch.isfinite(o).all()
    for j in range(64):
        mx, l2 = rel_err(o[j], ref[j])
        assert mx < _ROW_MAX_REL and l2 < _ROW_REL_L2, f"row {j} (opened at {j0}): max-rel {mx:.3e} rel-L2 {l2:.3e}"


# --------------------------------------------------------------------------- the execution-plan gate
#: `plan` names a phase-1 EXECUTION plan: 0 is the previous pass's (the runtime-width X earlier-block loop
#: with its sixteen-partial T16 tree, the 2/2/4/4 KK/KQ deal, runtime-trip-count copy and xu/xw loops),
#: 1 adds the ten-tile KK/KQ deal, 3 adds the compile-time widths. The kernel picks 3 for a prefill and 1
#: for a decode step by a deployment-fixed rule, so BOTH of those are production paths and 0 is the
#: before. The bits may not move between any of them, and this is the gate that says so -- a stronger
#: statement than the cross-shape relations, because it holds the inputs and the call shape fixed and
#: varies ONLY who evaluates what, in which order, with which loop widths. It is also what lets the
#: benchmark put two plans in ONE provider rotation, and what makes selecting a plan on speed legitimate.
_PLANS = [0, 1, 3, 4, 7]

#: THE DECODE PLANS, and after this pass they name TWO EVALUATORS rather than one schedule. 4 / 0 / 1 / 3 /
#: 7 are the CHUNK-FORM decode of `csrc/core.cu` -- the 64-row chunk kernel asked for one row, with the
#: other sixty-three predicated away -- and `rec_r1` / `rec_r2` / `rec_r4` / `rec_r8` are the RECURRENT step
#: of `csrc/decode.cu`, which reads the record's already-computed rows [0, j) and evaluates only row j, at
#: one / two / four / eight CTAs per head along the value rows. The shipped step is the recurrent one at the
#: split `ops.choose_decode_split` picks. They are a different SHAPE of the same recipe: the same chains
#: over the same operand vectors, with the operand roles exchanged (the transpose, a measured fact) and the
#: k-step bound the row's own 16-row block. NOTHING about the bits may move between them, outputs or
#: record, and these are the relations that say so -- the whole licence for shipping a second evaluator is
#: that it is not a second arithmetic.
_DECODE_PLANS = [G.PLANS["pred"], 0, 1, 3, 7,
                 G.PLANS["rec_r1"], G.PLANS["rec_r2"], G.PLANS["rec_r4"], G.PLANS["rec_r8"]]


@pytest.mark.parametrize("lanes", [[1], [17], [63], [64], [65], [129], [255], [64, 64], [13, 64, 129, 1]])
@pytest.mark.parametrize("label,hk,hv", GEOMETRIES)
def test_every_execution_plan_is_bit_identical(lanes, label, hk, hv):
    d = make(lanes, hk, hv, seed=600 + sum(lanes))
    ref = prefill(d, plan=_PLANS[0])
    for plan in _PLANS[1:]:
        assert biteq(ref, prefill(d, plan=plan)), f"plan {plan}"


@pytest.mark.parametrize("label,hk,hv", GEOMETRIES)
def test_every_decode_evaluator_agrees_on_a_decode_stream_and_its_record(label, hk, hv):
    """Seventy decode steps, so the stream crosses a chunk boundary and takes the in-launch resync and the `sh_cached` path on the other sixty-nine."""
    d = make([70], hk, hv, seed=777 + hv)
    outs, pools = {}, {}
    for plan in _DECODE_PLANS:
        pool = fresh(1, hv)
        slots = ids([0])
        outs[plan] = torch.cat([decode_step(d, [t], pool, slots, plan=plan) for t in range(70)])
        pools[plan] = pool
    for plan in _DECODE_PLANS[1:]:
        assert biteq(outs[_DECODE_PLANS[0]], outs[plan]), f"plan {plan}"
        assert state_equal(pools[_DECODE_PLANS[0]], pools[plan], [0]), f"plan {plan}"


@pytest.mark.parametrize("j0", list(range(0, 64)))
def test_every_decode_evaluator_agrees_at_every_open_chunk_offset(j0):
    """THE SHARPEST RELATION ON THE RECURRENT STEP, because `j0` is exactly what its shape is a function of."""
    hv = 8
    d = make([j0 + 1], 4, hv, seed=900 + j0)
    got = {}
    for plan in _DECODE_PLANS:
        pool = fresh(2, hv)
        slots = ids([1])
        if j0:
            prefill(d, slice(0, j0), cu=[0, j0], state=pool, slot_in=slots)
        got[plan] = (decode_step(d, [j0], pool, slots, plan=plan), pool)
    for plan in _DECODE_PLANS[1:]:
        assert biteq(got[_DECODE_PLANS[0]][0], got[plan][0]), f"output at j0 ={j0}, plan {plan}"
        assert state_equal(got[_DECODE_PLANS[0]][1], got[plan][1], [1]), f"record at j0 ={j0}, plan {plan}"


def test_every_decode_evaluator_agrees_on_slot_moves_null_lanes_and_forks():
    """Check batch-invariant output for the second evaluator."""
    d = make([1, 1, 1, 1], 4, 8, seed=4242)
    base = fresh(8, 8)
    warm = make([40], 4, 8, seed=11)
    prefill(warm, state=base, slot_in=ids([3]))          # slot 3 carries an open chunk at row 40
    got = {}
    for plan in _DECODE_PLANS:
        pool = fresh(8, 8)
        pool.data.copy_(base.data)
        o_move = decode_step(d, [0], pool, ids([3]), slot_out=ids([5]), plan=plan)
        o_null = decode_step(d, [1], pool, ids([-1, 5]), slot_out=ids([-1, 5]), plan=plan)             if False else decode_step(d, [1, 2], pool, ids([-1, 5]), plan=plan)
        o_fork = decode_step(d, [3, 3], pool, ids([5, 5]), slot_out=ids([6, 7]), plan=plan)
        got[plan] = (o_move, o_null, o_fork, pool)
    for plan in _DECODE_PLANS[1:]:
        for i, what in enumerate(("a slot move", "a NULL lane beside a real one", "a fork")):
            assert biteq(got[_DECODE_PLANS[0]][i], got[plan][i]), f"{what}, plan {plan}"
        assert state_equal(got[_DECODE_PLANS[0]][3], got[plan][3], [5, 6, 7]), f"records, plan {plan}"


def test_a_wide_decode_batch_agrees_with_the_chunk_form():
    """Check the one-record path used for production qualification."""
    hk, hv, n = 4, 8, 1024
    base = fresh(n, hv)
    warm = make([62] * n, hk, hv, seed=6200)
    slots = ids(range(n))
    prefill(warm, state=base, slot_in=slots)   # every one of the n slots opens its chunk at row 62
    steps = make([n] * 3, hk, hv, seed=6300)   # three decode rounds of n sequences each
    plans = (G.PLANS["pred"], G.PLANS["rec"], G.PLANS["rec_r2"])
    got = {}
    for plan in plans:
        pool = fresh(n, hv)
        pool.data.copy_(base.data)
        outs = [decode_step(steps, range(t * n, (t + 1) * n), pool, slots, plan=plan) for t in range(3)]
        got[plan] = (torch.cat(outs), pool)
    for plan in plans[1:]:
        assert biteq(got[plans[0]][0], got[plan][0]), f"plan {plan}"
        assert state_equal(got[plans[0]][1], got[plan][1], range(n)), f"records, plan {plan}"


#: THE PREPARED TWO-LAUNCH MODE (`ops.PLANS["prepared"]`, csrc/prepared.cu) is the same recipe with the
#: chunk-local work moved into a launch of its own: launch A runs `wy_chunk` for every (chunk, value
#: head) across the whole machine, launch B walks each sequence's chunks with the fp32 state in registers
#: and runs `slab_step`. It is selected by `ops._use_prepared`, a rule on the call's SHAPE alone, and the
#: licence for selecting it is exactly these relations -- the bits may not move, outputs or record, on
#: any call shape, because the only thing that changed is who evaluates a phase and when.
_PREPARED = G.PLANS["prepared"]
#: The ROUND-4 form of launch A -- the one-launch kernel's 194,816-byte shared-memory layout at one CTA
#: per SM, and the three-deep dependent global prologue -- kept in the shipped build as this pass's BEFORE
#: column. Every relation below runs it beside the shipped form, which is what says the two-CTA layout
#: (`u` and `w` overlaid on `L`/`X`/`P`, the carry moved out of `publish`) and the per-chunk-slot table
#: moved no bit. Which shared address a tile lives at and where a scalar was read from do not enter the
#: arithmetic, and these are the relations that say so rather than the argument.
_PREPARED_R4 = G.PLANS["prepared_round4"]
#: And the (chunk, KEY head) pre-kernel, which is built and gated and NOT shipped -- it measured a net
#: loss (`ops._use_prekernel`). Three forms of the same phase 1, differing in who evaluates the l2 norm and
#: the two 64 x 64 products, where the tiles live in shared memory, and whether the prologue is one 32-byte
#: table read or three dependent global loads. A rejected variant is gated so it can be re-measured.
_PREPARED_PREK = G.PLANS["prepared_prekernel"]
#: And launch B with its delta and output-seed chains UNFUSED. The fused form runs both as one k-loop over
#: the same `S^` B fragment; every accumulator sees the same addends in the same k order, and this is the
#: relation that says so rather than the argument.
_PREPARED_NOFUSE = G.PLANS["prepared_nofuse"]
#: And LAUNCH B SPLIT ACROSS CTAs along the value columns, two-way and four-way (`csrc/prepared.cu`,
#: `SPLIT`). A CTA of four or two warps owns 64 or 32 of the head's 128 value columns and runs the SAME
#: per-warp code: same chains, same operands, same k order, same seeds, same register-resident state share.
#: Only which CTA hosts a warp changes, which is placement invariance -- and these are the relations that
#: say so rather than the argument. They also gate that CTA 0 of a head owns the value-independent record
#: sections and the `pos` store, which is the one thing the split had to move.
_PREPARED_SPLIT2 = G.PLANS["prepared_split2"]
_PREPARED_SPLIT4 = G.PLANS["prepared_split4"]
_PREP_FORMS = (_PREPARED, _PREPARED_PREK, _PREPARED_NOFUSE, _PREPARED_R4, _PREPARED_SPLIT2,
               _PREPARED_SPLIT4)


@pytest.mark.parametrize("lanes", [[1], [17], [63], [64], [65], [129], [255], [64, 64], [13, 64, 129, 1]])
@pytest.mark.parametrize("label,hk,hv", GEOMETRIES)
def test_the_prepared_two_launch_mode_is_bit_identical(lanes, label, hk, hv):
    d = make(lanes, hk, hv, seed=600 + sum(lanes))
    ref = prefill(d, plan=0)
    for plan in _PREP_FORMS:
        assert biteq(ref, prefill(d, plan=plan)), f"plan {plan}"


@pytest.mark.parametrize("j0", [0, 1, 15, 16, 31, 47, 63])
@pytest.mark.parametrize("lanes", [[1], [64], [100], [257], [64, 100]])
def test_the_prepared_mode_agrees_on_a_stateful_pack_and_its_record(j0, lanes):
    """Every open-chunk offset, so chunk 0 reads the record's rows and the last chunk writes them back."""
    hk, hv, n = 4, 8, len(lanes)
    seeds = make([max(j0, 1)] * n, hk, hv, seed=11)
    d = make(lanes, hk, hv, seed=77 + j0 + sum(lanes))
    got = {}
    for plan in (0, *_PREP_FORMS):
        pool = fresh(n + 2, hv)
        slots = ids(range(1, n + 1))
        if j0:
            prefill(seeds, state=pool, slot_in=slots, plan=plan)
        got[plan] = (prefill(d, state=pool, slot_in=slots, plan=plan), pool)
    for plan in _PREP_FORMS:
        assert biteq(got[0][0], got[plan][0]), f"plan {plan}"
        assert state_equal(got[0][1], got[plan][1], range(1, n + 1)), f"plan {plan}"


def test_the_prepared_mode_agrees_on_a_slot_move_and_a_null_lane():
    d = make([130, 70], 4, 8, seed=5)
    got = {}
    for plan in (0, *_PREP_FORMS):
        pool = fresh(6, 8)
        o = prefill(d, state=pool, slot_in=ids([1, -1]), slot_out=ids([3, 4]), plan=plan)
        got[plan] = (o, pool)
    for plan in _PREP_FORMS:
        assert biteq(got[0][0], got[plan][0]), f"plan {plan}"
        assert state_equal(got[0][1], got[plan][1], [3, 4]), f"plan {plan}"


@pytest.mark.parametrize("label,hk,hv", GEOMETRIES)
def test_the_prepared_mode_agrees_on_a_decode_stream_and_its_record(label, hk, hv):
    """A prefill in the prepared mode, then seventy decode steps off the record it left, so the stream
    crosses a chunk boundary and takes the in-launch resync. Claim (c) has to survive the split."""
    pre = make([100], hk, hv, seed=21)
    d = make([70], hk, hv, seed=22)
    got = {}
    for plan in (0, *_PREP_FORMS):
        pool = fresh(2, hv)
        slots = ids([1])
        prefill(pre, state=pool, slot_in=slots, plan=plan)
        got[plan] = (torch.cat([decode_step(d, [t], pool, slots) for t in range(70)]), pool)
    for plan in _PREP_FORMS:
        assert biteq(got[0][0], got[plan][0]), f"plan {plan}"
        assert state_equal(got[0][1], got[plan][1], [1]), f"plan {plan}"


def test_the_prepared_mode_on_the_composition_shapes_the_rule_would_decline():
    """A fork, empty sequences, a stateful T = 0 pack and a 0xFF-poisoned pool, FORCED through the prepared mode and compared against the one-launch kernel."""
    hv = 8

    def run(plan):
        pool = fresh(6, hv)
        prefill(make([200], 4, hv, seed=31), state=pool, slot_in=ids([1]), plan=plan)
        fork = prefill(make([150, 150], 4, hv, seed=32), state=pool, slot_in=ids([1, 1]),
                       slot_out=ids([2, 3]), plan=plan)                       # a fork off one slot
        empt = prefill(make([0, 97, 0], 4, hv, seed=33), state=pool, slot_in=ids([4, 4, -1]),
                       slot_out=ids([4, 5, -1]), plan=plan)                   # empty sequences, NULL out
        prefill(make([0], 4, hv, seed=34), state=pool, slot_in=ids([5]), slot_out=ids([1]), plan=plan)
        bad = fresh(3, hv, poison=True)                                       # every byte 0xFF
        bad.reset(ids([1]))
        pois = prefill(make([103, 40], 4, hv, seed=35), state=bad, slot_in=ids([1, -1]),
                       slot_out=ids([1, 2]), plan=plan)
        return fork, empt, pois, pool, bad

    one = run(0)
    for plan in _PREP_FORMS:
        prep = run(plan)
        for i, what in enumerate(("fork", "empty sequences and a NULL lane", "poisoned pool")):
            assert biteq(one[i], prep[i]), f"{what}, plan {plan}"
        assert state_equal(one[3], prep[3], [1, 2, 3, 4, 5]), f"records after the fork, plan {plan}"
        assert state_equal(one[4], prep[4], [1, 2]), f"records off a poisoned pool, plan {plan}"


def test_the_prepared_selection_rule_reads_only_the_call_shape():
    """Check that the plan depends only on shape and device capacity."""
    dev = torch.device("cuda", torch.cuda.current_device())
    sms = G._sm_count(dev.index)
    assert G._use_prepared(8192, 1, 16, 32, False, dev)
    assert not G._use_prepared(8192, 1, 16, 32, True, dev)          # the trainer save path never takes it
    assert not G._use_prepared(64, 1, 16, 32, False, dev)           # one chunk: no serial chain to break
    assert not G._use_prepared(1 << 24, 1, 16, 32, False, dev)      # over the workspace budget
    #: ROUND 5 REMOVED THE CTA BOUND. Launch A is two CTAs per SM, so the split wins on the continuum term
    #: and not only on the one-launch kernel's rounded-up wave count: measured prepared / one-launch is
    #: 0.81-1.11 at 128 CTAs, 0.82-1.03 at 256 and 0.89-1.00 at 1024 where round 4 measured 1.07, 1.06 and
    #: 1.15. What is left is a CHUNK COUNT -- every point that lost had two chunks a sequence.
    assert G._use_prepared(8192, 8, 16, 32, False, dev)              # 256 CTAs, 16 chunks a sequence
    assert G._use_prepared(32768, 32, 16, 32, False, dev)            # 1024 CTAs, 16 chunks a sequence
    assert G._use_prepared(4096, 16, 16, 32, False, dev)             # 512 CTAs, FOUR chunks a sequence
    assert not G._use_prepared(4096, 32, 16, 32, False, dev)         # two chunks a sequence: a coin flip
    #: THE BOUNDARY IS A CHUNK COUNT AND NOT A TOKEN COUNT, which round 6 made literal. The rule reads
    #: `ceil(L / 64) >= 4` where `L` is the longest sequence when the host supplies it and the mean
    #: otherwise; round 5 wrote the same thing as `T >= 4 * 64 * nseq`, which is the same rule everywhere
    #: except for a mean length in [193, 256), where 193 tokens already ARE four chunks.
    assert G._use_prepared(256, 1, 16, 32, False, dev)               # four chunks
    assert G._use_prepared(193, 1, 16, 32, False, dev)               # still four chunks
    assert not G._use_prepared(192, 1, 16, 32, False, dev)           # exactly three
    #: And it depends on NOTHING but the shape -- no CTA count enters it any more, so the same prompt alone
    #: and beside seven others takes the same structure whenever each of them is four chunks long.
    assert sms > 0 and G._use_prepared(8192, 1, 16, 32, False, dev) == G._use_prepared(8192, 1, 2, 4,
                                                                                      False, dev)
    #: THE (chunk, KEY head) PRE-KERNEL is a second shape rule inside the same licence: it runs when a key
    #: head serves more than one value head, and at hv == hk it would do exactly what launch A does for one
    #: more launch and a 64 KB round trip. Nothing about it reads the call's CONTENTS either.
    assert G._use_prekernel(16, 32) and G._use_prekernel(4, 8) and G._use_prekernel(2, 4)
    assert not G._use_prekernel(32, 32)
    #: AND THE RULE READS THE LONGEST SEQUENCE WHEN THE HOST SUPPLIES IT. `[8192] + 100 x [64]` has a MEAN
    #: of 144 tokens, so the mean-based rule declined it and the 8,192-token prompt walked 128 serial chunks
    #: on one CTA while the other hundred finished in one wave -- the worst shape this mode exists for. The
    #: maximum comes off `cu_seqlens_cpu`, a HOST tensor, so nothing synchronizes.
    lens = [8192] + [64] * 100
    T, nseq = sum(lens), len(lens)
    assert not G._use_prepared(T, nseq, 16, 32, False, dev)                       # the mean: declined
    assert G._use_prepared(T, nseq, 16, 32, False, dev, max(lens))                # the maximum: taken
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
    assert G._max_len_of(cu, nseq) == 8192
    assert G._max_len_of(None, nseq) is None
    assert G._max_len_of(cu.cuda(), nseq) is None                                 # a device copy is ignored
    assert G._max_len_of(cu[:5], nseq) is None                                    # a short copy is ignored
    #: THE CPU ASSERTION (round 6, plan section 2.2): the composed call exactly as `ops.prefill` makes it
    #: -- `_max_len_of` on the HOST `cu_seqlens_cpu` tensor piped straight into `_use_prepared` -- rather
    #: than the maximum passed by hand above. `[8192] + 100 x [64]` supplied is taken; withheld it is the
    #: mean again and declined.
    assert G._use_prepared(T, nseq, 16, 32, False, dev, G._max_len_of(cu, nseq))
    assert not G._use_prepared(T, nseq, 16, 32, False, dev, G._max_len_of(None, nseq))


@pytest.mark.parametrize("lanes", [[64], [64, 64], [37], [200]])
def test_the_rows_the_prune_calls_zero_are_exactly_zero(lanes):
    """The k-step prune's PREMISE, bitwise, on real data: X_jm is EXACTLY +0 for every m > j."""
    d = make(lanes, 4, 8, seed=800 + sum(lanes))
    q, k, v, a, b = rows(d, slice(None))
    _, saved = G.trainer_forward(q, k, v, a, b, d["A_log"], d["dt_bias"], d["cut"])
    A = saved["A"].view(torch.int16)            # [T, hv, 64], rn(X) row j of its chunk
    upper = torch.triu(torch.ones(64, 64, dtype=torch.bool, device=A.device), diagonal=1)
    for lo, hi in zip(d["cu"][:-1], d["cu"][1:]):
        for t in range(lo, hi):
            j = (t - lo) % 64
            assert not bool(A[t, :, upper[j]].any()), f"token {t} (row {j}) has a nonzero X above its diagonal"


# --------------------------------------------------------------------------- round 5's coverage holes
# THE FIFTH FABLE REVIEW FOUND THAT THE PREPARED COVERAGE HAD MOVED FROM "STOPS WHERE THE RULE DOES" TO
# "STOPS BEFORE THE RULE STARTS". Round 4's rule was `T >= 128 nseq` and the forced-`plan=11` relations top
# out at 257 tokens, five chunks and four sequences -- while the mode SHIPS at 128 to 512 chunks and, since
# round 5 dropped the CTA bound, at any sequence count at all. Nothing in the kernel depends on the chunk
# count or the sequence count, which is why these pass; they exist so that the next change to the plan
# kernel's prefix sum, its binary search or the chunk-slot grid bound cannot pass unseen.


@pytest.mark.parametrize("label,hk,hv", [("tp8", 2, 4), ("tp4", 4, 8)])
def test_the_prepared_mode_agrees_at_the_chunk_counts_it_actually_ships_at(label, hk, hv):
    """Sixty-four and 128 chunks in one sequence, forced against the one-launch kernel."""
    for T in (4096, 8192):
        d = make([T], hk, hv, seed=900 + T)
        ref = prefill(d, plan=0)
        for plan in _PREP_FORMS:
            assert biteq(ref, prefill(d, plan=plan)), f"T={T} plan={plan}"


def test_the_prepared_mode_agrees_at_a_sequence_count_the_rule_no_longer_bounds():
    """320 sequences with mixed and empty lengths, forced against the one-launch kernel."""
    hk, hv = 4, 8
    lanes = []
    for i in range(320):
        lanes.append(0 if i in (0, 7, 160, 319) else (64 + 17 * (i % 9)))
    d = make(lanes, hk, hv, seed=931)
    ref = prefill(d, plan=0)
    for plan in _PREP_FORMS:
        assert biteq(ref, prefill(d, plan=plan)), f"plan {plan}"


def test_the_prepared_mode_agrees_out_of_a_0xff_poisoned_workspace(monkeypatch):
    """The hand-off, the carry and the chunk table are `torch.empty`, and "the stale rows travel verbatim"
    was tested against a poisoned RECORD and never against a poisoned WORKSPACE.

    Every tile row launch B reads is either written by launch A, zero-filled by the load, or reached only
    through a `@p`-guarded `chain_step` or a select -- so 0xFF everywhere must not move a bit. That is the
    argument; this is the test. `torch.empty` is patched for the duration, which reaches exactly the three
    allocations `ops.prefill` makes for the prepared mode (and any the reference path makes, harmlessly).
    """
    hk, hv = 4, 8
    d = make([1000, 300, 0, 457], hk, hv, seed=944)
    ref = prefill(d, plan=0)
    real_empty = torch.empty

    def poisoned(*a, **kw):
        t = real_empty(*a, **kw)
        if t.dtype in (torch.uint8, torch.int32):
            t.fill_(-1 if t.dtype is torch.int32 else 0xFF)
        return t

    monkeypatch.setattr(torch, "empty", poisoned)
    for plan in _PREP_FORMS:
        assert biteq(ref, prefill(d, plan=plan)), f"plan {plan}"


def test_every_head_of_a_slot_holds_the_same_position():
    """`ChunkRec` derives `j0` for every head from head 0's `pos`, and that was argued and never checked."""
    hv = 8
    pool = fresh(6, hv)
    slots = ids([1, 2])
    prefill(make([200, 97], 4, hv, seed=951), state=pool, slot_in=slots)
    d = make([2], 4, hv, seed=952)
    for t in range(3):
        decode_step(d, [0, 1], pool, slots)
    prefill(make([0], 4, hv, seed=953), state=pool, slot_in=ids([1]), slot_out=ids([4]))
    for slot in (1, 2, 4):
        pos = pool.pos[slot]
        assert bool((pos == pos[:1, :1]).all()), (slot, pos)
