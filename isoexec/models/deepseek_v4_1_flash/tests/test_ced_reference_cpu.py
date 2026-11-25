"""The two CED programs on the CPU reference (``reference.py``), bitwise, on a tiny synthetic CED (window 8).

(i) bounded replay == "encoder on every token, decoder on the segment with its window truncated at the segment
start" (the flat program with the decoder windows floored); (ii) full decoder == the reference code's flat
program; the trainer's per-segment scoring == the engine's bounded replay (chunked prefill + decode), single-turn
and multi-turn, packed with other sequences; every chunk split of the prompt gives the same decoder rows, state
and logits.
"""
import pytest
import torch

from isoexec.models.deepseek_v4_1_flash import reference as R
from isoexec.models.deepseek_v4_1_flash.config import BOUNDED_REPLAY, FULL_DECODER
from isoexec.models.deepseek_v4_1_flash.program import DecoderSegments, Turn


@pytest.fixture(scope="module")
def ref():
    return R.tiny(seed=0, window=8)


def tokens(n, seed):
    return torch.randint(0, 64, (n,), generator=torch.Generator().manual_seed(seed))


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.int32) if a.dtype == torch.float32
                                                                      else a, b.view(torch.int32)
                                                                      if b.dtype == torch.float32 else b)


def test_full_decoder_is_the_reference_codes_flat_program(ref):
    seqs = [tokens(23, 1), tokens(9, 2), tokens(31, 3)]
    cu = [0, 23, 32, 63]
    logits, _ = R.ced_forward(ref, seqs, DecoderSegments.all_rows(cu))
    flat = torch.cat([R.flat_forward(ref, s) for s in seqs])
    assert same(logits, flat)


@pytest.mark.parametrize("prompt", [7, 8, 9, 21])
def test_bounded_replay_is_the_decoder_on_its_segment_with_the_window_truncated(ref, prompt):
    W, T = ref.W, prompt + 6
    seq = tokens(T, prompt)
    s = max(0, prompt - W)
    segs = DecoderSegments.replay([0, T], [[Turn(prompt, T)]], W)
    logits, trace = R.ced_forward(ref, [seq], segs)
    masked = R.flat_forward(ref, seq, decoder_window_floor=s)
    assert same(logits, masked[s:])
    exact = R.flat_forward(ref, seq)
    if prompt <= W:                 # the segment starts at 0: the two programs coincide
        assert same(logits, exact)
    else:                           # the replay changes every decoder row of the segment (report p.20)
        assert not torch.equal(logits[-1], exact[-1])
    # the handoff and the decoder global KV are the full programs' (every token), whatever the segment
    _, full_trace = R.ced_forward(ref, [seq], DecoderSegments.all_rows([0, T]))
    assert same(trace.handoff.streams, full_trace.handoff.streams) and same(trace.handoff.pre, full_trace.handoff.pre)
    assert same(trace.decoder_kv["main_kv"][0], full_trace.decoder_kv["main_kv"][0])
    assert same(trace.decoder_kv["index_k"][0], full_trace.decoder_kv["index_k"][0])


CHUNKINGS = {7: [[7], [1] * 7, [3, 4]], 8: [[8], [5, 3]], 9: [[9], [1] * 9, [4, 5], [8, 1], [1, 8]],
             21: [[21], [1] * 21, [5, 5, 5, 6], [13, 8], [12, 9], [2, 19], [20, 1]]}


@pytest.mark.parametrize("prompt", sorted(CHUNKINGS))
def test_engine_bounded_replay_is_chunk_invariant_and_equals_trainer_scoring(ref, prompt):
    gen = 6
    seq = tokens(prompt + gen, 100 + prompt)
    engine = R.ReferenceEngine(ref, BOUNDED_REPLAY)
    runs = [R.rollout(engine, seq[:prompt], seq[prompt:], chunks) for chunks in CHUNKINGS[prompt]]
    logits0, st0 = runs[0]
    for logits, st in runs[1:]:
        assert same(logits, logits0)
        for L in st0.window:
            assert same(st.window[L][:prompt + gen], st0.window[L][:prompt + gen])
        for L in st0.main_kv:
            assert all(same(a, b) for a, b in zip(st.main_kv[L], st0.main_kv[L])) and \
                len(st.main_kv[L]) == len(st0.main_kv[L])
            assert all(same(a, b) for a, b in zip(st.index_k[L], st0.index_k[L]))
        for L in st0.decoder_window:
            assert same(st.decoder_window[L], st0.decoder_window[L])
    # the decoder ran exactly on [max(0, P - W), end): its window store holds those rows and nothing before
    s = max(0, prompt - ref.W)
    any_layer = next(iter(st0.decoder_window.values()))
    assert not any_layer[:s].any() and any_layer[s:prompt + gen - 1].abs().sum(-1).gt(0).all()
    # trainer scoring of the same rollout, one packed forward, per decoder segment
    tl, targets, segs, trace = R.trainer_logits(ref, [seq], [[Turn(prompt, prompt + gen)]], BOUNDED_REPLAY)
    assert same(tl, logits0)
    assert same(R.Reference.logprobs(tl, targets), R.Reference.logprobs(logits0, seq[prompt:]))
    # the decoder window keys the trainer computed at its segment rows are the engine's cache records
    rows = segs.positions.tolist()
    for L, K in trace.decoder_window.items():
        assert same(K[:len(rows) - 1], st0.decoder_window[L][rows[:-1]])


def test_engine_full_decoder_is_the_flat_program(ref):
    prompt, gen = 13, 5
    seq = tokens(prompt + gen, 7)
    engine = R.ReferenceEngine(ref, FULL_DECODER)
    flat = R.flat_forward(ref, seq)
    for chunks in ([13], [6, 7], [1] * 13):
        logits, _ = R.rollout(engine, seq[:prompt], seq[prompt:], chunks)
        assert same(logits, flat[prompt - 1:prompt + gen - 1])
    tl, _, _, _ = R.trainer_logits(ref, [seq], [[Turn(prompt, prompt + gen)]], FULL_DECODER)
    assert same(tl, logits)


def test_multi_turn_and_packing_score_each_segment_as_the_engine_generated_it(ref):
    """Turn 1: prompt 11, 4 generated; turn 2 (a fresh engine request): turn 1 + 6 user tokens, 5 generated. The
    trainer packs the whole conversation once (plus an unrelated sequence) with two overlapping segments."""
    W = ref.W
    conv = tokens(11 + 4 + 6 + 5, 42)
    other = tokens(17, 43)
    engine = R.ReferenceEngine(ref, BOUNDED_REPLAY)
    e1, _ = R.rollout(engine, conv[:11], conv[11:15], [6, 5])
    e2, _ = R.rollout(engine, conv[:21], conv[21:], [9, 12])
    e3, _ = R.rollout(engine, other[:10], other[10:], [10])
    turns = [[Turn(10, 17)], [Turn(11, 15), Turn(21, 26)]]
    tl, _, segs, _ = R.trainer_logits(ref, [other, conv], turns, BOUNDED_REPLAY)
    assert len(segs.segments) == 3 and segs.segments[1].start == 11 - W and segs.segments[2].start == 21 - W
    assert same(tl, torch.cat([e3, e1, e2]))
    # the two turns' decoder rows overlap (positions 13..14 of turn 1 are replayed by turn 2) with different
    # decoder states: P-replay is a set of segments, P-exact is one
    exact, _, _, _ = R.trainer_logits(ref, [conv], [[Turn(11, 15), Turn(21, 26)]], FULL_DECODER)
    assert not torch.equal(exact[-5:], e2)
