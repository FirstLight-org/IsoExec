"""Bitwise invariance of the ported logprobs_sm100 kernel (the policy's schedule) against itself and its statement.

What is varied: repeated runs; batch size and row position; batch composition; logits dtype (bf16 = widened fp32);
storage (views, fresh clones); engine full rows = trainer sampled logprobs (no_grad and autograd); vocabulary
shards at TP 1 / 2 / 4 / 8 (emulated on one device, as the tree's own TP test does); shard lengths around every
segment boundary (S - 8 ... 2S + 8, a short last segment); masked (-inf) columns, whole masked segments and shards;
temperature; the entropy partial. Equality is on raw bits (NaN payloads included).
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.logprobs.logprobs_sm100 import ops, schedule
from isoexec.ops.logprobs.logprobs_sm100 import reference as R

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="requires B200/SM100"
)

DEV = "cuda"
S, G = schedule.LSE_SHORTEST_SEGMENT, schedule.VOCAB_SHARDS
#: shard lengths around the segment boundaries (a shard is whole 8-column strips), times G
BOUNDARY_VOCABS = tuple(G * n for n in (8, S - 8, S, S + 8, 2 * S - 8, 2 * S, 2 * S + 8, 5 * S + 24, 127 * 8, 1024, 1032,
                                         4 * 256 * S, 4 * 256 * S + 8))
MODEL_VOCABS = (248320, 154880, 151680, 129280)


def _ext():
    return ops._ext(None)


def _logits(rows, vocab, seed, dtype=torch.float32, scale=4.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(rows, vocab, device=DEV, generator=g) * scale).to(dtype)


def _states(x, shards, temperature=None, entropy=False):
    m = torch.empty(x.shape[0], shards, device=DEV)
    s, a = torch.empty_like(m), (torch.empty_like(m) if entropy else None)
    _ext().leaf_states(x, shards, m, s, temperature, a)
    return m, s, a


def test_the_compiled_schedule_is_the_policy_output():
    seg, shards = _ext().schedule()
    assert shards == schedule.VOCAB_SHARDS
    for vocab in MODEL_VOCABS + BOUNDARY_VOCABS:
        assert seg == schedule.vocab_sum(vocab).S, vocab


@pytest.mark.parametrize("vocab", BOUNDARY_VOCABS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16), ids=("fp32", "bf16"))
def test_shard_partials_and_rows_are_the_stated_schedule(vocab, dtype):
    """(M, S, A) per shard and the full row, bit for bit, at shard lengths around every segment boundary."""
    x = _logits(5, vocab, vocab, dtype)
    x[1, 3 : vocab // 3] = float("-inf")
    x[2, : vocab // G] = float("-inf")  # a whole shard masked
    x[3, S : 2 * S] = float("-inf") if vocab // G >= 2 * S else x[3, S : 2 * S]  # a whole segment masked
    m, s, a = _states(x, G, entropy=True)
    rm, rs, ra = R.shard_states(x, G)
    assert bitwise_equal(m, rm) and bitwise_equal(s, rs) and bitwise_equal(a, ra), vocab
    full = ops.full_logprobs(x)
    assert bitwise_equal(full, R.full_logprobs(x)), vocab
    for _ in range(2):
        assert bitwise_equal(ops.full_logprobs(x), full)
    ent = torch.empty(5, device=DEV)
    _ext().entropy_rows(m, s, a, ent)
    assert bitwise_equal(ent, R.entropy(x)), vocab


@pytest.mark.parametrize("vocab", MODEL_VOCABS)
def test_model_vocabularies_are_the_stated_schedule(vocab):
    x = _logits(3, vocab, 5)
    x[1, 1000:90000] = float("-inf")
    m, s, _ = _states(x, G)
    rm, rs, _ = R.shard_states(x, G)
    assert bitwise_equal(m, rm) and bitwise_equal(s, rs)
    assert bitwise_equal(ops.full_logprobs(x), R.full_logprobs(x))
    t = torch.tensor([0.7], device=DEV)
    assert bitwise_equal(ops.full_logprobs(x, temperature=t), R.full_logprobs(x, 0.7))


@pytest.mark.parametrize("vocab", (248320, 129280, 4096, 320))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16), ids=("fp32", "bf16"))
def test_a_row_alone_is_the_row_in_every_batch_at_every_position(vocab, dtype):
    probe = _logits(1, vocab, 11, dtype)
    target = torch.tensor([vocab // 3], device=DEV)
    alone = ops.full_logprobs(probe)
    alone_sampled = ops.sampled_logprobs(probe, target)
    assert bitwise_equal(alone[0, vocab // 3], alone_sampled[0])
    for batch in (2, 3, 7, 8, 9, 31, 32, 33, 128, 257):
        if batch * vocab > 40_000_000:
            continue
        x = _logits(batch, vocab, batch, dtype, scale=9.0)
        for pos in sorted({0, batch // 2, batch - 1}):
            y = x.clone()
            y[pos] = probe[0]
            assert bitwise_equal(ops.full_logprobs(y)[pos], alone[0]), (vocab, batch, pos)
            t = torch.randint(0, vocab, (batch,), device=DEV)
            t[pos] = target[0]
            assert bitwise_equal(ops.sampled_logprobs(y, t)[pos], alone_sampled[0]), (vocab, batch, pos)


@pytest.mark.parametrize("fill", ("nan", "inf", "neginf", "zero", "huge"))
def test_neighbours_do_not_change_a_row(fill):
    vocab = 4096
    x = _logits(9, vocab, 21)
    want = ops.full_logprobs(x)
    value = {"nan": float("nan"), "inf": float("inf"), "neginf": float("-inf"), "zero": 0.0, "huge": 3e38}[fill]
    mixed = torch.full((18, vocab), value, device=DEV)
    mixed[::2] = x
    got = ops.full_logprobs(mixed)
    assert bitwise_equal(got[::2], want), fill
    assert bitwise_equal(got[1::2], R.full_logprobs(mixed[1::2])), fill


@pytest.mark.parametrize("vocab", (248320, 154880, 151680, 129280, 4096, 320))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16), ids=("fp32", "bf16"))
def test_vocabulary_shards_at_every_tp_degree(vocab, dtype):
    """A rank's shard partials are the same numbers whatever the world size; the row tree is over all G."""
    x = _logits(17, vocab, 31, dtype)
    x[5, : vocab // G] = float("-inf")
    whole = _states(x, G)[:2]
    for world in (1, 2, 4, 8):
        local = vocab // world
        parts = [_states(x[:, r * local : (r + 1) * local].contiguous(), G // world)[:2] for r in range(world)]
        m, s = torch.cat([p[0] for p in parts], 1), torch.cat([p[1] for p in parts], 1)
        assert bitwise_equal(m, whole[0]) and bitwise_equal(s, whole[1]), (vocab, world)
        rM, rL = torch.empty(17, device=DEV), torch.empty(17, device=DEV)
        _ext().merge_rows(m, s, rM, rL, None, None, None, None, None, vocab, None)
        full = (x.float() - rM[:, None]) - rL[:, None]
        assert bitwise_equal(full, ops.full_logprobs(x)), (vocab, world)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16), ids=("fp32", "bf16"))
def test_engine_rows_equal_trainer_sampled_logprobs(dtype):
    vocab = 154880
    x = _logits(40, vocab, 41, dtype)
    t = torch.randint(0, vocab, (40,), device=DEV)
    with torch.no_grad():
        engine = ops.full_logprobs(x)
        sampled = ops.sampled_logprobs(x, t)
        chunks = torch.cat([ops.full_logprobs(x[a:b].clone()) for a, b in ((0, 1), (1, 24), (24, 40))])
    xg = x.detach().clone().requires_grad_(True)
    trainer = ops.sampled_logprobs(xg, t)
    assert bitwise_equal(engine.gather(1, t[:, None]).squeeze(1), sampled)
    assert bitwise_equal(trainer.detach(), sampled) and bitwise_equal(chunks, engine)
    assert bitwise_equal(ops.full_logprobs(x.float()), engine)  # bf16 logits are widened exactly
    view = torch.empty(40, 2 * vocab, device=DEV, dtype=dtype)
    view[:, ::2] = x
    assert bitwise_equal(ops.full_logprobs(view[:, ::2]), engine)
