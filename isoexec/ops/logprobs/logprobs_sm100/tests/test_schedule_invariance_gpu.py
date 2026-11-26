"""logprobs_sm100 on the device: the kernels are the lane programs ``schedule.py`` states (through
``reference.py``), and each kernel is bitwise invariant against itself. Both reducers (recipes 1 and 2).

What is varied: repeated runs; batch size and row position; batch composition; logits dtype (bf16 = widened
fp32); storage (strided views, fresh clones); engine full rows = trainer sampled logprobs (no_grad and
autograd); one call against chunked calls; vocabulary shards at TP 1 / 2 / 4 / 8 (emulated on one device, as
the tree's own TP test does); masked (-inf) columns, whole masked lanes and shards. Equality is on raw bits
(NaN payloads included), except against ``reference.py``, where no NaN is produced.
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
G = schedule.VOCAB_SHARDS
MODEL_VOCABS = (248320, 154880, 151680, 129280)
#: shard lengths of 1 strip, fewer strips than lanes, exactly one strip per lane, one more, and several visits
SMALL_VOCABS = tuple(G * 8 * strips for strips in (1, 5, 255, 256, 257, 512, 700))
recipes = pytest.mark.parametrize("recipe", sorted(schedule.REDUCERS))
dtypes = pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16), ids=("fp32", "bf16"))


def _logits(rows, vocab, seed, dtype=torch.float32, scale=4.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(rows, vocab, device=DEV, generator=g) * scale).to(dtype)


def _masked(vocab, seed, dtype):
    x = _logits(6, vocab, seed, dtype)
    s = vocab // G
    x[1, 3 : vocab // 3] = float("-inf")
    x[2, :s] = float("-inf")                       # a whole shard masked
    x[3, s + 8 : s + 16] = float("-inf")           # one strip (lane 1's first) of shard 1
    x[4, 2 * s : 3 * s : 16] = float("-inf")       # scattered columns of shard 2
    x[5] = float("-inf")
    x[5, vocab - 1] = 0.5                          # a single live column
    return x


@recipes
@dtypes
@pytest.mark.parametrize("vocab", SMALL_VOCABS + MODEL_VOCABS)
def test_the_kernel_is_the_stated_lane_program(vocab, dtype, recipe):
    """(M, S) per shard and the full rows, bit for bit against the transcription of schedule.py's DAG."""
    x = _masked(vocab, vocab + recipe, dtype)
    m, s = ops.leaf_states(x, G, recipe=recipe)
    rm, rs = R.shard_states(x, G, recipe)
    assert bitwise_equal(m, rm) and bitwise_equal(s, rs), (vocab, recipe)
    full = ops.full_logprobs(x, recipe=recipe)
    assert bitwise_equal(full, R.full_logprobs(x, recipe)), (vocab, recipe)
    for _ in range(2):
        assert bitwise_equal(ops.full_logprobs(x, recipe=recipe), full)


@recipes
@dtypes
@pytest.mark.parametrize("vocab", (248320, 129280, 4096, 320))
def test_a_row_alone_is_the_row_in_every_batch_at_every_position(vocab, dtype, recipe):
    probe = _logits(1, vocab, 11, dtype)
    probe[0, 5 : vocab // 7] = float("-inf")
    target = torch.tensor([vocab // 3], device=DEV)
    alone = ops.full_logprobs(probe, recipe=recipe)
    alone_sampled = ops.sampled_logprobs(probe, target, recipe=recipe)
    assert bitwise_equal(alone[0, vocab // 3], alone_sampled[0])
    for batch in (2, 3, 7, 8, 9, 31, 32, 33, 128, 257):
        if batch * vocab > 40_000_000:
            continue
        x = _logits(batch, vocab, batch, dtype, scale=9.0)
        for pos in sorted({0, batch // 2, batch - 1}):
            y = x.clone()
            y[pos] = probe[0]
            assert bitwise_equal(ops.full_logprobs(y, recipe=recipe)[pos], alone[0]), (vocab, batch, pos)
            t = torch.randint(0, vocab, (batch,), device=DEV)
            t[pos] = target[0]
            assert bitwise_equal(ops.sampled_logprobs(y, t, recipe=recipe)[pos], alone_sampled[0]), (vocab, batch, pos)


@recipes
@pytest.mark.parametrize("fill", ("nan", "inf", "neginf", "zero", "huge"))
def test_neighbours_do_not_change_a_row(fill, recipe):
    vocab = 4096
    x = _logits(9, vocab, 21)
    want = ops.full_logprobs(x, recipe=recipe)
    value = {"nan": float("nan"), "inf": float("inf"), "neginf": float("-inf"), "zero": 0.0, "huge": 3e38}[fill]
    mixed = torch.full((18, vocab), value, device=DEV)
    mixed[::2] = x
    got = ops.full_logprobs(mixed, recipe=recipe)
    assert bitwise_equal(got[::2], want), fill
    assert bitwise_equal(got[1::2], ops.full_logprobs(mixed[1::2].clone(), recipe=recipe)), fill


@recipes
@dtypes
@pytest.mark.parametrize("vocab", (248320, 154880, 151680, 129280, 4096, 320))
def test_vocabulary_shards_at_every_tp_degree(vocab, dtype, recipe):
    """A rank's shard partials are the same numbers whatever the world size; the row tree is over all G."""
    x = _logits(17, vocab, 31, dtype)
    x[5, : vocab // G] = float("-inf")
    whole = ops.leaf_states(x, G, recipe=recipe)
    for world in (1, 2, 4, 8):
        local = vocab // world
        parts = [ops.leaf_states(x[:, r * local : (r + 1) * local].contiguous(), G // world, recipe=recipe)
                 for r in range(world)]
        m, s = torch.cat([p[0] for p in parts], 1), torch.cat([p[1] for p in parts], 1)
        assert bitwise_equal(m, whole[0]) and bitwise_equal(s, whole[1]), (vocab, world)
        rM, rL = torch.empty(17, device=DEV), torch.empty(17, device=DEV)
        ops._ext(recipe).merge_rows(m, s, rM, rL, None, None, None, None, None, vocab, None)
        full = (x.float() - rM[:, None]) - rL[:, None]
        assert bitwise_equal(full, ops.full_logprobs(x, recipe=recipe)), (vocab, world)


@recipes
@dtypes
def test_engine_rows_equal_trainer_sampled_logprobs(dtype, recipe):
    vocab = 154880
    x = _logits(40, vocab, 41, dtype)
    t = torch.randint(0, vocab, (40,), device=DEV)
    with torch.no_grad():
        engine = ops.full_logprobs(x, recipe=recipe)
        sampled = ops.sampled_logprobs(x, t, recipe=recipe)
        chunks = torch.cat([ops.full_logprobs(x[a:b].clone(), recipe=recipe) for a, b in ((0, 1), (1, 24), (24, 40))])
    xg = x.detach().clone().requires_grad_(True)
    trainer = ops.sampled_logprobs(xg, t, recipe=recipe)
    assert bitwise_equal(engine.gather(1, t[:, None]).squeeze(1), sampled)
    assert bitwise_equal(trainer.detach(), sampled) and bitwise_equal(chunks, engine)
    assert bitwise_equal(ops.full_logprobs(x.float(), recipe=recipe), engine)  # bf16 logits are widened exactly
    view = torch.empty(40, 2 * vocab, device=DEV, dtype=dtype)
    view[:, ::2] = x
    assert bitwise_equal(ops.full_logprobs(view[:, ::2], recipe=recipe), engine)


@recipes
@dtypes
def test_masked_columns_do_not_depend_on_the_batch(dtype, recipe):
    vocab = 16448
    x = _masked(vocab, 51, dtype)
    full = ops.full_logprobs(x, recipe=recipe)
    for i in range(x.shape[0]):
        assert bitwise_equal(ops.full_logprobs(x[i : i + 1].clone(), recipe=recipe)[0], full[i]), i
    pad = torch.cat([x, torch.full((7, vocab), float("-inf"), device=DEV, dtype=dtype)])
    assert bitwise_equal(ops.full_logprobs(pad, recipe=recipe)[: x.shape[0]], full)
