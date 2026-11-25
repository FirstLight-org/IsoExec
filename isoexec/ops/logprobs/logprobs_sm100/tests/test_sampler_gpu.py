"""SM100 decode sampler: trainer-equal logprobs, exact pruning, determinism, batch invariance.

The sampled-token logprob the engine returns is bit for bit the trainer's on the same logits (both
recipes, fp32 and bf16 rows, with and without a scoring temperature); rank and top-1 agree with a
gather over the engine's full rows. Tokens must be deterministic, independent of the other rows, and
distributed as softmax(x / t).
"""

from __future__ import annotations

import pytest
import torch

V = 248320

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="requires SM100 (B200)"
)


def _fs():
    from isoexec.ops.logprobs.logprobs_sm100 import sampler

    return sampler


def _lp():
    from isoexec.ops.logprobs.logprobs_sm100 import ops

    return ops


def _bits(t):
    return t.contiguous().view(torch.int32)


def _same(a, b):
    return torch.equal(_bits(a), _bits(b)) if a.dtype == torch.float32 else torch.equal(a, b)


def _hazard_logits(rows, vocab, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, vocab, device="cuda", generator=g) * 4
    if rows > 1:
        x[1, : vocab // 3] = -float("inf")           # a masked prefix
    if rows > 2:
        x[2] = 1.5                                   # an all-equal row: every column ties
    if rows > 3:
        m = x[3].max()
        x[3, [7, vocab // 2, vocab - 1]] = m + 3     # a tied top value in three different splits
    if rows > 4:
        x[4, 100] = 3e30                             # a huge logit
        x[4, 200] = -3e30
    if rows > 5:
        x[5] = torch.round(x[5])                     # many exact ties everywhere
    if rows > 6:
        x[6, vocab // 4:] = -float("inf")
        x[6, 5] = 40.0
    return x.to(dtype).contiguous()


def _peaked(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, V, device="cuda", generator=g) * 2
    cols = torch.randint(0, V, (rows, 30), device="cuda", generator=g)
    x.scatter_add_(1, cols, torch.rand(rows, 30, device="cuda", generator=g) * 10 + 12)
    return x


@pytest.fixture(params=[1, 2], ids=["recipe_v1", "recipe_v2"])
def recipe(request, monkeypatch):
    monkeypatch.setattr(_lp(), "RECIPE", request.param)
    return request.param


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("rows", [1, 8, 256])
@pytest.mark.parametrize("score_temp", [None, 0.7], ids=["T1", "T0.7"])
@pytest.mark.parametrize("k", [0, 1])
def test_logprobs_equal_trainer_and_engine_rows(recipe, dtype, rows, score_temp, k):
    fs, lp = _fs(), _lp()
    x = _hazard_logits(rows, V, dtype, seed=rows * 7 + recipe)
    st = None if score_temp is None else torch.tensor(score_temp, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(rows)
    tok = torch.randint(0, V, (rows,), device="cuda", generator=g)
    if rows > 1:
        tok[1] = 0                                   # a column under the -inf prefix
    if rows > 3:
        tok[3] = 7                                   # the tied top
    idx, lps, ranks = fs.sampled_logprobs(x, tok, k, st)
    assert idx.shape == (rows, 1 + k) and lps.shape == (rows, 1 + k)

    # Trainer: the scorer divides by the tensor temperature, then the sampled-token kernel.
    xt = x.float() if st is None else (x.float() / st)
    assert _same(lps[:, 0], lp.sampled_logprobs(xt, tok))

    # Engine full rows (what vLLM gathers from): sampled value, rank and top-1.
    full = lp.full_logprobs(x, temperature=st)
    tok_lp = full.gather(1, tok[:, None])
    assert _same(lps[:, 0], tok_lp[:, 0])
    assert torch.equal(idx[:, 0].long(), tok)
    assert torch.equal(ranks, (full >= tok_lp).sum(-1))
    if k == 1:
        top_v, top_i = torch.topk(full, 1, dim=-1)
        assert _same(lps[:, 1], top_v[:, 0])
        assert torch.equal(idx[:, 1].long(), top_i[:, 0])


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_engine_logprob_equals_the_trainer_entry_point(recipe, temperature):
    """Trainer through its bound ``logprobs.log_softmax`` op, engine through ``decode_step``."""
    from isoexec.ops.logprobs.eager import log_softmax_rowinv
    from isoexec.ops.logprobs.logprobs_sm100 import integration

    x = torch.cat([_peaked(100, 31), _hazard_logits(28, V, torch.float32, seed=2)]).contiguous()
    st = None if temperature == 1.0 else torch.tensor(temperature, device="cuda")
    tok, (_, lps, _) = _fs().decode_step(x, torch.ones(128, device="cuda"), 77, 400, 1, st)
    trainer = log_softmax_rowinv(x, tok, temperature, _sampled_logprobs=integration.sampled_logprobs)
    assert _same(lps[:, 0], trainer)


def test_nan_row_top1_follows_topk():
    x = _hazard_logits(4, V, torch.float32, seed=3)
    x[0, 999] = float("nan")
    x[0, 50] = float("nan")
    tok = torch.tensor([3, 4, 5, 6], device="cuda")
    idx, lps, ranks = _fs().sampled_logprobs(x, tok, 1, None)
    full = _lp().full_logprobs(x)
    top_v, top_i = torch.topk(full, 1, dim=-1)
    assert _same(lps[:, 1], top_v[:, 0])
    assert torch.equal(idx[:, 1].long(), top_i[:, 0])
    assert torch.equal(ranks, (full >= full.gather(1, tok[:, None])).sum(-1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_logprobs_are_batch_invariant(dtype):
    fs = _fs()
    x = _hazard_logits(64, V, dtype, seed=11)
    tok = torch.arange(64, device="cuda") * 1000
    whole = fs.sampled_logprobs(x, tok, 1, None)
    for i in (0, 3, 17, 63):
        one = fs.sampled_logprobs(x[i:i + 1].clone(), tok[i:i + 1], 1, None)
        for a, b in zip(whole, one):
            assert _same(a[i:i + 1], b)


@pytest.mark.parametrize("kind", ["flat", "peaked", "hazard", "bf16"])
def test_prune_is_exact(kind):
    fs = _fs()
    if kind == "flat":
        x = torch.randn(32, V, device="cuda") * 0.5
    elif kind == "peaked":
        x = _peaked(32, 5)
    elif kind == "hazard":
        x = _hazard_logits(32, V, torch.float32, seed=9)
    else:
        x = _peaked(32, 6).to(torch.bfloat16)
    temps = torch.tensor([1.0, 0.7, 0.0, 1.3] * 8, device="cuda")
    for seed, off in [(0, 0), (1, 4), (12345, 4 * 999), (2**63 + 5, 2**40)]:
        assert torch.equal(fs.sample(x, temps, seed, off, prune=True), fs.sample(x, temps, seed, off, prune=False))


def test_tokens_deterministic_and_independent_of_other_rows():
    fs = _fs()
    x = _peaked(128, 1)
    t = torch.ones(128, device="cuda")
    a = fs.sample(x, t, 42, 8)
    assert torch.equal(a, fs.sample(x, t, 42, 8))
    y = x.clone()
    y[::2] = torch.randn_like(y[::2]) * 7       # rewrite every even row
    assert torch.equal(a[1::2], fs.sample(y, t, 42, 8)[1::2])
    # A row's noise is keyed by its row index: a prefix batch samples the same tokens.
    assert torch.equal(a[:37], fs.sample(x[:37].contiguous(), t[:37].contiguous(), 42, 8))
    # The next step's offset draws different noise.
    assert not torch.equal(a, fs.sample(x, t, 42, 8 + fs.OFFSET_STEP))


def test_greedy_rows_take_the_first_argmax():
    fs = _fs()
    x = _hazard_logits(16, V, torch.float32, seed=21)
    want = x.argmax(-1)
    assert torch.equal(fs.sample(x, None, 0, 0), want)
    got = fs.sample(x, torch.tensor([0.0, 1.0] * 8, device="cuda"), 3, 4)
    assert torch.equal(got[0::2], want[0::2])


def test_sampling_distribution():
    fs = _fs()
    vocab = 64
    g = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn(1, vocab, device="cuda", generator=g)
    xb = logits.expand(4096, vocab).contiguous()
    for temp in (1.0, 0.7):
        t = torch.full((4096,), temp, device="cuda")
        counts = torch.zeros(vocab, device="cuda", dtype=torch.float64)
        for step in range(200):
            counts += torch.bincount(fs.sample(xb, t, 99, fs.OFFSET_STEP * step), minlength=vocab).double()
        n = counts.sum()
        p = torch.softmax(logits[0].double() / temp, -1)
        chi2 = float(((counts - n * p) ** 2 / (n * p)).sum())
        assert chi2 < 130, (temp, chi2)  # 63 dof; p < 1e-6 above 130


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("score_temp", [None, 0.7], ids=["T1", "T0.7"])
def test_decode_step_matches_the_separate_passes(recipe, dtype, score_temp):
    """The leaf maxima as prune bound (plain step) and a separate sampling tensor change nothing."""
    fs, lp = _fs(), _lp()
    x = torch.cat([_peaked(40, 8), _hazard_logits(24, V, torch.float32, seed=4)]).to(dtype).contiguous()
    st = None if score_temp is None else torch.tensor(score_temp, device="cuda")
    temps = torch.tensor([1.0, 0.7, 0.0, 1.0] * 16, device="cuda")
    tok, got = fs.decode_step(x, temps, 5, 40, 1, st)
    assert torch.equal(tok, fs.sample(x, temps, 5, 40, prune=False))
    for a, b in zip(got, fs.sampled_logprobs(x, tok, 1, st)):
        assert _same(a, b)
    xt = x.float() if st is None else x.float() / st
    # Processors applied to a copy: tokens from xs, logprobs from x.
    xs = x.float().clone()
    xs[:, :1000] = -float("inf")
    tok2, (_, lps2, _) = fs.decode_step(x, temps, 5, 40, 1, st, xs=xs)
    assert torch.equal(tok2, fs.sample(xs, temps, 5, 40, prune=False))
    assert bool((tok2 >= 1000).all())
    assert _same(lps2[:, 0], lp.sampled_logprobs(xt, tok2))
    # Greedy-only, logprob-free step.
    tok3, none = fs.decode_step(x, None, 0, 0, None, st)
    assert none is None and torch.equal(tok3, x.float().argmax(-1))
