"""The logprobs_sm100 schedule is the policy's output: CPU checks, no extension build.

Fails if the segment length or shard count the kernel is compiled with, or the executable statement in
``reference.py``, departs from :mod:`isoexec.ops.schedule_policy` on a deployed vocabulary.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
import torch

from isoexec.ops import schedule_policy as policy
from isoexec.ops.logprobs.logprobs_sm100 import ops, schedule, spec
from isoexec.ops.logprobs.logprobs_sm100 import reference as R

CSRC = Path(__file__).resolve().parents[1] / "csrc"
#: Qwen3.5 (248320), GLM (154880), MiMo (151680), DeepSeek (129280), and small test vocabularies.
VOCABS = (248320, 154880, 151680, 129280, 4096, 2048, 320, 64)


def test_policy_version():
    assert policy.VERSION == "7.1"


@pytest.mark.parametrize("vocab", VOCABS)
def test_segments_are_the_policy_output(vocab):
    assert ops.supports(vocab)
    n = vocab // schedule.VOCAB_SHARDS
    want = policy.reduce(n, b=1, L=schedule.LSE_SHORTEST_SEGMENT)
    assert schedule.vocab_sum(vocab) == want == policy.Segments(schedule.LSE_SHORTEST_SEGMENT, None, "tree")
    assert f"-DLOGPROBS_SM100_SEGMENT={want.S}" in schedule.cuda_defines()
    assert f"-DLOGPROBS_SM100_SHARDS={schedule.VOCAB_SHARDS}" in schedule.cuda_defines()


def test_d_max_is_the_trees_declaration():
    """d_max is spec.LEAVES, the declaration ``ops.supports`` already reads: every world that divides it."""
    assert schedule.VOCAB_SHARDS == spec.LEAVES == 8
    assert [w for w in range(1, 17) if ops.supports(248320, w)] == [1, 2, 4, 8]


def test_the_published_constant_has_one_definition():
    assert schedule.LSE_SHORTEST_SEGMENT == 32
    text = "\n".join(p.read_text() for p in sorted(CSRC.iterdir()))
    assert not re.search(r"#\s*define\s+LOGPROBS_SM100_(SEGMENT|SHARDS)", text)
    kernel = (CSRC / "logprobs.cu").read_text()
    assert "#error" in kernel and "constexpr int SEG = LOGPROBS_SM100_SEGMENT;" in kernel
    assert "constexpr int G = LOGPROBS_SM100_SHARDS;" in kernel
    assert "*cuda_defines()" in (CSRC.parent / "_extension.py").read_text()


def test_the_per_class_recipes_are_gone():
    assert not (CSRC / "logprobs_v2.cu").exists() and not (CSRC.parent / "_extension_v2.py").exists()
    assert not hasattr(spec, "DEFAULT_RECIPE_BY_CLASS") and not hasattr(spec, "IMPL_V2")
    kernel = (CSRC / "logprobs.cu").read_text()
    assert "q += LANES" not in kernel  # the strided lanes of recipes v1 / v2


def _merge(l, r):
    import numpy as np

    f32 = np.float32
    (ml, sl), (mr, sr) = l, r
    m = max(ml, mr)
    f = f32(0.0) if m == -math.inf else m
    with np.errstate(all="ignore"):
        return m, f32(f32(sl * np.exp(f32(ml - f), dtype=f32)) + f32(sr * np.exp(f32(mr - f), dtype=f32)))


def _statement(row: list[float], shards: int, seg: policy.Segments):
    """The schedule term by term in numpy fp32, independently of reference.py: (M, S) of one row."""
    import numpy as np

    f32 = np.float32
    cols = len(row) // shards
    tops = []
    for g in range(shards):
        parts = []
        for a in range(0, cols, seg.S):
            v = [f32(x) for x in row[g * cols + a : g * cols + min(a + seg.S, cols)]]
            m = f32(-math.inf)
            for x in v:
                m = max(m, x)
            f = f32(0.0) if m == -math.inf else m
            s = f32(0.0)
            for x in v:
                with np.errstate(all="ignore"):
                    s = f32(s + np.exp(f32(x - f), dtype=f32))
            parts.append((m, s))
        while len(parts) > 1:
            parts = [_merge(parts[i], parts[i + 1]) for i in range(0, len(parts) - 1, 2)] + parts[len(parts) - len(parts) % 2 :]
        tops.append(parts[0])
    while len(tops) > 1:
        tops = [_merge(tops[i], tops[i + 1]) for i in range(0, len(tops) - 1, 2)]
    return tops[0]


@pytest.mark.parametrize("vocab", (64, 320, 512, 2048, 4160))
def test_reference_is_segments_then_adjacent_trees(vocab):
    """reference.py against a term-by-term evaluation: same maximum bits, sums within the last few ulp
    (numpy's and torch's fp32 exp are not the same routine, so the sum is compared to 4 ulp per term chain)."""
    g = torch.Generator().manual_seed(vocab)
    x = torch.randn(3, vocab, generator=g) * 4
    x[1, 5 : vocab // 3] = float("-inf")
    seg = schedule.vocab_sum(vocab)
    m, s, _ = R.row_states(*R.shard_states(x, schedule.VOCAB_SHARDS))
    for r in range(3):
        wm, ws = _statement(x[r].tolist(), schedule.VOCAB_SHARDS, seg)
        assert m[r].item() == float(wm), (vocab, r)
        assert abs(s[r].item() - float(ws)) <= 1e-5 * abs(float(ws)), (vocab, r)


def test_a_short_last_segment_is_a_segment_of_its_own():
    """V / d_max = 40 columns: one segment of 32 and one of 8 per shard, merged in that order."""
    x = torch.randn(1, 320, generator=torch.Generator().manual_seed(1))
    m, s, a = R.shard_states(x, 8)
    for g in range(8):
        v = x[:, g * 40 : (g + 1) * 40]
        want = R.merge(R.segment_partials(v[:, :32]), R.segment_partials(v[:, 32:]))
        assert torch.equal(m[:, g], want[0]) and torch.equal(s[:, g], want[1]) and torch.equal(a[:, g], want[2])
