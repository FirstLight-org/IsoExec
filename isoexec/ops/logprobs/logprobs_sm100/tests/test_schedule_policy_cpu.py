"""logprobs_sm100: ``schedule.py`` against the policy and against what the kernels and the recipe table use.

CPU only, no extension build: the kernels' constants and launch geometry are read from ``csrc/*.cu`` as text.
Fails if the segment differs from :func:`isoexec.ops.schedule_policy.reduce`, or if the shard count, the lane
count, the lane program's text or the recipe binding differ from ``schedule.py``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from isoexec.ops import schedule_policy as policy
from isoexec.ops.logprobs.logprobs_sm100 import _extension, _extension_v2, ops, schedule, spec

PKG = Path(__file__).resolve().parents[1]
SRC = {v: " ".join((PKG / "csrc" / r.source).read_text().split()) for v, r in schedule.REDUCERS.items()}
MODEL_VOCABS = (248320, 154880, 151680)
VOCABS = MODEL_VOCABS + (64, 128, 320, 4096, 16384, 16448, 129280)


def _const(text: str, name: str) -> int:
    m = re.search(r"constexpr int " + name + r" = (\d+);", text)
    assert m, name
    return int(m.group(1))


def test_policy_version():
    assert policy.VERSION == "7.2"


@pytest.mark.parametrize("vocab", VOCABS)
def test_the_segment_is_the_policy_output(vocab):
    """One segment per shard at d_max = 8: G = 1, S = V / 8, shards under a tree; the lanes partition it."""
    seg = schedule.segment(vocab)
    n = vocab // schedule.VOCAB_SHARDS
    assert seg == policy.reduce(n) == policy.Segments(n, 1, "tree")
    for red in schedule.REDUCERS.values():
        assert red.lanes(seg.S) == schedule.LANES
        cols = [c for lane in range(schedule.LANES) for c in red.members(seg.S, lane)]
        assert sorted(cols) == list(range(seg.S))
        assert red.members(seg.S, 0)[: red.strip] == tuple(range(red.strip))
        if seg.S >= (schedule.LANES + 4) * red.strip:   # lane 3 has a second strip
            assert red.members(seg.S, 3)[red.strip] == red.strip * (3 + schedule.LANES)
            assert len(red.members(seg.S, schedule.LANES - 1)) <= len(red.members(seg.S, 3))


def test_the_shards_are_the_kernels_leaves_and_d_max_is_8():
    assert schedule.VOCAB_SHARDS == spec.LEAVES == ops.LEAVES
    for vocab in VOCABS:
        assert schedule.supported(vocab)
        assert [w for w in range(1, 33) if ops.supports(vocab, w)] == [1, 2, 4, 8], vocab
    assert not schedule.supported(100) and not ops.supports(100)


@pytest.mark.parametrize("version", (1, 2))
def test_the_lane_count_equals_the_kernels_constants(version):
    """Lanes, strip and shard count: schedule.py = spec.py = the constants and launch geometry of the source."""
    text = SRC[version]
    assert _const(text, "LANES") == schedule.LANES == spec.LANES
    assert _const(text, "STRIP") == schedule.STRIP == spec.STRIP
    assert _const(text, "G") == schedule.VOCAB_SHARDS
    # one CTA per (row, leaf), LANES threads, a leaf = V / leaves columns
    assert "int L = V / leaves_local;" in text and "dim3 grid(rows, leaves_local);" in text
    assert text.count("<<<grid, LANES, 0, stream>>>") == 2
    assert "leaf_state_split<T, TEMP><<<dim3(rows, G), LANES, 0, stream>>>" in text and "int L = V / G;" in text
    assert "__launch_bounds__(LANES) void leaf_state(" in text


@pytest.mark.parametrize("version", (1, 2))
def test_the_kernel_text_is_the_stated_lane_program(version):
    text = SRC[version]
    # membership and order: lane = thread; strips lane, lane + LANES, ...; columns ascending
    assert "int strips = L / STRIP;" in text
    assert "for (int q = threadIdx.x; q < strips; q += LANES) {" in text
    # the lanes combine by one ascending chain from lane 0
    assert "float S = lsum[0]; for (int l = 1; l < LANES; ++l) S = __fadd_rn(S, lsum[l]);" in text
    # the shard merge: fp64, adjacent tree, lower shard on the left
    assert "w[g] = double(sg[g]) * exp(double(mg[g]) - double(M));" in text
    assert "for (int step = 1; step < G; step *= 2) #pragma unroll for (int g = 0; g + step < G; g += 2 * step) " \
        "w[g] = w[g] + w[g + step];" in text
    assert "lsout = M == NINF ? double(__builtin_nan(\"\")) : log(w[0]);" in text
    assert "o.v[j] = __fsub_rn(__fsub_rn(value<T, TEMP>(p.v[j], t), M), L32);" in text
    if version == 2:
        assert schedule.V2.normalizer == "segment_max"
        assert "smax[1] = mm == NINF ? 0.f : mm;" in text
        assert "float d = __fsub_rn(v, F); float e = expf(d); s = __fadd_rn(s, e);" in text
    else:
        assert schedule.V1.normalizer == "online"
        assert "s = __fadd_rn(__fmul_rn(s, expf(__fsub_rn(m, v))), 1.f); m = v;" in text
        assert "} else if (v != NINF) { float d = __fsub_rn(v, m); float e = expf(d); s = __fadd_rn(s, e);" in text
        assert "lsum[threadIdx.x] = absent ? 0.f : __fmul_rn(s, expf(__fsub_rn(m, M)));" in text


def test_the_shard_tree_is_adjacent():
    assert schedule.shard_tree() == (((0, 1), (2, 3), (4, 5), (6, 7)), ((0, 2), (4, 6)), ((0, 4),))
    assert schedule.segment(MODEL_VOCABS[0]).merge == "tree"


def test_the_binding_is_the_trees_table():
    assert schedule.BINDING == spec.DEFAULT_RECIPE_BY_CLASS
    assert {v: r.impl_id for v, r in schedule.REDUCERS.items()} == spec.IMPL_BY_RECIPE
    assert set(ops._EXTENSIONS) == set(schedule.REDUCERS)
    assert schedule.V1.source in _extension.SOURCES and schedule.V2.source in _extension_v2.SOURCES
    for (dtype, vocab), version in schedule.BINDING.items():
        torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        assert ops.default_recipe(torch_dtype, vocab) == version == schedule.reducer(dtype, vocab).version
    assert sorted({v for _, v in schedule.BINDING}) == sorted(MODEL_VOCABS)
    # an unlisted class runs the default reducer
    assert ops.default_recipe(torch.bfloat16, 129280) == schedule.DEFAULT_VERSION == spec.RECIPE_VERSION
    assert schedule.reducer("bf16", 129280) is schedule.V1


def test_the_policy_is_not_on_a_call_path():
    """No module of the package imports ``schedule`` or the policy; importing the ops does not load them."""
    for path in sorted(PKG.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".h", ".cpp") and "tests" not in path.parts \
                and path.name not in ("schedule.py", "reference.py"):
            text = path.read_text()
            assert "schedule_policy" not in text and "import schedule" not in text and "schedule." not in text, path
    code = ("import sys; import isoexec.ops.logprobs.logprobs_sm100.ops, isoexec.ops.logprobs.logprobs_sm100.integration, "
            "isoexec.ops.logprobs.logprobs_sm100.sampler; "
            "assert not [m for m in sys.modules if m.endswith('logprobs_sm100.schedule') or "
            "m.endswith('logprobs_sm100.reference')]")
    subprocess.run([sys.executable, "-c", code], check=True)
