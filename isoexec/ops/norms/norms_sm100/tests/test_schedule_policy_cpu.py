"""The norms_sm100 schedule is the policy's output: CPU checks, no extension build.

Fails if a segment length the kernels are compiled with, or the executable statement in ``reference.py``, departs
from :mod:`isoexec.ops.schedule_policy` on any shape the package serves.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from isoexec.ops import schedule_policy as policy
from isoexec.ops.norms.norms_sm100 import ops as O
from isoexec.ops.norms.norms_sm100 import reference as R
from isoexec.ops.norms.norms_sm100 import schedule

CSRC = Path(__file__).resolve().parents[1] / "csrc"
#: Every width the forward has a kernel for (csrc/launch_fwd.cu::FWD_WIDTHS), the GDN gated heads included.
WIDTHS = (128, 384, 256, 512, 768, 1024, 2048, 6144, 8192, 1536, 4096, 5120, 7168)
ROWS = (1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 128, 1000, 4096, 16384, 65536)


def test_policy_version():
    assert policy.VERSION == "7.1"


@pytest.mark.parametrize("n", WIDTHS)
def test_hidden_axis_segments_are_the_policy_output(n):
    assert O.supported_width(n)
    want = policy.reduce(n, b=1, L=schedule.SUM_SHORTEST_SEGMENT)
    assert schedule.hidden_sum(n) == want == policy.Segments(schedule.SUM_SHORTEST_SEGMENT, None, "tree")
    assert f"-DNORMS_SM100_HIDDEN_SEGMENT={want.S}" in schedule.cuda_defines()


def test_the_published_constant_has_one_definition():
    """L_r is stated once (schedule.py); the kernels receive S from the build and define it nowhere."""
    assert schedule.SUM_SHORTEST_SEGMENT == 8
    text = "\n".join(p.read_text() for p in sorted(CSRC.iterdir()))
    assert not re.search(r"#\s*define\s+NORMS_SM100_HIDDEN_SEGMENT", text)
    common = (CSRC / "common.cuh").read_text()
    assert "#error" in common and "NORMS_SM100_HIDDEN_SEGMENT" in common
    ext = (CSRC.parent / "_extension.py").read_text()
    assert "*cuda_defines()" in ext


def test_no_width_keyed_schedule_table_remains():
    """The forward's strided-lane recipes (v1 / v2 per width) are gone; the backward is the base tree's."""
    text = "\n".join(p.read_text() for p in sorted(CSRC.iterdir()))
    for gone in ("fwd_recipe", "fwd_lanes_v2", "fwd_v2_lmax", "wide_fold", "lane_fold"):
        assert gone not in text, gone
    assert "row_fold" not in (CSRC / "fwd.cuh").read_text()  # the base fold survives in the backward only
    assert not (CSRC.parent / "reference_v2.py").exists()


def _chain_tree(values: list[float], seg: policy.Segments) -> float:
    """The schedule written out term by term in numpy fp32, independently of reference.py."""
    import numpy as np

    parts = []
    for start in range(0, len(values), seg.S):
        acc = np.float32(0.0)
        for v in values[start : start + seg.S]:
            acc = np.float32(acc + np.float32(v))
        parts.append(acc)
    while len(parts) > 1:
        nxt = [np.float32(parts[i] + parts[i + 1]) for i in range(0, len(parts) - 1, 2)]
        parts = nxt + parts[len(parts) - len(parts) % 2 :]
    return float(parts[0])


@pytest.mark.parametrize("n", (1, 7, 8, 9, 16, 17, 24, 40, 63, 64, 65, 128, 384, 640))
def test_reference_sum_is_segments_then_adjacent_tree(n):
    g = torch.Generator().manual_seed(n)
    terms = torch.randn(3, n, generator=g) * torch.logspace(-3, 3, n)
    seg = policy.reduce(n, b=1, L=schedule.SUM_SHORTEST_SEGMENT)
    got = R.segment_sum(terms, seg)
    for r in range(3):
        assert got[r].item() == _chain_tree(terms[r].tolist(), seg), (n, r)


def test_adjacent_tree_carries_an_unpaired_last_partial():
    p = torch.tensor([1.0, 2.0**-30, -1.0, 2.0**-30, 3.0])
    want = ((p[0] + p[1]) + (p[2] + p[3])) + p[4]
    assert R.adjacent_tree(p).item() == want.item()
