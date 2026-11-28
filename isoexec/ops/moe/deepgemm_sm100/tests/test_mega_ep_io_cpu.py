"""CPU contract of the MegaMoE EP data-movement protocol: canonical homes, equal chunks, the output-region layout
and the kernels' one-writer / fail-closed protocol (source checks; the bits are GPU-gated with the recipes that
use them, ``moe/deepseek_v41_sm100/tests``)."""
from pathlib import Path

from isoexec.ops.moe.deepgemm_sm100 import mega_ep

SRC = (Path(__file__).resolve().parents[1] / "csrc" / "mega_ep_io.cu").read_text()


def test_home_slices_partition_rows():
    for M in (0, 1, 7, 64, 1000):
        for W in (1, 2, 3, 4, 8):
            spans = [mega_ep.home_slice(M, W, r) for r in range(W)]
            assert spans[0][0] == 0 and spans[-1][1] == M
            assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
            assert max(b - a for a, b in spans) == -(-M // W)


def test_chunks_equal_and_covering():
    assert mega_ep.chunks(0, 64) == [1]
    assert mega_ep.chunks(64, 64) == [64]
    for width, cap in ((65, 64), (8193, 8192), (4096, 1000)):
        c = mega_ep.chunks(width, cap)
        assert len(set(c)) == 1 and c[0] <= cap and sum(c) >= width and sum(c) - c[0] < width


def test_gather_protocol_is_fail_closed():
    assert 'm.def("gather"' in SRC and 'm.def("prep_bf16"' in SRC and 'm.def("prep"' in SRC
    assert "ld.acquire.sys" in SRC and "st.release.sys" in SRC
    assert "__trap()" in SRC                       # a flag ahead of the tag is a protocol violation
