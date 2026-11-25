"""CPU contract of ``route_mega``: the vendored MegaGate's gate IS this package's selection program (byte-identical
copies of ``select.cuh`` and the segment tree), and what it serves is the logits program at the schedule policy's
segment count (one)."""
from pathlib import Path

import pytest
import torch

from isoexec.ops.moe.router_sm100 import ops, spec

CSRC = Path(ops.__file__).resolve().parent / "csrc"
VENDOR = Path(ops.__file__).resolve().parents[2] / "_vendor" / "deep_gemm" / "include" / "deep_gemm"
EPILOGUE = VENDOR / "epilogue" / "sm100_mega_gate.cuh"


def _between(text: str, begin: str, end: str) -> str:
    a = text.index(begin) + len(begin)
    return text[a:text.index(end, a)]


def test_megagate_selection_is_select_cuh_verbatim():
    sel = (CSRC / "select.cuh").read_text()
    core = sel[sel.index("namespace rt100 {"):sel.index("} // namespace rt100") + len("} // namespace rt100")]
    copied = _between(EPILOGUE.read_text(), "// ---- BEGIN verbatim isoexec/ops/moe/router_sm100/csrc/select.cuh ----\n",
                      "\n// ---- END verbatim isoexec/ops/moe/router_sm100/csrc/select.cuh ----")
    assert copied == core


def test_megagate_segment_tree_is_router_verbatim():
    router = (CSRC / "router.cu").read_text()
    a = router.index("//: THE segment tree")
    tree = router[a:router.index("return a[0];\n}\n", a) + len("return a[0];\n}\n")]
    copied = _between(EPILOGUE.read_text(),
                      "// ---- BEGIN verbatim isoexec/ops/moe/router_sm100/csrc/router.cu segment_tree ----\n",
                      "// ---- END verbatim isoexec/ops/moe/router_sm100/csrc/router.cu segment_tree ----")
    assert copied == tree


def test_megagate_compiles_with_the_router_arithmetic_profile():
    host = (VENDOR.parents[1] / "csrc" / "jit_kernels" / "impls" / "sm100_bf16_mega_gate.hpp").read_text()
    for flag in ("--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true", ".fast_math = false"):
        assert flag in host, flag


def test_mega_serves_the_schedule_policys_segment_count():
    """The MegaGate's compiled segment count (``layout/mega_gate.cuh``) is the schedule policy's, and the path is
    refused when the two differ."""
    from isoexec.ops import schedule_policy as P

    layout = (VENDOR / "layout" / "mega_gate.cuh").read_text()
    assert "static constexpr uint32_t kNumSegments = 1;" in layout
    for h in (2048, 4096, 5120, 7168, 64 * 6):
        assert ops.mega_gate_segments() == P.reduce(h, b=16).G == spec.logit_segments(h) == 1
        assert ops.mega_serves(384, h)
    assert ops.mega_serves(384, 5120) and ops.mega_serves(256, 7168) and ops.mega_serves(512, 4096)
    assert not ops.mega_serves(384, 100)             # not whole 64-wide K blocks
    heur = (VENDOR.parents[1] / "csrc" / "jit_kernels" / "heuristics" / "mega_gate.hpp").read_text()
    assert "const bool deterministic = kNumSegments == 1;" in heur      # one segment: K is never split
    assert not ops.mega_serves(1024, 5120)           # above MegaGate's 512 experts
    saved = ops.mega_gate_segments
    try:                                             # a header at another count is not served
        ops.mega_gate_segments = lambda: 4
        assert not ops.mega_serves(384, 5120)
    finally:
        ops.mega_gate_segments = saved
    assert not ops.mega_serves(386, 5120)            # E % 4 != 0


def test_route_mega_refuses_host_tensors():
    w = torch.zeros(384, 5120, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        ops.route_mega(torch.zeros(4, 5120, dtype=torch.bfloat16), w, routing=ops.SQRTSOFTPLUS, top_k=6,
                       num_shared=1)
