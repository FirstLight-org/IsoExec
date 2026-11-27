"""CPU: the index score's sum over the heads runs the schedule policy's lanes.

Per (query row, key) the logit is ``sum_h bf16(relu(d[h])) * w[h]`` over the H heads, computed by ONE thread that
keeps FOUR independent BF16 ``fma.rn`` accumulators (two ``bf16x2`` registers), so its lane count is
``schedule_policy.lanes(4)``: lane c takes the heads c, c + 4, c + 8, ... in ascending order, and the lanes are then
merged ``(c0 + c2, c1 + c3)``, low + high. The kernel is the vendored DeepGEMM MQA logits kernel
(``isoexec/ops/_vendor/deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh``), instantiated by ``csrc/mqa/mqa.cu``.
This module fails when the kernel's constants stop agreeing with that; the policy is not called on any forward call.
"""
from __future__ import annotations

import re
from pathlib import Path

from isoexec.ops import schedule_policy

PKG = Path(__file__).resolve().parents[1]
OPS = PKG.parents[1]
KERNEL = OPS / "_vendor" / "deep_gemm" / "include" / "deep_gemm" / "impls" / "sm100_mqa_logits.cuh"

#: Independent accumulators one thread keeps for one logit: four BF16 fma chains over the heads.
HEAD_ACCUMULATORS = 4
#: Head counts the operator serves (``index_logits.py``: H = 32 or 64).
HEADS = (32, 64)


def _reduce_pairs() -> str:
    src = KERNEL.read_text()
    start = src.index("const auto reduce_pairs = [&]")
    return src[start:src.index("};", start)]


def test_the_head_sum_runs_the_policys_four_lanes():
    lanes = schedule_policy.lanes(HEAD_ACCUMULATORS)
    assert lanes == 4
    src = KERNEL.read_text()
    body = _reduce_pairs()
    # one loop step consumes one head per lane ...
    step = re.search(r"for \(uint32_t head_offset = 0; head_offset < num_heads; head_offset \+= (\d+)\)", body)
    assert step and int(step.group(1)) == lanes
    assert int(re.search(r"DG_STATIC_ASSERT\(kNumHeads % (\d+) == 0", src).group(1)) == lanes
    # ... into the accumulators: bf16x2 registers, each updated once per step by a packed fma (two chains each)
    registers = re.findall(r"nv_bfloat162& (sum_\d+)", body)
    updates = re.findall(r"(sum_\d+) = __hfma2\(ptx::cvt_relu_bf16x2_f32\((accum_pair_\d+)\), chunk_weights\[[^\]]+\], (sum_\d+)\);",
                         body)
    assert len(updates) == len(registers) and [u[0] for u in updates] == [u[2] for u in updates] == registers
    assert 2 * len(registers) == lanes
    # lane c of a step reads head_offset + c: the pairs are (0, 1) and (2, 3)
    pairs = re.findall(r"const auto (accum_pair_\d+) = make_float2\(chunk\[head_offset(?: \+ (\d+))?\], "
                       r"chunk\[head_offset \+ (\d+)\]\);", body)
    assert [(name, int(a or 0), int(b)) for name, a, b in pairs] == [("accum_pair_0", 0, 1), ("accum_pair_1", 2, 3)]
    assert [u[1] for u in updates] == [p[0] for p in pairs]
    # the merge of the lanes: (c0 + c2, c1 + c3), then low + high
    assert "const auto sum = __hadd2_rn(sum_0, sum_1);" in src
    assert "const auto result = __hadd2_rn(__low2bfloat162(sum), __high2bfloat162(sum));" in src


def test_the_stated_program_names_the_same_lanes():
    """The package's own statements of the program (``csrc/mqa/mqa.cu``, ``index_logits.py``) carry the lane
    count and the membership the kernel has: heads ``4i + c``, ``c = 0..3``, ``i`` ascending."""
    lanes = schedule_policy.lanes(HEAD_ACCUMULATORS)
    mqa = (PKG / "csrc" / "mqa" / "mqa.cu").read_text()
    stated = re.search(r"four BF16 fma\.rn chains over the heads h = (\d+)i \+ c \(c = 0\.\.(\d+), i ascending\)", mqa)
    assert stated and (int(stated.group(1)), int(stated.group(2)) + 1) == (lanes, lanes)
    doc = (PKG / "index_logits.py").read_text()
    stated = re.search(r"four BF16 ``fma\.rn`` chains over the heads ``(\d+)i \+ c`` in ascending ``i``", doc)
    assert stated and int(stated.group(1)) == lanes
    assert "sm100_mqa_logits.cuh" in mqa
    for heads in HEADS:
        assert heads % lanes == 0
        members = [list(range(c, heads, lanes)) for c in range(lanes)]
        assert sorted(h for lane in members for h in lane) == list(range(heads))
        assert all(len(lane) == heads // lanes for lane in members)


def test_no_call_path_reads_the_policy():
    for path in sorted(PKG.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".h", ".cpp") and "tests" not in path.parts:
            assert "schedule_policy" not in path.read_text(), path
