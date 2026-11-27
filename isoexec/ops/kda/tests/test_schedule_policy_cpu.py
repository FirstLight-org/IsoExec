"""The KDA kernels' schedule parameters are the schedule policy's output (no GPU, no build).

Rule 1 of ``isoexec.ops.schedule_policy`` (a reducer with no merge): segments of the reducer's exact block, one
continued chain. These tests fail if a kernel's block or segment parameter stops coming from the policy: a literal
chunk length in the kernel source, a build that passes another value, a loaded module compiled with another value,
or a host-side constant that disagrees. They also pin the rule that the two reducers are not mixed.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

from isoexec.ops import schedule_policy as policy
from isoexec.ops.kda import schedule

KDA = Path(schedule.__file__).resolve().parent
CPR = KDA / "glm_sm100" / "csrc"


def _strip_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", text, flags=re.S))


def test_chunk_reducer_segments_are_rule_1_on_the_largest_built_block_cpu():
    seg = schedule.segments(schedule.CHUNK)
    assert seg == policy.no_merge(max(schedule.CHUNK.blocks))
    assert (seg.S, seg.G, seg.merge) == (64, None, "continued")
    assert schedule.DEFAULT is schedule.CHUNK and schedule.segments() == seg


def test_built_blocks_are_whole_k_steps_up_to_the_tile_cpu():
    text = _strip_comments((CPR / "common.cuh").read_text())
    tile = int(re.search(r"constexpr int C = (\d+);", text).group(1))
    step = int(re.search(r"constexpr int BLK = (\d+);", text).group(1))
    assert schedule.CHUNK.blocks == (16, 32, 64)
    assert all(b % step == 0 and step <= b <= tile for b in schedule.CHUNK.blocks)
    assert schedule.CHUNK.block == tile          # the largest built block is the tile this layout can hold


def test_kernel_has_no_chunk_length_of_its_own_cpu():
    text = _strip_comments((CPR / "common.cuh").read_text())
    assert not re.search(r"#\s*define\s+KDA_CL\b", text), "the kernel defines its own chunk length"
    assert re.search(r"#ifndef KDA_CL\s*\n\s*#error", text), "a build without the policy's definition must not compile"
    assert "constexpr int CL = KDA_CL;" in text
    for path in sorted(CPR.iterdir()):
        body = _strip_comments(path.read_text())
        assert not re.search(r"#\s*define\s+KDA_CL\b", body), path.name
        # chunk arithmetic on a position or a row count goes through CL, never through the tile height C
        assert not re.search(r"pos(_in)?\s*&\s*\(C - 1\)|pp\s*&\s*\(C - 1\)|/\s*C\b|\*\s*C\b|%\s*C\b", body), path.name
    plan = _strip_comments((CPR / "plan.cu").read_text())
    assert "pos_in % CL" in plan and "(j0 + len + CL - 1) / CL" in plan and "int chunk_len() { return CL; }" in plan
    assert "pos % CL" in _strip_comments((CPR / "decode_rec.cu").read_text())


def test_the_build_receives_exactly_the_policy_segment_cpu(monkeypatch):
    assert schedule.cuda_defines(schedule.CHUNK) == (f"-DKDA_CL={policy.no_merge(64).S}",)
    ext_mod = importlib.import_module("isoexec.ops.kda.glm_sm100._extension")
    seen = {}

    class Fake:
        __file__ = "fake.so"

        def __init__(self, cl):
            self.cl = cl

        def chunk_len(self):
            return self.cl

    def fake_build(family, module_file, sources, **kw):
        seen.update(family=family, flags=tuple(kw["extra_cuda_flags"]))
        return Fake(seen["cl"])

    monkeypatch.setattr(ext_mod, "build_sm100_extension", fake_build)
    for cl, ok in ((64, True), (32, False), (16, False)):
        ext_mod.extension.cache_clear()
        seen["cl"] = cl
        if ok:
            assert ext_mod.extension().chunk_len() == schedule.segments().S
        else:
            with pytest.raises(RuntimeError, match="schedule policy gives 64"):
                ext_mod.extension()
        assert seen["family"] == "kda_glm"
        assert [f for f in seen["flags"] if f.startswith("-DKDA_")] == ["-DKDA_CL=64"]
    ext_mod.extension.cache_clear()


def test_a_changed_policy_output_changes_the_build_cpu(monkeypatch):
    other = schedule.Reducer(name="x", impl_id="x", package="glm_sm100", family="kda_glm", blocks=(16, 32),
                             block_define="KDA_CL")
    assert schedule.cuda_defines(other) == ("-DKDA_CL=32",)
    monkeypatch.setattr(policy, "no_merge", lambda b: policy.Segments(b // 2, None, "continued"))
    assert schedule.cuda_defines(schedule.CHUNK) == ("-DKDA_CL=32",)


def test_host_side_chunk_length_is_the_policy_segment_cpu():
    spec = importlib.import_module("isoexec.ops.kda.glm_sm100.spec")
    assert spec.CHUNK_LENGTH == schedule.segments(schedule.CHUNK).S
    src = (KDA / "glm_sm100" / "ops.py").read_text()
    assert "return T // CHUNK_LENGTH + nseq" in src
    for name in ("ops.py", "engine.py", "integration.py", "engine_training.py"):
        assert "schedule_policy" not in (KDA / "glm_sm100" / name).read_text(), name   # the policy is off the call path


def test_token_reducer_is_block_one_and_has_no_block_parameter_cpu():
    seg = schedule.segments(schedule.TOKEN)
    assert seg == policy.no_merge(1) and schedule.TOKEN.blocks == (1,)
    assert schedule.cuda_defines(schedule.TOKEN) == ()
    for path in sorted((KDA / "glm_rec_sm100" / "csrc").iterdir()):
        assert "KDA_CL" not in path.read_text(), path.name


def test_the_two_reducers_have_different_manifests_and_are_refused_together_cpu():
    a, b = schedule.manifest(schedule.CHUNK), schedule.manifest(schedule.TOKEN)
    assert a["schedule"] == {"split": "anchored(64)", "combine": "continued"}
    assert b["schedule"] == {"split": "anchored(1)", "combine": "continued"}
    assert a["reducer"] != b["reducer"] and a["builds"] != b["builds"]
    assert schedule.manifest_hash(a) != schedule.manifest_hash(b)
    assert schedule.manifest_hash(a) == schedule.manifest_hash(schedule.manifest())      # engine == trainer by default
    schedule.assert_same_manifest(a, schedule.manifest(schedule.CHUNK))
    with pytest.raises(ValueError, match="one deployment runs one reducer"):
        schedule.assert_same_manifest(a, b)


def test_selection_is_explicit_and_pools_are_not_interchangeable_cpu():
    cpr, rec = schedule.select(), schedule.select(schedule.TOKEN)
    assert cpr.__name__ == "isoexec.ops.kda.glm_sm100.engine" and rec.__name__ == "isoexec.ops.kda.glm_rec_sm100.engine"
    assert cpr.KdaStatePool.reducer is schedule.CHUNK and rec.KdaStatePool.reducer is schedule.TOKEN

    class Pool:
        reducer = schedule.TOKEN

    schedule.check_pool(schedule.TOKEN, Pool())
    schedule.check_pool(schedule.CHUNK, None)
    with pytest.raises(ValueError, match="not mixed inside one deployment"):
        schedule.check_pool(schedule.CHUNK, Pool())
    for mod, red in ((cpr, "schedule.CHUNK"), (rec, "schedule.TOKEN")):
        assert f"schedule.check_pool({red}, pool)" in Path(mod.__file__).read_text()
    assert "isoexec_kda_glm_sm100_cuda" not in sys.modules and "isoexec_kda_glm_rec_sm100_cuda" not in sys.modules
