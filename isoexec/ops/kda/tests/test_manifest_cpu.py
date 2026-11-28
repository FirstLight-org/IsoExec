"""The KDA manifest covers every KDA source its engine compiles and the flags that reach the arithmetic; the engine's
o_proj leaf count is the GEMM's row-parallel tree (no GPU, no build).

Two parties compare manifests before they exchange state or bits. The manifest used to hash only the reducer's own
csrc tree, but the token engine also runs glm_sm100's extension (gated norm, leaf-tree fallback, trainer gate and
beta: csrc/layer.cu), so two parties with different layer.cu compared equal.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from isoexec.ops import build_cache
from isoexec.ops.kda import schedule

KDA = Path(schedule.__file__).resolve().parent


@pytest.fixture
def tree(tmp_path):
    """A copy of both reducers' csrc trees."""
    for r in schedule.REDUCERS:
        shutil.copytree(schedule.csrc(r), tmp_path / r.package / "csrc")
    return tmp_path


def _files(r):
    return sorted(p.relative_to(KDA).as_posix() for p in schedule.csrc(r).rglob("*") if p.is_file())


def test_the_copy_has_the_tree_manifests_cpu(tree):
    for r in schedule.REDUCERS:
        assert schedule.manifest(r, root=tree) == schedule.manifest(r)


@pytest.mark.parametrize("path", _files(schedule.CHUNK) + _files(schedule.TOKEN))
def test_changing_any_compiled_source_changes_the_manifest_of_every_engine_that_compiles_it_cpu(tree, path):
    before = {r.name: schedule.manifest_hash(schedule.manifest(r, root=tree)) for r in schedule.REDUCERS}
    f = tree / path
    f.write_bytes(f.read_bytes() + b"\n// changed\n")
    package = path.split("/")[0]
    for r in schedule.REDUCERS:
        compiles = package in {b.package for b in schedule.ENGINE_BUILDS[r.name]}
        after = schedule.manifest_hash(schedule.manifest(r, root=tree))
        assert (after != before[r.name]) == compiles, (path, r.name)
        if compiles:
            with pytest.raises(ValueError, match="manifests differ"):
                schedule.assert_same_manifest(schedule.manifest(r), schedule.manifest(r, root=tree))


def test_the_token_engine_manifest_covers_the_chunk_packages_layer_stages_cpu(tree):
    """The reviewed case: a different glm_sm100/csrc/layer.cu (which the token engine runs) is a different manifest."""
    assert (KDA / "glm_sm100" / "csrc" / "layer.cu").is_file()
    before = schedule.manifest(schedule.TOKEN, root=tree)
    f = tree / "glm_sm100" / "csrc" / "layer.cu"
    f.write_bytes(f.read_bytes().replace(b"rn(sigm(bf(b[i])))", b"rn(sigm(bf(b[i]) ))", 1))
    with pytest.raises(ValueError, match="manifests differ"):
        schedule.assert_same_manifest(before, schedule.manifest(schedule.TOKEN, root=tree))


def test_the_numerics_flags_and_defines_are_in_the_manifest_cpu(monkeypatch):
    for r in schedule.REDUCERS:
        for b in schedule.manifest(r)["builds"]:
            assert all(f in b["flags"] for f in build_cache.NUMERICS_FLAGS), b
            assert "-gencode=arch=compute_100a,code=sm_100a" in b["flags"]
    chunk = next(b for b in schedule.manifest(schedule.CHUNK)["builds"] if b["package"] == "glm_sm100")
    assert "-DKDA_CL=64" in chunk["flags"]
    before = {r.name: schedule.manifest_hash(schedule.manifest(r)) for r in schedule.REDUCERS}
    monkeypatch.setattr(build_cache, "NUMERICS_FLAGS", ("--ftz=true", "--fmad=false"))
    for r in schedule.REDUCERS:
        assert schedule.manifest_hash(schedule.manifest(r)) != before[r.name], r.name


def test_the_engines_build_what_the_manifest_names_cpu():
    """ENGINE_BUILDS is what the engine modules import: the chunk engine glm_sm100 only, the token engine both."""
    chunk = (KDA / "glm_sm100" / "engine.py").read_text()
    token = (KDA / "glm_rec_sm100" / "engine.py").read_text()
    assert "glm_rec_sm100" not in chunk
    assert "isoexec.ops.kda.glm_sm100" in token and "glm_rec_sm100._extension" in token
    assert [b.package for b in schedule.ENGINE_BUILDS[schedule.CHUNK.name]] == ["glm_sm100"]
    assert sorted(b.package for b in schedule.ENGINE_BUILDS[schedule.TOKEN.name]) == ["glm_rec_sm100", "glm_sm100"]


def test_the_o_proj_leaves_are_the_gemms_row_parallel_tree_cpu():
    from isoexec.ops.kda.glm_rec_sm100 import engine as token
    from isoexec.ops.kda.glm_sm100 import engine as chunk
    from isoexec.ops.mm.mm_sm100.segments import row_parallel_leaves

    assert chunk.TP_LEAVES == token.TP_LEAVES == 8
    for heads in range(8, 257, 8):                       # every head count shard_layer takes
        assert row_parallel_leaves(heads * 128, max(chunk.TP_DEGREES)) == chunk.TP_LEAVES, heads
    assert not re.search(r"^TP_LEAVES\s*=\s*\d", (KDA / "glm_sm100" / "engine.py").read_text(), re.M)   # no literal
