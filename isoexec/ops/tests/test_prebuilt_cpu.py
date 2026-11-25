"""The prebuilt extension store: path-free keys, publish and hit, prebuilt-only refusals."""

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from isoexec.ops import prebuilt


def _package(root: Path, body: str = "kernel one") -> Path:
    csrc = root / "isoexec" / "ops" / "fam" / "csrc"
    (csrc / "inc").mkdir(parents=True)
    (csrc / "k.cu").write_text(body)
    (csrc / "inc" / "h.cuh").write_text("header")
    return root / "isoexec"


def _inputs(pkg: Path, **kw):
    csrc = pkg / "ops" / "fam" / "csrc"
    return prebuilt.extension_inputs("isoexec_fam_sm100_cuda", [str(csrc / "k.cu")],
                                     extra_include_paths=[str(csrc)], extra_cuda_cflags=kw.pop("flags", ["-O3"]),
                                     **kw)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv(prebuilt.ENV_DIR, str(tmp_path / "cache"))
    monkeypatch.delenv(prebuilt.ENV_ONLY, raising=False)
    prebuilt._tree_digest.cache_clear()
    prebuilt._file_digest.cache_clear()
    return tmp_path / "cache"


def test_key_is_independent_of_the_checkout_path(tmp_path, monkeypatch, store):
    a, b = _package(tmp_path / "A"), _package(tmp_path / "deeper" / "B")
    monkeypatch.setattr(prebuilt, "PACKAGE_ROOT", a)
    ia = _inputs(a)
    monkeypatch.setattr(prebuilt, "PACKAGE_ROOT", b)
    ib = _inputs(b)
    assert prebuilt.key_of(ia) == prebuilt.key_of(ib)
    assert prebuilt.path_dependent_inputs(ia) == []
    assert str(tmp_path) not in json.dumps(ia)


def test_key_folds_sources_include_trees_and_flags(tmp_path, monkeypatch, store):
    pkg = _package(tmp_path / "A")
    monkeypatch.setattr(prebuilt, "PACKAGE_ROOT", pkg)
    base = prebuilt.key_of(_inputs(pkg))
    assert prebuilt.key_of(_inputs(pkg, flags=["-O3", "--fmad=true"])) != base
    (pkg / "ops" / "fam" / "csrc" / "inc" / "h.cuh").write_text("header two")
    prebuilt._tree_digest.cache_clear()
    changed = _inputs(pkg)
    assert prebuilt.key_of(changed) != base
    old = _inputs(pkg)
    (pkg / "ops" / "fam" / "csrc" / "k.cu").write_text("kernel two")
    prebuilt._tree_digest.cache_clear()
    prebuilt._file_digest.cache_clear()
    diff = prebuilt.describe_difference(old, _inputs(pkg))
    assert any("k.cu" in d for d in diff) and any(d.startswith("tree") for d in diff)


def _fake_load(tmp_path, calls):
    def load(**kwargs):
        calls.append(kwargs)
        build = Path(kwargs.get("build_directory") or tmp_path / "build")
        build.mkdir(parents=True, exist_ok=True)
        path = build / f"{kwargs['name']}.py"
        path.write_text("VALUE = 7\n")
        module = types.ModuleType(kwargs["name"])
        module.__file__ = str(path)
        return module
    return load


def test_miss_builds_publishes_and_the_next_load_hits(tmp_path, monkeypatch, store):
    import torch.utils.cpp_extension as cpp

    pkg = _package(tmp_path / "A")
    monkeypatch.setattr(prebuilt, "PACKAGE_ROOT", pkg)
    calls = []
    monkeypatch.setattr(cpp, "load", _fake_load(tmp_path, calls))
    monkeypatch.setattr(prebuilt, "_classify_dependencies", lambda *a: ([], []))
    csrc = pkg / "ops" / "fam" / "csrc"
    args = dict(name="isoexec_fam_sm100_cuda", sources=[str(csrc / "k.cu")], extra_include_paths=[str(csrc)],
                build_directory=str(tmp_path / "build"))
    prebuilt.load_extension(**args)
    assert len(calls) == 1 and calls[0]["sources"] == [str(csrc / "k.cu")]
    entries = list((store / "ext" / "isoexec_fam_sm100_cuda").iterdir())
    assert len(entries) == 1 and (entries[0] / prebuilt.COMMITTED).exists()
    module = prebuilt.load_extension(**args)
    assert len(calls) == 1 and module.VALUE == 7 and str(store) in module.__file__
    # prebuilt-only: the same key still hits
    monkeypatch.setenv(prebuilt.ENV_ONLY, "1")
    assert prebuilt.load_extension(**args).VALUE == 7


def test_prebuilt_only_refuses_by_name_and_names_the_stale_input(tmp_path, monkeypatch, store):
    import torch.utils.cpp_extension as cpp

    pkg = _package(tmp_path / "A")
    monkeypatch.setattr(prebuilt, "PACKAGE_ROOT", pkg)
    calls = []
    monkeypatch.setattr(cpp, "load", _fake_load(tmp_path, calls))
    monkeypatch.setattr(prebuilt, "_classify_dependencies", lambda *a: ([], []))
    csrc = pkg / "ops" / "fam" / "csrc"
    args = dict(name="isoexec_fam_sm100_cuda", sources=[str(csrc / "k.cu")], extra_include_paths=[str(csrc)])
    prebuilt.load_extension(**args)
    (csrc / "k.cu").write_text("edited kernel")
    prebuilt._tree_digest.cache_clear()
    prebuilt._file_digest.cache_clear()
    monkeypatch.setenv(prebuilt.ENV_ONLY, "1")
    with pytest.raises(prebuilt.PrebuiltMissing) as error:
        prebuilt.load_extension(**args)
    text = str(error.value)
    assert text.startswith(prebuilt.MARKER) and "isoexec_fam_sm100_cuda" in text
    assert "<isoexec>/ops/fam/csrc/k.cu (changed)" in text
    assert len(calls) == 1                    # nothing was compiled for the refused load


def test_every_registry_extension_loads_through_the_store():
    """Every SM100 extension's key is computable without a build (no GPU, no compiler run)."""
    for label, *_ in prebuilt.EXTENSIONS:
        name, key, inputs = prebuilt.probe(label)
        assert name.startswith(("isoexec_", "_isoexec_")) and len(key) == 24, label
        assert prebuilt.path_dependent_inputs(inputs) == [], (label, prebuilt.path_dependent_inputs(inputs))


def test_dg_jit_refusal_keeps_the_version_and_names_the_kernel(tmp_path, monkeypatch):
    fake = tmp_path / "cuda" / "bin" / "nvcc"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 13.2, V13.2.51'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "cuda"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    shim = prebuilt._nvcc_refusal_shim("/cache/dg-jit")
    out = subprocess.run([shim, "--version"], capture_output=True, text=True)
    assert out.returncode == 0 and "release 13.2" in out.stdout
    kernel = tmp_path / "kernel.cu"
    kernel.write_text("static void __instantiate_kernel() {\n"
                      "    auto ptr = reinterpret_cast<void*>(&sm100_bf16_gemm_impl<128, 64>);\n};\n")
    out = subprocess.run([shim, str(kernel), "--cubin"], capture_output=True, text=True)
    assert out.returncode == 86 and out.stdout.startswith(prebuilt.MARKER)
    assert "sm100_bf16_gemm_impl<128, 64>" in out.stdout


def test_activate_places_the_jit_caches_under_the_root(tmp_path, monkeypatch):
    monkeypatch.setenv(prebuilt.ENV_DIR, str(tmp_path / "pre"))
    monkeypatch.delenv(prebuilt.ENV_ONLY, raising=False)
    for name in ("ISOEXEC_DG_JIT_CACHE_DIR", "TRITON_CACHE_DIR", "TRITON_CACHE_MANAGER",
                 "ISOEXEC_DG_JIT_NVCC_COMPILER"):
        monkeypatch.delenv(name, raising=False)
    prebuilt.activate()
    assert os.environ["ISOEXEC_DG_JIT_CACHE_DIR"] == str(tmp_path / "pre" / "dg-jit")
    assert os.environ["TRITON_CACHE_DIR"] == str(tmp_path / "pre" / "triton")
    assert "TRITON_CACHE_MANAGER" not in os.environ
    monkeypatch.setenv(prebuilt.ENV_ONLY, "1")
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    prebuilt.activate()
    assert os.environ["ISOEXEC_DG_JIT_CACHE_DIR"].endswith(":" + str(tmp_path / "pre" / "dg-jit"))
    assert os.environ["TRITON_CACHE_MANAGER"] == "isoexec.ops.prebuilt:TritonPrebuiltOnly"
    assert os.access(os.environ["ISOEXEC_DG_JIT_NVCC_COMPILER"], os.X_OK)


def test_triton_manager_refuses_a_missing_kernel(tmp_path, monkeypatch):
    pytest.importorskip("triton")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    manager = prebuilt.TritonPrebuiltOnly("abc")
    with pytest.raises(prebuilt.PrebuiltMissing) as error:
        manager.get_group("my_kernel.json")
    assert "triton my_kernel" in str(error.value)
    manager.put_group("my_kernel.json", {"my_kernel.json": manager.put("{}", "my_kernel.json")})
    assert manager.get_group("my_kernel.json") is not None


def test_manifest_and_verify_detect_a_changed_artefact(tmp_path, monkeypatch, store):
    import torch.utils.cpp_extension as cpp

    pkg = _package(tmp_path / "A")
    monkeypatch.setattr(prebuilt, "PACKAGE_ROOT", pkg)
    monkeypatch.setattr(cpp, "load", _fake_load(tmp_path, []))
    monkeypatch.setattr(prebuilt, "_classify_dependencies", lambda *a: ([], []))
    csrc = pkg / "ops" / "fam" / "csrc"
    prebuilt.load_extension(name="isoexec_fam_sm100_cuda", sources=[str(csrc / "k.cu")])
    data = prebuilt.write_manifest(store)
    assert [e["name"] for e in data["ext"]] == ["isoexec_fam_sm100_cuda"] and len(data["digest"]) == 64
    monkeypatch.setattr(prebuilt, "EXTENSIONS", ())
    assert prebuilt.verify(store) == []
    entry = data["ext"][0]
    (store / "ext" / entry["name"] / entry["key"] / entry["file"]).write_text("VALUE = 8\n")
    problems = prebuilt.verify(store)
    assert len(problems) == 1 and problems[0].startswith("STALE torch-extension isoexec_fam_sm100_cuda")


def test_cli_lists_the_registry():
    out = subprocess.run([sys.executable, "-m", "isoexec.ops.prebuilt", "list"], capture_output=True, text=True,
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert out.returncode == 0 and out.stdout.split() == [e[0] for e in prebuilt.EXTENSIONS]


def test_deepgemm_signatures_are_recorded_once_while_a_cache_is_recorded(tmp_path, monkeypatch):
    from isoexec.ops.mm.deepgemm_sm100 import jit_warm

    monkeypatch.setattr(jit_warm, "_STATE", {"on": None, "seen": set()})
    monkeypatch.setenv(prebuilt.ENV_DIR, str(tmp_path))
    monkeypatch.delenv(prebuilt.ENV_ONLY, raising=False)
    for _ in range(3):
        jit_warm.note("gemm", n=5120, k=1024)
    jit_warm.note("bhr", g=8, r=1024, d=512)
    rows = jit_warm.shapes(tmp_path)
    assert [r["kind"] for r in rows] == ["bhr", "gemm"] and rows[1] == {"kind": "gemm", "n": 5120, "k": 1024,
                                                                        "inference": False}
    # prebuilt-only: nothing is recorded (the hot path is one boolean test)
    monkeypatch.setattr(jit_warm, "_STATE", {"on": None, "seen": set()})
    monkeypatch.setenv(prebuilt.ENV_ONLY, "1")
    jit_warm.note("gemm", n=1, k=64)
    assert len(jit_warm.shapes(tmp_path)) == 2


def _args(pkg):
    csrc = pkg / "ops" / "fam" / "csrc"
    return dict(name="isoexec_fam_sm100_cuda", sources=[str(csrc / "k.cu")], extra_include_paths=[str(csrc)])


@pytest.fixture
def fake_build(tmp_path, monkeypatch, store):
    import torch.utils.cpp_extension as cpp

    pkg = _package(tmp_path / "A")
    monkeypatch.setattr(prebuilt, "PACKAGE_ROOT", pkg)
    calls = []
    monkeypatch.setattr(cpp, "load", _fake_load(tmp_path, calls))
    monkeypatch.setattr(prebuilt, "_classify_dependencies", lambda *a: ([], []))
    return pkg, calls


@pytest.mark.parametrize("name,value", [("NVCC_PREPEND_FLAGS", "-DX=1"), ("NVCC_APPEND_FLAGS", "--use_fast_math"),
                                        ("LIBRARY_PATH", "/opt/lib")])
def test_compiler_environment_is_a_key_input(fake_build, monkeypatch, name, value):
    pkg, _ = fake_build
    monkeypatch.delenv(name, raising=False)
    prebuilt.toolchain.cache_clear()
    base = prebuilt.key_of(_inputs(pkg))
    monkeypatch.setenv(name, value)
    prebuilt.toolchain.cache_clear()
    try:
        assert prebuilt.key_of(_inputs(pkg)) != base
    finally:
        prebuilt.toolchain.cache_clear()


def test_include_path_environment_is_keyed_by_content(fake_build, tmp_path, monkeypatch):
    pkg, _ = fake_build
    extra = tmp_path / "cpath"
    extra.mkdir()
    (extra / "x.h").write_text("one")
    monkeypatch.delenv("CPATH", raising=False)
    without = prebuilt.key_of(_inputs(pkg))
    monkeypatch.setenv("CPATH", str(extra))
    with_dir = prebuilt.key_of(_inputs(pkg))
    (extra / "x.h").write_text("two")
    prebuilt._tree_digest.cache_clear()
    assert len({without, with_dir, prebuilt.key_of(_inputs(pkg))}) == 3


def test_a_build_reading_an_unhashed_header_is_refused_not_published(fake_build, store, monkeypatch):
    pkg, calls = fake_build
    monkeypatch.setattr(prebuilt, "_classify_dependencies", lambda *a: (["<abs>/opt/vendor/x.cuh"], []))
    with pytest.raises(prebuilt.PrebuiltUncovered) as error:
        prebuilt.load_extension(**_args(pkg))
    assert "ISOEXEC_PREBUILT_UNCOVERED" in str(error.value) and "<abs>/opt/vendor/x.cuh" in str(error.value)
    assert not (store / "ext" / "isoexec_fam_sm100_cuda").exists() or not any(
        (p / prebuilt.COMMITTED).exists() for p in (store / "ext" / "isoexec_fam_sm100_cuda").iterdir())


def test_dependency_classes_from_the_ninja_log(tmp_path, monkeypatch, store):
    pkg = _package(tmp_path / "A")
    monkeypatch.setattr(prebuilt, "PACKAGE_ROOT", pkg)
    csrc = pkg / "ops" / "fam" / "csrc"
    sysdir = tmp_path / "sysinc"
    sysdir.mkdir()
    (sysdir / "stdio.h").write_text("")
    stray = tmp_path / "stray" / "y.h"
    stray.parent.mkdir()
    stray.write_text("")
    log = f"k.o: #deps 3\n    {csrc / 'inc' / 'h.cuh'}\n    {sysdir / 'stdio.h'}\n    {stray}\n"
    inputs = _inputs(pkg)
    monkeypatch.setattr(prebuilt, "_system_include_dirs", lambda: (str(sysdir.resolve()),))
    monkeypatch.setattr(prebuilt.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=log, stderr=""))
    missing, external = prebuilt._classify_dependencies(str(tmp_path), inputs, [str(csrc / "k.cu")])
    assert missing == ["<abs>" + str(stray.resolve())]
    assert external == [str((sysdir / "stdio.h").resolve())]
    # an unreadable / empty dependency log is itself uncovered (refused, never published blind)
    monkeypatch.setattr(prebuilt.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    assert prebuilt._classify_dependencies(str(tmp_path), inputs, [])[0] == ["<ninja deps unavailable>"]


def test_an_entry_of_the_previous_schema_is_a_miss_never_a_hit(fake_build, store, monkeypatch):
    pkg, calls = fake_build
    monkeypatch.setattr(prebuilt, "SCHEMA", "isoexec-prebuilt-v1")
    prebuilt.load_extension(**_args(pkg))
    assert len(calls) == 1
    monkeypatch.setattr(prebuilt, "SCHEMA", "isoexec-prebuilt-v2")
    monkeypatch.setenv(prebuilt.ENV_ONLY, "1")
    with pytest.raises(prebuilt.PrebuiltMissing) as error:
        prebuilt.load_extension(**_args(pkg))
    assert "differs in: schema" in str(error.value)
    monkeypatch.delenv(prebuilt.ENV_ONLY)
    prebuilt.load_extension(**_args(pkg))
    assert len(calls) == 2                         # rebuilt, not loaded from the v1 entry
    assert prebuilt.SCHEMA == "isoexec-prebuilt-v2"


def test_verify_on_load_catches_a_changed_object_and_toolchain_header(fake_build, store, tmp_path, monkeypatch):
    pkg, calls = fake_build
    header = tmp_path / "sys" / "a.h"
    header.parent.mkdir()
    header.write_text("v1")
    monkeypatch.setattr(prebuilt, "_classify_dependencies", lambda *a: ([], [str(header)]))
    prebuilt.load_extension(**_args(pkg))
    entry = next(p for p in (store / "ext" / "isoexec_fam_sm100_cuda").iterdir() if p.name[0] != ".")
    so = entry / json.loads((entry / "entry.json").read_text())["file"]
    so.write_text(so.read_text().replace("7", "9"))   # same size, different bytes
    assert prebuilt.load_extension(**_args(pkg)).VALUE == 9 and len(calls) == 1   # size check only: a hit
    monkeypatch.setenv("ISOEXEC_PREBUILT_VERIFY", "1")
    monkeypatch.setenv(prebuilt.ENV_ONLY, "1")
    with pytest.raises(prebuilt.PrebuiltMissing, match="fails the load check"):
        prebuilt.load_extension(**_args(pkg))
    so.write_text(so.read_text().replace("9", "7"))
    assert prebuilt.load_extension(**_args(pkg)).VALUE == 7       # restored: passes the deep check
    header.write_text("v2")                                        # a toolchain header edited in place
    with pytest.raises(prebuilt.PrebuiltMissing, match="fails the load check"):
        prebuilt.load_extension(**_args(pkg))
