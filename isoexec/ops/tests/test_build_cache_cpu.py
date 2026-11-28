"""Native extension caches must not depend on one host's private mount layout."""

import importlib
import tomllib
from pathlib import Path

import pytest

from isoexec.ops import build_cache
from isoexec.ops.build_cache import NUMERICS_FLAGS, sm90_build_dir

MANIFEST = Path(__file__).resolve().parents[3] / "environments" / "default.toml"


def _fake_cutlass(root: Path, revision: str | None = None) -> Path:
    (root / "include" / "cutlass").mkdir(parents=True)
    (root / "include" / "cutlass" / "version.h").write_text(
        "#define CUTLASS_MAJOR 4\n#define CUTLASS_MINOR 3\n#define CUTLASS_PATCH 5\n"
    )
    if revision is not None:
        (root / build_cache.CUTLASS_REVISION_FILE).write_text(revision + "\n")
    return root


def test_default_cache_is_writable_and_source_isolated(tmp_path, monkeypatch):
    cache = tmp_path / "torch-extensions"
    source = tmp_path / "source"
    source.mkdir()
    kernel = source / "kernel.cu"
    kernel.write_text("version one")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(cache))
    monkeypatch.delenv("ISOEXEC_GDN_CPR_SM90_BUILD_DIR", raising=False)

    first = sm90_build_dir("ISOEXEC_GDN_CPR_SM90_BUILD_DIR", "gdn_cpr-sm90", "torch-test", source)
    # The requested cache itself may legitimately live under any mount.
    # Portability means honoring that root, not forbidding a particular mount.
    assert first.is_relative_to(cache / "isoexec" / "gdn_cpr-sm90")
    same_sources = tmp_path / "second-checkout"
    same_sources.mkdir()
    (same_sources / "kernel.cu").write_text("version one")
    assert sm90_build_dir("ISOEXEC_GDN_CPR_SM90_BUILD_DIR", "gdn_cpr-sm90", "torch-test", same_sources) == first
    kernel.write_text("version two")
    assert sm90_build_dir("ISOEXEC_GDN_CPR_SM90_BUILD_DIR", "gdn_cpr-sm90", "torch-test", source) != first


def test_explicit_cache_override_preserves_existing_contract(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    override = tmp_path / "explicit"
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "unused"))
    monkeypatch.setenv("ISOEXEC_GDN_CPR_SM90_BUILD_DIR", str(override))
    assert sm90_build_dir("ISOEXEC_GDN_CPR_SM90_BUILD_DIR", "gdn_cpr-sm90", "tag", source) == override
    assert sm90_build_dir(
        "ISOEXEC_GDN_CPR_SM90_BUILD_DIR", "gdn_cpr-sm90", "tag", source, append_tag_to_override=True
    ) == override / "tag"


def test_source_records_have_unambiguous_boundaries(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("ISOEXEC_GDN_CPR_SM90_BUILD_DIR", raising=False)
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "a.h").write_bytes(b"Xb.h")
    (left / "b.h").write_bytes(b"Y")
    (right / "a.h").write_bytes(b"X")
    (right / "b.h").write_bytes(b"b.hY")
    assert sm90_build_dir("ISOEXEC_GDN_CPR_SM90_BUILD_DIR", "gdn_cpr-sm90", "tag", left) != sm90_build_dir(
        "ISOEXEC_GDN_CPR_SM90_BUILD_DIR", "gdn_cpr-sm90", "tag", right
    )


def test_cache_uses_xdg_fallback_when_torch_directory_is_absent(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.delenv("TORCH_EXTENSIONS_DIR", raising=False)
    monkeypatch.delenv("ISOEXEC_ROPE_SM90_BUILD_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    path = sm90_build_dir("ISOEXEC_ROPE_SM90_BUILD_DIR", "rope-sm90", "torch-test", source)
    assert path.is_relative_to(tmp_path / "xdg" / "torch_extensions" / "isoexec" / "rope-sm90")


@pytest.mark.parametrize(
    "module_name,override,family",
    [
        ("gdn.gdn_cpr_sm90._extension", "ISOEXEC_GDN_CPR_SM90_BUILD_DIR", "gdn_cpr-sm90"),
        ("attention.rope_sm90._extension", "ISOEXEC_ROPE_SM90_BUILD_DIR", "rope-sm90"),
        ("gdn.conv_sm90._extension", "ISOEXEC_CONV_SM90_BUILD_DIR", "conv-sm90"),
        ("logprobs.logprobs_sm90._extension", "ISOEXEC_LOGPROBS_SM90_BUILD_DIR", "logprobs-sm90"),
        ("norms.norms_sm90._extension", "ISOEXEC_NORMS_SM90_BUILD_DIR", "norms-sm90"),
        ("moe.moe_sm90._extension", "ISOEXEC_MOE_SM90_BUILD_DIR", "moe-sm90"),
        ("attention.mla_sm90.fwd", "ISOEXEC_MLA_SM90_BUILD_DIR", "mla-sm90"),
    ],
)
def test_extensions_load_from_resolved_cache(tmp_path, monkeypatch, module_name, override, family):
    import torch.utils.cpp_extension as cpp

    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "torch-extensions"))
    monkeypatch.delenv(override, raising=False)
    seen = {}

    def fake_load(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(cpp, "load", fake_load)
    if module_name.startswith("attention.mla_sm90"):
        monkeypatch.setenv("ISOEXEC_MLA_SM90_CUTLASS", str(_fake_cutlass(tmp_path / "cutlass")))
        monkeypatch.setenv("ISOEXEC_MLA_SM90_CUDA_HOME", str(tmp_path / "cuda"))
    module = importlib.import_module(f"isoexec.ops.{module_name}")
    module.extension.cache_clear()
    try:
        module.extension()
    finally:
        module.extension.cache_clear()
    build = Path(seen["build_directory"])
    assert build.is_dir()
    assert build.is_relative_to(tmp_path / "torch-extensions" / "isoexec" / family)
    # Check non-FTZ and no-contraction flags at the actual nvcc call site.
    flags = seen["extra_cuda_cflags"]
    assert "-gencode=arch=compute_90a,code=sm_90a" in flags, (module_name, flags)
    # Check the shared numerical profile at each nvcc call site.
    # MLA is separate because its selected profile is included in profile_digest.
    if not module_name.startswith("attention.mla_sm90"):
        assert set(NUMERICS_FLAGS) <= set(flags), (module_name, flags)


def test_cutlass_pin_matches_the_setup_manifest():
    manifest = tomllib.loads(MANIFEST.read_text())
    assert manifest["cutlass"]["revision"] == build_cache.CUTLASS_REVISION


def test_bundled_cutlass_is_where_the_wheel_puts_it():
    # setup.py checks the headers out into isoexec/_runtime/cutlass of a built wheel.
    package = Path(build_cache.__file__).resolve().parents[1]
    assert build_cache.CUTLASS_BUNDLED == package / "_runtime" / "cutlass"
    assert build_cache.CUTLASS_CANDIDATES[:2] == (
        str(build_cache.CUTLASS_INSTALLED / "include"), str(build_cache.CUTLASS_BUNDLED / "include"))


def test_pinned_cutlass_needs_no_environment_when_setup_installed_it(tmp_path, monkeypatch):
    installed = _fake_cutlass(tmp_path / "venv-cutlass", build_cache.CUTLASS_REVISION)
    monkeypatch.delenv("ISOEXEC_MLA_SM90_CUTLASS", raising=False)
    monkeypatch.delenv("ISOEXEC_SM90_CUTLASS", raising=False)
    monkeypatch.setattr(build_cache, "CUTLASS_CANDIDATES", (str(installed / "include"),))
    assert build_cache.require_pinned_cutlass("mla", "SM90") == installed / "include"


def test_missing_cutlass_names_make_setup(tmp_path, monkeypatch):
    monkeypatch.delenv("ISOEXEC_MLA_SM90_CUTLASS", raising=False)
    monkeypatch.delenv("ISOEXEC_SM90_CUTLASS", raising=False)
    monkeypatch.setattr(build_cache, "CUTLASS_CANDIDATES", (str(tmp_path / "absent"),))
    with pytest.raises(RuntimeError, match="make setup"):
        build_cache.require_pinned_cutlass("mla", "SM90")


def test_installed_cutlass_at_another_revision_is_refused(tmp_path, monkeypatch):
    other = _fake_cutlass(tmp_path / "other", "0" * 40)
    monkeypatch.setenv("ISOEXEC_MLA_SM90_CUTLASS", str(other))
    with pytest.raises(RuntimeError, match="not 147f5673"):
        build_cache.require_pinned_cutlass("mla", "SM90")


@pytest.mark.parametrize("suffix", ["", "include"])
def test_cutlass_override_accepts_checkout_or_include_root(tmp_path, monkeypatch, suffix):
    root = _fake_cutlass(tmp_path / "checkout")
    monkeypatch.setenv("ISOEXEC_MLA_SM90_CUTLASS", str(root / suffix))
    assert build_cache.require_pinned_cutlass("mla", "SM90") == root / "include"


def test_cutlass_override_without_headers_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("ISOEXEC_MLA_SM90_CUTLASS", str(tmp_path))
    with pytest.raises(RuntimeError, match="does not hold a CUTLASS include tree"):
        build_cache.require_pinned_cutlass("mla", "SM90")


def test_extra_sources_join_the_build_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("ISOEXEC_GDN_CPR_SM90_BUILD_DIR", raising=False)
    source, shared = tmp_path / "source", tmp_path / "shared"
    source.mkdir()
    shared.mkdir()
    (source / "kernel.cu").write_text('#include "../shared/pair.cuh"\n')
    (shared / "pair.cuh").write_text("version one")

    def key(**kw):
        return sm90_build_dir("ISOEXEC_GDN_CPR_SM90_BUILD_DIR", "gdn_cpr-sm90", "tag", source, **kw)

    # No extra trees keeps every existing key.
    assert key(extra_sources=()) == key()
    first = key(extra_sources=(shared,))
    assert first != key()
    (shared / "pair.cuh").write_text("version two")
    assert key(extra_sources=(shared,)) != first
    (shared / "notes.md").write_text("not a source")
    assert key(extra_sources=(shared,)) == key(extra_sources=(shared,))


def test_extra_sources_join_the_sm100_profile_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(build_cache, "cutlass_identity", lambda family="": "cutlass-test")
    source, shared = tmp_path / "source", tmp_path / "shared"
    source.mkdir()
    shared.mkdir()
    (source / "kernel.cu").write_text("kernel")
    (shared / "pair.cuh").write_text("version one")
    plain = build_cache.sm100_profile_digest("fam", source)
    first = build_cache.sm100_profile_digest("fam", source, extra_sources=(shared,))
    assert first != plain
    (shared / "pair.cuh").write_text("version two")
    assert build_cache.sm100_profile_digest("fam", source, extra_sources=(shared,)) != first
    assert build_cache.sm100_profile_digest("fam", source) == plain


def test_shared_sm100_headers_are_in_the_package():
    assert (build_cache.SM100_COMMON / "tmem_pair.cuh").is_file()
