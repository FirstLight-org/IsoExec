"""Require collected tests to declare CPU/GPU lanes in their filenames and every CPU file to match
collection globs. Suffixes allow lane selection without importing device-dependent modules.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Globs pytest collects, from ``python_files`` in ``pyproject.toml``. Kept in sync by
# ``test_the_scan_matches_the_configured_python_files``.
_COLLECTED_GLOBS = ("test_*.py", "*_test.py")

# Collected files without lane suffixes, with reasons for each exception.
UNSUFFIXED: dict[str, str] = {
    "isoexec/ops/moe/moe_sm90/tests/test_sm90.py": (
        "GPU-only (skips without SM90). Cited by name in TWIN_PROOFS['moe.expert_mlp.sm90_wgmma_v1'] "
        "in core/registry.py; renaming it means editing that table"
    ),
    "isoexec/ops/moe/moe_sm90/tests/test_integration.py": (
        "GPU-only (skips without SM90). Also cited in TWIN_PROOFS['moe.expert_mlp.sm90_wgmma_v1']"
    ),
    "isoexec/ops/moe/moe_sm90/tests/test_tp8.py": (
        "GPU-only (skips without SM90). Cited by name in moe_sm90/tests/README.md, which is not "
        "a test file"
    ),
    "isoexec/ops/norms/norms_sm90/tests/test_sm90.py": (
        "GPU-only (skips without SM90). Cited by name in norms_sm90/reference.py's docstring and in "
        "norms_sm90/README.md, neither of which is a test file"
    ),
}


# CPU files excluded from collection require explicit reasons; shared helpers use unsuffixed
# names.
UNCOLLECTED_CPU: dict[str, str] = {}


def _cpu_lane_files() -> list[str]:
    """Every ``*_cpu.py`` under ``isoexec/``, repo-relative, sorted."""
    return sorted(
        path.relative_to(_REPO_ROOT).as_posix()
        for path in (_REPO_ROOT / "isoexec").rglob("*_cpu.py")
        if "__pycache__" not in path.parts
    )


def _collected_test_files() -> list[str]:
    """Every file a default ``pytest isoexec`` collects as a test module, repo-relative, sorted."""
    found: set[str] = set()
    for glob in _COLLECTED_GLOBS:
        for path in (_REPO_ROOT / "isoexec").rglob(glob):
            if "__pycache__" in path.parts:
                continue
            found.add(path.relative_to(_REPO_ROOT).as_posix())
    return sorted(found)


def test_every_collected_test_file_names_its_lane():
    on_disk = _collected_test_files()
    assert len(on_disk) >= 250, f"only {len(on_disk)} test files found; the scan is probably wrong"

    unnamed = [p for p in on_disk if not p.endswith(("_cpu.py", "_gpu.py")) and p not in UNSUFFIXED]
    assert not unnamed, (
        "test files with no lane suffix, which `pytest -k cpu` therefore deselects and no lane "
        f"runs: {unnamed}. Rename each to `_cpu.py` (passes with no CUDA device) or `_gpu.py`, "
        "with `git mv`, and update the references to it. If a rename is blocked, add it to "
        "UNSUFFIXED with the reason."
    )


def test_every_cpu_lane_file_is_collected():
    """A file named ``_cpu`` that ``python_files`` never matches runs in no lane at all."""
    on_disk = _cpu_lane_files()
    assert len(on_disk) >= 250, f"only {len(on_disk)} `_cpu.py` files found; the scan is probably wrong"

    collected = set(_collected_test_files())
    uncollected = [p for p in on_disk if p not in collected and p not in UNCOLLECTED_CPU]
    assert not uncollected, (
        "files claim the CPU lane in their name but pytest's `python_files` "
        f"{list(_COLLECTED_GLOBS)} never matches them, so no lane runs them: {uncollected}. "
        "Rename each to `test_*_cpu.py` with `git mv` and give it real test functions (a `main()` "
        "script is not collected even once the name matches), or add it to UNCOLLECTED_CPU with "
        "the reason it must stay out."
    )


def test_uncollected_cpu_list_has_no_stale_entries():
    """An entry that outlived its file would quietly re-license the next uncollected file."""
    on_disk = set(_cpu_lane_files())
    stale = sorted(p for p in UNCOLLECTED_CPU if p not in on_disk)
    assert not stale, f"UNCOLLECTED_CPU names files that are gone or already renamed: {stale}"
    for path, reason in UNCOLLECTED_CPU.items():
        assert reason.strip(), f"{path} is exempted with no reason"


def test_unsuffixed_list_has_no_stale_entries():
    """An entry that outlived its file would quietly re-license the next unnamed file."""
    on_disk = set(_collected_test_files())
    stale = sorted(p for p in UNSUFFIXED if p not in on_disk)
    assert not stale, f"UNSUFFIXED names files that are gone or already renamed: {stale}"
    for path, reason in UNSUFFIXED.items():
        assert reason.strip(), f"{path} is exempted with no reason"


def test_the_scan_matches_the_configured_python_files():
    """Keep the scanner’s filename globs aligned with pytest collection."""
    import tomllib

    with open(_REPO_ROOT / "pyproject.toml", "rb") as handle:
        configured = tomllib.load(handle)["tool"]["pytest"]["ini_options"]["python_files"]
    assert sorted(configured) == sorted(_COLLECTED_GLOBS), (
        f"pyproject's python_files is {configured}, but this file scans {list(_COLLECTED_GLOBS)}"
    )
