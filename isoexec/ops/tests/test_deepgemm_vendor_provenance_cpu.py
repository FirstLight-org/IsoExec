"""Independent provenance and dependency checks for the DeepGEMM source snapshot.

These checks require neither an installed DeepGEMM wheel nor a GPU. Set
ISOEXEC_DEEPGEMM_UPSTREAM_ROOT to a checkout to enable the additional upstream
re-diff; the ordinary hash/coverage checks always run.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


OPS = Path(__file__).resolve().parents[1]
VENDOR = OPS / "_vendor" / "deep_gemm"


def _manifest():
    return json.loads((VENDOR / "upstream_manifest.json").read_text())


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pinned_upstreams_retain_their_license_notices():
    manifest = _manifest()
    assert {"deep_gemm", "deep_jit", "cutlass"} <= manifest["sources"].keys()
    entries = {entry["path"]: entry for entry in manifest["files"]}
    for source, identity in manifest["sources"].items():
        assert identity["url"].startswith("https://")
        assert len(identity["revision"]) == 40
        assert identity["license"]
        notice = f"licenses/{source}.txt"
        assert notice in entries, f"Missing recorded license for {source}"
        assert entries[notice]["source"] == source
        body = (VENDOR / notice).read_text()
        assert "Copyright" in body or "copyright" in body
        assert len(body) > 500


def test_every_recorded_source_matches_its_digest_and_edit_declaration():
    manifest = _manifest()
    seen = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        assert str(relative) not in seen
        seen.add(str(relative))
        assert entry["source"] in manifest["sources"]
        assert entry["upstream_path"]
        assert len(entry["upstream_sha256"]) == 64
        assert _digest(VENDOR / relative) == entry["sha256"], entry["path"]
        changed = entry["sha256"] != entry["upstream_sha256"]
        assert changed == entry["modified"], entry["path"]
        assert bool(entry["modifications"]) == changed, entry["path"]


def test_snapshot_source_trees_have_complete_provenance_coverage():
    recorded = {entry["path"] for entry in _manifest()["files"]}
    for folder in ["csrc", "include", "mega", "utils", "licenses"]:
        for path in (VENDOR / folder).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            assert path.relative_to(VENDOR).as_posix() in recorded, path
    for filename in ["_api.py", "locality_domain.py"]:
        assert filename in recorded
    forbidden = {".so", ".o", ".a", ".cubin", ".whl"}
    assert not [p for p in VENDOR.rglob("*") if p.is_file() and p.suffix in forbidden]


def test_python_sources_do_not_bind_an_external_deepgemm_installation():
    offenders = []
    roots = [VENDOR, *sorted(OPS.glob("*/deepgemm_sm100"))]
    for root in roots:
        assert root.is_dir(), root
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                if isinstance(node, ast.Import):
                    modules = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules = [node.module or ""]
                else:
                    continue
                if any(name == "deep_gemm" or name.startswith("deep_gemm.") for name in modules):
                    offenders.append(f"{path}:{node.lineno}")
    assert not offenders, "External runtime imports: " + ", ".join(offenders)


def test_optional_rediff_matches_pinned_upstream_checkouts():
    checkout = os.environ.get("ISOEXEC_DEEPGEMM_UPSTREAM_ROOT")
    if not checkout:
        pytest.skip("Set ISOEXEC_DEEPGEMM_UPSTREAM_ROOT for upstream re-diff")
    checkout = Path(checkout)
    roots = {"deep_gemm": checkout, "deep_jit": checkout / "third-party/deep_jit",
             "cutlass": checkout / "third-party/cutlass", "fmt": checkout / "third-party/fmt"}
    manifest = _manifest()
    for source, identity in manifest["sources"].items():
        revision = subprocess.check_output(["git", "-C", str(roots[source]), "rev-parse", "HEAD"], text=True).strip()
        assert revision == identity["revision"], source
    for entry in manifest["files"]:
        assert _digest(roots[entry["source"]] / entry["upstream_path"]) == entry["upstream_sha256"], entry["path"]
