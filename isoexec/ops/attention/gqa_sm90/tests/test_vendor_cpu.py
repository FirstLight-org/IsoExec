"""The vendored FA3 tree matches its own PROVENANCE.json (recorded hashes, rewrite rules) and is complete for the build."""

from __future__ import annotations

import hashlib
import json
import re


from isoexec.ops.attention.gqa_sm90 import spec
from isoexec.ops.attention.gqa_sm90._extension import FA3_DIR, SOURCES

_REPO_LEVEL = ("LICENSE", "AUTHORS")


def _provenance() -> dict:
    return json.loads((FA3_DIR / "PROVENANCE.json").read_text())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _vendored_name(name: str) -> str:
    """LICENSE/AUTHORS carry a .txt suffix so the wheel's package-data globs include them."""
    return f"{name}.txt" if name in _REPO_LEVEL else name


def test_provenance_pins_the_commit_spec_states():
    prov = _provenance()
    assert prov["upstream"]["commit"] == spec.FA3_COMMIT
    assert prov["upstream"]["license"] == "BSD-3-Clause"


def test_every_vendored_file_matches_its_recorded_hash_and_nothing_else_is_there():
    prov = _provenance()
    listed = {_vendored_name(e["file"]) for e in prov["files"]}
    on_disk = {p.relative_to(FA3_DIR).as_posix() for p in FA3_DIR.rglob("*") if p.is_file()} - {"PROVENANCE.json"}
    assert on_disk == listed
    edited = {e["file"] for e in prov["isoexec_edits"]}
    added = {e["file"] for e in prov["isoexec_files"]}
    for entry in prov["files"]:
        data = (FA3_DIR / _vendored_name(entry["file"])).read_bytes()
        assert _sha256(data) == entry["vendored_sha256"], entry["file"]
        if entry["file"] in added:
            assert entry["upstream_sha256"] is None, f"{entry['file']} is IsoExec's, not upstream's"
        elif entry["file"] not in edited and not any(r["file"] == entry["file"] for r in prov["rewrite_rules"]):
            assert entry["upstream_sha256"] == entry["vendored_sha256"], f"{entry['file']} must be unmodified"


def test_isoexec_edits_are_the_segment_split_and_declared():
    """Only the segment split touches upstream files, and each edit is stated."""
    prov = _provenance()
    assert {e["file"] for e in prov["isoexec_edits"]} == {
        "block.h", "flash.h", "flash_fwd_launch_template.h", "flash_prepare_scheduler.cu", "flash_api_stable.cpp",
        "mainloop_fwd_sm90_tma_gmma_ws.hpp"}
    assert all(e["what"] for e in prov["isoexec_edits"] + prov["isoexec_files"])


def test_only_the_operator_namespace_is_rewritten():
    prov = _provenance()
    assert {r["file"] for r in prov["rewrite_rules"]} == {"flash_api_stable.cpp"}
    for rule in prov["rewrite_rules"]:
        assert rule["old"].replace("flash_attn_3", "isoexec_gqa_sm90") == rule["new"]


def test_the_include_closure_is_vendored():
    """Every quoted include of a compiled source resolves inside fa3/ or names a CUTLASS/CUDA header."""
    seen, todo = set(), list(SOURCES)
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        path = FA3_DIR / rel
        assert path.is_file(), rel
        for inc in re.findall(r'^\s*#include\s+"([^"]+)"', path.read_text(), re.M):
            local = next((c.resolve() for c in (path.parent / inc, FA3_DIR / inc) if c.is_file()), None)
            if local is not None and FA3_DIR in local.parents:
                todo.append(local.relative_to(FA3_DIR).as_posix())
            else:
                assert inc.startswith(("cutlass/", "cute/", "cub/")), f"{rel} includes {inc}, which is not vendored"


