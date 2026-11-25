"""Independent provenance and dependency checks for the vendored FlashAttention-4 SM100 backward.

These checks need neither an installed ``flash_attn`` / ``quack`` nor a GPU: hashes of our copy,
coverage, license notices and imports. The copy is cited, not compared with upstream.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

OPS = Path(__file__).resolve().parents[1]
VENDOR = OPS / "_vendor" / "flash_attn_cute"


def _manifest():
    return json.loads((VENDOR / "upstream_manifest.json").read_text())


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_pinned_upstreams_retain_their_license_notices():
    manifest = _manifest()
    assert set(manifest["sources"]) == {"flash_attention", "quack"}
    entries = {entry["path"]: entry for entry in manifest["files"]}
    for source, identity in manifest["sources"].items():
        assert identity["url"].startswith("https://github.com/Dao-AILab/")
        assert len(identity["revision"]) == 40
        assert identity["license"] in ("BSD-3-Clause", "Apache-2.0")
        notice = f"licenses/{source}.txt"
        assert notice in entries and entries[notice]["source"] == source and not entries[notice]["modified"]
        body = (VENDOR / notice).read_text()
        assert "Copyright" in body or "copyright" in body.lower()
        assert len(body) > 500
    assert "BSD 3-Clause License" in (VENDOR / "licenses/flash_attention.txt").read_text()
    assert "Apache License" in (VENDOR / "licenses/quack.txt").read_text()
    # FlashAttention's license names its contributors by reference to AUTHORS, so it travels with it.
    assert "Tri Dao" in (VENDOR / "licenses/flash_attention_AUTHORS.txt").read_text()
    assert manifest["sources"]["flash_attention"]["revision"] == "0bbb25a3a5ad3c58c029b3d287d6c9af56a5cad5"


def test_every_recorded_file_matches_its_digest_and_edit_declaration():
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
        assert _digest((VENDOR / relative).read_bytes()) == entry["sha256"], entry["path"]
        changed = entry["sha256"] != entry["upstream_sha256"]
        assert changed == entry["modified"], entry["path"]
        assert bool(entry["modifications"]) == changed, entry["path"]


def test_the_snapshot_has_complete_provenance_coverage():
    manifest = _manifest()
    recorded = {entry["path"] for entry in manifest["files"]}
    authored = set(manifest["authored_files"])
    assert not recorded & authored
    assert all((VENDOR / name).is_file() for name in authored), "a declared authored file is missing"
    assert "PROVENANCE.md" in authored
    for path in VENDOR.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(VENDOR).as_posix()
        assert rel in recorded or rel in authored, f"{rel} is neither recorded nor declared authored"
    forbidden = {".so", ".o", ".a", ".cubin", ".ptx", ".whl", ".pyc"}
    assert not [p for p in VENDOR.rglob("*") if p.is_file() and p.suffix in forbidden and "__pycache__" not in p.parts]


def test_an_import_rewritten_file_is_upstream_plus_exactly_its_declared_edits():
    """Undo the recorded rewrites line by line and drop the note: the bytes must be upstream's.

    This is what proves, without an upstream checkout, that the only edits are the declared imports,
    the declared CuTe DSL 4.8 API replacements (``cute_dsl_rewrites``), IsoExec's declared kernel edits
    (``isoexec_edits``) and the one header line -- the
    copyright header included -- and that the note sits directly after the upstream's leading
    comment block.
    """
    checked = 0
    for entry in _manifest()["files"]:
        if "import_rewrites" not in entry and "cute_dsl_rewrites" not in entry:
            continue
        lines = (VENDOR / entry["path"]).read_text().split("\n")
        # [line, upstream, local]: put upstream's lines back at their recorded place, bottom up.
        edits = entry.get("cute_dsl_rewrites", []) + entry.get("isoexec_edits", [])
        for line, upstream, local in sorted(edits, reverse=True):
            span = slice(line - 1, line - 1 + len(local.split("\n")))
            assert lines[span] == local.split("\n"), (entry["path"], line, local)
            lines[span] = upstream.split("\n")
        note = [i for i, line in enumerate(lines) if line.startswith("# Modified for IsoExec:")]
        assert len(note) == 1, entry["path"]
        assert all(line.startswith("#") for line in lines[:note[0]]), entry["path"]
        del lines[note[0]]
        text = "\n".join(lines)
        for upstream, local in entry.get("import_rewrites", []):
            assert text.count(local + "\n") >= 1, (entry["path"], local)
            text = text.replace(local + "\n", upstream + "\n", 1)
        assert _digest(text.encode()) == entry["upstream_sha256"], entry["path"]
        checked += 1
    assert checked >= 18


def test_every_modified_file_says_so_and_why():
    for entry in _manifest()["files"]:
        if entry["modified"] and entry["path"].endswith(".py"):
            lines = (VENDOR / entry["path"]).read_text().splitlines()
            lead = next(i for i, line in enumerate(lines) if not line.startswith("#"))
            assert any(line.startswith("# Modified for IsoExec:") for line in lines[:lead]), entry["path"]
            assert entry["modifications"], entry["path"]


def _imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            for item in node.names:
                yield node.lineno, item.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.lineno, node.module or ""


def test_no_source_binds_an_external_flash_attention_or_quack_installation():
    offenders = []
    for path in sorted((OPS.parent).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for lineno, module in _imports(path):
            if module.split(".")[0] in ("flash_attn", "flash_attn_4", "quack"):
                rel = path.relative_to(OPS.parent).as_posix()
                if rel.startswith(("ops/attention/gqa_sm90/", "ops/attention/mla_sm90/")):
                    continue  # FA3 comparisons in sm90 test harnesses, not the vendored FA4
                offenders.append(f"{rel}:{lineno}: {module}")
    assert not offenders, "imports of an installed FlashAttention/QuACK: " + ", ".join(offenders)


def test_the_vendored_package_imports_only_itself_cutlass_and_the_standard_runtime():
    allowed = {"cutlass", "cuda", "torch", "tvm_ffi", "math", "operator", "functools", "typing", "dataclasses",
               "enum", "inspect", "hashlib", "os", "re", "importlib", "typing_extensions", "triton", "itertools",
               "abc", "collections", "warnings", "sys"}
    stray = []
    for path in VENDOR.rglob("*.py"):
        for lineno, module in _imports(path):
            if module.split(".")[0] not in allowed:
                stray.append(f"{path.relative_to(VENDOR)}:{lineno}: {module}")
    assert not stray, stray


def test_the_copy_calls_no_cute_dsl_api_that_4_8_removed():
    """nvidia-cutlass-dsl 4.8.0 dropped these deprecated names; the copy compiles at first use, so an
    import-clean tree could still fail at JIT time if one came back."""
    removed = ("cute.core.ThrMma", "cute.core.ThrCopy", "cute.make_fragment(", "vector.extractelement(",
               "vector.insertelement(")
    hits = [f"{path.relative_to(VENDOR)}:{n}: {name}"
            for path in VENDOR.rglob("*.py")
            for n, line in enumerate(path.read_text().splitlines(), 1)
            for name in removed if name in line]
    assert not hits, hits
