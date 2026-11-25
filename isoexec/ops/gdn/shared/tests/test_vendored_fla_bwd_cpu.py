"""The vendored FLA 0.5.2 backward tree is unedited, cited, and the only FLA source IsoExec runs.

Own-copy checks only: nothing here imports or reads an installed flash-linear-attention.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.dirname(_HERE)
TREE = os.path.join(_SHARED, "vendored_fla_bwd")
_ISOEXEC = os.path.dirname(os.path.dirname(os.path.dirname(_SHARED)))  # .../isoexec
_REPO = os.path.dirname(_ISOEXEC)
_PKG = "isoexec.ops.gdn.shared.vendored_fla_bwd"
_FLA_COPYRIGHT = "# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li"


def _doc() -> dict:
    with open(os.path.join(TREE, "PROVENANCE.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _read(rel: str) -> str:
    with open(os.path.join(TREE, rel), "r", encoding="utf-8") as fh:
        return fh.read()


def _header(doc: dict, entry: dict) -> str:
    up = doc["upstream"]
    return doc["header_template"].format(module=entry["upstream_module"], version=up["version"], tag=up["tag"],
                                         commit=up["commit"], repo=up["repo"], provenance=doc["provenance_label"])


_ENTRIES = _doc()["files"]


@pytest.mark.parametrize("entry", _ENTRIES, ids=[e["file"] for e in _ENTRIES])
def test_vendored_files_match_their_recorded_hashes(entry):
    """A hand-edited vendored kernel is a silent change to the backward's rounding."""
    with open(os.path.join(TREE, entry["file"]), "rb") as fh:
        got = hashlib.sha256(fh.read()).hexdigest()
    assert got == entry["vendored_sha256"], (
        f"vendored_fla_bwd/{entry['file']} was edited in place. Revert it, or re-vendor the whole tree from a "
        "new upstream release, re-qualify the backward bitwise, and rewrite PROVENANCE.json."
    )


@pytest.mark.parametrize("entry", _ENTRIES, ids=[e["file"] for e in _ENTRIES])
def test_each_file_is_the_header_then_the_upstream_copyright(entry):
    doc = _doc()
    text = _read(entry["file"])
    header = _header(doc, entry)
    assert text.startswith(header), f"{entry['file']}: does not start with the rendered header_template"
    body = text[len(header):]
    assert body.startswith(_FLA_COPYRIGHT + "\n"), f"{entry['file']}: upstream copyright header missing"
    assert "MIT license" in body.split("\n\n", 1)[0], f"{entry['file']}: upstream license notice missing"


def test_provenance_names_the_upstream_artefact():
    doc = _doc()
    up = doc["upstream"]
    for key in ("project", "distribution", "version", "tag", "commit", "repo", "url", "license", "package",
                "owner", "copyright"):
        assert up.get(key), f"PROVENANCE.json is missing upstream.{key}"
    assert (up["project"], up["version"], up["tag"], up["license"]) == ("flash-linear-attention", "0.5.2", "v0.5.2",
                                                                        "MIT")
    assert len(up["commit"]) == 40 and all(c in "0123456789abcdef" for c in up["commit"])
    assert doc.get("_README") and doc["files"]
    for entry in doc["files"]:
        assert entry["upstream_module"].startswith(up["package"] + ".")
        assert entry["upstream_path"] == f"fla/{entry['file']}"
        assert len(entry["upstream_sha256"]) == 64 and len(entry["vendored_sha256"]) == 64


def test_every_recorded_rewrite_rule_was_applied_and_only_redirects_imports():
    doc = _doc()
    bodies = {e["file"]: _read(e["file"])[len(_header(doc, e)):] for e in doc["files"]}
    for old, new in doc["rewrite_rules"]:
        assert not any(old in body for body in bodies.values()), f"rule {old!r} was not applied"
        assert any(new in body for body in bodies.values()), f"rule {old!r} -> {new!r} matched nothing; delete it"
        assert "fla" in old and _PKG in new, f"rule {old!r} -> {new!r} is not an import redirect into this tree"


def test_every_python_file_in_the_tree_is_vendored_or_declared_authored():
    doc = _doc()
    declared = {e["file"] for e in doc["files"]} | set(doc["authored_files"])
    on_disk = set()
    for root, _, names in os.walk(TREE):
        for name in names:
            if name.endswith(".py"):
                on_disk.add(os.path.relpath(os.path.join(root, name), TREE).replace(os.sep, "/"))
    assert on_disk == declared, f"undeclared: {sorted(on_disk - declared)}; missing: {sorted(declared - on_disk)}"


def test_notice_cites_the_tree():
    with open(os.path.join(_REPO, "NOTICE"), "r", encoding="utf-8") as fh:
        notice = fh.read()
    assert "isoexec/ops/gdn/shared/vendored_fla_bwd/" in notice
    assert _doc()["upstream"]["commit"] in notice


def _fla_imports(path: str) -> list[str]:
    """``import fla...`` / ``from fla... import`` / ``importlib.import_module("fla...")`` in one file."""
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits += [f"{node.lineno}: import {a.name}" for a in node.names if a.name.split(".")[0] == "fla"]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").split(".")[0] == "fla":
            hits.append(f"{node.lineno}: from {node.module} import ...")
        elif (isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None)) in
              ("import_module", "__import__") and node.args):
            arg = node.args[0]
            head = arg.value if isinstance(arg, ast.Constant) else (
                arg.values[0].value if isinstance(arg, ast.JoinedStr) and arg.values
                and isinstance(arg.values[0], ast.Constant) else None)
            if isinstance(head, str) and (head == "fla" or head.startswith("fla.")):
                hits.append(f"{node.lineno}: import_module({head!r}...)")
    return hits


def test_no_isoexec_module_imports_the_fla_package():
    """The GDN backward runs vendored_fla_bwd; nothing in IsoExec may bind the pip ``fla`` package."""
    offenders = {}
    for root, dirs, names in os.walk(_ISOEXEC):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in names:
            if name.endswith(".py"):
                path = os.path.join(root, name)
                hits = _fla_imports(path)
                if hits:
                    offenders[os.path.relpath(path, _REPO)] = hits
    assert not offenders, f"modules importing the fla package: {offenders}"


def test_loader_resolves_backward_pins_in_the_vendored_tree(monkeypatch):
    """The sm90 pin table names kernels that exist in vendored_fla_bwd, and the loader's resolver finds them."""
    from isoexec.ops.gdn.shared import gdn_batch_invariant as gbi
    from isoexec.ops.gdn.shared import gdn_fla_backward as loader

    for module_name, kernel_name in gbi._FLA_BACKWARD_PINS["sm90"]:
        module = loader._import_vendored(module_name)
        assert module.__name__ == f"{_PKG}.ops.{module_name}"
        assert gbi._autotuner(loader._import_vendored, module_name, kernel_name) is not None, kernel_name


def test_backward_stages_come_from_the_vendored_chunk_module(monkeypatch):
    """``fla_backward.load`` reads its stages from the vendored chunk module's namespace."""
    from isoexec.ops.gdn.gdn_cpr_sm90 import fla_backward
    from isoexec.ops.gdn.shared.vendored_fla_bwd.ops.gated_delta_rule import chunk

    monkeypatch.setattr(fla_backward, "get_fla_chunk_backward", lambda: chunk.chunk_gated_delta_rule)
    fla_backward.load.cache_clear()
    try:
        ns = fla_backward.load()
    finally:
        fla_backward.load.cache_clear()
    for name in fla_backward._CHUNK_NAMES:
        assert getattr(ns, name) is vars(chunk)[name], name
    source = ns.reverse_cumsum_kernel.fn.__code__.co_filename
    assert os.path.realpath(source) == os.path.realpath(os.path.join(TREE, "ops", "utils", "cumsum.py"))


def test_context_parallel_is_refused_not_computed():
    from isoexec.ops.gdn.shared.vendored_fla_bwd import _compat

    for name in ("chunk_gated_delta_rule_bwd_dhu_pre_process", "chunk_gated_delta_rule_fwd_h_pre_process",
                 "compress_h0", "expand_h0"):
        with pytest.raises(NotImplementedError, match="context-parallel"):
            getattr(_compat, name)()
    assert _compat.dispatch("common")(len) is len
