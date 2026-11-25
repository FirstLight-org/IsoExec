"""Check both vendored kernel trees' own copies: recorded hashes, no runtime imports, a complete
machine-readable upstream citation, and rewrite rules that were actually applied. The trees are not
re-diffed against an installed vLLM."""

from __future__ import annotations

import json
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_OPS = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # .../isoexec/ops

TREES = {
    "ops/gdn/shared/vendored_fla": os.path.join(_OPS, "gdn", "shared", "vendored_fla"),
    "ops/gdn/conv_sm90/vendored": os.path.join(_OPS, "gdn", "conv_sm90", "vendored"),
}


def _provenance():
    """The shared checker. Imported by path so this test does not depend on the package __init__."""
    from isoexec.ops.gdn.shared.vendored_fla import _provenance as P

    return P


@pytest.mark.parametrize("name", sorted(TREES))
def test_vendored_files_match_their_recorded_hashes(name):
    """A hand-edited vendored kernel is a silent change to our advertised rounding schedule."""
    rep = _provenance().check_vendored_hashes(TREES[name])
    assert rep.ok, (
        f"{rep.summary()}\n"
        "A vendored kernel was edited in place. That is never the right move: it silently changes the "
        "qualified numerics. Either revert, or re-vendor the whole tree from a new upstream commit, "
        "update PROVENANCE.json, and requalify it (gdn_vendored_determinism_gpu.py, "
        "gdn_native_kernel_parity_test_gpu.py and the full-shape exact-zero logprob gate)."
    )


@pytest.mark.parametrize("name", sorted(TREES))
def test_no_vendored_file_imports_a_runtime(name):
    """The point of vendoring. `_compat.py` is what absorbs the three utilities upstream reaches for."""
    tree = TREES[name]
    offenders = []
    for entry in os.listdir(tree):
        if not entry.endswith(".py"):
            continue
        with open(os.path.join(tree, entry), "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                stripped = line.strip()
                if stripped.startswith(("import vllm", "from vllm", "import megatron", "from megatron")):
                    offenders.append(f"{entry}:{lineno}: {stripped}")
    assert not offenders, "vendored tree still binds a runtime:\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize("name", sorted(TREES))
def test_provenance_records_the_upstream_artefact_machine_readably(name):
    doc = _provenance().load(TREES[name])
    up = doc["upstream"]
    for key in ("distribution", "version", "commit", "repo", "package", "owner"):
        assert up.get(key), f"{name}: PROVENANCE.json is missing upstream.{key}"
    assert doc["files"], f"{name}: PROVENANCE.json lists no files"
    for entry in doc["files"]:
        assert entry["upstream_module"].startswith(up["package"] + ".")
        assert len(entry["upstream_sha256"]) == 64 and len(entry["vendored_sha256"]) == 64


@pytest.mark.parametrize("name", sorted(TREES))
def test_every_recorded_rewrite_rule_is_one_that_was_actually_applied(name):
    """Require every rewrite rule to remove its old text and insert its replacement in at least one file."""
    tree = TREES[name]
    doc = _provenance().load(tree)
    bodies = {}
    for entry in doc["files"]:
        with open(os.path.join(tree, entry["file"]), "r", encoding="utf-8") as fh:
            bodies[entry["file"]] = fh.read()
    for old, new in doc["rewrite_rules"]:
        assert not any(old in body for body in bodies.values()), f"{name}: rule {old!r} was not applied"
        assert any(new in body for body in bodies.values()), (
            f"{name}: rule {old!r} -> {new!r} is recorded but its replacement appears nowhere. "
            "A rule that matched nothing must be deleted from PROVENANCE.json, not carried."
        )


def test_the_two_trees_do_not_disagree_about_their_upstream():
    """Both copies come from ONE vLLM build. Two commits would mean two different Tritons agreed to."""
    docs = {name: _provenance().load(path) for name, path in TREES.items()}
    commits = {name: d["upstream"]["commit"] for name, d in docs.items()}
    assert len(set(commits.values())) == 1, (
        f"vendored trees taken from different upstream commits: {commits}. The GDN forward runs "
        "conv.short and gdn.core back to back; gating them against two upstreams gates nothing."
    )
    versions = {d["upstream"]["version"] for d in docs.values()}
    assert len(versions) == 1, f"vendored trees taken from different vLLM versions: {versions}"


def test_provenance_json_is_valid_json_with_a_readme():
    for name, path in TREES.items():
        with open(os.path.join(path, "PROVENANCE.json"), "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        assert doc.get("_README"), f"{name}: PROVENANCE.json has no _README explaining what it is for"
