"""Synthetic-tree checks of component-tier import boundaries and the existing core-to-ops
prohibition.
"""

from __future__ import annotations

import pytest

from isoexec.checks import lint


def test_the_third_tier_is_scanned_and_carries_no_exemptions():
    assert "nn" in lint.SCANNED_ROOTS
    assert "nn" in lint.ZERO_EXEMPTION_ROOTS
    assert not [ex for ex in lint.load_exemptions() if ex.path.split("/", 1)[0] == "nn"]


def test_the_tier_below_may_not_import_ops():
    """Reject operation imports from the lower core tier."""
    assert ("core", "ops") in lint.FORBIDDEN_LAYER_EDGES
    assert "core" in lint.ZERO_EXEMPTION_ROOTS
    assert not [ex for ex in lint.load_exemptions() if ex.path.startswith("core/")]


@pytest.mark.parametrize(("importer", "imported"), [("ops", "nn"), ("nn", "runtimes"), ("core", "ops")])
def test_each_forbidden_edge_is_declared_with_a_reason(importer, imported):
    assert (importer, imported) in lint.FORBIDDEN_LAYER_EDGES
    reason = lint.LAYER_EDGE_REASONS[(importer, imported)]
    assert len(reason) > 80, f"a rule whose message does not say what to do instead is routed around: {reason!r}"


@pytest.mark.parametrize("statement", [
    "from isoexec.nn.attention import Attention",
    "from ...nn import Attention",
    "import isoexec.nn",
    "import importlib\ndef f():\n    return importlib.import_module('isoexec.nn.attention')",
])
def test_ops_may_not_import_the_component_tier(tmp_path, statement):
    """Including from inside a function body and through importlib -- a rule a string walks around
    is not a rule, and ``packed_proj.attach`` is duck-typed precisely so none is needed."""
    path = tmp_path / "m.py"
    path.write_text(statement + "\n")
    found = lint.scan_file_layering(str(path), "ops/mm/packed_proj.py")
    assert [(v.importer_root, v.imported_root) for v in found] == [("ops", "nn")], statement


@pytest.mark.parametrize("statement", [
    "from isoexec.runtimes.parallel import make_row_parallel",
    "from ..runtimes import parallel",
    "import importlib\nimportlib.import_module('isoexec.runtimes.parallel')",
])
def test_the_component_tier_may_not_import_a_host(tmp_path, statement):
    path = tmp_path / "m.py"
    path.write_text(statement + "\n")
    found = lint.scan_file_layering(str(path), "nn/attention.py")
    assert [(v.importer_root, v.imported_root) for v in found] == [("nn", "runtimes")], statement


def test_the_component_tier_may_import_ops(tmp_path):
    path = tmp_path / "m.py"
    path.write_text("from isoexec.ops.collectives.tp_bytes import make_kv_gather\n")
    assert lint.scan_file_layering(str(path), "nn/attention.py") == []


def test_the_component_tier_may_not_import_test_code(tmp_path):
    """Production components must not import ``isoexec/tests``; tests are a separate allowed boundary."""
    path = tmp_path / "m.py"
    path.write_text("from isoexec.tests.helpers import resolve_rope\n")
    found = lint.scan_file_layering(str(path), "nn/attention.py")
    assert [(v.importer_root, v.imported_root) for v in found] == [("nn", "tests")]


def test_a_test_may_import_test_code(tmp_path):
    """Allow tests to import shared test helpers."""
    path = tmp_path / "m.py"
    path.write_text("from isoexec.tests.helpers import Attention\n")
    assert lint.scan_file_layering(str(path), "nn/tests/test_attention_cpu.py") == []
    assert lint.scan_file_layering(str(path), "nn/tests/_fixtures.py") == []


def test_an_exemption_under_the_component_tier_is_refused(tmp_path):
    """The zero-exemption rule is what makes the import purity a theorem rather than an audit."""
    ledger = tmp_path / "exemptions.json"
    ledger.write_text(
        '{"exemptions": [{"path": "nn/attention.py", "prefix": "vllm", "category": "c",'
        ' "reason": "a reason long enough to pass the justification check, but on a zero-exemption root"}]}'
    )
    (tmp_path / "nn").mkdir()
    (tmp_path / "nn" / "attention.py").write_text("import vllm\n")
    failures = lint.check(isoexec_dir=str(tmp_path), exemptions_path=str(ledger))
    assert any("FORBIDDEN EXEMPTION" in f and "nn/attention.py" in f for f in failures), failures


def test_the_live_tree_is_clean():
    assert lint.check() == []
