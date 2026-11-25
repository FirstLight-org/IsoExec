"""Run the runtime-import lint in the CPU suite without importing optional engines."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap

import pytest

_LINT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHECKER = os.path.join(_LINT_DIR, "lint.py")

# Load the checker by path: no sys.path games, and no dependency on how pytest was invoked.
_spec = importlib.util.spec_from_file_location("_isoexec_runtime_imports_lint", _CHECKER)
ri = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ri  # dataclasses resolves annotations through sys.modules
_spec.loader.exec_module(ri)


# Minimum scan size guards against missing roots without pinning the exact file count.
MIN_SCANNED = 200

# Printed when the floor trips: the result below is "nothing was read", never "nothing is wrong".
_DEAD_SCAN = (
    "the real-tree scan visited almost no files, so every clean result in this test is a scan that "
    "saw nothing -- see test_the_real_tree_scan_actually_visits_files"
)


def _scanned() -> list:
    """Every file the production scans actually visit, so a dead scan is visible rather than clean."""
    return [rel for _, rel in ri._walk_py(ri.ISOEXEC_DIR, ri.SCANNED_ROOTS, ri.SCANNED_FILES)]


def test_the_real_tree_scan_actually_visits_files():
    """Reject vacuous success from an empty source scan."""
    seen = _scanned()
    assert len(seen) >= MIN_SCANNED, (
        f"the layering/runtime scan visited {len(seen)} files under {ri.SCANNED_ROOTS}; a scan that "
        f"sees nothing reports a clean tree. Check ISOEXEC_DIR and SCANNED_ROOTS before trusting "
        f"any other result in this file."
    )
    for root in ri.SCANNED_ROOTS:
        assert any(rel.startswith(root + "/") for rel in seen), f"root {root!r} contributed no files"


def test_tree_is_clean():
    """Every runtime import under a scanned tier is classified, and nothing is stale."""
    assert len(_scanned()) >= MIN_SCANNED, _DEAD_SCAN
    failures = ri.check()
    assert not failures, "\n\n".join(failures)


def test_the_zero_exemption_roots_have_no_runtime_imports_at_all():
    """Core, nn, and schema must remain free of runtime imports, without exemptions."""
    for root in ri.ZERO_EXEMPTION_ROOTS:
        assert [rel for _, rel in ri._walk_py(ri.ISOEXEC_DIR, (root,), ())], f"{root}/ scanned no files"
        violations = ri.scan(roots=(root,), files=())
        assert violations == [], f"{root}/ must not import a runtime:\n" + "\n".join(str(v) for v in violations)


def test_the_zero_exemption_roots_carry_no_exemptions():
    assert ri.load_exemptions(), "the exemption ledger is empty; this loop asserts nothing"
    for ex in ri.load_exemptions():
        for root in ri.ZERO_EXEMPTION_ROOTS:
            assert not ex.path.startswith(f"{root}/"), f"{root}/ must stay exemption-free, found: {ex}"


def test_every_exemption_names_an_existing_file():
    assert ri.load_exemptions(), "the exemption ledger is empty; this loop asserts nothing"
    for ex in ri.load_exemptions():
        assert os.path.isfile(os.path.join(ri.ISOEXEC_DIR, ex.path)), f"exemption names a missing file: {ex.path}"


def test_every_exemption_is_classified_and_justified():
    assert ri.load_exemptions(), "the exemption ledger is empty; this loop asserts nothing"
    for ex in ri.load_exemptions():
        assert ex.category in ri.CATEGORIES
        assert len(ex.reason) > 40, f"reason too thin to be a justification: {ex}"


def test_function_local_imports_are_caught(tmp_path):
    """The common shape in this tree is an import inside a function body, not at module top."""
    f = tmp_path / "m.py"
    f.write_text(
        textwrap.dedent(
            """
            def go():
                from vllm.model_executor.determinism import batch_invariant
                return batch_invariant
            """
        )
    )
    found = ri.scan_file(str(f), "ops/fake.py")
    assert [v.target for v in found] == ["vllm.model_executor.determinism.batch_invariant"]
    assert found[0].lineno == 3


def test_dynamic_imports_are_caught(tmp_path):
    """importlib.import_module binds a runtime just as hard as an import statement.

    ops/gdn/shared/gdn_batch_invariant.py::_autotuner is a live example, and an import-statement-only
    scan waves it straight through.
    """
    f = tmp_path / "m.py"
    f.write_text(
        textwrap.dedent(
            """
            import importlib

            def a():
                return importlib.import_module("vllm.model_executor.determinism.batch_invariant")

            def b(name):
                return importlib.import_module(f"vllm.third_party.flash_linear_attention.ops.{name}")

            def c(name):
                return importlib.import_module(name)          # unresolvable, not reportable

            def d():
                return importlib.import_module("torch.nn")    # not a runtime
            """
        )
    )
    assert sorted(v.target for v in ri.scan_file(str(f), "ops/fake.py")) == [
        "vllm.model_executor.determinism.batch_invariant",
        "vllm.third_party.flash_linear_attention.ops",
    ]


def test_relative_imports_are_not_violations(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("from ..megatron_thing import x\nfrom . import vllm_helper\n")
    assert ri.scan_file(str(f), "ops/fake.py") == []


def _run_lint_against(tmp_root: str, exemptions_path: str) -> list[str]:
    return ri.check(isoexec_dir=tmp_root, exemptions_path=exemptions_path)


@pytest.fixture()
def sandbox(tmp_path):
    """A miniature isoexec tree: ops/ + core/ + a copy of the real ledger."""
    (tmp_path / "ops").mkdir()
    (tmp_path / "core").mkdir()
    (tmp_path / "ops" / "real.py").write_text("from vllm.device_allocator.cumem import CuMemAllocator\n")
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "exemptions": [
                    {
                        "path": "ops/real.py",
                        "prefix": "vllm.device_allocator.cumem",
                        "category": "c",
                        "reason": "memory plumbing, named in kernel-ownership.md section 2c -- permanently allowed",
                    }
                ]
            }
        )
    )
    return tmp_path, str(ledger)


def test_sandbox_starts_clean(sandbox):
    root, ledger = sandbox
    assert _run_lint_against(str(root), ledger) == []


def test_a_new_runtime_import_fails(sandbox):
    """ACCEPTANCE: adding `import vllm` to an ops/ file must fail."""
    root, ledger = sandbox
    (root / "ops" / "fresh.py").write_text("import vllm\n")
    failures = _run_lint_against(str(root), ledger)
    assert any("NEW RUNTIME IMPORT" in f and "ops/fresh.py" in f for f in failures), failures


def test_a_new_runtime_import_in_a_zero_exemption_root_fails(sandbox):
    root, ledger = sandbox
    (root / "core" / "fresh.py").write_text("import megatron.core\n")
    failures = _run_lint_against(str(root), ledger)
    assert any("NEW RUNTIME IMPORT" in f and "core/fresh.py" in f for f in failures), failures


def test_removing_a_violation_without_updating_the_ledger_fails(sandbox):
    """ACCEPTANCE: a stale exemption is as bad as a missing one."""
    root, ledger = sandbox
    (root / "ops" / "real.py").write_text("# cleaned up, vendored into ops/mm\n")
    failures = _run_lint_against(str(root), ledger)
    assert any("STALE EXEMPTION" in f and "ops/real.py" in f for f in failures), failures


def test_an_exemption_under_a_zero_exemption_root_is_refused(sandbox):
    root, ledger = sandbox
    data = json.loads(open(ledger).read())
    data["exemptions"].append(
        {
            "path": "core/sneaky.py",
            "prefix": "vllm",
            "category": "c",
            "reason": "trying to smuggle a runtime import into the clean root, which must not be possible",
        }
    )
    (root / "core" / "sneaky.py").write_text("import vllm\n")
    open(ledger, "w").write(json.dumps(data))
    failures = _run_lint_against(str(root), ledger)
    assert any("FORBIDDEN EXEMPTION" in f for f in failures), failures


def test_cli_exits_nonzero_on_failure(sandbox, monkeypatch):
    root, ledger = sandbox
    (root / "ops" / "fresh.py").write_text("import vllm\n")
    monkeypatch.setattr(ri, "ISOEXEC_DIR", str(root))
    monkeypatch.setattr(ri, "EXEMPTIONS_PATH", ledger)
    assert ri.main([]) == 1


def test_module_runs_as_a_script():
    """The pre-commit hook shells out to this file; it must work standalone."""
    proc = subprocess.run(
        [sys.executable, _CHECKER, "--summary"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "outstanding debt" in proc.stdout


def test_core_does_not_import_ops():
    """The live property. ``core/`` is the shared-machinery tier; ``ops/`` is the
    op-implementation and registry tier, and the dependency runs one way only."""
    assert len(_scanned()) >= MIN_SCANNED, _DEAD_SCAN
    violations = ri.scan_layering()
    assert violations == [], "isoexec/core/ must not import from isoexec/ops/:\n" + "\n".join(
        str(v) for v in violations
    )


def test_the_whole_tree_check_covers_layering():
    """``check()`` is what the pre-commit hook and the CLI run, so the rule has to be in it."""
    assert len(_scanned()) >= MIN_SCANNED, _DEAD_SCAN
    assert ri.check() == []


@pytest.mark.parametrize(
    "path, source, where",
    [
        ("core/moe/layer.py", "from ...ops.moe.moe_sm90 import packed_experts as _pack\n", "relative"),
        ("core/moe/layer.py", "import isoexec.ops.moe.moe_sm90.packed_experts\n", "absolute"),
        (
            "core/moe/layer.py",
            "from isoexec.ops.moe.moe_sm90 import packed_experts\n",
            "absolute from",
        ),
        ("core/x.py", "def f():\n    from ..ops.moe.moe_sm90 import packed_experts\n", "function-local"),
        (
            "core/x.py",
            "import importlib\n\n\ndef f():\n    return importlib.import_module("
            '"isoexec.ops.moe.moe_sm90.packed_experts")\n',
            "importlib",
        ),
    ],
)
def test_a_core_to_ops_import_fails(sandbox, path, source, where):
    """ACCEPTANCE: adding a ``core/ -> ops/`` import must fail the lint, however it is written.

    A layering rule an inner function or a string literal can walk around is not a rule.
    """
    root, ledger = sandbox
    f = root / path
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(source)
    failures = _run_lint_against(str(root), ledger)
    assert any("FORBIDDEN LAYERING IMPORT" in m and path in m for m in failures), (where, failures)


def test_the_permitted_direction_is_permitted(sandbox):
    """``ops -> core`` is the whole point of the rule: it must stay legal, both spellings."""
    root, ledger = sandbox
    (root / "ops" / "moe").mkdir(parents=True)
    (root / "ops" / "moe" / "w.py").write_text(
        "from ...core.linear import packed_view\n"
        "from isoexec.core.bindings import BoundOp\n"
    )
    assert _run_lint_against(str(root), ledger) == []


def test_within_core_relative_imports_are_not_violations(sandbox):
    """The check resolves relative imports against the importing file's package; a sibling or a
    parent inside ``core/`` must not be mistaken for a cross-root one."""
    root, ledger = sandbox
    (root / "core" / "moe").mkdir(parents=True)
    (root / "core" / "moe" / "layer.py").write_text(
        "from . import routing\n"
        "from ..bindings import OpTable\n"
        "from ...schema import ModelConfig\n"  # up to the isoexec package: a different root, allowed
    )
    assert _run_lint_against(str(root), ledger) == []


def test_layering_has_no_exemption_mechanism(sandbox):
    """The runtime rule has a ledger; this one does not, on purpose. An entry naming the import
    must not silence it -- it is refused as a forbidden exemption AND the import still fails."""
    root, ledger = sandbox
    (root / "core" / "sneaky.py").write_text("from ..ops.moe.moe_sm90 import packed_experts\n")
    data = json.loads(open(ledger).read())
    data["exemptions"].append(
        {
            "path": "core/sneaky.py",
            "prefix": "isoexec.ops",
            "category": "c",
            "reason": "an attempt to buy an exemption for the layering rule, which has no such thing",
        }
    )
    open(ledger, "w").write(json.dumps(data))
    failures = _run_lint_against(str(root), ledger)
    assert any("FORBIDDEN LAYERING IMPORT" in m for m in failures), failures


def test_cli_exits_nonzero_on_a_layering_violation(sandbox, monkeypatch):
    root, ledger = sandbox
    (root / "core" / "fresh.py").write_text("from ..ops.moe.moe_sm90 import packed_experts\n")
    monkeypatch.setattr(ri, "ISOEXEC_DIR", str(root))
    monkeypatch.setattr(ri, "EXEMPTIONS_PATH", ledger)
    assert ri.main([]) == 1


@pytest.mark.parametrize("path", ["checks/new.py", "compiler/new.py", "runtimes/new.py", "models/new.py", "config.py"])
def test_every_shipping_root_refuses_test_only_imports(sandbox, path):
    root, ledger = sandbox
    source = root / path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("from isoexec.tests.helpers import fixture\n")
    failures = _run_lint_against(str(root), ledger)
    assert any("FORBIDDEN LAYERING IMPORT" in message and path in message for message in failures)
