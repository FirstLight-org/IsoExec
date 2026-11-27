"""Dead code and test helpers of the KDA packages (no GPU, no build).

One shared test-helper module (the two packages' copies had diverged), no dead ``dtype``/``device`` dicts, no
compile-time knob whose every other value is wrong, an import-builds-nothing check that can fail, and SASS gates that
skip cleanly without the CUDA toolchain.
"""

from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path

import pytest

KDA = Path(__file__).resolve().parents[1]


def test_one_shared_fixtures_module_cpu():
    copies = sorted(p.relative_to(KDA).as_posix() for p in KDA.rglob("_fixtures.py"))
    assert copies == ["tests/_fixtures.py"], copies
    for test in (p for p in KDA.rglob("test_*.py") if p.name != Path(__file__).name):
        for m in re.finditer(r"from (\S+)\._fixtures import", test.read_text()):
            assert m.group(1) == "isoexec.ops.kda.tests", (test.name, m.group(1))


def test_no_function_builds_a_dict_it_never_reads_cpu():
    for path in KDA.rglob("*.py"):
        if "tests" in path.parts:
            continue
        for fn in ast.walk(ast.parse(path.read_text())):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            read = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            for node in ast.walk(fn):
                if not isinstance(node, ast.Assign):
                    continue
                pairs = []
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        pairs.append((target, node.value))
                    elif isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                        pairs += list(zip(target.elts, node.value.elts))
                for t, v in pairs:
                    if (isinstance(t, ast.Name) and isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                            and v.func.id == "dict"):
                        assert t.id in read, f"{path.relative_to(KDA)}:{node.lineno} {fn.name}: {t.id} is never read"


def test_the_decode_state_pipeline_depth_is_not_a_knob_cpu():
    src = (KDA / "glm_sm100" / "csrc" / "decode_rec.cu").read_text()
    assert "KDA_NSTAGE" not in src                         # no -D override: any value but 2 was wrong
    assert re.search(r"constexpr int NSTAGE = 2\b", src)
    assert 'static_assert(NSTAGE == 2' in src


def test_the_import_check_can_fail_cpu():
    """``assert_imports_build_nothing`` fails when an import fills an extension cache (here with a fake build, so
    nothing is compiled); the ``sys.modules`` check it replaced could not, since the module never enters it."""
    from isoexec.ops.kda.tests._fixtures import assert_imports_build_nothing

    assert_imports_build_nothing("import isoexec.ops.kda.glm_sm100.engine")
    fake = ("import isoexec.ops.kda.glm_sm100._extension as e;"
            "e.build_sm100_extension = lambda *a, **k: type('M', (), {'chunk_len': lambda s: 64, '__file__': 'x'})();"
            "e.extension()")
    with pytest.raises(AssertionError, match="glm_sm100 was built"):
        assert_imports_build_nothing(fake)


def test_sass_gates_skip_without_the_toolchain_cpu(monkeypatch):
    from isoexec.ops.kda.tests import _fixtures

    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(pytest.skip.Exception, match="CUDA toolchain"):
        _fixtures.require_cuda_toolchain()
    for gate in (KDA / "glm_sm100" / "tests" / "test_sass_cpu.py", KDA / "glm_rec_sm100" / "tests" / "test_sass_cpu.py"):
        assert "require_cuda_toolchain()" in gate.read_text(), gate
