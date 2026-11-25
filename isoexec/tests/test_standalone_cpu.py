"""The installed public package has no dependency on a training framework."""

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_public_import_has_no_framework_or_cuda_side_effect():
    program = """
import sys
for name in ('skyrl', 'verl', 'vllm', 'sglang', 'megatron'):
    sys.modules[name] = None
import isoexec
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
assert sorted(isoexec.__all__) == ['IsoExecConfig', 'ModelSource', 'ModelSourceError']
from isoexec import IsoExecConfig, ModelSource, ModelSourceError
assert 'torch' not in sys.modules
assert ModelSource('isoexec.models.qwen3_5').package == 'isoexec.models.qwen3_5'
assert issubclass(ModelSourceError, ValueError)
assert IsoExecConfig().scoring == 'contracted'
import torch
from isoexec.config import IsoExecConfig as Config
assert Config().policy(True).bound == 0.0
assert not torch.cuda.is_initialized()
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_public_package_has_no_other_attributes():
    import isoexec

    assert sorted(isoexec.__all__) == ["IsoExecConfig", "ModelSource", "ModelSourceError"]
    # ``isoexec.weights`` is a subpackage, so it is bound once anything imports it.
    for name in ("api", "build", "register", "load", "names", "handshake", "check"):
        with pytest.raises(AttributeError):
            getattr(isoexec, name)


def test_owned_reduction_kernels_match_recorded_source_bytes():
    root = Path(__file__).resolve().parents[1] / "ops/mm"
    source = (root / "batch_invariant_reductions.py").read_text()
    lines = source.splitlines()
    recorded = json.loads((root / "batch_invariant_reductions.json").read_text())["symbols_sha256"]
    actual = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        actual[node.name] = hashlib.sha256("\n".join(lines[start - 1:node.end_lineno]).encode()).hexdigest()
    assert actual == recorded


def test_owned_conv_metadata_preserves_recorded_source():
    root = Path(__file__).resolve().parents[1] / "ops/gdn/shared"
    source = (root / "conv_metadata.py").read_text()
    recorded = json.loads((root / "conv_metadata.json").read_text())
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    function = "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])
    for original, replacement in recorded["substitutions"].items():
        assert function.count(replacement) == 1
        function = function.replace(replacement, original)
    assert hashlib.sha256(function.encode()).hexdigest() == recorded["sha256"]


def _framework_import_violations(root):
    violations = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if relative.parts[0] == "integrations":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            targets = ([node.module] if isinstance(node, ast.ImportFrom)
                       else [item.name for item in node.names] if isinstance(node, ast.Import) else [])
            if any(name and name.split(".")[0] in ("skyrl", "verl") for name in targets):
                violations.append(f"{relative}:{node.lineno}")
    return violations


def test_framework_imports_stay_inside_their_adapters():
    violations = _framework_import_violations(Path(__file__).resolve().parents[1])
    assert not violations, violations


def test_framework_import_boundary_covers_shared_runtime(tmp_path):
    for relative in (
        "core/model.py", "runtimes/megatron/model.py",
        "integrations/skyrl/adapter.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("import skyrl\n")
    assert set(_framework_import_violations(tmp_path)) == {
        "core/model.py:1", "runtimes/megatron/model.py:1",
    }
