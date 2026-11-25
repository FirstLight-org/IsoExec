"""The verl integration must not import the SkyRL adapter (which imports SkyRL itself)."""

from __future__ import annotations

import ast
from pathlib import Path

VERL = Path(__file__).resolve().parents[1]


def _imports(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)


def test_verl_integration_never_imports_skyrl():
    offenders = sorted(
        f"{path.relative_to(VERL)}: {name}"
        for path in VERL.rglob("*.py")
        if "tests" not in path.parts
        for name in _imports(path)
        if name == "skyrl" or name.startswith(("skyrl.", "isoexec.integrations.skyrl"))
    )
    assert not offenders, offenders
