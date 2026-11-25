"""Check that no module in checks/ reads ambient environment."""

from __future__ import annotations

import ast
import os

CHECKS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_NAMES = frozenset({"environ", "getenv", "environb"})


def _env_reads(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _ENV_NAMES:
            if isinstance(node.value, ast.Name) and node.value.id == "os":
                hits.append(f"line {node.lineno}: os.{node.attr}")
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            for alias in node.names:
                if alias.name in _ENV_NAMES:
                    hits.append(f"line {node.lineno}: from os import {alias.name}")
    return hits


def test_no_check_reads_the_environment():
    # Check scan coverage separately from environment-read detection.
    scanned = [n for n in sorted(os.listdir(CHECKS_DIR)) if n.endswith(".py")]
    assert len(scanned) > 5, f"only {len(scanned)} checks/ modules were scanned: {scanned}"
    offenders = {}
    for name in scanned:
        hits = _env_reads(os.path.join(CHECKS_DIR, name))
        if hits:
            offenders[name] = hits
    assert not offenders, (
        f"checks/ modules reading the environment: {offenders}. "
        f"A check that reads ambient env is not comparing what it says it compares; "
        f"the caller reads it and passes the value in."
    )
