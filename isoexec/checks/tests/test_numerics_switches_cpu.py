"""Require every environment read under ops/ and runtimes/ to be declared in core/flags as a
numerics switch or a side-local setting.
"""

from __future__ import annotations

import ast
import os

from isoexec.core import flags

ISOEXEC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The tree a read can select numerics from. ``core/``, ``debug/`` and the gates are deliberately out:
# they read environment to decide what to RUN, not what a kernel does.
NUMERICS_ROOTS = ("ops", "runtimes")


def _string_constants(tree: ast.AST) -> dict[str, str]:
    """Collect string assignments so the scanner also recognizes environment keys read through
    constants.
    """
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                out[target.id] = value.value
    return out


def _key_of(node: ast.AST, constants: dict[str, str]) -> str | None:
    """The environment key an expression names: a literal, a module constant, or ``self.ATTR``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.Attribute):
        return constants.get(node.attr)
    return None


def _env_reads(path: str) -> set[str]:
    """Keys read from the environment in one file: ``os.environ[...]``,
    ``os.environ.get/pop/setdefault(...)`` and ``os.getenv(...)``, under any module alias, whether
    the key is a literal or a string constant the module defines."""
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    constants = _string_constants(tree)
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr not in ("get", "getenv", "pop", "setdefault"):
                continue
            if not node.args:
                continue
            source = ast.unparse(func.value)
            if "environ" not in source and func.attr != "getenv":
                continue
            key = _key_of(node.args[0], constants)
            if key is not None:
                keys.add(key)
        elif isinstance(node, ast.Subscript):
            if not ast.unparse(node.value).endswith("environ"):
                continue
            key = _key_of(node.slice, constants)
            if key is not None:
                keys.add(key)
    return keys


def _numerics_env_reads() -> dict[str, list[str]]:
    """``{env key: [files]}`` over the production modules of ``ops/`` and ``runtimes/``."""
    found: dict[str, list[str]] = {}
    for root in NUMERICS_ROOTS:
        base = os.path.join(ISOEXEC_DIR, root)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in ("__pycache__", "tests"))
            for filename in sorted(filenames):
                if not filename.endswith(".py") or filename.startswith("test_"):
                    continue
                path = os.path.join(dirpath, filename)
                for key in _env_reads(path):
                    found.setdefault(key, []).append(os.path.relpath(path, ISOEXEC_DIR))
    return found


def test_every_numerics_env_read_is_classified():
    table = set(flags.NUMERICS_SWITCHES) | set(flags.SIDE_LOCAL_SWITCHES)
    reads = _numerics_env_reads()
    # A missing scan root would otherwise pass without examining files.
    assert reads, "the numerics env scan found no reads; the scan is broken, not the tree clean"
    assert table, "neither switch table names a key; there is nothing to classify against"
    unclassified = {key: sites for key, sites in sorted(reads.items()) if key not in table}
    assert not unclassified, (
        "these environment keys are read on a numerics path and appear in neither "
        "core/flags.NUMERICS_SWITCHES nor core/flags.SIDE_LOCAL_SWITCHES, so nothing states "
        f"whether two processes may differ in them: {unclassified}"
    )


def test_the_two_classes_are_disjoint():
    assert not (set(flags.NUMERICS_SWITCHES) & set(flags.SIDE_LOCAL_SWITCHES))


def test_every_side_local_exemption_carries_a_reason():
    for name, reason in flags.SIDE_LOCAL_SWITCHES.items():
        assert isinstance(reason, str) and len(reason) > 12, name


def test_every_compared_switch_has_a_declared_default():
    assert flags.NUMERICS_SWITCHES, "no compared switch; this loop asserts nothing"
    for name in flags.NUMERICS_SWITCHES:
        flag = flags.get(name)
        assert not flag.default_dynamic, name
        assert isinstance(flag.default, str), name
