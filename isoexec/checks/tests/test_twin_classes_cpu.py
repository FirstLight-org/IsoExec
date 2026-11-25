"""Check that equivalence classes cite existing local batteries or immutable external evidence."""

from __future__ import annotations

import os
import subprocess
from pathlib import PurePosixPath

import pytest

from isoexec.core.registry import (
    DEBUG_PROOF_REVISION,
    DEBUG_PROOF_ROOT,
    TWIN_PROOFS,
    runnable_batteries,
)
from isoexec.core.registry_build import build_registry

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 3))

# Classes with only archived evidence, with the missing local proof stated explicitly.
ARCHIVED_ONLY_EVIDENCE: dict[str, str] = {
    # Every equivalence class currently has a runnable local battery.
}


@pytest.mark.parametrize("path", sorted({path for paths in TWIN_PROOFS.values() for path in paths}))
def test_every_named_battery_exists(path):
    if not path.startswith(DEBUG_PROOF_ROOT):
        assert os.path.isfile(os.path.join(_REPO_ROOT, path)), path
        return
    relative = path.removeprefix(DEBUG_PROOF_ROOT)
    assert relative.startswith("examples/validation/") and relative.endswith(".py")
    assert ".." not in PurePosixPath(relative).parts
    # Source archives and shallow clones may omit the debug snapshot. Never fetch
    # during CPU tests; the immutable link still identifies the external battery.
    try:
        present = subprocess.run(
            ["git", "-C", _REPO_ROOT, "cat-file", "-e", f"{DEBUG_PROOF_REVISION}^{{commit}}"],
            capture_output=True, check=False,
        )
    except FileNotFoundError:
        pytest.skip("git unavailable; external debug battery is pinned by revision")
    if present.returncode:
        pytest.skip("external debug snapshot absent from this source checkout")
    subprocess.run(
        ["git", "-C", _REPO_ROOT, "cat-file", "-e", f"{DEBUG_PROOF_REVISION}:{relative}"],
        capture_output=True, check=True,
    )


def test_every_class_names_at_least_one_battery():
    # Require nonempty evidence before applying all() or parametrizing sibling tests.
    assert TWIN_PROOFS, "TWIN_PROOFS is empty; no twin class names a battery because there are none"
    assert all(TWIN_PROOFS[name] for name in TWIN_PROOFS)


def test_every_multi_member_class_has_a_proof():
    registry = build_registry()
    members: dict[str, list[str]] = {}
    for op in registry.ops.values():
        for impl in op.impls.values():
            if impl.twin_class:
                members.setdefault(impl.twin_class, []).append(f"{op.name}:{impl.impl_id}")
    assert registry.ops, "the registry declares no ops; this proof has no subject"
    assert members, "no declared impl carries a twin_class; this proof has no subject"
    unproven = {name: ms for name, ms in members.items() if len(ms) > 1 and not TWIN_PROOFS.get(name)}
    assert not unproven, f"twin classes with more than one member and no battery: {unproven}"


@pytest.mark.parametrize("twin_class", sorted(TWIN_PROOFS))
def test_every_class_has_a_runnable_battery(twin_class):
    """Require runnable local evidence or an explicit archived-only exception."""
    local = runnable_batteries(twin_class)
    if twin_class in ARCHIVED_ONLY_EVIDENCE:
        assert not local, (
            f"{twin_class} now has a runnable battery ({list(local)}); remove it from "
            f"ARCHIVED_ONLY_EVIDENCE rather than leaving a debt recorded against a class that paid it"
        )
        pytest.xfail(f"{twin_class}: no runnable battery. Owed: {ARCHIVED_ONLY_EVIDENCE[twin_class]}")
    assert local, (
        f"twin class {twin_class!r} names only archived permalinks "
        f"({list(TWIN_PROOFS[twin_class])}); no battery in this tree proves it. Add one, or record "
        f"the debt in ARCHIVED_ONLY_EVIDENCE saying exactly what is going unchecked"
    )
    for path in local:
        assert os.path.isfile(os.path.join(_REPO_ROOT, path)), path
