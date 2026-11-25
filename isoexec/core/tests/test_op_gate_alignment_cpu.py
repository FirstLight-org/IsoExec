"""Check equivalence-class membership and colocated evidence for each implementation claim."""

from __future__ import annotations

import importlib
import pathlib
import re

from isoexec.core.registry import Registry
from isoexec.core.registry_build import (
    _FAMILIES,
    _ISOEXEC_ROOT,
    build_registry,
    declare_into,
)

_OPS_DIR = pathlib.Path(__file__).resolve().parents[2] / "ops"

# Exact equivalence claims whose tests live outside the owning family’s tests directory.
UNTESTED_CLAIMS: dict[tuple[str, str, str], str] = {
    ("moe", "moe.combine", "pik_leaf_tree"): (
        "named by nn/tests/test_accelerator_resolution_cpu.py, runtimes/tests/"
        "test_ep_combine_class_cpu.py and checks/tests/test_engagement_cpu.py"
    ),
    ("moe", "moe.combine", "pik_leaf_tree_ep_owner_fold"): (
        "named by nn/tests/test_moe_seams_cpu.py, runtimes/tests/test_ep_combine_class_cpu.py and "
        "checks/tests/test_fleet_resolution_cpu.py"
    ),
    ("moe", "moe.route_mechanics", "deterministic_sorted_topk"): (
        "named by nn/tests/test_accelerator_resolution_cpu.py and "
        "checks/tests/test_fleet_resolution_cpu.py"
    ),
}


def _family_registries():
    """{family: Registry with only that family's ops} -- attributes each op to its directory."""
    out = {}
    for fam in _FAMILIES:
        try:
            mod = importlib.import_module(f"{_ISOEXEC_ROOT}.ops.{fam}.spec")
        except ModuleNotFoundError:
            continue
        out[fam] = declare_into(Registry(), mod)
    return out


def _tests_sources(fam: str) -> dict:
    """{filename: source text} for the family's colocated tests (helpers included)."""
    d = _OPS_DIR / fam
    if not d.is_dir():
        return {}
    return {str(p.relative_to(d)): p.read_text(errors="replace") for p in sorted(d.glob("**/tests/*.py"))}


def _findings():
    """Return violations, missing-test reports, and scanned claims so empty scans are detectable."""
    violations, reports, claims = [], [], []
    fams = _family_registries()
    sources = {fam: _tests_sources(fam) for fam in fams}

    # Twin-class membership is registry-wide: `nn` and the host tables select across families, so a
    # class's siblings are counted over the whole registry rather than within one ops/ directory.
    members: dict[str, list[str]] = {}
    for _fam, reg in fams.items():
        for op_name, op in reg.ops.items():
            for impl_id, impl in op.impls.items():
                if impl.twin_class:
                    members.setdefault(impl.twin_class, []).append(f"{op_name}:{impl_id}")

    for fam, reg in fams.items():
        blob = "\n".join(sources[fam].values())
        fam_has_tests = bool(sources[fam])
        for op_name, op in sorted(reg.ops.items()):
            for impl_id, impl in sorted(op.impls.items()):
                if not impl.twin_class:
                    continue
                claims.append((fam, op_name, impl_id))

                # An equivalence class requires at least two implementations.
                if len(members[impl.twin_class]) < 2:
                    violations.append(
                        (fam, op_name, impl_id, f"twin class {impl.twin_class!r} has one member", "lone claim")
                    )

                # (b) a twin-class membership -- the claim that decides whether two hosts may digest
                # the same table -- needs a colocated test naming the claiming impl.
                if not fam_has_tests:
                    reports.append((fam, op_name, impl_id, "family has no tests/ dir", "claim untested"))
                elif not re.search(re.escape(impl_id), blob):
                    reports.append(
                        (fam, op_name, impl_id, f"no colocated test names impl {impl_id!r}", "claim untested")
                    )

    return violations, reports, claims


def _print_table(rows, title):
    if not rows:
        return
    print(f"\n== {title} ({len(rows)}) ==")
    for row in rows:
        print("  " + " | ".join(str(c) for c in row))


def test_registry_builds():
    reg = build_registry()
    assert reg.ops, "registry is empty"


def test_every_equivalence_claim_has_a_counterparty():
    violations, _reports, claims = _findings()
    assert len(claims) >= 5, f"only {len(claims)} twin-class claims were found; the scan is not reading them"
    _print_table(violations, "LONE equivalence claims")
    assert not violations, (
        f"{len(violations)} twin class(es) have a single member, so the equivalence they claim has "
        "nothing to be checked against. Either the sibling was renamed or deleted -- restore the "
        "membership -- or the claim is over and `twin_class` should come off the impl."
    )


def test_claims_have_colocated_tests():
    _, reports, claims = _findings()
    assert len(claims) >= 5, f"only {len(claims)} twin-class claims were found; the scan is not reading them"
    rows = [r for r in reports if r[-1] == "claim untested"]
    _print_table(rows, "equivalence claims without a colocated test")
    found = {(fam, op, impl) for fam, op, impl, _why, _kind in rows}
    recorded = set(UNTESTED_CLAIMS)
    assert found - recorded == set(), (
        f"equivalence claim(s) with no colocated test and no entry in UNTESTED_CLAIMS: "
        f"{sorted(found - recorded)}. Add a test in the family's own tests/ naming the impl id, or "
        "record the debt with the reason it is owed."
    )
    assert recorded - found == set(), (
        f"UNTESTED_CLAIMS records debts that are paid: {sorted(recorded - found)}. Delete the "
        "entries rather than leaving a licence behind for the next untested claim."
    )
    for key, reason in UNTESTED_CLAIMS.items():
        assert reason.strip(), f"{key} is recorded with no reason"


def test_every_family_with_claims_has_tests_dir():
    """Every family declaring equivalence must own a tests directory."""
    fams = _family_registries()
    assert len(fams) >= 5, f"only {len(fams)} families were read; the ops/ walk is not finding them"
    claiming, missing = [], []
    for fam, reg in fams.items():
        if any(impl.twin_class for op in reg.ops.values() for impl in op.impls.values()):
            claiming.append(fam)
            if not _tests_sources(fam):
                missing.append(fam)
    assert claiming, "no family declares a twin_class; this rule has no subject"
    _print_table([(f, "no tests/ directory") for f in missing], "claiming families without tests/")
    assert not missing, f"families declaring an equivalence with no tests/ directory: {missing}"


def _run():
    import traceback

    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = []
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed.append(name)
            traceback.print_exc()
            print(f"FAIL {name}")
    print(f"{len(fns) - len(failed)}/{len(fns)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _run()
