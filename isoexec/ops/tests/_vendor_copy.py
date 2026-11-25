"""Shared own-copy checks and script execution for the vendored MM, RoPE and gate copies.

A vendored copy was qualified for IsoExec's purpose (determinism) when it went in, so these checks
look only at OUR copy: its provenance record names the upstream it cites, and its source still
hashes to what was recorded. Nothing here compares against an installed upstream package."""

from __future__ import annotations


def assert_provenance_record_is_complete(V, *, upstream_module, keys, holder_key=None) -> None:
    """A copy must cite its origin: project, module, version/commit and URL."""
    p = V.provenance()
    for key in keys:
        assert p.get(key), f"provenance is missing {key!r}"
    assert p["module"] == upstream_module
    assert p["symbols"], "no symbols recorded"
    holder = V if holder_key is None else getattr(V, p[holder_key])
    for dotted in p["symbols"]:
        # A method is recorded as "Class.method"; the holder is the class in that case.
        assert hasattr(holder, dotted.split(".", 1)[-1]), f"{dotted} is recorded as vendored but is not defined"


def assert_copy_matches_recorded_digests(V, *, requalify: str) -> None:
    """Nobody has edited the copy.

    ``requalify`` names the determinism checks an intended edit must pass before new digests are
    recorded.
    """
    recorded = V.provenance()["symbols"]
    actual = V.source_digests()
    assert set(actual) == set(recorded), (
        f"vendored symbol set changed: missing={sorted(set(recorded) - set(actual))} "
        f"extra={sorted(set(actual) - set(recorded))}"
    )
    moved = sorted(k for k in recorded if recorded[k] != actual[k])
    assert not moved, (
        f"the vendored copy no longer matches its recorded digests -- somebody edited a VERBATIM "
        f"copy: {moved}. If the edit is intended it is no longer a copy; re-derive the provenance "
        f"and re-run {requalify} before recording new digests."
    )


def run_as_script(namespace: dict) -> int:
    """Run every ``test_*`` in ``namespace`` without pytest, and return a process exit code."""
    import traceback

    tests = [v for k, v in sorted(namespace.items()) if k.startswith("test_") and callable(v)]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            name = type(e).__name__
            if name in ("Skipped", "_Skipped", "OutcomeException") or "skip" in name.lower():
                print(f"SKIP  {fn.__name__}: {e}")
                continue
            failed.append(fn.__name__)
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\nRESULT: {'FAIL ' + str(failed) if failed else 'PASS'}")
    return 1 if failed else 0
