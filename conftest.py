"""Report unrun GPU batteries at session end. ISOEXEC_GPU_BATTERIES=required turns skips into
failures.
"""

from __future__ import annotations

_UNWITNESSED: list[tuple[str, str]] = []
_MARKER = "UNWITNESSED: "


def pytest_runtest_logreport(report):
    """Remember every skip whose reason names an unwitnessed claim."""
    if report.skipped and isinstance(getattr(report, "longrepr", None), tuple):
        reason = report.longrepr[2]
        if _MARKER in reason:
            _UNWITNESSED.append((report.nodeid, reason.split(_MARKER, 1)[1].strip()))


def pytest_terminal_summary(terminalreporter):
    if not _UNWITNESSED:
        return
    terminalreporter.write_sep("=", f"{len(_UNWITNESSED)} GPU BATTERIES DID NOT RUN", yellow=True, bold=True)
    terminalreporter.write_line(
        "These claims were NOT checked by this run. A skip is not a pass. "
        "Set ISOEXEC_GPU_BATTERIES=required to make each one a failure instead."
    )
    for nodeid, claim in _UNWITNESSED:
        terminalreporter.write_line(f"  {nodeid.split('::')[-1]}")
        terminalreporter.write_line(f"      {claim}")
