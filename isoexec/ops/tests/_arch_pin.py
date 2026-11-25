"""Pin the dispatch architecture to the reference arm for composition tests.
Accelerator-resolution tests explicitly undo the pin; importing a reference proof must not depend on visible GPUs."""

from __future__ import annotations

from fnmatch import fnmatch

import pytest


def pin_reference_arch(monkeypatch) -> str:
    """Point ``ops/_dispatch.ARCH`` at the non-accelerator arch for the caller's scope."""
    from isoexec.core.arch import NON_ACCELERATOR_ARCH
    from isoexec.ops import _dispatch

    monkeypatch.setattr(_dispatch, "ARCH", NON_ACCELERATOR_ARCH)
    return NON_ACCELERATOR_ARCH


@pytest.fixture(autouse=True)
def _reference_composition(monkeypatch):
    """Per test. Import this into a conftest to pin a whole folder."""
    pin_reference_arch(monkeypatch)


@pytest.fixture(scope="module", autouse=True)
def _reference_composition_module():
    """Per module, for files that place their trees once in a module-scoped fixture.

    A function-scoped pin would be undone before the module-scoped fixture that needs it ever ran,
    which is not a pin.
    """
    with pytest.MonkeyPatch.context() as mp:
        pin_reference_arch(mp)
        yield


def reference_composition_for(pattern: str):
    """Build an autouse reference-architecture fixture restricted to filenames matching pattern."""

    @pytest.fixture(autouse=True)
    def _reference_composition(request, monkeypatch):
        if fnmatch(request.path.name, pattern):
            pin_reference_arch(monkeypatch)

    return _reference_composition
