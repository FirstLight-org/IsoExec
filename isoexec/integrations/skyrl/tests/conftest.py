"""The SkyRL adapter's tests exercise the SkyRL that environments/default.toml pins.

Without SkyRL, or with a SkyRL that predates the ``isoexec`` config field the adapter is called through,
none of them can run; they are left uncollected (or skipped, when named explicitly) instead of failing.
"""

import importlib.util


def _pinned_skyrl() -> bool:
    try:
        if importlib.util.find_spec("skyrl.train.config") is None:
            return False
        from skyrl.train.config import SkyRLTrainConfig
    except ImportError:
        return False
    return "isoexec" in SkyRLTrainConfig.__dataclass_fields__


_PINNED = _pinned_skyrl()


def pytest_ignore_collect(collection_path, config):
    if not _PINNED and collection_path.name.startswith("test_") and collection_path.suffix == ".py":
        return True
    return None


def pytest_collection_modifyitems(config, items):
    # Files named explicitly on the command line bypass pytest_ignore_collect; skip their tests instead.
    if _PINNED:
        return
    import pytest
    from pathlib import Path

    here = Path(__file__).parent
    skip = pytest.mark.skip(reason="needs the SkyRL commit environments/default.toml pins")
    for item in items:
        if Path(str(item.fspath)).parent == here:
            item.add_marker(skip)
