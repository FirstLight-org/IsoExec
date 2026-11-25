"""The CPR adapter propagates shared FLA loader admission failures."""

import pytest

from isoexec.ops.gdn.gdn_cpr_sm90 import fla_backward


def test_owned_loader_error_propagates_without_fallback_import(monkeypatch):
    import importlib

    failure = RuntimeError("FLA warmup failed")

    def load():
        raise failure

    def alternate_import(*args, **kwargs):
        raise AssertionError("the adapter must not bypass its shared loader")

    monkeypatch.setattr(fla_backward, "get_fla_chunk_backward", load)
    monkeypatch.setattr(importlib, "import_module", alternate_import)
    fla_backward.load.cache_clear()
    try:
        with pytest.raises(RuntimeError) as caught:
            fla_backward.load()
        assert caught.value is failure
    finally:
        fla_backward.load.cache_clear()
