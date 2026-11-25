"""The shared tiny Qwen3.5 configs build through their model packages."""

import pytest

from isoexec.compiler.source import ModelSource
from isoexec.models.qwen3_5.tests.tiny import TINY_DENSE_HF
from isoexec.models.qwen3_5_moe.tests.tiny import TINY_MOE_HF
from isoexec.nn import GDN, MoE


@pytest.mark.parametrize("package,raw", [("isoexec.models.qwen3_5", TINY_DENSE_HF),
                                         ("isoexec.models.qwen3_5_moe", TINY_MOE_HF)])
def test_tiny_config_builds_on_meta(package, raw):
    model = ModelSource(package).build_meta(raw)
    assert len(model.layers) == 4
    assert [isinstance(layer.mixer, GDN) for layer in model.layers] == [True, True, True, False]
    assert all(isinstance(layer.mlp, MoE) == package.endswith("_moe") for layer in model.layers)
    assert all(p.is_meta for p in model.parameters())
