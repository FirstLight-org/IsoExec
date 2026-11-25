"""HF architecture selection and private local configuration without Hub access."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from isoexec.compiler.source import ModelSource, ModelSourceError


@pytest.fixture
def hf_package(tmp_path: Path, monkeypatch):
    package = tmp_path / "private_hf_fixture"
    package.mkdir()
    (package / "model.py").write_text(
        "from torch import nn\n"
        "from transformers import AutoConfig, PreTrainedConfig\n"
        "from isoexec.nn import Embedding, LMHead\n"
        "class PrivateConfig(PreTrainedConfig):\n"
        "    model_type = 'isoexec_private_fixture'\n"
        "AutoConfig.register(PrivateConfig.model_type, PrivateConfig, exist_ok=True)\n"
        "class PrivateDecoder(nn.Module):\n"
        "    def __init__(self, config):\n"
        "        super().__init__()\n"
        "        self.config = config\n"
        "        self.embed = Embedding(vocab=config.vocab_size, hidden=config.hidden_size)\n"
        "        self.head = LMHead(vocab=config.vocab_size, hidden=config.hidden_size, tied_to=self.embed.weight)\n"
        "    def forward(self, input_ids):\n"
        "        return self.head(self.embed(input_ids))\n"
        "class AnotherDecoder(PrivateDecoder):\n"
        "    pass\n"
    )
    raw = {"model_type": "isoexec_private_fixture", "architectures": ["PrivateDecoder"],
           "vocab_size": 32, "hidden_size": 8, "text_config": {"a": 1, "b": 2}}
    (package / "config.json").write_text(json.dumps(raw))
    monkeypatch.syspath_prepend(str(tmp_path))
    yield ModelSource(package.name), raw
    for name in tuple(sys.modules):
        if name == package.name or name.startswith(package.name + "."):
            sys.modules.pop(name)


def test_private_config_and_architecture_preserve_the_original_document(hf_package):
    source, raw = hf_package
    before = deepcopy(raw)
    model = source.build_meta(raw)
    assert type(model).__name__ == "PrivateDecoder"
    assert model.get_parameter("embed.weight").is_meta
    assert model.get_parameter("embed.weight") is model.get_parameter("head.weight")
    assert raw == before
    patched = source.apply_overrides(raw, {"text_config": {"a": 3}})
    assert patched["text_config"] == {"a": 3, "b": 2} and raw == before
    raw["architectures"] = ["MissingDecoder", "AnotherDecoder", "PrivateDecoder"]
    assert type(source.build_meta(raw)).__name__ == "AnotherDecoder"


@pytest.mark.parametrize("architectures", [None, [], "PrivateDecoder", ["MissingDecoder"], ["bad.name"]])
def test_invalid_architecture_does_not_fall_back_to_a_default_class(hf_package, architectures):
    source, raw = hf_package
    raw["architectures"] = architectures
    with pytest.raises(ModelSourceError, match="architectures"):
        source.build_meta(raw)


def test_python_config_is_never_imported_for_hf_construction_or_overrides(hf_package, tmp_path):
    source, raw = hf_package
    (tmp_path / source.package / "config.py").write_text(
        "raise AssertionError('Python config hooks are unsupported')\n"
    )
    patched = source.apply_overrides(raw, {"text_config": {"a": 3}})
    assert source.package not in sys.modules
    assert patched["text_config"] == {"a": 3, "b": 2}
    assert type(source.build_meta(patched)).__name__ == "PrivateDecoder"
    assert source.package + ".config" not in sys.modules


@pytest.mark.parametrize("model_type", [None, "", 7])
def test_model_type_is_required(hf_package, model_type):
    source, raw = hf_package
    raw["model_type"] = model_type
    with pytest.raises(ModelSourceError, match="requires model_type"):
        source.build_meta(raw)
