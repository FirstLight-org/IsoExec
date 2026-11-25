"""Model source reconstruction without native runtimes or checkpoint loading."""

import json
import os
from collections.abc import Iterator
from pathlib import Path
import subprocess
import sys
from typing import cast

import pytest
import torch

from isoexec.compiler.source import ModelSource, ModelSourceError


@pytest.fixture
def author_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    package = tmp_path / "source_test_model"
    package.mkdir()
    # Namespace package: the author needs no __init__.py export file.
    (package / "model.py").write_text(
        "from torch import nn\n"
        "from isoexec.nn import Embedding, LMHead\n"
        "class Model(nn.Module):\n"
        "    def __init__(self, config):\n"
        "        super().__init__()\n"
        "        config.shape['hidden'] = int(config.shape['hidden'])\n"
        "        self.embedding = Embedding(**config.shape)\n"
        "        self.readout = LMHead(**config.shape,\n"
        "                              tied_to=self.embedding.weight)\n"
        "    def forward(self, input_ids, seq):\n"
        "        return self.readout(self.embedding(input_ids) * 0.5)\n"
    )
    (package / "weight.py").write_text("CHECKPOINT_FORMAT = 'fixture'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield package
    for name in tuple(sys.modules):
        if name == package.name or name.startswith(package.name + "."):
            sys.modules.pop(name)


def test_json_worker_reconstructs_the_authored_module_and_preserves_ties(author_package: Path) -> None:
    source = ModelSource(author_package.name)
    assert source.to_dict() == {"package": author_package.name}
    assert author_package.name not in sys.modules
    script = """
import json
import sys
from isoexec.compiler.source import ModelSource
from isoexec.models import ModelSource as RootSource, build_component_model
from isoexec.models import ModelSource as CatalogSource

data = json.load(sys.stdin)
source = ModelSource.from_dict(data['source'])
assert CatalogSource is ModelSource
assert RootSource is ModelSource
assert set(sys.modules['isoexec.models'].__all__) == {'ModelSource', 'ModelSourceError', 'build_component_model'}
assert not any(name.startswith('isoexec.models.') for name in sys.modules)
assert 'torch' not in sys.modules
assert source.package not in sys.modules
raw = data['raw_config']
model = build_component_model(source, raw)
assert raw['shape']['hidden'] == '8', 'model construction mutated its caller input'
assert type(model).__module__ == source.package + '.model'
assert model.embedding.weight.is_meta
assert model.embedding.weight.shape == (16, 8)
assert model.embedding.weight is model.readout.weight
assert source.package + '.weight' not in sys.modules
assert source.weight_module().CHECKPOINT_FORMAT == 'fixture'
assert not any(name in sys.modules for name in ('vllm', 'megatron'))
import torch
assert not torch.cuda.is_initialized()
"""
    repo = Path(__file__).resolve().parents[3]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(author_package.parent), str(repo))))
    payload = {"source": source.to_dict(), "raw_config": {
        "model_type": "llama", "architectures": ["Model"], "shape": {"hidden": "8", "vocab": 16}}}
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(payload),
        env=env,
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_source_data_refuses_invalid_fields_without_importing() -> None:
    data = {"package": "unimported_author.decoder"}
    assert ModelSource.from_dict(data) == ModelSource(data["package"])
    for invalid in (None, {}, {**data, "api_version": 2}, {**data, "model": {}}):
        with pytest.raises(ModelSourceError, match="exactly 'package'"):
            ModelSource.from_dict(invalid)
    with pytest.raises(ModelSourceError, match="importable dotted Python package"):
        ModelSource.from_dict({**data, "package": "__main__"})
    assert data["package"] not in sys.modules


def test_source_and_capture_pin_float_factories_and_restore_native_default_after_failure(
    author_package: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from isoexec.compiler.capture.capture import capture
    from isoexec.compiler.tests.test_capture_cpu import Model, example_inputs
    import importlib

    source = ModelSource(author_package.name)
    constructor = importlib.import_module(author_package.name + '.model')
    original_init = constructor.Model.__init__
    seen = []
    def initialize(self, config):
        seen.append(torch.get_default_dtype())
        original_init(self, config)
    monkeypatch.setattr(constructor.Model, '__init__', initialize)
    raw = {'model_type': 'llama', 'architectures': ['Model'], 'shape': {'hidden': 8, 'vocab': 16}}
    prior = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = source.build_meta(raw)
        assert seen == [torch.float32] and model.get_parameter('embedding.weight').dtype == torch.bfloat16
        assert torch.get_default_dtype() == torch.bfloat16
        def fail(self, config):
            assert torch.get_default_dtype() == torch.float32
            raise RuntimeError('construction failed')
        monkeypatch.setattr(constructor.Model, '__init__', fail)
        with pytest.raises(RuntimeError, match='construction failed'):
            source.build_meta(raw)
        assert torch.get_default_dtype() == torch.bfloat16
        class ConstantModel(Model):
            def forward(self, input_ids, seq, *, factor=1.0):
                hidden = self.lexicon(input_ids) * torch.tensor(0.1)
                for block in self.units:
                    hidden = block(hidden, seq)
                return self.readout(self.finish(hidden))
        captured = capture(ConstantModel(), example_inputs())
        torch.set_default_dtype(torch.float32)
        repeated = capture(ConstantModel(), example_inputs())
        assert str(captured.program.graph) == str(repeated.program.graph)
        constants = [str(node.target) for node in captured.program.graph.nodes if node.op == 'get_attr']
        assert constants
        for name in constants:
            value, other = captured.program.get_buffer(name), repeated.program.get_buffer(name)
            assert value.dtype == other.dtype == torch.float32 and torch.equal(value, other)
            assert torch.equal(value, torch.tensor(0.1, dtype=torch.float32))
    finally:
        torch.set_default_dtype(prior)

def test_package_errors_distinguish_a_missing_role_config_field_and_dependency(author_package: Path) -> None:
    from isoexec.models import build_component_model

    # The constructor takes a ModelSource, never a bare package string or checkpoint path.
    with pytest.raises(TypeError, match="requires a ModelSource"):
        build_component_model(cast(ModelSource, author_package.name), {})
    with pytest.raises(ModelSourceError, match="model.py and weight.py"):
        ModelSource("missing_source_test_model").build_meta({})
    with pytest.raises(ModelSourceError, match="requires model_type"):
        ModelSource(author_package.name).build_meta({})
    # A missing dependency in an existing weight role keeps its original error.
    (author_package / "weight.py").write_text("import missing_source_test_dependency\n")
    with pytest.raises(ModuleNotFoundError) as error:
        ModelSource(author_package.name).weight_module()
    assert error.value.name == "missing_source_test_dependency"


@pytest.mark.parametrize(
    "factory, message",
    [
        ("def Model(config):\n    return object()\n", "must define an nn.Module"),
        (
            "from torch import nn\nclass Model(nn.Linear):\n"
            "    def __init__(self, config): super().__init__(8, 8, device='cpu')\n",
            "parameter 'weight' is on cpu",
        ),
        (
            "import torch\nfrom torch import nn\nclass Model(nn.Module):\n"
            "    def __init__(self, config):\n"
            "        super().__init__()\n"
            "        self.register_buffer('cache', torch.ones(1, device='cpu'))\n",
            "buffer 'cache' is on cpu",
        ),
    ],
)
def test_construction_rejects_non_modules_and_materialized_tensors(
    author_package: Path, factory: str, message: str,
) -> None:
    (author_package / "model.py").write_text(factory)
    with pytest.raises(ModelSourceError, match=message):
        ModelSource(author_package.name).build_meta({
            "model_type": "llama", "architectures": ["Model"], "shape": {"hidden": 8, "vocab": 16}})
