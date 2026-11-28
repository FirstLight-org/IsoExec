"""The trainer worker's hf_config: a package's own config class when it ships one (DeepSeek V4.1, whose model_type
the installed transformers does not register), AutoConfig otherwise (Qwen, unchanged)."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("megatron.core")
pytest.importorskip("skyrl.train.config")

from isoexec.compiler.source import ModelSource  # noqa: E402
from isoexec.integrations.skyrl import megatron, model_folder  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[3] / "models/deepseek_v4_1_flash/tests/fixtures"


def _worker():
    return NS(cfg=NS(_isoexec_config=NS(model_descriptor={}), algorithm=NS(enable_sample_support_replay=False)),
              strategy=NS(), megatron_config=None)


def test_v41_uses_the_package_config_class(monkeypatch):
    raw = json.loads((FIXTURES / "debug_config.json").read_text())
    import skyrl.utils.tok as tok
    monkeypatch.setattr(tok, "get_tokenizer", lambda path, **kwargs: object())
    source = ModelSource("isoexec.models.deepseek_v4_1_flash")
    monkeypatch.setattr(model_folder, "source_config", lambda selected, cfg, path: (source, raw))
    worker = _worker()
    megatron.init_configs(worker, str(FIXTURES), "mc")
    from isoexec.models.deepseek_v4_1_flash.model import CONFIG_CLASS
    assert isinstance(worker.strategy.hf_config, CONFIG_CLASS)
    assert worker.strategy.hf_config.model_type == "deepseek_v41"


def test_package_without_config_class_keeps_autoconfig(monkeypatch):
    import transformers
    source = ModelSource("isoexec.models.qwen3_5")
    monkeypatch.setattr(model_folder, "source_config", lambda selected, cfg, path: (source, {}))
    monkeypatch.setattr(model_folder, "configuration_path", lambda path: path)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda path, **kw: ("auto", path))
    import skyrl.utils.tok as tok
    monkeypatch.setattr(tok, "get_tokenizer", lambda path, **kw: "tokenizer")
    worker = _worker()
    megatron.init_configs(worker, "/qwen", "mc")
    assert worker.strategy.hf_config == ("auto", "/qwen")
