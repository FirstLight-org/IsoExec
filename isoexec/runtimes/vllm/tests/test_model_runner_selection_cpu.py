"""The V4.1 engine consumes the Engram token history supplied by vLLM Model Runner V2."""
import json
from pathlib import Path

from isoexec.compiler.source import ModelSource
from isoexec.runtimes.environment import resolved_environment


def test_v41_checkpoint_selects_the_model_with_host_state_on_runner_v2():
    fixtures = Path(__file__).resolve().parents[3] / "models/deepseek_v4_1_flash/tests/fixtures"
    source = ModelSource("isoexec.models.deepseek_v4_1_flash")
    author = source.build_meta(json.loads((fixtures / "debug_config.json").read_text()))
    assert callable(author.engine_model_state_cls)
    assert resolved_environment(str(fixtures))["VLLM_USE_V2_MODEL_RUNNER"] == "1"
