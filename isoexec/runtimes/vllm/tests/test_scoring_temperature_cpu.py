"""The reporting temperature cannot mutate the native sampling distribution."""

import pytest
import torch
from isoexec.checks import Refusal
from isoexec.runtimes.vllm import scoring_temperature as scoring


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.setattr(scoring, "_OWNER", None)
    monkeypatch.setattr(scoring, "_TEMPERATURE", 1.0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_tempered_reporting_preserves_sampling_source_and_registers_scalar(dtype):
    model = torch.nn.Linear(2, 2, dtype=dtype, device="cpu")
    scoring.install(model, 0.7)
    logits = torch.tensor([[1.125, -2.25]], dtype=dtype, device="cpu")
    original = logits.clone()
    scored = scoring.prepare_logits(logits)
    assert torch.equal(logits, original)
    assert scored.dtype == torch.float32 and scored.data_ptr() != logits.data_ptr()
    assert torch.equal(scored, original.float() / torch.tensor(0.7, dtype=torch.float32, device="cpu"))
    assert "_isoexec_score_temperature" in dict(model.named_buffers())
    assert "_isoexec_score_temperature" not in model.state_dict()


def test_unit_temperature_retains_original_storage():
    model = torch.nn.Linear(2, 2, dtype=torch.bfloat16, device="cpu")
    scoring.install(model, 1.0)
    logits = torch.ones((1, 2), dtype=torch.bfloat16, device="cpu")
    assert scoring.prepare_logits(logits) is logits


@pytest.mark.parametrize("temperature", [0, -1, float("inf"), float("nan")])
def test_invalid_scoring_temperature_refuses(temperature):
    model = torch.nn.Linear(2, 2, dtype=torch.bfloat16, device="cpu")
    with pytest.raises(Refusal, match="score_temperature"):
        scoring.install(model, temperature)
