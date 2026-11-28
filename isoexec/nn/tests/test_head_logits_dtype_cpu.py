"""An LMHead may produce FP32 logits from the head GEMM's unrounded root."""

import pytest

import isoexec.nn as inn
from isoexec.nn.head import head_selection


def test_fp32_logits_select_the_fp32_head_and_enter_the_facts():
    bf16 = inn.LMHead(vocab=16, hidden=8)
    fp32 = inn.LMHead(vocab=16, hidden=8, logits_dtype="fp32")
    assert head_selection(bf16) == {} and "logits_dtype" not in bf16.facts()
    assert head_selection(fp32) == {"facts": {"logits": "fp32"}} and fp32.facts()["logits_dtype"] == "fp32"
    with pytest.raises(ValueError, match="logits_dtype"):
        inn.LMHead(vocab=16, hidden=8, logits_dtype="fp16")


def test_fp32_head_binds_the_fp32_root_gemm_on_sm100_only(monkeypatch):
    from isoexec.ops import _dispatch
    from isoexec.ops.table import OpResolutionError, OpTable
    fp32 = inn.LMHead(vocab=16, hidden=8, logits_dtype="fp32")
    inputs = {"tp": 1, "boundary_out_dtype": "fp32"}
    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    bound = OpTable().bind("logprobs.head", site="head.head", inputs=inputs, **head_selection(fp32))
    assert bound.impl == "fp32_root"
    assert OpTable().bind("logprobs.head", site="head.head", inputs=inputs).impl == "eager"
    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    with pytest.raises(OpResolutionError, match="SM100"):
        OpTable().bind("logprobs.head", site="head.head", inputs=inputs, **head_selection(fp32))
