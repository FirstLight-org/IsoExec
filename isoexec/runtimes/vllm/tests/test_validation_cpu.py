"""Native construction refuses grammar-masked logprobs and non-BF16 model or KV dtypes."""

from types import SimpleNamespace

import pytest
import torch

from isoexec.runtimes.vllm.validation import kv_cache_refusal, structured_outputs_refusal


def _config(*, backend="auto", cache_dtype="auto", dtype=torch.bfloat16):
    return SimpleNamespace(
        structured_outputs_config=SimpleNamespace(backend=backend),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        model_config=SimpleNamespace(dtype=dtype),
    )


def test_clean_config_admits():
    config = _config()
    assert structured_outputs_refusal(config) is None
    assert kv_cache_refusal(config) is None
    assert kv_cache_refusal(_config(cache_dtype="bfloat16")) is None


def test_an_explicit_grammar_backend_is_refused():
    assert "grammar bitmask" in structured_outputs_refusal(_config(backend="xgrammar"))


@pytest.mark.parametrize("over", [{"cache_dtype": "fp8"}, {"cache_dtype": "nvfp4"}, {"dtype": torch.float16}])
def test_non_bf16_model_or_kv_dtype_is_refused(over):
    assert "A36" in kv_cache_refusal(_config(**over))


def test_kv_cache_accepts_only_the_encoding_the_model_declares():
    assert kv_cache_refusal(_config(cache_dtype="fp8_ds_mla"), model_kv_dtype="fp8_ds_mla") is None
    assert "A36" in kv_cache_refusal(_config(cache_dtype="auto"), model_kv_dtype="fp8_ds_mla")
    assert "A36" in kv_cache_refusal(_config(cache_dtype="fp8"), model_kv_dtype="fp8_ds_mla")
