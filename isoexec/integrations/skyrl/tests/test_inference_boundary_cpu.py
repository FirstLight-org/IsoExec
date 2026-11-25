"""The full-vocabulary request and response handling SkyRL's ``/skyrl/v1/generate`` endpoint calls."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from isoexec.integrations.skyrl import inference

RAW = NS(logprobs_mode="raw_logprobs")


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_full_comparison_flag_requires_an_explicit_boolean(value):
    with pytest.raises(ValueError, match="full_logprob_comparison must be a boolean"):
        inference.full_comparison_enabled({"full_logprob_comparison": value})


def test_one_isoexec_flag_drives_the_client_and_engine_policy():
    assert not inference.full_comparison_enabled(None) and not inference.full_comparison_enabled({})
    assert inference.full_comparison_enabled({"full_logprob_comparison": True})
    options = {}
    inference.engine_defaults(options)
    assert options == {"max_logprobs": -1, "logprobs_mode": "raw_logprobs"}


@pytest.mark.parametrize("options", [{"max_logprobs": 20}, {"logprobs_mode": "processed_logprobs"}])
def test_full_logprobs_reject_conflicting_engine_settings(options):
    with pytest.raises(ValueError, match="full logprobs"):
        inference.engine_defaults(options)


@pytest.mark.parametrize("body,params,model_config,lora,match", [
    ({"return_sample_support": True}, {}, RAW, None, "sample-support"),
    ({"features": {"image": "..."}}, {}, RAW, None, "text-only"),
    ({}, {}, RAW, object(), "LoRA"),
    ({}, {"n": 2}, RAW, None, "n=1"),
    ({}, {}, NS(logprobs_mode="processed_logprobs"), None, "raw_logprobs"),
])
def test_a_request_that_cannot_return_full_rows_is_refused_before_generation(body, params, model_config, lora, match):
    with pytest.raises(ValueError, match=match):
        inference.prepare_full_logprobs_capture(body, params, model_config, lora)


def test_capture_sets_the_sampling_params_that_make_the_engine_return_every_row():
    params = {"temperature": 1.0, "logprobs": 1}
    inference.prepare_full_logprobs_capture({}, params, RAW, None)
    assert params == {"temperature": 1.0, "logprobs": -1, "detokenize": False}
    # Stop strings are matched on text, so detokenization stays on.
    with_stop = {"logprobs": 0, "stop": ["END"]}
    inference.prepare_full_logprobs_capture({}, with_stop, RAW, None)
    assert with_stop == {"logprobs": -1, "stop": ["END"]}


def test_engine_output_becomes_fp32_rows_skyrl_can_pack_without_changing_a_bit():
    from skyrl.backends.skyrl_train.inference_servers.generate_wire import pack_full_logprobs

    rows = np.array([[-0.0, -2.5, -3.0], [-4.0, -1.25, -5.0]], dtype=np.float32)
    # SkyRL's endpoint passes the request's CompletionOutput (final_res.outputs[0]), not the RequestOutput.
    completion = NS(token_ids=[0, 1], logprobs=[
        {index: NS(logprob=float(value)) for index, value in enumerate(row)} for row in rows
    ])
    full_logprobs = inference.full_logprobs_from_output(completion, 3)
    assert full_logprobs.dtype == np.float32 and full_logprobs.shape == (2, 3)
    assert full_logprobs.tobytes() == rows.tobytes()
    pack_full_logprobs(full_logprobs)  # SkyRL's packer accepts float32 [num_tokens, vocab_size]
