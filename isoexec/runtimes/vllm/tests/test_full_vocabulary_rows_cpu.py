"""Complete-vocabulary rows stay float32 arrays in vLLM's output processor."""

from __future__ import annotations

import base64
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from isoexec.integrations.full_distribution import FullVocabularyRow, pack_full_logprobs
from isoexec.runtimes.vllm import vllm_patches as patches


@pytest.fixture(autouse=True)
def reset_census(monkeypatch):
    monkeypatch.setattr(patches, "_FULL_ROW_STATE", {"calls": 0, "rows": 0, "fallback": 0, "reported": 0})


@pytest.fixture
def processor(monkeypatch):
    """vLLM's LogprobsProcessor surface the hook reads, hooked by the production patch."""
    calls = []

    class LogprobsProcessor:
        def __init__(self, num_logprobs=-1, tokenizer=None, logprobs=None):
            self.num_logprobs = num_logprobs
            self.tokenizer = tokenizer
            self.logprobs = [] if logprobs is None else logprobs
            self.cumulative_logprob = 0.0

        def _update_sample_logprobs(self, logprobs_lists):
            calls.append(logprobs_lists)
            self.logprobs.append("stock")

    module = ModuleType("vllm.v1.engine.logprobs")
    module.LogprobsProcessor = LogprobsProcessor
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.logprobs", module)
    assert patches.patch_vllm_full_vocabulary_logprobs()
    hooked = LogprobsProcessor._update_sample_logprobs
    assert patches.patch_vllm_full_vocabulary_logprobs() and LogprobsProcessor._update_sample_logprobs is hooked
    return LogprobsProcessor, calls


def _lists(values, sampled):
    """One request's ``LogprobsLists``: sampled-first rank-ordered rows, as the engine core sends them."""
    order = np.argsort(-values, axis=1, kind="stable")
    token_ids = np.concatenate((np.asarray(sampled)[:, None], order), axis=1).astype(np.int32)
    first = values[np.arange(len(values)), sampled][:, None]
    logprobs = np.concatenate((first, np.take_along_axis(values, order, axis=1)), axis=1).astype(np.float32)
    ranks = np.array([1 + int((row > row[tid]).sum()) for row, tid in zip(values, sampled)])
    return token_ids, logprobs, ranks, None


def test_complete_rows_stay_arrays_and_pack_to_the_values_the_engine_sent(processor):
    cls, calls = processor
    values = np.random.default_rng(3).standard_normal((2, 9)).astype(np.float32)
    sampled = [5, 0]
    request = cls()

    assert request._update_sample_logprobs(_lists(values, sampled)) is None

    assert calls == [] and all(type(row) is FullVocabularyRow for row in request.logprobs)
    assert [row.sampled_token_id for row in request.logprobs] == sampled
    assert request.cumulative_logprob == float(values[0, 5]) + float(values[1, 0])
    assert request.logprobs[0][5].rank == 1 + int((values[0] > values[0, 5]).sum())
    packed = pack_full_logprobs(sampled, request.logprobs, vocab_size=9)
    np.testing.assert_array_equal(
        np.frombuffer(base64.b64decode(packed["data"]), "<f4").reshape(2, 9).view(np.uint32),
        values.view(np.uint32),
    )
    assert patches._FULL_ROW_STATE == {"calls": 1, "rows": 2, "fallback": 0, "reported": 2}


@pytest.mark.parametrize(
    "request_kwargs",
    [{"num_logprobs": 5}, {"tokenizer": object()}, {"logprobs": SimpleNamespace(append=lambda row: None)}],
    ids=["top-k", "detokenizing", "flat-container"],
)
def test_every_other_request_keeps_vllms_own_container(processor, request_kwargs):
    cls, calls = processor
    lists = _lists(np.zeros((1, 4), np.float32), [0])

    cls(**request_kwargs)._update_sample_logprobs(lists)

    assert calls == [lists]
    assert patches._FULL_ROW_STATE["calls"] == 0


def test_an_incomplete_row_falls_back_whole_without_a_partial_append(processor):
    cls, calls = processor
    values = np.random.default_rng(5).standard_normal((2, 6)).astype(np.float32)
    token_ids, logprobs, ranks, _ = _lists(values, [1, 2])
    token_ids[1, 3] = token_ids[1, 4]  # the second position names one token twice
    request = cls()

    request._update_sample_logprobs((token_ids, logprobs, ranks, None))

    assert len(calls) == 1 and request.logprobs == ["stock"] and request.cumulative_logprob == 0.0
    assert patches._FULL_ROW_STATE == {"calls": 1, "rows": 0, "fallback": 1, "reported": 1}


def test_a_shorter_ranks_list_bounds_the_positions_as_vllm_does(processor):
    cls, calls = processor
    values = np.array([[-3.0, -1.0, -2.0], [-2.0, -3.0, -1.0]], dtype=np.float32)
    token_ids, logprobs, ranks, _ = _lists(values, [2, 1])
    request = cls()

    request._update_sample_logprobs((token_ids, logprobs, ranks[:1], None))

    assert calls == [] and len(request.logprobs) == 1
    row = request.logprobs[0]
    assert row.sampled_token_id == 2 and [row[tid].logprob for tid in range(3)] == values[0].tolist()
    assert request.cumulative_logprob == float(values[0, 2])
    assert patches._FULL_ROW_STATE == {"calls": 1, "rows": 1, "fallback": 0, "reported": 1}
