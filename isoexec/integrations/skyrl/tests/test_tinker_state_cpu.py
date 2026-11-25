"""The server must never train against stale, ambiguous or partial rollout evidence."""

import numpy as np
import pytest

from isoexec.checks import Refusal
from isoexec.integrations.skyrl.tinker_state import SampleRows, TinkerState


def sample(*, version=0, prompt=(1, 2), response=(3, 4), value=-1.0):
    return SampleRows("policy", version, prompt, response,
                      np.full((len(response), 5), value, dtype=np.float32),
                      np.full(len(response), value, dtype=np.float32))


def put(state, *records):
    size = sum(record.nbytes for record in records)
    with state.reserve(size):
        state.commit(records, size)


def test_update_requires_current_completed_comparison_and_new_sync():
    state = TinkerState()
    with pytest.raises(Refusal, match="acknowledged"):
        state.require_sampler("w0")
    state.acknowledge_sync("w0")
    state.acknowledge_sync("w0-alias")
    assert state.require_sampler("w0") == state.require_sampler("w0-alias") == 0
    put(state, sample())
    with pytest.raises(Refusal, match="forward_backward"):
        state.require_update()
    state.completed_backward_calls = 2
    state.require_update()
    state.updated()
    assert state.version == 1 and state.completed_backward_calls == 0
    assert state.find("policy", (1, 2, 3, 4), required=False) is None
    with pytest.raises(Refusal, match="acknowledged"):
        state.require_sampler("w0")
    state.acknowledge_sync("w1")
    assert state.require_sampler("w1") == 1


def test_model_prefix_and_weight_version_are_part_of_evidence_identity():
    state = TinkerState()
    state.acknowledge_sync("w0")
    put(state, sample())
    assert state.find("policy", (1, 2, 3, 4), required=True) is not None
    for model, tokens in (("other", (1, 2, 3, 4)), ("policy", (9, 2, 3, 4)), ("policy", (1, 2, 3))):
        with pytest.raises(Refusal, match="no matching"):
            state.find(model, tokens, required=True)
    with pytest.raises(Refusal, match="Weights changed"):
        put(state, sample(version=1))


def test_failed_accumulation_cannot_reuse_an_earlier_success():
    state = TinkerState()
    state.acknowledge_sync("w0")
    put(state, sample())
    state.completed_backward_calls = 1
    state.fail()
    with pytest.raises(Refusal, match="unload and recreate"):
        state.require_update()
    with pytest.raises(Refusal, match="unload and recreate"):
        state.require_sampler("w0")
    assert not state._rows and state.completed_backward_calls == 0


def test_concurrent_reservations_are_bounded_and_released_on_failure():
    record = sample()
    state = TinkerState(byte_limit=record.nbytes)
    state.acknowledge_sync("w0")
    with state.reserve(record.nbytes):
        with pytest.raises(Refusal, match="diagnostic bound"):
            with state.reserve(1):
                pass
        with pytest.raises(Refusal, match="in flight"):
            state.acknowledge_sync("w0")
    assert state._reserved == 0
    with pytest.raises(RuntimeError):
        with state.reserve(record.nbytes):
            raise RuntimeError("generation failed")
    put(state, record)
    assert state._bytes == record.nbytes and state._reserved == 0


def test_a_sample_request_is_committed_atomically():
    state = TinkerState()
    state.acknowledge_sync("w0")
    old = sample()
    put(state, old)
    new = sample(prompt=(7, 8))
    with pytest.raises(Refusal, match="ambiguous"):
        put(state, new, sample(value=-2.0))
    assert state.find("policy", (7, 8, 3, 4), required=False) is None
    assert state.find("policy", (1, 2, 3, 4), required=True) is old
    assert state._bytes == old.nbytes and state._reserved == 0


def test_identical_responses_deduplicate_but_prompt_boundary_cannot_change():
    state = TinkerState()
    state.acknowledge_sync("w0")
    first = sample()
    put(state, first, sample())
    assert state._bytes == first.nbytes
    with pytest.raises(Refusal, match="ambiguous"):
        put(state, sample(prompt=(1,), response=(2, 3, 4)))
    assert not first.rows.flags.writeable


def test_store_cannot_bypass_reservation_with_a_forged_budget():
    state = TinkerState(byte_limit=1)
    state.acknowledge_sync("w0")
    with pytest.raises(Refusal, match="active byte reservation"):
        state.commit([sample()], 1_000_000)


def test_empty_responses_do_not_accumulate_unbounded_metadata():
    state = TinkerState()
    state.acknowledge_sync("w0")
    put(state, sample(response=()))
    assert not state._rows and state._bytes == 0
