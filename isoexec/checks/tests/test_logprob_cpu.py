"""L0 (CPU) -- check 5: the logprob gate.

The negative that matters is the last one: the two env vars that used to demote a refusal are set,
and the gate still raises. There is no strict mode any more.
"""

from __future__ import annotations

import pytest

from isoexec.checks import Refusal
from isoexec.checks.logprob import (
    CONTRACTED,
    FRAMEWORK_OWNED,
    MAX_KEY,
    MEAN_KEY,
    LogprobPolicy,
    check_logprob_gate,
)


def _metrics(mean: float, maximum: float) -> dict:
    return {MEAN_KEY: mean, MAX_KEY: maximum}


def test_bitwise_agreement_passes_under_the_contracted_bound():
    receipt = check_logprob_gate(_metrics(0.0, 0.0), CONTRACTED, audited=True, step=1)
    assert receipt == {"scoring": "contracted", "bound": 0.0, "step": 1, "audited": True, "mean": 0.0, "max": 0.0}


def test_a_tiny_disagreement_refuses_under_a_zero_bound():
    with pytest.raises(Refusal, match="scoring: mean"):
        check_logprob_gate(_metrics(1e-7, 1e-7), CONTRACTED, audited=True)


def test_the_kill_switches_no_longer_demote(monkeypatch, tmp_path):
    monkeypatch.setenv("ISOEXEC_MANIFEST_STRICT", "0")
    monkeypatch.setenv("ISOEXEC_DEBUG_TRACE", str(tmp_path / "trace"))
    with pytest.raises(Refusal, match="scoring: mean"):
        check_logprob_gate(_metrics(1e-7, 1e-7), CONTRACTED, audited=True)


def test_the_framework_owned_floor_admits_the_bridge_path_and_refuses_beyond_it():
    check_logprob_gate(_metrics(9e-6, 9e-5), FRAMEWORK_OWNED, audited=True)
    with pytest.raises(Refusal, match="scoring: max"):
        check_logprob_gate(_metrics(9e-6, 2e-4), FRAMEWORK_OWNED, audited=True)


def test_an_audited_step_with_no_measurement_refuses():
    with pytest.raises(Refusal, match="carries no logprob statistic"):
        check_logprob_gate({}, CONTRACTED, audited=True)


def test_an_unaudited_step_records_that_it_was_not_measured():
    receipt = check_logprob_gate({}, CONTRACTED, audited=False, step=3)
    assert receipt["audited"] is False and "mean" not in receipt


def test_the_declared_interval_always_audits_step_one():
    policy = LogprobPolicy("contracted", 0.0, interval=4)
    assert [policy.audits(s) for s in (1, 2, 3, 4, 5, 9)] == [True, False, False, False, True, True]


def test_a_malformed_policy_is_refused_at_construction():
    with pytest.raises(ValueError, match="scoring="):
        LogprobPolicy("whatever", 0.0)
    with pytest.raises(ValueError, match="bound="):
        LogprobPolicy("contracted", -1.0)
    with pytest.raises(ValueError, match="interval="):
        LogprobPolicy("contracted", 0.0, interval=0)
