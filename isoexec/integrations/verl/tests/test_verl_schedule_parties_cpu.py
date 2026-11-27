"""The verl weight sync compares the attention schedule of the parties that compute: every Megatron training
worker's record against every vLLM engine worker's. The task runner (the process that runs the comparison)
computes no attention and its environment is not a party. Needs no verl: fake workers write the records."""

from __future__ import annotations

import json
import os

import pytest

from isoexec.checks import Refusal
from isoexec.integrations.verl import schedule
from isoexec.ops.attention import target_batch


def _other(record: dict) -> dict:
    """The record of a process launched with another target batch (other spans)."""
    return {**record, "mla_sm100": {**record["mla_sm100"], "target_batch": 64, "seg_tokens": 5120}}


def _trainer(root, rank: int, record: dict | None, pid: int = 1000) -> None:
    """A fake training worker of ``rank`` that bound ``record`` (``None``: an entry without one)."""
    directory = schedule.trainer_schedule_directory(root)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"trainer-rank{rank:04d}-pid{pid + rank}.json").write_text(
        json.dumps({"rank": rank, "pid": pid + rank, "attention_schedule": record}))


def _acks(records: list[dict | None]) -> list[dict]:
    return [{"replica_rank": 0, "dp_rank": 0, "rank": rank, "attention_schedule": record}
            for rank, record in enumerate(records)]


def test_a_training_worker_records_the_schedule_it_bound(tmp_path):
    record = schedule.record_trainer_schedule(tmp_path, 3)
    assert record == {"rank": 3, "pid": os.getpid(), "attention_schedule": target_batch.schedule_record()}
    (path,) = schedule.trainer_schedule_directory(tmp_path).glob("*.json")
    assert path.name == f"trainer-rank0003-pid{os.getpid()}.json" and json.loads(path.read_text()) == record
    # the reference engine of the same process records the same bytes into the same file
    schedule.record_trainer_schedule(tmp_path, 3)
    assert len(list(schedule.trainer_schedule_directory(tmp_path).iterdir())) == 1


def test_training_workers_and_engine_workers_that_bound_one_schedule_pass(tmp_path):
    bound = target_batch.schedule_record()
    for rank in range(2):
        _trainer(tmp_path, rank, bound)
    assert schedule.require_same_schedules(tmp_path, _acks([bound, bound]), 2) == bound


def test_the_comparing_process_is_not_a_party(tmp_path, monkeypatch):
    """Both failures of comparing the task runner's own record: workers and engines that agree with each other
    start whatever this process bound, and a training worker that differs is refused although this process and
    every engine agree."""
    bound = target_batch.schedule_record()
    # (i) every computing party bound another declaration than this process: a valid launch
    for rank in range(2):
        _trainer(tmp_path / "valid", rank, _other(bound))
    monkeypatch.setattr(target_batch, "schedule_record", lambda: pytest.fail("the comparing process was read"))
    assert schedule.require_same_schedules(tmp_path / "valid", _acks([_other(bound)] * 2), 2) == _other(bound)
    # (ii) this process and both engines agree; training worker 1 bound another span
    _trainer(tmp_path / "split", 0, bound)
    _trainer(tmp_path / "split", 1, _other(bound))
    with pytest.raises(Refusal, match=r"attention_schedule\.mla_sm100\.seg_tokens: training worker rank 0 \d+, "
                                      r"training worker rank 1 \(pid 1001\) 5120"):
        schedule.require_same_schedules(tmp_path / "split", _acks([bound, bound]), 2)


def test_an_engine_worker_that_differs_from_the_training_workers_is_refused(tmp_path):
    bound = target_batch.schedule_record()
    for rank in range(2):
        _trainer(tmp_path, rank, bound)
    with pytest.raises(Refusal, match=r"attention_schedule\.mla_sm100\.seg_tokens: training worker rank 0 .*"
                                      r"engine worker replica 0 dp 0 rank 1 5120"):
        schedule.require_same_schedules(tmp_path, _acks([bound, _other(bound)]), 2)
    with pytest.raises(Refusal, match=r"attention_schedule\.missing: engine worker replica 0 dp 0 rank 1"):
        schedule.require_same_schedules(tmp_path, _acks([bound, None]), 2)


def test_a_training_worker_that_recorded_nothing_is_refused(tmp_path):
    bound = target_batch.schedule_record()
    with pytest.raises(Refusal, match=r"attention_schedule\.missing: 2 training workers .* ranks \[\] recorded"):
        schedule.require_same_schedules(tmp_path, _acks([bound, bound]), 2)
    _trainer(tmp_path, 0, bound)
    with pytest.raises(Refusal, match=r"attention_schedule\.missing: 2 training workers .* ranks \[0\] recorded"):
        schedule.require_same_schedules(tmp_path, _acks([bound, bound]), 2)
    _trainer(tmp_path, 1, None)
    with pytest.raises(Refusal, match=r"attention_schedule\.missing: training worker rank 1"):
        schedule.require_same_schedules(tmp_path, _acks([bound, bound]), 2)


def test_a_restarted_training_worker_cannot_hide_what_the_first_process_bound(tmp_path):
    bound = target_batch.schedule_record()
    _trainer(tmp_path, 0, bound)
    _trainer(tmp_path, 1, bound)
    _trainer(tmp_path, 1, _other(bound), pid=2000)  # the same rank, another process, another declaration
    with pytest.raises(Refusal, match=r"training worker rank 1 \(pid 2001\) 5120"):
        schedule.require_same_schedules(tmp_path, _acks([bound, bound]), 2)


# --- the flag reaches every Ray worker the way the other IsoExec flags do (config.apply)

_KEY = f"ray_kwargs.ray_init.runtime_env.env_vars.{target_batch.FLAG}"


def _forward(monkeypatch, driver: str | None, job: str | None):
    from omegaconf import OmegaConf

    from isoexec.integrations.verl.config import _forward_target_batch

    if driver is None:
        monkeypatch.delenv(target_batch.FLAG, raising=False)
    else:
        monkeypatch.setenv(target_batch.FLAG, driver)
    config = OmegaConf.create({"ray_kwargs": {"ray_init": {"runtime_env": {}}}})
    if job is not None:
        OmegaConf.update(config, _KEY, job, force_add=True)
    changes = _forward_target_batch(config)
    return OmegaConf.select(config, _KEY, default=None), changes


@pytest.mark.parametrize("driver, job, want, changed", [
    ("8", None, "8", True),      # the driver's declaration rides the job's runtime environment
    ("", None, "", True),        # an explicit "no target batch" overrides whatever a node's Ray daemon inherited
    (None, None, None, False),   # nothing declared: nothing forwarded
    (None, "8", "8", False),     # declared in the job's environment only: every Ray worker already gets it
    ("8", "8", "8", False),
    ("8", " 8 ", " 8 ", False),  # the same declaration
], ids=["driver", "driver-none", "undeclared", "job-only", "both", "both-spelled-differently"])
def test_the_drivers_target_batch_is_forwarded_to_every_ray_worker(monkeypatch, driver, job, want, changed):
    value, changes = _forward(monkeypatch, driver, job)
    assert value == want and bool(changes) is changed
    assert all(target_batch.FLAG in change for change in changes)


@pytest.mark.parametrize("driver, job", [("8", "16"), ("", "8"), ("8", "")], ids=["8-16", "none-8", "8-none"])
def test_a_job_environment_that_contradicts_the_driver_is_refused(monkeypatch, driver, job):
    with pytest.raises(ValueError, match=f"{target_batch.FLAG}.*must bind one target batch"):
        _forward(monkeypatch, driver, job)


@pytest.mark.parametrize("driver, job", [("zero", None), ("0", None), (None, "-1")], ids=["word", "zero", "job"])
def test_a_malformed_target_batch_is_refused_in_the_driver(monkeypatch, driver, job):
    with pytest.raises(ValueError, match=f"{target_batch.FLAG} must be a positive integer"):
        _forward(monkeypatch, driver, job)
