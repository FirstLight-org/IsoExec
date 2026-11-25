"""The unchanged SkyRL worker seam reaches IsoExec's two-phase online gate."""

import sys
from types import SimpleNamespace as NS

import pytest
import torch

from isoexec.checks import Refusal
from isoexec.checks.prescore import PrescoreReport
from isoexec.integrations.skyrl import online_prescore as online


class Batch(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = {}

    def __len__(self):
        return self["sequences"].shape[0]

    def __getitem__(self, key):
        if isinstance(key, slice):
            batch = Batch({name: value[key] for name, value in self.items()})
            batch.metadata = self.metadata
            return batch
        return super().__getitem__(key)


class RemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class LocalActor:
    def __init__(self, collector=None):
        self.collector = collector or online.PrescoreCollector()

    def __getattr__(self, name):
        return RemoteMethod(getattr(self.collector, name))


@pytest.fixture(autouse=True)
def fake_ray(monkeypatch):
    monkeypatch.setitem(sys.modules, "ray", NS(get=lambda value, timeout=None: value, is_initialized=lambda: True))


def data():
    batch = Batch(
        sequences=torch.tensor([[0, 10, 11, 3], [20, 21, 4, 5]]),
        attention_mask=torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]),
        loss_mask=torch.tensor([[0, 1], [1, 1]]),
        rollout_logprobs=torch.zeros((2, 2)),
        rollout_full_logprobs=torch.zeros((2, 2, 8)),
        sample_indices=torch.tensor([6, 7]),
    )
    batch.metadata = {}
    return batch


def stage():
    axis = lambda size=1, rank=0: NS(size=lambda: size, rank=lambda: rank)
    return NS(pg_collection=NS(dp=axis(), pp=axis(), cp=axis(), tp=axis()))


def test_plan_uses_exact_rows_boundaries_and_global_step():
    batch = data()
    boundaries = [(0, 1), (1, 2)]
    expected = tuple(online._row_identities(batch[start:stop]) for start, stop in boundaries)
    plan = online._plan(batch, boundaries, 1)
    assert plan.step == 1 and plan.vocab == 8
    assert plan.calls == expected
    assert [[row.sample for row in call] for call in plan.calls] == [[6], [7, 7]]
    assert [row.response for row in plan.calls[1]] == [0, 1]


def test_plan_constructs_each_identity_once_and_reuses_it_across_boundaries(monkeypatch):
    batch = data()
    original = online.RowIdentity
    constructed = []

    def counted(*args):
        identity = original(*args)
        constructed.append(identity)
        return identity

    monkeypatch.setattr(online, "RowIdentity", counted)
    plan = online._plan(batch, [(0, 1), (1, 2)], 1)

    assert len(constructed) == 3
    assert plan.rows == tuple(constructed)
    assert all(actual is expected for actual, expected in zip(plan.rows, constructed, strict=True))


@pytest.mark.parametrize(
    "boundaries",
    [[], [(1, 2)], [(0, 0), (0, 2)], [(0, 1)], [(0, 2), (1, 2)], [(0, 3)], [(0, 1.0), (1, 2)]],
)
def test_plan_refuses_missing_overlapping_or_out_of_range_boundaries(boundaries):
    with pytest.raises((Refusal, IndexError)):
        online._plan(data(), boundaries, 1)


def test_collector_preflight_and_finalization_are_one_shot():
    plan = online._plan(data(), [(0, 2)], 1)
    collector = online.PrescoreCollector()
    collector.begin(plan)
    rows = plan.calls[0]
    preflight = PrescoreReport(plan.id, 1, 0, (0, 0, 0, 0), (1, 1, 1, 1), rows, (), False)
    collector.preflight(preflight)
    assert collector.preflight_status(plan.id, 0)["ready"]
    collector.report(PrescoreReport(
        **{**preflight.__dict__, "shards": tuple(
            online.ComparedShard(row, 0, 8, 8) for row in rows
        ), "completed": True}
    ))
    assert collector.finalize(plan.id)["rows"] == len(rows)
    with pytest.raises(Refusal, match="closed"):
        collector.finalize(plan.id)


def test_collector_refuses_completion_without_matching_preflight():
    plan = online._plan(data(), [(0, 2)], 1)
    collector = online.PrescoreCollector()
    collector.begin(plan)
    rows = plan.calls[0]
    completion = PrescoreReport(
        plan.id,
        1,
        0,
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        rows,
        tuple(online.ComparedShard(row, 0, 8, 8) for row in rows),
        True,
    )
    with pytest.raises(Refusal, match="without coordinated preflight"):
        collector.report(completion)
    preflight = PrescoreReport(**{**completion.__dict__, "shards": (), "completed": False})
    collector.preflight(preflight)
    changed = PrescoreReport(**{**completion.__dict__, "input_rows": tuple(reversed(rows))})
    with pytest.raises(Refusal, match="changed identity"):
        collector.report(changed)


def test_driver_begin_and_finish_uses_online_receipt_not_loss_mask_count(monkeypatch):
    batch, actor = data(), LocalActor()
    monkeypatch.setattr(online, "_collector", lambda: actor)
    token = online.begin_online_prescore(batch, [(0, 2)], 1)
    assert batch.metadata[online.TOKEN] is token
    plan = token["plan"]
    rows = plan.calls[0]
    base = PrescoreReport(plan.id, 1, 0, (0, 0, 0, 0), (1, 1, 1, 1), rows, (), False)
    actor.preflight.remote(base)
    actor.report.remote(PrescoreReport(**{
        **base.__dict__, "shards": tuple(online.ComparedShard(row, 0, 8, 8) for row in rows), "completed": True,
    }))
    receipt = online.finish_online_prescore(batch)
    assert receipt["rows"] == 3
    assert online.TOKEN not in batch.metadata


def _worker(monkeypatch, *, emit=True, fail=False):
    local_stage = stage()
    monkeypatch.setattr("isoexec.integrations.skyrl.scoring._find_isoexec_stage", lambda _: local_stage)
    calls = []

    def original(batch):
        calls.append("native")
        if fail:
            raise RuntimeError("native failure")
        if emit:
            mask = batch["loss_mask"] > 0
            coords = mask.nonzero(as_tuple=False)
            response = coords[:, 1]
            online.record_compared_rows(
                sequences=batch["sequences"], attention_mask=batch["attention_mask"],
                row_indices=coords[:, 0], response_columns=response,
                target_columns=batch["sequences"].shape[1] - mask.shape[1] + response,
                sample_indices=batch["sample_indices"], vocab_start=0, vocab_end=8,
            )
        return torch.ones(1)

    worker = NS(actor_module=object(), _forward_logprobs=original)
    online.install_worker(worker)
    return worker, calls


def test_real_worker_wrapper_preflights_runs_native_and_reports(monkeypatch):
    batch, collector = data(), LocalActor()
    plan = online._plan(batch, [(0, 2)], 1)
    collector.begin.remote(plan)
    batch.metadata["isoexec_online_prescore"] = {"collector": collector, "plan": plan}
    worker, calls = _worker(monkeypatch)
    assert worker._forward_logprobs(batch).item() == 1
    assert calls == ["native"]
    receipt = collector.finalize.remote(plan.id)
    assert receipt["rows"] == 3 and receipt["compared_elements"] == 24


def test_missing_comparator_engagement_fails_online_finalization(monkeypatch):
    batch, collector = data(), LocalActor()
    plan = online._plan(batch, [(0, 2)], 1)
    collector.begin.remote(plan)
    batch.metadata["isoexec_online_prescore"] = {"collector": collector, "plan": plan}
    worker, _ = _worker(monkeypatch, emit=False)
    worker._forward_logprobs(batch)
    with pytest.raises(Refusal, match="complete expected batch"):
        collector.finalize.remote(plan.id)


def test_native_failure_is_reported_and_never_becomes_coverage_pass(monkeypatch):
    batch, collector = data(), LocalActor()
    plan = online._plan(batch, [(0, 2)], 1)
    collector.begin.remote(plan)
    batch.metadata["isoexec_online_prescore"] = {"collector": collector, "plan": plan}
    worker, _ = _worker(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="native failure"):
        worker._forward_logprobs(batch)
    with pytest.raises(Refusal, match="did not complete"):
        collector.finalize.remote(plan.id)


def test_install_is_idempotent(monkeypatch):
    worker, _ = _worker(monkeypatch)
    wrapped = worker._forward_logprobs
    online.install_worker(worker)
    assert worker._forward_logprobs is wrapped
