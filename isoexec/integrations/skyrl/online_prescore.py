"""IsoExec-owned online coverage gate over SkyRL's existing full-mode hooks."""

from __future__ import annotations

import contextvars
import json
import time
import uuid
from dataclasses import replace
from functools import wraps
from typing import cast

import torch

from isoexec.checks import Refusal
from isoexec.checks.prescore import (
    ComparedShard,
    PrescorePlan,
    PrescoreReport,
    RowIdentity,
    verify_preflight,
    verify_prescore,
)

_ACTIVE = contextvars.ContextVar("isoexec_online_prescore", default=None)
#: Metadata key of the batch SkyRL sends to the policy workers: the collector handle and the plan.
TOKEN = "isoexec_online_prescore"
_DRIVER_COLLECTOR = None
_REMOTE_COLLECTOR = None
PREFLIGHT_TIMEOUT_SECONDS = 120.0


def _row_identities_by_batch_row(batch) -> tuple[tuple[RowIdentity, ...], ...]:
    """Exact selected identities grouped by their position in the incoming batch."""
    sequences = batch["sequences"].detach().cpu()
    attention = batch["attention_mask"].detach().cpu().bool()
    mask = batch["loss_mask"].detach().cpu() > 0
    sample_indices = batch.get("sample_indices")
    if sample_indices is None:
        raise Refusal("prescore.coverage: global sample indices are required")
    samples = sample_indices.detach().cpu().tolist()
    if len(samples) != len(set(samples)):
        # Padding may copy a sample index, but its all-zero loss row is ignored.
        active = [samples[row] for row in mask.any(dim=1).nonzero(as_tuple=False).flatten().tolist()]
        if len(active) != len(set(active)):
            raise Refusal("prescore.coverage: active global sample indices are duplicated")
    response_length = mask.shape[1]
    out: list[list[RowIdentity]] = [[] for _ in samples]
    for row, response in mask.nonzero(as_tuple=False).tolist():
        target = sequences.shape[1] - response_length + response
        real = attention[row].nonzero(as_tuple=False).flatten()
        first = int(real[0]) if real.numel() else 0
        if target <= first or not bool(attention[row, target]):
            raise Refusal("prescore.coverage: selected response token has no real predictor prefix")
        out[row].append(RowIdentity(
            int(samples[row]), int(response), int(sequences[row, target]),
            tuple(int(token) for token in sequences[row, first:target].tolist()),
        ))
    return tuple(tuple(rows) for rows in out)


def _row_identities(batch) -> tuple[RowIdentity, ...]:
    """Exact selected identities from driver or DP-local worker tensors."""
    return tuple(identity for rows in _row_identities_by_batch_row(batch) for identity in rows)


def _plan(data_fwd_pass, boundaries, step) -> PrescorePlan:
    """What the policy forward over ``data_fwd_pass`` has to compare, one call per mini-batch."""
    if type(step) is not int or step < 1:
        raise Refusal("prescore.coverage: global_step must be a positive integer")
    full_rows = data_fwd_pass["rollout_full_logprobs"]
    if full_rows.ndim != 3 or full_rows.shape[2] < 1:
        raise Refusal("prescore.coverage: expected [batch,response,vocab] rollout rows")
    if boundaries is None:
        boundaries = [(0, len(data_fwd_pass))]
    if (
        not boundaries
        or any(len(pair) != 2 for pair in boundaries)
        or any(
            type(start) is not int
            or type(stop) is not int
            or not 0 <= start < stop <= len(data_fwd_pass)
            for start, stop in boundaries
        )
        or [pair[0] for pair in boundaries] != [0, *[pair[1] for pair in boundaries[:-1]]]
        or boundaries[-1][1] != len(data_fwd_pass)
    ):
        raise Refusal("prescore.coverage: policy mini-batch boundaries must tile the batch in order")
    rows_by_batch = _row_identities_by_batch_row(data_fwd_pass)
    calls = tuple(
        tuple(identity for rows in rows_by_batch[start:stop] for identity in rows)
        for start, stop in boundaries
    )
    return PrescorePlan(uuid.uuid4().hex, step, int(full_rows.shape[2]), calls)


class PrescoreCollector:
    """Small serial state machine; Ray is only its production transport."""

    def __init__(self):
        self.states = {}

    def begin(self, plan):
        if not isinstance(plan, PrescorePlan) or plan.id in self.states or len(self.states) >= 8:
            raise Refusal("prescore.coverage: duplicate/invalid plan or too many in-flight plans")
        self.states[plan.id] = {"plan": plan, "preflight": {}, "reports": {}}
        return plan.id

    def preflight(self, report):
        if not isinstance(report, PrescoreReport):
            raise Refusal("prescore.coverage: invalid preflight report type")
        state = self.states.get(report.plan_id)
        if state is None:
            raise Refusal("prescore.coverage: preflight references an unknown/closed plan")
        key = (report.call, report.coordinate)
        if key in state["preflight"]:
            raise Refusal(f"prescore.coverage: duplicate preflight participant {key}")
        state["preflight"][key] = report
        return True

    def preflight_status(self, plan_id, call):
        state = self.states.get(plan_id)
        if state is None:
            return {"error": "unknown/closed plan"}
        reports = [report for (index, _), report in state["preflight"].items() if index == call]
        if not reports:
            return {"ready": False, "received": 0}
        shape = reports[0].shape
        if len(shape) != 4 or any(type(n) is not int or n < 1 for n in shape):
            return {"error": f"invalid resolved shape {shape}"}
        expected = shape[0] * shape[1] * shape[2] * shape[3]
        if len(reports) < expected:
            return {"ready": False, "received": len(reports), "expected": expected}
        try:
            return {"ready": True, **verify_preflight(state["plan"], reports, call)}
        except Exception as error:  # noqa: BLE001 -- the same refusal must reach every polling rank
            return {"error": str(error)}

    def report(self, report):
        if not isinstance(report, PrescoreReport):
            raise Refusal("prescore.coverage: invalid completion report type")
        state = self.states.get(report.plan_id)
        if state is None:
            raise Refusal("prescore.coverage: report references an unknown/closed plan")
        key = (report.call, report.coordinate)
        preflight = state["preflight"].get(key)
        if preflight is None:
            raise Refusal(f"prescore.coverage: participant {key} completed without coordinated preflight")
        if replace(report, shards=(), completed=False, errors=()) != preflight:
            raise Refusal(f"prescore.coverage: participant {key} changed identity after preflight")
        if key in state["reports"]:
            raise Refusal(f"prescore.coverage: duplicate completed participant {key}")
        state["reports"][key] = report
        return True

    def finalize(self, plan_id):
        state = self.states.pop(plan_id, None)
        if state is None:
            raise Refusal("prescore.coverage: cannot finalize an unknown/closed plan")
        return verify_prescore(state["plan"], list(state["reports"].values()))


def _collector():
    global _DRIVER_COLLECTOR, _REMOTE_COLLECTOR
    import ray

    if not ray.is_initialized():
        raise Refusal("prescore.coverage: Ray must be initialized before full-mode prescore")
    if _DRIVER_COLLECTOR is None:
        if _REMOTE_COLLECTOR is None:
            _REMOTE_COLLECTOR = ray.remote(num_cpus=0, max_restarts=0, max_task_retries=0)(PrescoreCollector)
        _DRIVER_COLLECTOR = _REMOTE_COLLECTOR.remote()
    return _DRIVER_COLLECTOR


def close_online_prescore() -> None:
    """Release this host's collector before a Tinker model tears down Ray."""
    global _DRIVER_COLLECTOR, _REMOTE_COLLECTOR
    import ray
    from ray.actor import ActorHandle

    collector = _DRIVER_COLLECTOR
    _DRIVER_COLLECTOR = None
    _REMOTE_COLLECTOR = None
    if collector is not None and ray.is_initialized():
        ray.kill(cast(ActorHandle, collector), no_restart=True)


def begin_online_prescore(data_fwd_pass, mini_batch_boundaries, global_step):
    """Register the plan with the collector and put it in the metadata the workers receive with the batch."""
    import ray

    if not ray.is_initialized():
        # Production trainers use Ray; standalone checks may lack its actor lifecycle.
        token = {"disabled": "no Ray runtime (CPU contract test only)"}
        data_fwd_pass.metadata[TOKEN] = token
        return token
    plan, collector = _plan(data_fwd_pass, mini_batch_boundaries, global_step), _collector()
    ray.get(collector.begin.remote(plan), timeout=PREFLIGHT_TIMEOUT_SECONDS)
    token = {"collector": collector, "plan": plan}
    data_fwd_pass.metadata[TOKEN] = token
    return token


def finish_online_prescore(data_fwd_pass):
    """The collector's verdict on the plan ``begin_online_prescore`` registered; raises unless it is covered."""
    import ray

    token = data_fwd_pass.metadata.get(TOKEN)
    if isinstance(token, dict) and token.get("disabled"):
        data_fwd_pass.metadata.pop(TOKEN, None)
        return {"rows": int(data_fwd_pass["loss_mask"].count_nonzero()), "test_only_disabled": token["disabled"]}
    if not isinstance(token, dict) or not isinstance(token.get("plan"), PrescorePlan):
        raise Refusal("prescore.coverage: driver lost the live plan before optimizer admission")
    try:
        receipt = ray.get(token["collector"].finalize.remote(token["plan"].id), timeout=PREFLIGHT_TIMEOUT_SECONDS)
    except Exception as error:
        raise Refusal(f"prescore.coverage: online finalization failed: {error}") from error
    data_fwd_pass.metadata.pop(TOKEN, None)
    print("[ISOEXEC-PRESCORE] PASS " + json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def _coordinate(stage):
    groups = stage.pg_collection
    axes = [(group.size(), group.rank()) for group in (groups.dp, groups.pp, groups.cp, groups.tp)]
    shape = tuple(int(size) for size, _ in axes)
    coordinate = tuple(int(rank) for _, rank in axes)
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != shape[0] * shape[1] * shape[2] * shape[3]:
        raise Refusal(f"prescore.coverage: live world does not equal DP*PP*CP*TP={shape}")
    return coordinate, shape


def _wait_preflight(collector, report):
    import ray

    ray.get(collector.preflight.remote(report), timeout=PREFLIGHT_TIMEOUT_SECONDS)
    deadline = time.monotonic() + PREFLIGHT_TIMEOUT_SECONDS
    while True:
        status = ray.get(collector.preflight_status.remote(report.plan_id, report.call), timeout=PREFLIGHT_TIMEOUT_SECONDS)
        if status.get("error"):
            raise Refusal(f"prescore.coverage: coordinated preflight failed: {status['error']}")
        if status.get("ready"):
            return
        if time.monotonic() >= deadline:
            raise Refusal(f"prescore.coverage: preflight timed out: {status}")
        time.sleep(0.05)


class _ActiveReport:
    def __init__(self, base):
        self.base, self.shards = base, []


def record_compared_rows(*, sequences, attention_mask, row_indices, response_columns,
                         target_columns, sample_indices, vocab_start, vocab_end):
    """Called only after every raw bit in these actual slices compared equal."""
    active = _ACTIVE.get()
    if active is None:
        # Standalone scorers may run without SkyRL's batch lifecycle.
        return
    batch = {
        "sequences": sequences,
        "attention_mask": attention_mask,
        "loss_mask": torch.zeros((sequences.shape[0], int(sequences.shape[1])), dtype=torch.bool),
        "sample_indices": sample_indices,
    }
    # Reuse the exact identity definition, but selected response columns use a
    # response-width coordinate rather than absolute sequence columns.
    response_width = int(active.response_width)
    expected_targets = sequences.shape[1] - response_width + response_columns
    if not torch.equal(target_columns.detach().cpu(), expected_targets.detach().cpu()):
        raise Refusal("prescore.coverage: comparator target columns disagree with response identities")
    batch["loss_mask"] = torch.zeros((sequences.shape[0], response_width), dtype=torch.bool)
    batch["loss_mask"][row_indices.detach().cpu(), response_columns.detach().cpu()] = True
    identities = _row_identities(batch)
    if len(identities) != len(row_indices):
        raise Refusal("prescore.coverage: comparator identities do not match selected rows")
    for identity in identities:
        active.shards.append(ComparedShard(identity, int(vocab_start), int(vocab_end), int(vocab_end - vocab_start)))


def install_worker(worker):
    """Wrap the existing internal prescore seam; Ray actor APIs stay unchanged."""
    if getattr(worker, "_isoexec_online_prescore_installed", False):
        return
    from .scoring import _find_isoexec_stage

    original = worker._forward_logprobs
    stage = _find_isoexec_stage(worker.actor_module)
    if stage is None:
        raise Refusal("prescore.coverage: cannot install without an IsoExec trainer stage")
    calls = {}

    @wraps(original)
    def wrapped(data):
        token = (data.metadata or {}).get(TOKEN)
        if token is None:
            return original(data)
        plan, collector = token.get("plan"), token.get("collector")
        if not isinstance(plan, PrescorePlan) or collector is None:
            raise Refusal("prescore.coverage: worker received malformed plan metadata")
        call = calls.get(plan.id, 0)
        if call >= len(plan.calls):
            raise Refusal("prescore.coverage: worker received an extra native forward call")
        coordinate, shape = _coordinate(stage)
        base = PrescoreReport(plan.id, plan.step, call, coordinate, shape, _row_identities(data), (), False)
        _wait_preflight(collector, base)
        active = _ActiveReport(base)
        active.response_width = int(data["loss_mask"].shape[1])
        context = _ACTIVE.set(active)
        try:
            result = original(data)
        except Exception as error:
            failed = replace(base, shards=tuple(active.shards), errors=(f"{type(error).__name__}: {error}",))
            try:
                import ray
                ray.get(collector.report.remote(failed), timeout=PREFLIGHT_TIMEOUT_SECONDS)
            finally:
                _ACTIVE.reset(context)
            raise
        _ACTIVE.reset(context)
        complete = replace(base, shards=tuple(active.shards), completed=True)
        import ray
        ray.get(collector.report.remote(complete), timeout=PREFLIGHT_TIMEOUT_SECONDS)
        calls[plan.id] = call + 1
        return result

    worker._forward_logprobs = wrapped
    worker._isoexec_online_prescore_installed = True
