"""Exercise PrescoreCollector serialization and barriers through real Ray actors. Use a private
CPU-only local cluster and temporary directory to avoid joining an ambient training cluster.
"""

from __future__ import annotations

import dataclasses
import shutil
import tempfile
import time

import pytest

from isoexec.checks.prescore import ComparedShard, PrescorePlan, PrescoreReport, RowIdentity
from isoexec.integrations.skyrl.online_prescore import PrescoreCollector

ray = pytest.importorskip("ray", reason="UNWITNESSED: the multi-process Ray transport for PrescoreCollector")

TP = 2
VOCAB = 8
# Each participant compares a contiguous quarter-open slice of the row; together they tile ``VOCAB``.
SHARD = VOCAB // TP


@ray.remote(num_cpus=0)
def participant(collector, plan, tp_rank):
    """One TP rank: preflight, wait for the whole group, then report its compared slice."""
    row = plan.calls[0][0]
    base = PrescoreReport(
        plan.id,
        1,
        0,
        (0, 0, 0, tp_rank),
        (1, 1, 1, TP),
        (row,),
        (),
        False,
    )
    ray.get(collector.preflight.remote(base), timeout=60)
    deadline = time.monotonic() + 60
    while True:
        status = ray.get(collector.preflight_status.remote(plan.id, 0), timeout=60)
        if status.get("error"):
            raise RuntimeError(status["error"])
        if status.get("ready"):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"preflight timeout: {status}")
        time.sleep(0.02)
    start, stop = tp_rank * SHARD, (tp_rank + 1) * SHARD
    ray.get(
        collector.report.remote(
            dataclasses.replace(base, shards=(ComparedShard(row, start, stop, SHARD),), completed=True)
        ),
        timeout=60,
    )
    return status


@pytest.fixture(scope="module")
def private_ray():
    """A fresh, private, CPU-only Ray instance -- never the ambient cluster (see the module docstring)."""
    if ray.is_initialized():
        pytest.skip("UNWITNESSED: Ray was already initialized in this process; refusing to reuse it")
    # Short path on purpose: Ray's worker sockets are AF_UNIX, whose sun_path caps at ~107 bytes.
    temp_dir = tempfile.mkdtemp(prefix="rayix-")
    try:
        ray.init(
            address="local",  # never read /tmp/ray/ray_current_cluster; always a new instance
            num_cpus=TP,
            num_gpus=0,
            object_store_memory=128 * 1024 * 1024,
            include_dashboard=False,
            log_to_driver=False,
            configure_logging=False,
            _temp_dir=temp_dir,
        )
    except Exception as e:  # noqa: BLE001 -- a box that cannot host a private instance is not a failure
        shutil.rmtree(temp_dir, ignore_errors=True)
        pytest.skip(f"UNWITNESSED: cannot start a private Ray instance here ({type(e).__name__}: {e})")
    try:
        yield
    finally:
        ray.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_the_collector_tiles_one_row_across_two_ray_worker_processes(private_ray):
    """Two real Ray workers preflight, are released together, and their slices finalize as one row."""
    remote_collector = ray.remote(num_cpus=0, max_restarts=0, max_task_retries=0)(PrescoreCollector)
    collector = remote_collector.remote()
    row = RowIdentity(7, 0, 3, (10, 11))
    plan = PrescorePlan("ray-transport", 1, VOCAB, ((row,),))

    assert ray.get(collector.begin.remote(plan), timeout=60) == plan.id

    statuses = ray.get([participant.remote(collector, plan, rank) for rank in range(TP)], timeout=60)
    assert all(status["ready"] for status in statuses)

    receipt = ray.get(collector.finalize.remote(plan.id), timeout=60)
    assert receipt["rows"] == 1
    assert receipt["compared_elements"] == VOCAB
    assert receipt["participant_completions"] == TP


def test_finalize_refuses_a_plan_whose_participants_never_reported(private_ray):
    """An incomplete participant barrier must prevent finalization."""
    collector = ray.remote(num_cpus=0, max_restarts=0, max_task_retries=0)(PrescoreCollector).remote()
    row = RowIdentity(1, 0, 0, (5,))
    plan = PrescorePlan("ray-transport-empty", 1, VOCAB, ((row,),))
    assert ray.get(collector.begin.remote(plan), timeout=60) == plan.id

    with pytest.raises(ray.exceptions.RayTaskError, match="prescore.coverage"):
        ray.get(collector.finalize.remote(plan.id), timeout=60)
