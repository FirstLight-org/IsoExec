"""Which attention schedule the computing parties of a verl run bound, compared before a token is generated.

Two kinds of process compute the bits that must agree: the Megatron training workers (the trainer forward that
rescoring and the update run) and the vLLM engine workers (the rollout). Each binds its attention schedule at
import from its own environment (``ops/attention/target_batch.py``). The task runner, which hosts verl's
trainer class and reads the receipts, computes no attention and is not a party.

A training worker writes the record it bound when its model is built (``engine.IsoExecMegatronEngine.initialize``);
an engine worker puts its record in every weight acknowledgement (``worker_extension``). At every weight sync
:func:`require_same_schedules` compares all of them and refuses a difference or a party that recorded nothing.
This module imports nothing of verl.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from isoexec.checks import Refusal


def trainer_schedule_directory(root: Path) -> Path:
    return root / "schedules"


def record_trainer_schedule(root: Path, rank: int) -> dict[str, Any]:
    """Write the attention schedule THIS process bound as training worker ``rank``; returns the record.

    One file per (rank, process): a second engine of the same process (the reference policy) rewrites the same
    bytes, and a process that bound something else can never overwrite or hide another's record."""
    from isoexec.ops.attention.target_batch import schedule_record

    record = {"rank": int(rank), "pid": os.getpid(), "attention_schedule": schedule_record()}
    directory = trainer_schedule_directory(root)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"trainer-rank{int(rank):04d}-pid{os.getpid()}.json"
    tmp = directory / f".{name}.tmp"
    tmp.write_text(json.dumps(record, sort_keys=True))
    os.replace(tmp, directory / name)
    return record


def require_same_schedules(root: Path, acks: Iterable[Mapping[str, Any]], trainer_ranks: int) -> dict[str, Any]:
    """Refuse unless every one of the ``trainer_ranks`` training workers recorded an attention schedule and all of
    them and every engine worker of ``acks`` bound the same one. Returns the schedule they share."""
    from isoexec.ops.attention.target_batch import require_same_schedule

    directory = trainer_schedule_directory(root)
    records = ([json.loads(path.read_text()) for path in sorted(directory.glob("trainer-rank*.json"))]
               if directory.exists() else [])
    ranks = sorted({record["rank"] for record in records})
    if len(ranks) != trainer_ranks:
        raise Refusal(f"attention_schedule.missing: {trainer_ranks} training workers compute the trainer forward, "
                      f"but ranks {ranks} recorded the attention schedule they bound (under {directory})")
    first = records[0]
    shared, here = first["attention_schedule"], f"training worker rank {first['rank']}"
    for record in records[1:]:
        require_same_schedule(shared, record.get("attention_schedule"), here=here,
                              who=f"training worker rank {record['rank']} (pid {record['pid']})")
    for ack in acks:
        require_same_schedule(shared, ack.get("attention_schedule"), here=here,
                              who=f"engine worker replica {ack['replica_rank']} dp {ack['dp_rank']} rank {ack['rank']}")
    return shared


__all__ = ["record_trainer_schedule", "require_same_schedules", "trainer_schedule_directory"]
