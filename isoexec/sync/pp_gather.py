"""Gather pipeline stages into a complete logical weight stream on every rank."""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import torch

from isoexec.schema import ShardError

Meta = tuple[str, str, list[int]]


def stage_schedule(metadata_by_stage: Sequence[Sequence[Meta]]) -> list[tuple[int, str, str, list[int]]]:
    """Flatten per-stage ``(name, dtype, shape)`` advertisements into the lockstep broadcast order:
    stage 0's stream, then stage 1's, ... -- each in its own stream order. A name
    advertised by two stages refuses (a tied alias emitted twice would double-count the digest)."""
    seen: dict[str, int] = {}
    out: list[tuple[int, str, str, list[int]]] = []
    for stage, metas in enumerate(metadata_by_stage):
        for name, dtype, shape in metas:
            if name in seen:
                raise ShardError(f"pipeline gather: {name!r} advertised by stages {seen[name]} and {stage}")
            seen[name] = stage
            out.append((stage, str(name), str(dtype), [int(d) for d in shape]))
    return out


def _torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name.split(".")[-1], None)
    if not isinstance(dtype, torch.dtype):
        raise ShardError(f"pipeline gather: unknown dtype name {name!r}")
    return dtype


def broadcast_stage_stream(
    schedule: Sequence[tuple[int, str, str, list[int]]],
    local: Iterator[tuple[str, torch.Tensor]],
    *,
    group,
    device: Any,
) -> Iterator[tuple[int, str, torch.Tensor]]:
    """Broadcast every stage's tensors in ``stage_schedule`` order on every rank of ``group``.

    The owning stage's ``local`` stream must yield exactly its advertised names, shapes and dtypes,
    in order; a mismatch or a tensor beyond the advertisement refuses.
    """
    import torch.distributed as dist

    rank = dist.get_rank(group)
    group_ranks = dist.get_process_group_ranks(group)
    for stage, name, dtype, shape in schedule:
        if stage == rank:
            try:
                got_name, tensor = next(local)
            except StopIteration as exc:
                raise ShardError(f"pipeline gather: stage {stage} advertised {name!r} but its stream ended") from exc
            if got_name != name:
                raise ShardError(f"pipeline gather: stage {stage} yielded {got_name!r} where {name!r} was advertised")
            buf = tensor.detach().contiguous()
            del tensor
            if list(buf.shape) != shape or buf.dtype != _torch_dtype(dtype):
                raise ShardError(
                    f"pipeline gather: {name!r} is {tuple(buf.shape)}/{buf.dtype}, advertised {shape}/{dtype}"
                )
        else:
            buf = torch.empty(shape, dtype=_torch_dtype(dtype), device=device)
        dist.broadcast(buf, src=group_ranks[stage], group=group)
        yield stage, name, buf
    leftover = next(local, None)
    if leftover is not None:
        raise ShardError(f"pipeline gather: stage {rank} yielded {leftover[0]!r} beyond its advertisement")


__all__ = ["broadcast_stage_stream", "stage_schedule"]
