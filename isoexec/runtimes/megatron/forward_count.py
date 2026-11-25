"""Every rank of an expert-parallel group runs the same number of forwards in each schedule.

An MoE forward under EP issues the group's dispatch / combine collectives once per layer, so a rank that runs a
different number of microbatch forwards than its EP peers issues a different collective sequence: its peers
wait in a collective it never joins (or pair one layer's exchange with another's), which hangs or corrupts the
step instead of failing by name. The trainer driver is responsible for equal counts: SkyRL splits the batch
into equal data-parallel shards (``pad_training_input_batch``) and, with token-based microbatching, all-reduces
the count with MAX and appends padding microbatches (``TokenBasedBatchIterator._sync_num_microbatches``).
``require_equal_forward_count`` checks that contract before the schedule issues its first collective: one
all-gather of the local count over the EP group, so every rank sees the same counts and refuses together
(``EP_FORWARD_COUNT_MISMATCH``) rather than one rank deciding alone.
"""
from __future__ import annotations

from typing import Any

import torch

from isoexec.checks import Refusal


def require_equal_forward_count(count: int, group: Any, *, site: str) -> int:
    """Refuse unless every rank of ``group`` is about to run ``count`` forwards; returns ``count``.

    A collective over ``group`` (none for a missing or single-rank group); every rank must call it once per
    schedule, before the schedule's first forward."""
    import torch.distributed as dist

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{site}: forward count must be a nonnegative int, got {count!r}")
    if group is None or not dist.is_initialized() or dist.get_world_size(group) == 1:
        return count
    size = dist.get_world_size(group)
    device = (torch.device("cuda", torch.cuda.current_device())
              if dist.get_backend(group) == "nccl" else torch.device("cpu"))
    every = torch.empty(size, dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(every, torch.tensor([count], dtype=torch.int64, device=device), group=group)
    counts = [int(c) for c in every.tolist()]
    if len(set(counts)) != 1:
        ranks = dist.get_process_group_ranks(group)
        by_count = {c: [r for r, n in zip(ranks, counts) if n == c] for c in sorted(set(counts))}
        raise Refusal(f"EP_FORWARD_COUNT_MISMATCH: {site}: the ranks of the expert-parallel group would run "
                      f"different numbers of forwards ({{forwards: global ranks}} = {by_count}), so their MoE "
                      "dispatch/combine collectives would not pair; the trainer driver must give every EP rank "
                      "the same number of microbatches (padding microbatches as SkyRL's token-based iterator does)")
    return count


__all__ = ["require_equal_forward_count"]
