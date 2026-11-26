"""The numerical-schedule policy (design version 7.2): three rules, no tables.

Canonical copy: project-x/policy_validation/port/schedule_policy.py. It is installed unchanged as
isoexec/ops/schedule_policy.py; kernels take every segment count and segment length from here.
Mirrors alloy_new/policy.py; the port's CPU test compares the two on every deployed shape.

A reducer is the fixed way one segment is reduced: a DAG over the segment's terms that depends only on their
positions inside the segment (a chain of blocks, interleaved lanes, an MMA accumulation). The policy never
looks inside it. It decides only how a reduction is cut into segments and how the segments merge.

For one reduction over n positions of a shard (n = axis length, capacity or declared longest length, divided
by d_max), with exact block b:

  rule 1  reducer has no merge (recurrence, sparse-attention tile):  segments of b, one continued chain
  rule 2  sum over data-selected coordinates:                        one segment per coordinate, chain
  rule 3  otherwise G equal segments per shard, adjacent tree inside the shard, then over the d_max shards:
              G = pow2floor(min(p, cap, n // L)), at least 1
              S = b * ceil(n / (b * G))

L, cap and the occupancy behind p are the policy's cost parameters for a reducer, measured on a GPU
generation: L is the shortest segment worth a partial result of its own, p the workers free for the segments
of one output. A reducer without them (L absent) gets G = 1: one segment per shard. So does a deployment that
declares no target batch (p absent): it is scheduled for throughput, where the outputs of a batch already
fill the device.

7.2 against 7.1: the form "L without occupancy gives S = L" is gone. The scalar sums that used it (the sum of
squares of a norm, the log-sum-exp of the output) are one segment per shard; their interleaved lanes are the
reducer's own DAG. The smallest batch is now an optional target batch.
"""
from dataclasses import dataclass
from typing import Optional

VERSION = "7.2"


def pow2floor(x: int) -> int:
    return 1 << (x.bit_length() - 1)


def workers(sms: int, occupancy: Optional[int] = None) -> int:
    """p for one device: workers free for the segments of one output. The deployment takes the minimum over
    its devices. occupancy is the reducer's measured function of (model shape, target batch); None means the
    deployment declared no target batch, which schedules for throughput (p = 1)."""
    if occupancy is None:
        return 1
    if occupancy < 1:
        raise ValueError(f"occupancy {occupancy}")
    return max(1, sms // occupancy)


@dataclass(frozen=True)
class Segments:
    S: int            # segment length, in positions
    G: Optional[int]  # segments per shard; None when the number of segments follows the data
    merge: str        # "tree" | "chain" | "continued"


def no_merge(b: int) -> Segments:                       # rule 1
    return Segments(b, None, "continued")


def selected() -> Segments:                             # rule 2
    return Segments(1, None, "chain")


def reduce(n: int, b: int = 1, L: Optional[int] = None, p: Optional[int] = None,
           cap: Optional[int] = None) -> Segments:      # rule 3
    if n < 1 or b < 1:
        raise ValueError(f"n = {n}, b = {b}")
    G = 1
    if L is not None:
        if L < 1 or L % b:
            raise ValueError(f"L = {L} is not whole blocks of {b}")
        if p is not None:
            G = pow2floor(max(1, min(p, n // L) if cap is None else min(p, cap, n // L)))
    return Segments(b * -(-n // (b * G)), G, "tree")
