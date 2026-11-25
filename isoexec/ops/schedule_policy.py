"""The numerical-schedule policy (design version 7.1): three rules, no tables.

Canonical copy: project-x/policy_validation/port/schedule_policy.py. It is installed unchanged as
isoexec/ops/schedule_policy.py; kernels take every segment count and segment length from here.
Mirrors alloy_next/policy.py (`segment_length`, `constants`, rule 3 of `Schedule`); the port's CPU test
compares the two on every deployed shape.

For one reduction over n positions of a shard (n = axis length, capacity or declared longest length, divided
by d_max), with exact block b:

  rule 1  reducer has no merge (recurrence, DSA tile):   segments of b, one continued chain
  rule 2  sum over data-selected coordinates:            one segment per coordinate, chain
  rule 3  otherwise G segments per shard, adjacent tree inside the shard, then over the d_max shards:
              S = b * ceil(n / (b * G))
              G = 1                                      reducer publishes no shortest segment
              G = pow2floor(min(p, cap, n // L)), >= 1   reducer publishes L and an occupancy
              S = L (G not used, last segment short)     reducer publishes L and no occupancy (scalar step)
"""
from dataclasses import dataclass
from typing import Optional

VERSION = "7.1"


def pow2floor(x: int) -> int:
    return 1 << (x.bit_length() - 1)


def workers(sms: int, occupancy: int) -> int:
    """p for one device: workers free for the segments of one output. The deployment takes the minimum over
    its devices; occupancy is the reducer's published function of (model shape, smallest batch)."""
    if occupancy < 1:
        raise ValueError(f"occupancy {occupancy}")
    return max(1, sms // occupancy)


@dataclass(frozen=True)
class Segments:
    S: int            # segment length, in positions
    G: Optional[int]  # segments per shard; None when S is fixed and the last segment may be short
    merge: str        # "tree" | "chain" | "continued"


def no_merge(b: int) -> Segments:                       # rule 1
    return Segments(b, None, "continued")


def selected() -> Segments:                             # rule 2
    return Segments(1, None, "chain")


def reduce(n: int, b: int = 1, L: Optional[int] = None, p: Optional[int] = None,
           cap: Optional[int] = None) -> Segments:      # rule 3
    if n < 1 or b < 1:
        raise ValueError(f"n = {n}, b = {b}")
    if L is None:
        G = 1
    elif p is None:
        if L % b:
            raise ValueError(f"L = {L} is not whole blocks of {b}")
        return Segments(L, None, "tree")
    else:
        if L % b:
            raise ValueError(f"L = {L} is not whole blocks of {b}")
        G = pow2floor(max(1, min(p, n // L) if cap is None else min(p, cap, n // L)))
    return Segments(b * -(-n // (b * G)), G, "tree")
