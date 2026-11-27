"""The numerical-schedule policy (version 7.4): three rules, no tables.

Kernels take every segment count and segment length from here, so that a trainer and an inference engine
that run the same version cut every reduction in the same places.

A reducer is the fixed way one segment is reduced: a DAG over the segment's terms that depends only on their
positions inside the segment (a chain of blocks, interleaved lanes, an MMA accumulation). The policy decides
how a reduction is cut into segments, how the segments merge, and whether a segment is reduced sequentially
or on interleaved lanes.

For one reduction over n positions of a shard (n = axis length, capacity or declared longest length, divided
by d_max), with exact block b:

  rule 1  reducer has no merge (recurrence, sparse-attention tile):  segments of b, one continued chain
  rule 2  sum over data-selected coordinates:                        one segment per coordinate, chain
  rule 3  otherwise G equal segments per shard, adjacent tree inside the shard, then over the d_max shards:
              G = pow2floor(min(p, max_segments, n // min_segment)), at least 1
              S = b * ceil(n / (b * G))

Lanes: a reducer that steps one term on one thread gets as many lanes as the threads it has for one output
(lane l of L takes the terms l, l + L, ...: neighbouring threads read neighbouring terms). A reducer that
steps a whole block with one instruction of all its threads, and a reducer without a merge, are sequential
(L = 1). Lanes stay inside a segment, so they are independent of the three rules.

min_segment, max_segments, the occupancy behind p and the threads behind L are the hardware specifications
of a reducer on a GPU generation: min_segment is the shortest segment worth a partial result of its own,
max_segments the most partial results the kernel holds for one output, p the workers free for the segments
of one output. A reducer without them (min_segment absent) gets G = 1: one segment per shard. So does a
deployment that declares no target batch (p absent): it is scheduled for throughput, where the outputs of a
batch already fill the device.
"""
from dataclasses import dataclass
from typing import Optional

VERSION = "7.4"


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


def lanes(threads: Optional[int] = None) -> int:
    """L for one segment: the number of interleaved lanes. threads is the number of threads the reducer has for
    one output when it steps one term on one thread; None for a reducer that steps a block with one instruction
    of all its threads or that has no merge, which is sequential (L = 1). A reducer whose lanes must be equal
    runs the largest count up to L that its segment admits."""
    if threads is None:
        return 1
    if threads < 1:
        raise ValueError(f"threads {threads}")
    return threads


@dataclass(frozen=True)
class Segments:
    S: int            # segment length, in positions
    G: Optional[int]  # segments per shard; None when the number of segments follows the data
    merge: str        # "tree" | "chain" | "continued"


def no_merge(b: int) -> Segments:                       # rule 1
    return Segments(b, None, "continued")


def selected() -> Segments:                             # rule 2
    return Segments(1, None, "chain")


def reduce(n: int, b: int = 1, min_segment: Optional[int] = None, p: Optional[int] = None,
           max_segments: Optional[int] = None) -> Segments:      # rule 3
    if n < 1 or b < 1:
        raise ValueError(f"n = {n}, b = {b}")
    G = 1
    if min_segment is not None:
        if min_segment < 1 or min_segment % b:
            raise ValueError(f"min_segment = {min_segment} is not whole blocks of {b}")
        if p is not None:
            G = min(p, n // min_segment)
            G = pow2floor(max(1, G if max_segments is None else min(G, max_segments)))
    return Segments(b * -(-n // (b * G)), G, "tree")
