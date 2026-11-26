"""Executable statement of the lane programs that ``schedule.py`` states, independent of the CUDA kernels.

Written with torch tensor ops, one rounded operation per line, vectorised over rows, shards and lanes only,
never over a lane's chain, the lane chain or a tree level's operand order. Lane membership comes from
``schedule.LaneReducer.members``. Run it on the CUDA device: the comparison with the kernels is bitwise and
relies on ``torch.exp`` / ``torch.log`` there being the kernels' ``expf`` / ``exp`` / ``log``
(``tests/test_schedule_invariance_gpu.py`` holds that comparison). No temperature, no entropy partial.

Not on a call path: only tests import this module.
"""

from __future__ import annotations

import torch

from isoexec.ops.logprobs.logprobs_sm100 import schedule

NINF = float("-inf")


def _lanes(v: torch.Tensor, red: schedule.LaneReducer):
    """``v [..., S]`` as ``(u [..., visits, LANES, strip], present [visits, LANES])`` in the lanes' own order."""
    s = v.shape[-1]
    lanes = red.lanes(s)
    visits = max(len(red.members(s, lane)) for lane in range(lanes)) // red.strip
    idx = torch.zeros(visits, lanes, red.strip, dtype=torch.long, device="cpu")
    present = torch.zeros(visits, lanes, dtype=torch.bool, device="cpu")
    for lane in range(lanes):
        cols = red.members(s, lane)
        if cols:
            n = len(cols) // red.strip
            idx[:n, lane] = torch.tensor(cols).view(n, red.strip)
            present[:n, lane] = True
    return v[..., idx.to(v.device)], present.to(v.device)


def _lane_chain(lane: torch.Tensor) -> torch.Tensor:
    """One fp32 chain over the lanes ``[..., LANES]``, ascending from lane 0."""
    total = lane[..., 0]
    for i in range(1, lane.shape[-1]):
        total = total + lane[..., i]
    return total


def segment_partial_v2(v: torch.Tensor):
    """Reducer v2 on segments ``v [..., S]`` (fp32): ``(M, S)`` fp32 ``[...]``."""
    red = schedule.V2
    u, present = _lanes(v, red)
    m = torch.where(torch.isnan(v), torch.full_like(v, NINF), v).amax(-1)   # fmaxf: a NaN never wins
    f = torch.where(m == NINF, torch.zeros_like(m), m)
    s = torch.zeros(u.shape[:-3] + u.shape[-2:-1], device=v.device, dtype=torch.float32)
    for i in range(u.shape[-3]):
        for j in range(red.strip):
            d = u[..., i, :, j] - f[..., None]
            s = torch.where(present[i], s + torch.exp(d), s)
    return m, _lane_chain(s)


def segment_partial_v1(v: torch.Tensor):
    """Reducer v1 on segments ``v [..., S]`` (fp32): ``(M, S)`` fp32 ``[...]``."""
    red = schedule.V1
    u, present = _lanes(v, red)
    shape = u.shape[:-3] + u.shape[-2:-1]
    m = torch.full(shape, NINF, device=v.device, dtype=torch.float32)
    s = torch.zeros(shape, device=v.device, dtype=torch.float32)
    for i in range(u.shape[-3]):
        for j in range(red.strip):
            x = u[..., i, :, j]
            rescaled = s * torch.exp(m - x)
            rescaled = rescaled + 1.0
            added = s + torch.exp(x - m)
            new = torch.where(x > m, rescaled, torch.where(x != NINF, added, s))
            s = torch.where(present[i], new, s)
            m = torch.where(present[i] & (x > m), x, m)
    top = m.amax(-1)
    scaled = s * torch.exp(m - top[..., None])
    lane = torch.where((m == NINF) & (s == 0), torch.zeros_like(s), scaled)
    return top, _lane_chain(lane)


PARTIAL = {1: segment_partial_v1, 2: segment_partial_v2}


def shard_states(x: torch.Tensor, shards: int, version: int):
    """``(M, S)`` fp32 ``[rows, shards]`` for rows ``x [rows, shards * S]`` holding ``shards`` whole shards."""
    rows, width = x.shape
    seg = schedule.policy.reduce(width // shards)
    if (seg.G, seg.merge) != (1, "tree"):
        raise ValueError(f"logprobs_sm100 states one segment per shard, not {seg}")
    return PARTIAL[version](x.float().view(rows, shards, seg.S))


def row_state(m: torch.Tensor, s: torch.Tensor):
    """The shard merge over all ``schedule.VOCAB_SHARDS`` partials ``[rows, G]``: ``(M fp32, logS fp64)``."""
    if m.shape[-1] != schedule.VOCAB_SHARDS:
        raise ValueError("the row tree is over all d_max shards")
    top = m.amax(-1)
    nan = torch.full(s.shape, float("nan"), device=s.device, dtype=torch.float64)
    w = s.double() * torch.exp(m.double() - top.double()[:, None])
    w = torch.where(m == NINF, torch.where(s == 0, torch.zeros_like(w), nan), w)
    parts = [w[:, g] for g in range(w.shape[1])]
    for level in schedule.shard_tree(len(parts)):
        for left, right in level:
            parts[left] = parts[left] + parts[right]
    return top, torch.where(top == NINF, nan[:, 0], torch.log(parts[0]))


def full_logprobs(x: torch.Tensor, version: int) -> torch.Tensor:
    """fp32 ``[rows, V]`` logprobs: ``(v - M) - RN32(logS)``."""
    m, ls = row_state(*shard_states(x, schedule.VOCAB_SHARDS, version))
    return (x.float() - m[:, None]) - ls.float()[:, None]


__all__ = ["PARTIAL", "full_logprobs", "row_state", "segment_partial_v1", "segment_partial_v2", "shard_states"]
