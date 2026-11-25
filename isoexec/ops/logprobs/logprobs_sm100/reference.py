"""Executable statement of the logprobs_sm100 numerical schedule, independent of the CUDA kernels.

Written with torch tensor ops in fp32, one rounded operation per line, vectorised over rows, shards and segments
only, never over a segment's chain or a tree level's operand order. Run it on the CUDA device: ``torch.exp`` there
is the kernels' ``expf``.

    shard g of G = d_max:  columns [g L, (g + 1) L),  L = V / G
    segment:   S consecutive columns (the last of a shard short), ascending
               m = fmaxf chain from -inf;  F = m, or +0 when m == -inf;  s = add chain from +0 of exp(v - F)
               (entropy: a = add chain from +0 of exp(v - F) * (v - F), +0 for a -inf column)
    merge (l, r):  M = fmaxf(m_l, m_r);  F as above;  s = s_l * exp(m_l - F) + s_r * exp(m_r - F)
               (entropy: a = e_l * (a_l + s_l * (m_l - F)) + e_r * (...);  a side with m == -inf adds s * 0)
    shard partial = adjacent tree over its segments (an unpaired last partial moves up)
    row partial   = adjacent tree over the G shard partials
    L32 = RN32(log(double(S)));  logp = (v - M) - L32;  H = log(S) - A / S in double

S comes from :mod:`isoexec.ops.schedule_policy` through ``schedule.py``. This is not logprobs_sm90's recipe (an
online normalizer on 256 strided lanes per leaf, merged in double), so the two architectures no longer agree bit
for bit.
"""

from __future__ import annotations

import torch

from isoexec.ops.logprobs.logprobs_sm100 import schedule

NINF = float("-inf")


def _fmax(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C ``fmaxf``: a NaN operand loses."""
    return torch.where(torch.isnan(b) | (a >= b), a, b).where(~torch.isnan(a), b)


def merge(left, right):
    """The reducer's merge on ``(m, s, a)`` tuples of equal-shape fp32 tensors, left operand first."""
    ml, sl, al = left
    mr, sr, ar = right
    m = _fmax(ml, mr)
    f = torch.where(m == NINF, torch.zeros_like(m), m)
    dl, dr = ml - f, mr - f
    el, er = torch.exp(dl), torch.exp(dr)
    ua = torch.where(ml == NINF, sl * 0.0, el * (al + sl * dl))
    ub = torch.where(mr == NINF, sr * 0.0, er * (ar + sr * dr))
    return m, sl * el + sr * er, ua + ub


def adjacent_tree(parts):
    """Adjacent tree over the LAST axis of ``(m, s, a)``: pair (0, 1), (2, 3), ...; an unpaired last one moves up."""
    while parts[0].shape[-1] > 1:
        g = parts[0].shape[-1]
        merged = merge(tuple(p[..., 0 : g - 1 : 2] for p in parts), tuple(p[..., 1:g:2] for p in parts))
        parts = tuple(torch.cat([q, p[..., g - 1 :]], dim=-1) for q, p in zip(merged, parts)) if g % 2 else merged
    return tuple(p[..., 0] for p in parts)


def segment_partials(v: torch.Tensor):
    """``(m, s, a)`` of ONE segment per trailing row of ``v [..., n]`` (n <= S columns, ascending)."""
    m = torch.full(v.shape[:-1], NINF, device=v.device, dtype=torch.float32)
    for j in range(v.shape[-1]):
        m = _fmax(m, v[..., j])
    f = torch.where(m == NINF, torch.zeros_like(m), m)
    s, a = torch.zeros_like(m), torch.zeros_like(m)
    for j in range(v.shape[-1]):
        d = v[..., j] - f
        e = torch.exp(d)
        s = s + e
        a = a + torch.where(v[..., j] == NINF, torch.zeros_like(e), e * d)
    return m, s, a


def shard_states(x: torch.Tensor, shards: int, temperature: float | torch.Tensor | None = None):
    """``(M, S, A)`` fp32 ``[rows, shards]`` for rows ``x [rows, shards * L]`` holding ``shards`` whole shards."""
    v = x.float()
    if temperature is not None:
        v = v / torch.as_tensor(temperature, dtype=torch.float32, device=v.device)
    rows, width = v.shape
    cols = width // shards
    seg = schedule.policy.reduce(cols, b=1, L=schedule.LSE_SHORTEST_SEGMENT)
    if seg.merge != "tree" or seg.G is not None:
        raise ValueError(f"logprobs_sm100 states fixed-length segments under a tree, not {seg}")
    v = v.view(rows, shards, cols)
    full = cols // seg.S
    parts = []
    if full:
        parts.append(segment_partials(v[..., : full * seg.S].reshape(rows, shards, full, seg.S)))
    if cols % seg.S:  # the short last segment
        parts.append(tuple(p.unsqueeze(-1) for p in segment_partials(v[..., full * seg.S :])))
    return adjacent_tree(tuple(torch.cat([p[k] for p in parts], dim=-1) for k in range(3)))


def row_states(m: torch.Tensor, s: torch.Tensor, a: torch.Tensor | None = None):
    """The row's ``(M, S, A)`` from all ``schedule.VOCAB_SHARDS`` shard partials ``[rows, G]``."""
    if m.shape[-1] != schedule.VOCAB_SHARDS:
        raise ValueError("the row tree is over all d_max shards")
    return adjacent_tree((m, s, torch.zeros_like(s) if a is None else a))


def full_logprobs(x: torch.Tensor, temperature: float | torch.Tensor | None = None) -> torch.Tensor:
    """fp32 ``[rows, V]`` logprobs: ``(v - M) - L32``."""
    m, s, _ = row_states(*shard_states(x, schedule.VOCAB_SHARDS, temperature))
    l32 = torch.where(m == NINF, torch.full_like(s, float("nan"), dtype=torch.float64), torch.log(s.double())).float()
    v = x.float()
    if temperature is not None:
        v = v / torch.as_tensor(temperature, dtype=torch.float32, device=v.device)
    return (v - m[:, None]) - l32[:, None]


def entropy(x: torch.Tensor, temperature: float | torch.Tensor | None = None) -> torch.Tensor:
    """fp32 ``[rows]`` row entropy ``log(S) - A / S`` (double, rounded once)."""
    m, s, a = row_states(*shard_states(x, schedule.VOCAB_SHARDS, temperature))
    h = torch.log(s.double()) - a.double() / s.double()
    return torch.where(m == NINF, torch.full_like(h, float("nan")), h).float()


__all__ = ["adjacent_tree", "entropy", "full_logprobs", "merge", "row_states", "segment_partials", "shard_states"]
