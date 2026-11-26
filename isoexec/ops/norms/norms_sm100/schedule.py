"""The numerical schedule of the norms_sm100 FORWARD sum of squares, stated through the policy (design 7.4).

Nothing here is on a call path: the kernels (``csrc/``) are the tree's own and read nothing from this module.
It states their schedule as data, and ``tests/test_schedule_policy_cpu.py`` fails if ``csrc/plan.h`` or
``csrc/launch_fwd.cu`` stop agreeing with it.

The reduction: one row's ``sum_h x[h] * x[h]`` over the hidden axis (RMSNorm, and the fused add + RMSNorm on
``h = rn_bf16(x + r)``). The tree declares no shard degree for the hidden axis of a norm (``isoexec/nn/norm.py``
declares only ``CONTEXT_PARALLEL``: per token; the kernels take whole rows), so ``d_max = 1`` and ``n`` is the
row's width. The segment rule has no hardware specification here, so rule 3 of
:mod:`isoexec.ops.schedule_policy` gives ONE segment per shard: ``G = 1``, ``S = width``. The lane count comes from
``policy.lanes``: a lane steps one term on one thread, so a reducer's most lanes are ``policy.lanes(threads)``
with ``threads`` the threads it has for one row (32, one warp; 128, four warps), and a segment runs the largest
count up to it that it admits. Everything below the segment is the reducer's own fixed DAG, which depends only on a
term's position inside the segment:

    leaf        8 consecutive columns (one 16-byte bf16 load)
    lanes       L = the largest power of two <= cap that divides S / 8;  K = S / (8 L) visits per lane
    membership  lane l owns columns  k * 8L + 8l + j,  k = 0..K-1, j = 0..7
    lane chain  fp32, from +0, k ascending then j ascending:  acc = add_rn(acc, mul_rn(v, v))  (no fma)
    combine     butterfly over the lanes, strides L/2, L/4, ..., 1:  acc[l] = add_rn(acc[l], acc[l ^ s]),
                own value on the left; the row's sum is lane 0's value. For L > 32 the kernel evaluates the
                strides >= 32 as v[q] = add_rn(v[q], v[q + s]) on the L / 32 values of one in-warp position
                (``csrc/common.cuh::wide_fold``): the same DAG read at lane 0.

Two reducers differ only in the threads behind the cap (:data:`V1`, :data:`V2`). Which one a width runs is a
BINDING, not a schedule: :data:`BINDING` lists the base tree's choice (``csrc/plan.h::fwd_recipe_default``) and is
not changed here. The GDN gated out-norm always runs :data:`V1` (``launch_gated_fwd``); it is not stated further here.

The backward's weight-gradient sum is over the rows of a call, has no contract, and is not scheduled by the policy.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from isoexec.ops import schedule_policy as policy

#: d_max of the hidden axis: it is not sharded.
HIDDEN_SHARDS = 1
#: Columns per leaf: one 16-byte load of bf16 (``csrc/fwd.cuh::load8``).
LEAF = 8
#: Widths at which reducer v2 was swept (``csrc/plan.h::fwd_v2_lmax``): its cap is 128 there and 32 elsewhere.
V2_SWEPT = (128, 384, 512, 768, 1536, 2048, 4096, 5120, 7168)


def pow2_dividing(m: int, cap: int) -> int:
    """The largest power of two ``<= cap`` that divides ``m``."""
    lanes = 1
    while lanes * 2 <= cap and m % (lanes * 2) == 0:
        lanes *= 2
    return lanes


@dataclass(frozen=True)
class LaneReducer:
    """A sum of squares over one segment on interleaved lanes. The fields are the whole DAG."""

    version: int                     # the tree's forward recipe version
    impl_id: str                     # the declared impl (spec.py)
    threads: int                     # threads for one row at ``cap_at``: most lanes = policy.lanes(threads)
    threads_elsewhere: int           # threads for one row at a segment length outside ``cap_at``
    cap_at: tuple[int, ...] | None   # segment lengths that get ``cap``; None: every length
    reference: str                   # the executable statement of this DAG (torch, fp32, the stated order)
    leaf: int = LEAF
    accumulator: str = "fp32"
    lane_start: str = "+0"
    lane_order: str = "visit k ascending, then column j ascending"
    lane_step: str = "acc = add_rn(acc, mul_rn(v, v))"
    combine: str = "butterfly: for s = L/2 .. 1: acc[l] = add_rn(acc[l], acc[l ^ s]); result = acc[0]"

    def supported(self, s: int) -> bool:
        """Segment lengths the forward has a fold for (``csrc/plan.h::fwd_supported``)."""
        return s > 0 and (s % 256 == 0 or s in (128, 384))

    @property
    def cap(self) -> int:
        """Most lanes: the policy's lane count for :attr:`threads`."""
        return policy.lanes(self.threads)

    @property
    def cap_elsewhere(self) -> int:
        """Most lanes at a segment length outside ``cap_at``: the policy's lane count for :attr:`threads_elsewhere`."""
        return policy.lanes(self.threads_elsewhere)

    def lane_cap(self, s: int) -> int:
        return self.cap if self.cap_at is None or s in self.cap_at else self.cap_elsewhere

    def lanes(self, s: int) -> int:
        """L as the function of the segment length that the kernel uses."""
        if not self.supported(s):
            raise ValueError(f"norms_sm100 has no forward fold for a segment of {s}")
        return pow2_dividing(s // self.leaf, self.lane_cap(s))

    def visits(self, s: int) -> int:
        """K: leaves per lane."""
        return s // (self.leaf * self.lanes(s))

    def members(self, s: int, lane: int) -> tuple[int, ...]:
        """The positions of ``lane`` inside the segment, in the order its chain adds them."""
        ell, k = self.lanes(s), self.visits(s)
        return tuple(v * self.leaf * ell + self.leaf * lane + j for v in range(k) for j in range(self.leaf))

    def strides(self, s: int) -> tuple[int, ...]:
        """The butterfly's strides, in order."""
        ell = self.lanes(s)
        return tuple(ell >> i for i in range(1, ell.bit_length()))

    def reference_fn(self):
        """``row_sum_of_squares(x [M, S]) -> fp32 [M]`` of :attr:`reference`."""
        module, name = self.reference.rsplit(".", 1)
        return getattr(importlib.import_module(module), name)


#: Recipe v1: at most policy.lanes(32) = 32 lanes (one warp), at every supported width. Its executable statement
#: is the tree's ``norms_sm90/reference.py`` (re-exported by ``norms_sm100/reference.py``).
V1 = LaneReducer(version=1, impl_id="sm100_rms_v1", threads=32, threads_elsewhere=32, cap_at=None,
                 reference="isoexec.ops.norms.norms_sm100.reference.row_sum_of_squares")
#: Recipe v2: at most policy.lanes(128) = 128 lanes (four warps) at the swept widths, v1's lanes elsewhere.
#: Statement: ``reference_v2.py``.
V2 = LaneReducer(version=2, impl_id="sm100_rms_v2", threads=128, threads_elsewhere=32, cap_at=V2_SWEPT,
                 reference="isoexec.ops.norms.norms_sm100.reference_v2.row_sum_of_squares")

REDUCERS = {1: V1, 2: V2}

#: The binding (``csrc/plan.h::fwd_recipe_default``): the widths that run v2. Every other supported width runs v1.
BINDING: dict[int, int] = {1536: 2, 4096: 2, 5120: 2, 7168: 2}
DEFAULT_VERSION = 1

#: Widths the tree's own batteries cover (``tests/test_sm100_gpu.py::WIDTHS`` and the 1536-7168 battery).
WIDTHS = (128, 384, 512, 768, 1536, 2048, 4096, 5120, 7168)


def reducer(width: int) -> LaneReducer:
    """The reducer bound to ``width``."""
    return REDUCERS[BINDING.get(width, DEFAULT_VERSION)]


def segment(width: int) -> policy.Segments:
    """Rule 3 with no hardware specification: one segment per shard of the hidden axis."""
    return policy.reduce(width // HIDDEN_SHARDS)


def lanes(width: int) -> tuple[int, int]:
    """``(L, K)`` the bound reducer runs on the policy's segment of ``width``."""
    red, s = reducer(width), segment(width).S
    return red.lanes(s), red.visits(s)


def evaluate(x, red: LaneReducer | None = None):
    """The stated DAG, evaluated from the data above: fp32 ``[M]`` sums of squares of ``x [M, S]``.

    Reads only :meth:`LaneReducer.members` and :meth:`LaneReducer.strides`. It exists so that the data can be
    compared with :attr:`LaneReducer.reference` (which the tree's GPU tests compare with the kernels).
    """
    import torch

    s = x.shape[-1]
    red = reducer(s) if red is None else red
    ell = red.lanes(s)
    idx = torch.tensor([red.members(s, lane) for lane in range(ell)], device=x.device)
    v = x.float()[:, idx]
    acc = torch.zeros(x.shape[0], ell, device=x.device, dtype=torch.float32)
    for t in range(idx.shape[1]):
        acc = acc + v[:, :, t] * v[:, :, t]
    lane = torch.arange(ell, device=x.device)
    for stride in red.strides(s):
        acc = acc + acc[:, lane ^ stride]
    return acc[:, 0]


__all__ = ["BINDING", "DEFAULT_VERSION", "HIDDEN_SHARDS", "LEAF", "LaneReducer", "REDUCERS", "V1", "V2",
           "V2_SWEPT", "WIDTHS", "evaluate", "lanes", "pow2_dividing", "reducer", "segment"]
