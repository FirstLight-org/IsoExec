"""SM90 forward API, with host metadata validation and graph-safe device plans.

Arithmetic version 1: WGMMA k16 atoms in ascending K order, no split K.
FC1 rounds to the operand dtype; fused SiLU rounds its result, the up product,
and the routing-probability product separately. expf is CUDA libdevice, FTZ and
FMA contraction disabled. FC2 resets the FP32 accumulator for each fixed K leaf,
rounds each leaf RNE to the operand dtype, and merges lower-index subtrees on the
left using FP32 add.rn. FC2 stores only the root; the explicit single-leaf API
can store it losslessly in operand dtype. The arithmetic does not depend
on token counts, expert populations, tile positions, or gradient mode.

These are standalone forward primitives. They reject differentiable calls rather
than silently discarding gradients. Routing scores/top-k are supplied by the caller.
"""

from dataclasses import dataclass

import torch

from ._extension import extension


def _check(t, *, dtype=None, device=None, ndim=None, contiguous=True):
    if not isinstance(t, torch.Tensor) or not t.is_cuda:
        raise ValueError("moe_sm90 requires CUDA tensors")
    if torch.cuda.get_device_capability(t.device) != (9, 0):
        raise ValueError("moe_sm90 requires an SM90 GPU")
    if dtype is not None and t.dtype not in (dtype if isinstance(dtype, tuple) else (dtype,)):
        raise ValueError(f"expected dtype {dtype}, got {t.dtype}")
    if device is not None and t.device != device:
        raise ValueError("all operands must be on the same device")
    if ndim is not None and t.ndim != ndim:
        raise ValueError(f"expected {ndim} dimensions")
    if contiguous and not t.is_contiguous():
        raise ValueError("expected a contiguous tensor")
    if torch.is_grad_enabled() and t.requires_grad:
        raise ValueError("moe_sm90 is forward-only; use torch.no_grad() for inference")


@dataclass(frozen=True)
class GroupedPlan:
    offsets: torch.Tensor
    tiles: torch.Tensor
    total: torch.Tensor
    rows: int

    @property
    def experts(self):
        return self.offsets.numel() - 1


@dataclass(frozen=True)
class RoutePlan:
    grouped: GroupedPlan
    indices: torch.Tensor
    inverse: torch.Tensor
    restore: torch.Tensor


def make_plan(offsets: torch.Tensor, rows: int) -> GroupedPlan:
    """Build compact expert tiles from CUDA int32 [E+1] offsets.

    Offsets must start at zero, be nondecreasing, and end at ``rows``. Device
    assertions enforce the value contract without synchronizing the host.
    Rebuild after changing offsets; graph replay rebuilds from their current values.
    """
    _check(offsets, dtype=torch.int32, ndim=1)
    if not isinstance(rows, int) or not 2 <= offsets.numel() < 2**31 - 256 or not 0 <= rows < 2**31 - 256:
        raise ValueError("require int32 expert counts and 0 <= rows < 2**31-256")
    experts = offsets.numel() - 1
    block_m = 16 if rows <= experts * 16 else 64 if rows <= experts * 64 else 128
    bound = min(rows, (rows + block_m - 1) // block_m + offsets.numel() - 2)
    if bound >= (2**31 - 1) // 3:
        raise ValueError("tile metadata exceeds int32 indexing")
    tiles = torch.empty((bound, 3), device=offsets.device, dtype=torch.int32)
    total = torch.empty((), device=offsets.device, dtype=torch.int32)
    extension().plan(offsets, tiles, total, rows)
    return GroupedPlan(offsets, tiles, total, rows)


def make_plan_from_counts(counts, rows):
    _check(counts, dtype=(torch.int32, torch.int64), ndim=1)
    experts = counts.numel()
    if not isinstance(rows, int) or not 0 <= rows < 2**31 - 256 or not 0 < experts < 2**31 - 256:
        raise ValueError("invalid rows/experts")
    bm = 16 if rows <= experts * 16 else 64 if rows <= experts * 64 else 128
    bound = min(rows, (rows + bm - 1) // bm + experts - 1)
    offsets = torch.empty(experts + 1, device=counts.device, dtype=torch.int32)
    tiles = torch.empty((bound, 3), device=counts.device, dtype=torch.int32)
    total = torch.empty((), device=counts.device, dtype=torch.int32)
    extension().counts_plan(counts, offsets, tiles, total, rows)
    return GroupedPlan(offsets, tiles, total, rows)


def _gemm(x, w, plan, leaves, glu, probs=None, out=None, narrow=False):
    _check(x, dtype=(torch.bfloat16, torch.float16), ndim=2, contiguous=False)
    _check(w, dtype=x.dtype, device=x.device, ndim=3, contiguous=False)
    if not isinstance(plan, GroupedPlan):
        raise TypeError("expected a GroupedPlan from make_plan")
    if x.stride(1) != 1 or w.stride(2) != 1 or any(s <= 0 or s % 8 for s in (x.stride(0), w.stride(1), w.stride(0))):
        raise ValueError("K must be contiguous; row and expert strides must be positive multiples of 8")
    if x.data_ptr() % 16 or w.data_ptr() % 16:
        raise ValueError("operands must be 16-byte aligned")
    if x.size(1) != w.size(2) or w.size(0) != plan.experts or x.size(0) != plan.rows:
        raise ValueError("incompatible grouped GEMM dimensions")
    if max(x.stride(0), w.stride(1), x.size(1), w.size(1)) >= 2**31 - 256:
        raise ValueError("K, N and row strides must be smaller than 2**31-256")
    if leaves not in (1, 2, 4, 8) or x.size(1) == 0 or x.size(1) % (8 * leaves):
        raise ValueError("leaves must be 1/2/4/8 and each K leaf must be a positive multiple of 8")
    if glu and w.size(1) % 2:
        raise ValueError("gate_up must have an even output width")
    n = w.size(1) // (2 if glu else 1)
    dtype = x.dtype if glu or narrow else torch.float32
    for t in (plan.offsets, plan.tiles, plan.total):
        _check(t, dtype=torch.int32, device=x.device)
    bm = 16 if plan.rows <= plan.experts * 16 else 64 if plan.rows <= plan.experts * 64 else 128
    bound = min(plan.rows, (plan.rows + bm - 1) // bm + plan.experts - 1)
    if plan.offsets.ndim != 1 or plan.tiles.shape != (bound, 3) or plan.total.numel() != 1:
        raise ValueError("malformed grouped plan")
    if bound * ((n + 63) // 64) * (2 if bm == 128 else 1) >= 2**31:
        raise ValueError("GEMM exceeds the CUDA grid limit")
    if probs is None:
        probs = torch.empty(0, dtype=torch.float32, device=x.device)
    else:
        _check(probs, dtype=torch.float32, device=x.device, ndim=1)
        if probs.numel() != x.size(0):
            raise ValueError("probs must have one value per grouped row")
    if out is None:
        out = torch.empty((x.size(0), n), dtype=dtype, device=x.device)
    else:
        _check(out, dtype=dtype, device=x.device, ndim=2)
        if out.shape != (x.size(0), n):
            raise ValueError("incorrect output shape")
        for t in (x, w, probs, plan.offsets, plan.tiles, plan.total):
            if torch._C._overlaps(out, t):
                raise ValueError("output must not alias inputs or metadata")
    extension().gemm(x, w, plan.tiles, plan.total, probs, out, leaves, glu)
    return out


def fc1(x, weight, plan: GroupedPlan, *, out=None):
    """Grouped ``x @ weight[e].T`` -> operand dtype [R,N]."""
    return _gemm(x, weight, plan, 1, False, out=out, narrow=True)


def fc1_glu(x, gate_up, plan: GroupedPlan, probs=None, *, out=None):
    """Grouped FC1 + separately rounded SiLU, up multiply, optional probability multiply."""
    return _gemm(x, gate_up, plan, 1, True, probs=probs, out=out)


def fc2(x, down, plan: GroupedPlan, *, leaves=8, out=None):
    """Grouped FC2 with rounded balanced tree inside the GEMM -> FP32 [R,H]."""
    return _gemm(x, down, plan, leaves, False, out=out)


def route(indices: torch.Tensor, num_experts: int) -> RoutePlan:
    """CUDA int32 [T,topk] expert IDs -> compact, padding-free expert groups.

    IDs must be in [0,E), topk in [1,32]. Row placement uses integer atomics;
    within-expert row order is unspecified. The inverse map and fixed GEMM
    arithmetic make the final result bitwise independent of placement.
    """
    _check(indices, dtype=torch.int32, ndim=2)
    if (
        not isinstance(num_experts, int)
        or not 0 < num_experts < 2**31 - 256
        or not 1 <= indices.size(1) <= 32
        or indices.numel() >= 2**31 - 256
    ):
        raise ValueError("require int32 E > 0, topk in [1,32], and fewer than 2**31-256 routes")
    counts = torch.empty(num_experts, device=indices.device, dtype=torch.int32)
    cursors = torch.empty(num_experts, device=indices.device, dtype=torch.int32)
    offsets = torch.empty(num_experts + 1, device=indices.device, dtype=torch.int32)
    inverse = torch.empty(indices.numel(), device=indices.device, dtype=torch.int32)
    restore = torch.empty_like(indices)
    extension().route(indices, counts, offsets, cursors, inverse, restore)
    return RoutePlan(make_plan(offsets, indices.numel()), indices, inverse, restore)


def dispatch(x, probs, plan: RoutePlan):
    """Gather token activations and route probabilities to expert-major rows."""
    if not isinstance(plan, RoutePlan):
        raise TypeError("expected a RoutePlan from route")
    _check(x, dtype=(torch.bfloat16, torch.float16), ndim=2)
    if x.size(1) >= 2**31 - 256:
        raise ValueError("hidden size must fit int32")
    _check(probs, dtype=torch.float32, device=x.device, ndim=2)
    if probs.shape != plan.indices.shape or x.size(0) != probs.size(0) or x.size(1) % 8:
        raise ValueError("incompatible token/probability shapes or H not divisible by 8")
    _check(plan.inverse, dtype=torch.int32, device=x.device)
    if plan.inverse.shape != (probs.numel(),) or plan.grouped.rows != probs.numel():
        raise ValueError("malformed route plan")
    if x.data_ptr() % 16:
        raise ValueError("dispatch activations must be 16-byte aligned")
    out = torch.empty((plan.grouped.rows, x.size(1)), device=x.device, dtype=x.dtype)
    op = torch.empty(plan.grouped.rows, device=x.device, dtype=torch.float32)
    extension().gather(x, probs, plan.inverse, out, op, probs.size(1))
    return out, op


def combine(rows, plan: RoutePlan, *, dtype=torch.bfloat16):
    """Sum FP32 rows in ascending expert ID (then route slot) order; round once."""
    if not isinstance(plan, RoutePlan):
        raise TypeError("expected a RoutePlan from route")
    _check(rows, dtype=torch.float32, ndim=2)
    if rows.size(1) >= 2**31 - 256 or (rows.size(1) + 255) // 256 > 65535:
        raise ValueError("hidden size exceeds the CUDA grid limit")
    for t in (plan.indices, plan.restore):
        _check(t, dtype=torch.int32, device=rows.device)
    if plan.indices.ndim != 2 or plan.restore.shape != plan.indices.shape or not 1 <= plan.indices.size(1) <= 32:
        raise ValueError("malformed route plan")
    if rows.size(0) != plan.grouped.rows or dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("incorrect row count or output dtype")
    out = torch.empty((plan.indices.size(0), rows.size(1)), dtype=dtype, device=rows.device)
    extension().combine(rows, plan.indices, plan.restore, out)
    return out


def experts(x, gate_up, down, plan: GroupedPlan, probs=None, *, leaves=8):
    """Two grouped CUDA kernels: fused FC1/activation/probabilities, then FC2 tree."""
    h = fc1_glu(x, gate_up, plan, probs)
    return fc2(h, down, plan, leaves=leaves)


def moe(x, gate_up, down, indices, probs, *, leaves=8):
    """Dispatch, expert MLP and ordered combine from already-selected routing IDs."""
    if leaves == 1 and _can_gather_fc1(x, gate_up, indices, probs):
        _check(indices, dtype=torch.int32, ndim=2)
        _check(probs, dtype=torch.float32, device=x.device, ndim=2)
        if indices.numel() <= 256 and gate_up.size(0) <= 256:
            plan, sorted_indices, pp, rows = small_route_plan(indices, probs, gate_up.size(0))
        else:
            rp = route(indices, gate_up.size(0))
            plan = rp.grouped
            sorted_indices = torch.empty(indices.numel(), device=x.device, dtype=torch.int64)
            pp = torch.empty(indices.numel(), device=x.device, dtype=torch.float32)
            rows = torch.empty_like(indices)
            extension().route_metadata(indices, probs, rp.inverse, sorted_indices, pp, rows)
        h = _fc1_from_routes(x, gate_up, plan, sorted_indices, indices, pp)
        root = fc2_compact_root(h, down, plan)
        return combine_compact_roots(root, rows)
    plan = route(indices, gate_up.size(0))
    grouped_x, grouped_p = dispatch(x, probs, plan)
    return combine(experts(grouped_x, gate_up, down, plan.grouped, grouped_p, leaves=leaves), plan, dtype=x.dtype)


def combine_compact_roots(roots, rows):
    """Stable expert-order FP32 fold of compact roots, with one final dtype round.

    This is a local fold. In ETP execution the cross-rank tree must happen first;
    the production owner combine supplies that tree and does not call this helper.
    """
    _check(roots, dtype=(torch.bfloat16, torch.float16), ndim=2)
    _check(rows, dtype=torch.int32, device=roots.device, ndim=2)
    if roots.size(0) != rows.numel() or not 1 <= rows.size(1) <= 32:
        raise ValueError("incompatible roots and combine rows")
    out = torch.empty((rows.size(0), roots.size(1)), device=roots.device, dtype=roots.dtype)
    extension().combine_root_rows(roots, rows, out)
    return out


def small_route_plan(indices, probs, experts):
    _check(indices, dtype=(torch.int32, torch.int64), ndim=2)
    _check(probs, dtype=torch.float32, device=indices.device, ndim=2)
    if (
        indices.shape != probs.shape
        or not 0 < indices.numel() <= 256
        or not 0 < experts <= 256
        or not 1 <= indices.size(1) <= 32
    ):
        raise ValueError("small route plan requires 1..256 routes/experts")
    rows = indices.numel()
    bm = 16 if rows <= experts * 16 else 64 if rows <= experts * 64 else 128
    bound = min(rows, (rows + bm - 1) // bm + experts - 1)
    offsets = torch.empty(experts + 1, device=indices.device, dtype=torch.int32)
    tiles = torch.empty((bound, 3), device=indices.device, dtype=torch.int32)
    total = torch.empty((), device=indices.device, dtype=torch.int32)
    sorted_indices = torch.empty(rows, device=indices.device, dtype=torch.int64)
    pp = torch.empty(rows, device=indices.device, dtype=torch.float32)
    combine_rows = torch.empty_like(indices, dtype=torch.int32)
    extension().small_routes(indices, probs, sorted_indices, pp, combine_rows, offsets, tiles, total)
    return GroupedPlan(offsets, tiles, total, rows), sorted_indices, pp, combine_rows


def _can_gather_fc1(x, weight, indices, probs):
    """Metadata-only predicate; arithmetic and routing membership stay unchanged."""
    return (
        x.is_cuda
        and weight.is_cuda
        and indices.is_cuda
        and probs.is_cuda
        and x.ndim == 2
        and weight.ndim == 3
        and indices.ndim == 2
        and x.dtype in (torch.bfloat16, torch.float16)
        and weight.dtype == x.dtype
        and indices.dtype in (torch.int32, torch.int64)
        and probs.dtype == torch.float32
        and x.device == weight.device == indices.device == probs.device
        and x.size(1) == weight.size(2) == 2048
        and weight.size(1) == 128
        and x.size(0) == indices.size(0)
        and indices.shape == probs.shape
        and 1 <= indices.size(1) <= 32
        and 0 < indices.numel() <= weight.size(0) * 64
        and indices.numel() < 2**31 - 256
        and weight.size(0) > 0
        and x.stride(1) == weight.stride(2) == 1
        and all(
            0 < stride < 2**31 - 256 and stride % 8 == 0 for stride in (x.stride(0), weight.stride(1), weight.stride(0))
        )
        and x.data_ptr() % 16 == weight.data_ptr() % 16 == 0
    )


def _fc1_from_routes(x, weight, plan, sorted_indices, indices, probs, combine_rows=None):
    # The caller checks the shape specialization and builds the device plan.
    # Keep tensor checks here so the standalone path also rejects autograd.
    _check(x, dtype=(torch.bfloat16, torch.float16), ndim=2, contiguous=False)
    _check(weight, dtype=x.dtype, device=x.device, ndim=3, contiguous=False)
    _check(sorted_indices, dtype=torch.int64, device=x.device, ndim=1)
    if probs is None:
        probs = torch.empty(0, dtype=torch.float32, device=x.device)
    else:
        _check(probs, dtype=torch.float32, device=x.device, ndim=1)
        if probs.numel() != plan.rows:
            raise ValueError("routed FC1 probabilities must match the grouped plan")
    if sorted_indices.numel() != plan.rows:
        raise ValueError("routed FC1 metadata must match the grouped plan")
    h = torch.empty((plan.rows, 64), device=x.device, dtype=x.dtype)
    if combine_rows is None:
        combine_rows = torch.empty(0, device=x.device, dtype=torch.int32)
    extension().gathered_fc1(
        x, weight, plan.tiles, plan.total, sorted_indices, indices.contiguous(), probs, h, combine_rows
    )
    return h


def fc2_compact_root(x, down, plan, *, out=None):
    """One rounded local root stored in operand dtype, without widening to FP32.

    Consumers must widen to FP32 before any addition. The public ``fc2`` continues to
    return FP32; this explicit storage option cannot represent multi-leaf roots.
    """
    return _gemm(x, down, plan, 1, False, out=out, narrow=True)
