"""Rank-ordered gradient sums; transport never performs floating-point arithmetic."""
from __future__ import annotations

import torch
import torch.distributed as dist


def sum_partials(parts):
    """Accumulate rank-ordered partials in FP32 (FP64 for FP64 inputs), then round once."""
    dtype = parts[0].dtype
    total = parts[0].to(torch.float64 if dtype == torch.float64 else torch.float32).clone()
    for part in parts[1:]:
        total.add_(part)
    return total.to(dtype)


@torch.no_grad()
def ordered_sum(tensor, group=None, *, chunk_elements=262144):
    """Sum replicated-parameter or input gradients over ``group`` in ascending group-rank order.

    All ranks must enter, including ranks with no local tokens (zero partials). Chunking bounds
    scratch space by group size rather than parameter size. No averaging or floating-point NCCL
    reduction is involved. This promises repeatability for a fixed layout, not layout invariance.
    """
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    if not tensor.is_floating_point():
        raise TypeError("ordered_sum expects a floating-point gradient")
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return tensor
    world = dist.get_world_size(group)
    flat = tensor.contiguous().view(-1)
    result = torch.empty_like(flat)
    for start in range(0, flat.numel(), chunk_elements):
        local = flat[start:start + chunk_elements]
        parts = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(parts, local, group=group)
        result[start:start + local.numel()].copy_(sum_partials(parts))
    return result.view_as(tensor)
