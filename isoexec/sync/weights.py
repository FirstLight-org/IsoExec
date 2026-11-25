"""Gather TP shards and EP-held experts into full logical tensors."""

from __future__ import annotations

from typing import Sequence

import torch

from isoexec.runtimes.parallel import LocalSpec, assemble_full, units_replicate
from isoexec.schema import ShardError


def _gather_tp_to_full(shard: torch.Tensor, dim: int, stride: int, tp_group) -> torch.Tensor:
    """All-gather a TP shard across the group and rebuild the full tensor along ``dim``.

    Reconstruction matches Megatron's own sharding (``_initialize_affine_weight_*``): the full
    dim is split into ``world*stride`` equal pieces and rank r owns pieces ``[r, r+world, ...]``
    (strided). So master piece k came from rank ``k % world``'s ``(k // world)``-th sub-piece.
    stride==1 (the common case) collapses to a plain concat in rank order.
    """
    import torch.distributed as dist

    world = dist.get_world_size(tp_group)
    if world == 1:
        return shard
    shard = shard.contiguous()
    gathered = [torch.empty_like(shard) for _ in range(world)]
    dist.all_gather(gathered, shard, group=tp_group)
    if stride == 1:
        return torch.cat(gathered, dim=dim)
    subs = [torch.chunk(gathered[r], stride, dim=dim) for r in range(world)]  # [world][stride]
    pieces = [None] * (world * stride)
    for r in range(world):
        for j in range(stride):
            pieces[r + j * world] = subs[r][j]
    return torch.cat(pieces, dim=dim)


def gather_full(local: torch.Tensor, spec: LocalSpec, group) -> torch.Tensor:
    """The FULL logical tensor from this rank's shard, identical on every rank of ``group``."""
    if not spec.sharded:
        return local
    if group is None:
        raise ShardError(f"{spec.name}: gathering a tp={spec.tp} shard needs the Placement's group handle")
    import torch.distributed as dist

    world = dist.get_world_size(group)
    if world != spec.tp:
        raise ShardError(f"{spec.name}: group world {world} != the shard's tp {spec.tp}")
    local = local.contiguous()
    if spec.segments is None and not units_replicate(spec.tp, spec.units, spec.policy or "replicate"):
        assert spec.dim is not None
        return _gather_tp_to_full(local, spec.dim, spec.stride, group)
    shards = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(shards, local, group=group)
    return assemble_full(dict(enumerate(shards)), spec, spec.tp)


def gather_expert_stack(held: Sequence[torch.Tensor], group) -> torch.Tensor:
    """All-gather each EP rank's contiguous held experts into one stack in global expert order."""
    import torch.distributed as dist

    local = torch.stack(held)
    out = torch.empty((dist.get_world_size(group) * local.shape[0], *local.shape[1:]),
                      dtype=local.dtype, device=local.device)
    dist.all_gather_into_tensor(out, local.contiguous(), group=group)
    return out


__all__ = ["gather_expert_stack", "gather_full"]
