"""Preserve gradient dtypes at the pinned Megatron/TE multi-tensor boundary.

TE selects one element type per multi-tensor launch. Megatron 0.19.0 passes bf16 decoupled
gradients and fp32 router gradients together, producing invalid norms and corrupting clipping.
Grouping native calls avoids full gradient casts; Megatron still owns the single global norm
reduction, clipping coefficient, duplicate filtering and optimizer step.

TE also takes one NVTETensor handle per tensor per list from a fixed pool (20 MiB, 20321 handles in
TE 2.19), and a MoE model registers every expert as its own Parameter: Qwen3.5-35B-A3B holds 30720
expert gradients, 15360 per rank at EP2, so one clipping launch over two lists exceeds the pool. Each
native launch therefore takes at most ``HANDLES_PER_LAUNCH`` handles, in a fixed order.
"""

from collections.abc import Callable

import torch

from ...checks import Refusal

_INSTALLED: Callable | None = None

#: NVTETensor handles one native launch may hold (tensors x lists), well inside TE's default pool.
HANDLES_PER_LAUNCH = 8192


def _make_applier(
    native: Callable, l2_norm: Callable, scales: tuple[Callable, ...], *, handles: int = HANDLES_PER_LAUNCH
) -> Callable:
    def apply(op, overflow, tensor_lists: list[list[torch.Tensor]], *args):
        per_launch = max(1, handles // max(1, len(tensor_lists)))
        groups: dict[tuple, list[list[torch.Tensor]]] = {}
        for tensors in zip(*tensor_lists, strict=True):
            key = tuple((t.dtype, t.device) for t in tensors)
            group = groups.setdefault(key, [[] for _ in tensors])
            for destination, tensor in zip(group, tensors, strict=True):
                destination.append(tensor)
        if len(groups) <= 1 and (not tensor_lists or len(tensor_lists[0]) <= per_launch):
            return native(op, overflow, tensor_lists, *args)
        # Dtype groups, each cut into launches of at most ``per_launch`` tensors, in list order.
        launches = [[lists[i : i + per_launch] for lists in group]
                    for group in groups.values() for i in range(0, len(group[0]), per_launch)]
        if op is l2_norm:
            if args != (False,):
                raise Refusal(
                    "multi_tensor_applier.per_tensor: mixed-dtype norm requires per_tensor=False"
                )
            results = [native(op, overflow, lists, *args) for lists in launches]
            # Only scalar partial norms are allocated. Input gradient tensors remain aliases of
            # their original DDP buffers, and the caller performs the distributed reduction once.
            total = torch.zeros_like(
                results[0][0], dtype=torch.float32, device=results[0][0].device
            )
            for norm, _ in results:
                total.addcmul_(norm, norm)
            return total.sqrt_(), results[0][1]
        if op not in scales:
            raise Refusal(
                f"multi_tensor_applier.op: unsupported mixed-dtype or oversized operation {op}"
            )
        for lists in launches:
            native(op, overflow, lists, *args)
        return None

    return apply


def install_mixed_dtype_grad_kernels() -> None:
    """Adapt only Megatron's gradient-statistics calls; leave TE's global applier unchanged."""
    global _INSTALLED
    from megatron.core.optimizer import clip_grads

    if _INSTALLED is not None:
        if clip_grads.multi_tensor_applier is not _INSTALLED:
            raise Refusal(
                "clip_grads.multi_tensor_applier: installed dtype grouping was replaced"
            )
        return
    _INSTALLED = _make_applier(
        clip_grads.multi_tensor_applier,
        clip_grads.l2_norm_impl,
        (clip_grads.multi_tensor_scale_impl, clip_grads.multi_tensor_scale_tensor_impl),
    )
    clip_grads.multi_tensor_applier = _INSTALLED
