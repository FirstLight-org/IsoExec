"""Native Megatron optimizer construction and step completion for the registered model."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from megatron.core.optimizer import OptimizerConfig


def dtype_uniform_param_group_overrides(optimizer_config: OptimizerConfig) -> dict:
    """Keep FP32 parameters in separate optimizer groups without changing hyperparameters."""
    from megatron.core.optimizer import get_standard_config_overrides
    from megatron.core.optimizer.optimizer_config import ParamKey, ParamPredicate

    overrides: dict[Any, Any] = dict(get_standard_config_overrides(optimizer_config))
    fp32_key = ParamKey(predicate=ParamPredicate(name="fp32_held", fn=lambda p: p.dtype == torch.float32))
    # Pinned MCore retains lr_mult as a backwards-compatible group key while
    # its TypedDict describes only newer scheduler fields. Preserve the exact
    # grouping/value behavior through the native mapping interface.
    overrides[fp32_key] = {"lr_mult": 1.0}
    return overrides


@contextlib.contextmanager
def leaf_fp32_optimizer_shards(enabled: bool):
    """Keep optimizer FP32 shards as leaf tensors during optimizer construction."""
    if not enabled:
        yield []
        return
    from megatron.core import tensor_parallel
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    native: Any = DistributedOptimizer
    original = native.__dict__["_build_model_and_main_param_groups"]
    repaired: list[tuple[int, ...]] = []

    def patched(cls, gbuf_ranges, param_gbuf_map, opt_group_ranges, config):
        out = original.__func__(cls, gbuf_ranges, param_gbuf_map, opt_group_ranges, config)
        shard_fp32_groups = out[3]
        swap: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for group in shard_fp32_groups:
            for i, shard in enumerate(group):
                if shard is None or shard.is_leaf:
                    continue
                leaf = shard.detach()
                tensor_parallel.copy_tensor_model_parallel_attributes(leaf, shard)
                if hasattr(shard, "shared"):
                    leaf.shared = shard.shared
                swap[id(shard)] = (shard, leaf)
                group[i] = leaf
                repaired.append(tuple(shard.shape))
        for group_range in opt_group_ranges:
            params = group_range["orig_group"]["params"]
            for i, p in enumerate(params):
                hit = swap.get(id(p))
                if hit is not None and hit[0] is p:
                    params[i] = hit[1]
        return out

    native._build_model_and_main_param_groups = classmethod(patched)
    try:
        yield repaired
    finally:
        native._build_model_and_main_param_groups = original


def require_uniform_cpu_capability(group=None) -> str:
    """Refuse CPU-offloaded Adam across ranks whose ATen CPU kernels differ.

    Megatron's offloaded Adam is ``torch.optim.AdamW(fused=True)`` on the host (``HybridDeviceOptimizer``). Its
    bits depend on the ATen CPU capability (measured: DEFAULT differs from AVX2 / AVX512, which agree; thread
    count does not matter). Parameters every rank holds and updates itself (the EP-summed router and shared
    expert at EP > 1, ``mark_ep_summed``) stay equal across ranks only if every rank runs the same kernels.
    Collective over ``group`` (default: the world); pin ``ATEN_CPU_CAPABILITY`` on every node to fix a mix."""
    import torch.distributed as dist

    mine = str(torch.backends.cpu.get_cpu_capability())
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return mine
    every: list = [None] * dist.get_world_size(group)
    dist.all_gather_object(every, mine, group=group)
    if len(set(every)) != 1:
        ranks = {cap: [r for r, c in enumerate(every) if c == cap] for cap in sorted(set(every))}
        raise RuntimeError(f"CPU_ADAM_CAPABILITY_MIXED: optimizer_cpu_offload runs Adam on the host and its bits "
                           f"depend on the ATen CPU capability; ranks differ: {ranks} (set ATEN_CPU_CAPABILITY "
                           "to one value on every node)")
    return mine


def finish_native_optimizer_step(modules: Sequence[Any]) -> None:
    """Complete native writes and parameter gathers before the next model consumer.

    The optimizer writes parameters through the DDP buffers, which leaves their
    version counters unchanged; components that cache weight-derived copies
    (``refresh_derived_weights``) re-derive them in place here.
    """
    for module in modules:
        for bucket in (*module.bucket_groups, *module.expert_parallel_bucket_groups):
            if bucket.ddp_config.overlap_param_gather:
                bucket.finish_param_sync(skip_next_bucket_dispatch=True)
    torch.cuda.synchronize()
    refresh_derived_weights(modules)


def refresh_derived_weights(modules: Sequence[Any]) -> int:
    """Re-derive every weight-derived copy the components of ``modules`` cache, in place. Returns the count."""
    from isoexec.nn.component import Component

    count = 0
    for module in modules:
        for component in module.modules():
            refresh = getattr(component, "refresh_derived_weights", None)
            if isinstance(component, Component) and refresh is not None:
                count += refresh()
    if count:
        torch.cuda.synchronize()
    return count


__all__ = ["dtype_uniform_param_group_overrides", "finish_native_optimizer_step", "leaf_fp32_optimizer_shards",
           "refresh_derived_weights", "require_uniform_cpu_capability"]
