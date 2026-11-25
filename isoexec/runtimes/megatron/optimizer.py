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


def finish_native_optimizer_step(modules: Sequence[Any]) -> None:
    """Complete native writes and parameter gathers before the next model consumer."""
    for module in modules:
        for bucket in (*module.bucket_groups, *module.expert_parallel_bucket_groups):
            if bucket.ddp_config.overlap_param_gather:
                bucket.finish_param_sync(skip_next_bucket_dispatch=True)
    torch.cuda.synchronize()


__all__ = ["dtype_uniform_param_group_overrides", "finish_native_optimizer_step", "leaf_fp32_optimizer_shards"]
