"""Host adapters for the SM100 logprobs kernels, selected by `ops/_dispatch` (arch `sm100`)."""

from __future__ import annotations

import torch
import torch.distributed as dist

from isoexec.ops.logprobs.logprobs_sm100 import ops

TEMPERATURE_FUSION_POLICY = "rowinv_temperature_v1:full_rows>=128|vocab<=2048;otherwise=div_then_rowinv"

_S = {"full_calls": 0, "full_rows": 0, "sampled_calls": 0, "tokens": 0, "declined": 0, "decline_reason": ""}


def stats() -> dict:
    return dict(_S)


def _decline(reason: str):
    _S["declined"] += 1
    _S["decline_reason"] = reason
    return None


def full_logprobs(logits, *, src_dtype=None, reference=None, temperature=None):
    """Full ``[N, V]`` fp32 rows, or ``None`` to decline."""
    if logits.ndim != 2 or not logits.is_cuda:
        return _decline("layout/device")
    if logits.dtype not in (torch.float32, torch.bfloat16) or not ops.supports(logits.shape[-1]):
        return _decline(f"dtype {logits.dtype} / vocab {logits.shape[-1]}")
    _S["full_calls"] += 1
    _S["full_rows"] += logits.shape[0]
    with torch.no_grad():
        return ops.full_logprobs(logits, temperature=temperature)


def sampled_logprobs(logits, target, *, vocab_start_index, vocab_end_index, group, src_dtype=None, reference=None):
    """Sampled logprobs over a TP vocab shard + group, differentiable; ``None`` declines."""
    world = 1 if group is None else dist.get_world_size(group=group)
    V_local = logits.shape[-1]
    if not logits.is_cuda or logits.dtype not in (torch.float32, torch.bfloat16):
        return _decline("device/dtype")
    if not ops.supports(V_local * world, world) or vocab_end_index - vocab_start_index != V_local:
        return _decline(f"vocab {V_local}x{world} / partition")
    _S["sampled_calls"] += 1
    _S["tokens"] += target.numel()
    return ops.sampled_logprobs(logits, target, vocab_start_index=vocab_start_index, group=group)


@torch.no_grad()
def full_logprobs_for_check(logits_fp32):
    """Observe already-tempered rows using the production recipe, without counting a serve."""
    if logits_fp32.ndim != 2:
        raise RuntimeError("logprobs check observer requires a non-empty [N, V_local] matrix")
    return full_logprobs_for_check_sharded(
        logits_fp32, vocab_start_index=0, vocab_end_index=logits_fp32.shape[-1], group=None
    )


@torch.no_grad()
def full_logprobs_for_check_sharded(
    logits_fp32, *, vocab_start_index, vocab_end_index, group
):
    """Observe this TP rank's full-distribution slice without mutating adapter statistics."""
    if logits_fp32.ndim != 2 or logits_fp32.numel() == 0:
        raise RuntimeError("logprobs check observer requires a non-empty [N, V_local] matrix")
    if logits_fp32.dtype is not torch.float32 or not logits_fp32.is_contiguous():
        raise RuntimeError("logprobs check observer requires already-tempered contiguous fp32 logits")
    world = 1 if group is None else dist.get_world_size(group=group)
    rank = 0 if group is None else dist.get_rank(group=group)
    width = logits_fp32.shape[1]
    if (vocab_start_index, vocab_end_index) != (rank * width, (rank + 1) * width):
        raise RuntimeError("logprobs check observer vocabulary bounds do not match the TP rank")
    if not ops.supports(width * world, world):
        raise RuntimeError(f"logprobs check observer unsupported vocab {width * world} / world {world}")
    if world == 1:
        return ops.full_logprobs(logits_fp32)
    return ops.full_logprobs_sharded(logits_fp32, group=group)


select_recipe = ops.select_recipe
default_recipe = ops.default_recipe


def __getattr__(name: str):
    """``IMPL``: the registered log-softmax identity of the recipe currently pinned in ``ops``."""
    if name == "IMPL":
        return ops.impl_id()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
