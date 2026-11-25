"""Trainer scoring stages: vocabulary GEMM widened to fp32, identity row selection and log-softmax."""

from __future__ import annotations

import torch

from isoexec.ops.logprobs.temperature import temperature_divide
from isoexec.ops.mm.trainer import mm_trainer


def head_bf16(h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute vocabulary-head logits using the trainer's BF16 GEMM."""
    return mm_trainer(h, w)


def log_softmax_rowinv(logits_fp32: torch.Tensor, targets: torch.Tensor | None, temperature: float, *, group=None,
                       with_entropy: bool = False, _sampled_logprobs=None, _sampled_entropy=None):
    """Compute sampled-token logprobs using the trainer's fixed log-softmax recipe.

    ``with_entropy`` (no grad only) returns ``(logprobs, entropy)`` where the provider emits the row
    entropy of ``logits / T`` from the same leaf pass, or ``(logprobs, None)`` where it does not; the
    logprobs are the same bits either way.
    """
    if _sampled_logprobs is None:
        from isoexec.ops.logprobs import sampled_logprobs
    else:
        sampled_logprobs = _sampled_logprobs

    if targets is None:
        raise NotImplementedError("the trainer chain is sampled-only; full-vocab logprobs are not bound")
    logits_fp32 = temperature_divide(logits_fp32, temperature)
    lg = logits_fp32.unsqueeze(0).contiguous()
    tg = targets.unsqueeze(0).contiguous()
    V = lg.shape[-1]
    if group is None:
        world, rank = 1, 0
    else:
        import torch.distributed as dist

        world, rank = dist.get_world_size(group), dist.get_rank(group)
    vs, ve = rank * V, (rank + 1) * V

    def reference() -> torch.Tensor:
        full = lg
        if world > 1:
            import torch.distributed as dist

            parts = [torch.empty_like(lg) for _ in range(world)]
            dist.all_gather(parts, lg, group=group)
            full = torch.cat(parts, dim=-1)
        lp = torch.log_softmax(full, dim=-1)
        return torch.gather(lp, -1, tg.unsqueeze(-1)).squeeze(-1)

    if with_entropy and _sampled_entropy is not None and not (torch.is_grad_enabled() and lg.requires_grad):
        pair = _sampled_entropy(
            lg, tg, vocab_start_index=vs, vocab_end_index=ve, group=group, src_dtype=torch.bfloat16, reference=reference
        )
        if pair is not None:
            return pair[0].squeeze(0), pair[1].squeeze(0)
    out = sampled_logprobs(
        lg, tg, vocab_start_index=vs, vocab_end_index=ve, group=group, src_dtype=torch.bfloat16, reference=reference
    )
    if out is None:
        raise RuntimeError(
            "selected logprobs provider declined the layout; the trainer chain would have fallen back -- refusing "
            "silently different bits"
        )
    return (out.squeeze(0), None) if with_entropy else out.squeeze(0)


def lm_head_slice_identity(h: torch.Tensor) -> torch.Tensor:
    """``logprobs.lm_head_slice:identity``: the trainer scores every row, so the slice is the identity."""
    return h


__all__ = ["head_bf16", "lm_head_slice_identity", "log_softmax_rowinv"]
