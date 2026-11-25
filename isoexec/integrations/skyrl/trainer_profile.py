"""Resolve framework-owned persistence and packed capture bounds before workers."""

from __future__ import annotations

from typing import Any


def trainer_capture_bound(cfg: Any) -> int:
    """Bound native packed THD, including its soft token budget and padding.

    The standard batched producer emits at most prompt + generated tokens.
    Custom/environment trajectories need an explicit bound on their *packed*
    microbatch; the runtime export guard still validates every actual input.
    """
    trainer, generator = cfg.trainer, cfg.generator
    if getattr(trainer, "update_ref_every_epoch", False) and (
        getattr(trainer.algorithm, "use_kl_loss", False) or getattr(trainer.algorithm, "use_kl_in_reward", False)
    ):
        raise ValueError("IsoExec does not support trainer.update_ref_every_epoch: it requires native HF export")
    for field in ("ckpt_interval", "hf_save_interval"):
        interval = getattr(trainer, field)
        if type(interval) is not int or interval > 0:
            raise ValueError(
                f"Authored weight mappings do not define native checkpoint/HF export inversion; "
                f"set trainer.{field}<=0 before workers start (got {interval!r})"
            )
    if getattr(trainer, "resume_mode", None) not in (None, "none") or getattr(trainer, "resume_path", None):
        raise ValueError("Native training checkpoint restore is unsupported; set trainer.resume_mode='none' "
                         "and clear trainer.resume_path")
    selected = cfg.isoexec or {}
    explicit = selected.get("trainer_capture_max_tokens")
    if explicit is not None:
        if type(explicit) is not int or explicit < 4:
            raise ValueError("isoexec.trainer_capture_max_tokens must be an integer >=4 for packed capture")
        return explicit
    if not generator.batched:
        raise ValueError("Custom/unbatched trajectories need isoexec.trainer_capture_max_tokens: "
                         "set a finite upper bound on the padded tokens in one train/prescore microbatch")

    def positive(value: Any, name: str) -> int:
        if type(value) is not int or value < 1:
            raise ValueError(f"Trainer capture needs positive integer {name}, got {value!r}")
        return value

    prompt = positive(trainer.max_prompt_length, "trainer.max_prompt_length")
    generated = positive(generator.sampling_params.max_generate_length, "generator.sampling_params.max_generate_length")
    # This adapter uses SkyRL's own THD alignment policy, without importing a
    # model definition or duplicating its FP8/CP packing implementation.
    from skyrl.backends.skyrl_train.distributed.megatron.packing_utils import get_packed_seq_align_size

    budget = trainer.max_tokens_per_microbatch
    if type(budget) is not int:
        raise ValueError("trainer.max_tokens_per_microbatch must be an integer")
    forward_count = positive(trainer.micro_forward_batch_size_per_gpu, "trainer.micro_forward_batch_size_per_gpu")
    train_count = positive(trainer.micro_train_batch_size_per_gpu, "trainer.micro_train_batch_size_per_gpu")
    roles = [("policy", max(train_count, forward_count))]
    if getattr(trainer.algorithm, "use_kl_loss", False) or getattr(trainer.algorithm, "use_kl_in_reward", False):
        roles.append(("ref", forward_count))
    bounds = [4]
    for role, count in roles:
        mc = getattr(trainer, role).megatron_config
        prefix = f"trainer.{role}.megatron_config"
        tp = positive(mc.tensor_model_parallel_size, f"{prefix}.tensor_model_parallel_size")
        cp = positive(mc.context_parallel_size, f"{prefix}.context_parallel_size")
        # validate_model_options rejects FP8 for both native roles; their
        # constructed carriers therefore use the framework's non-FP8 rule.
        alignment = int(get_packed_seq_align_size(tp, cp, fp8_enabled=False))
        longest = ((prompt + generated + alignment - 1) // alignment) * alignment
        # Native Balanced caps the real-token sum except for one oversized
        # sample. Bound each role's global padding; its CP-local input cannot
        # exceed that bound. Reference performs forward microbatches only.
        bounds.append(max(budget * alignment, longest) if budget > 0 else count * longest)
    return max(bounds)


__all__ = ["trainer_capture_bound"]
