"""SkyRL's Megatron model wrapper, scoring an IsoExec model with the model's own LM-head logprobs.

SkyRL keeps micro-batching, packing, the pipeline schedule, losses and metrics. Two numerics must be
the engine's for the trainer's logprobs to equal the rollout's bit for bit, and SkyRL's wrapper leaves
each to a subclass:

- the temperature (``_scale_logits``): SkyRL divides logits by a Python scalar, which CUDA lowers to a
  reciprocal multiply; the engine divides by an FP32 tensor. Entropy and every loss then see the
  tempered logits;
- the sampled-token logprobs (``_target_logprob_fn``): the model's registered ``LMHead.log_probs``, the
  producer the engine reports from, takes the place of SkyRL's distributed log-softmax. SkyRL still
  finds the targets in the packed batch and scatters the results back.

With the full comparison on, SkyRL hands the policy forward the rollout's full-vocabulary rows and
calls ``_verify_full_logprobs`` with that forward's logits; every trainable row is compared bit for bit.
"""

from functools import partial

import torch
from megatron.core.utils import get_model_config
from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import MegatronModelWrapper

from isoexec.ops.logprobs.temperature import temperature_divide
from isoexec.runtimes.megatron.forward_count import require_equal_forward_count

from .rollout import microbatch_requests
from .scoring import _find_isoexec_stage, lm_head_log_probs, target_logprobs, verify_full_distribution


class IsoExecModelWrapper(MegatronModelWrapper):
    """Keep SkyRL's batching and scheduling while supplying IsoExec's scorer and temperature."""

    def __init__(self, config, actor_module, **kwargs):
        model_config = get_model_config(actor_module[0])
        # SkyRL replaces the model's finalizer with its deferrer; replay the one IsoExec installed
        # (it also synchronizes cross-PP tied parameters) at the optimizer boundary.
        self._isoexec_finalize_model_grads = model_config.finalize_model_grads_func
        super().__init__(config=config, actor_module=actor_module, **kwargs)
        self._isoexec_log_probs = lm_head_log_probs(self)

    def _stage(self):
        modules = self.actor_module if isinstance(self.actor_module, (list, tuple)) else [self.actor_module]
        return modules, _find_isoexec_stage(modules)

    def _rollout_requests(self, micro_batches):
        modules, stage = self._stage()
        return microbatch_requests(stage, micro_batches, chunks=len(modules))

    def _agree_forward_count(self, micro_batches, site):
        # A rank with a different number of forwards would issue a different EP collective sequence: refuse by
        # name, agreed over the EP group, before the schedule's first collective.
        placement = getattr(self._stage()[1], "placement", None)
        if placement is not None and placement.ep.size > 1:
            require_equal_forward_count(len(micro_batches), placement.ep.group, site=site)

    def forward(self, micro_batches, *args, **kwargs):
        self._agree_forward_count(micro_batches, "forward")
        # Stages that read the engine requests behind each sequence (V4.1's decoder segments) get them per forward.
        with self._rollout_requests(micro_batches):
            return super().forward(micro_batches, *args, **kwargs)

    def forward_backward_mini_batch(self, micro_batches, *args, **kwargs):
        self._agree_forward_count(micro_batches, "forward_backward_mini_batch")
        with self._rollout_requests(micro_batches):
            return super().forward_backward_mini_batch(micro_batches, *args, **kwargs)

    def run_pending_grad_sync(self) -> None:
        if self._isoexec_finalize_model_grads is None:
            return super().run_pending_grad_sync()
        # Replay the model's original finalizer at the native optimizer boundary,
        # after all accumulated microbatches, including ranks with no local batch.
        pending = self._pending_grad_sync
        self._pending_grad_sync = None
        self._isoexec_finalize_model_grads(self.actor_module, pending["num_tokens"] if pending else None)

    def _scale_logits(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature == 1.0:
            return logits
        return temperature_divide(logits.float(), temperature)

    def _target_logprob_fn(self):
        if self._isoexec_log_probs is None:
            return None
        return partial(target_logprobs, self._isoexec_log_probs, chunk_size=self.cfg.logprobs_chunk_size)

    def _verify_full_logprobs(self, logits: torch.Tensor, data: dict) -> None:
        verify_full_distribution(self, logits, data)
