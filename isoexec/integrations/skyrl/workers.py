"""SkyRL's Megatron policy and reference workers with the model, optimizer and scorer built by IsoExec.

SkyRL keeps the worker lifecycle, ``init_model`` included: process groups, dispatch, micro-batching,
offload/backload, the optimizer window and weight-sync transport. These subclasses fill the places SkyRL
leaves to a subclass: the configs and module (``init_configs``, ``make_megatron_module``), the optimizer
and model wrapper (``_build_optimizer``, ``_build_model_wrapper``), the strategy (``strategy_cls``) and
the weight source (``_build_weight_source``).
"""

import ray

from skyrl.backends.skyrl_train.distributed.megatron.megatron_strategy import MegatronStrategy
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
    MegatronRefWorkerBase,
)

from isoexec.runtimes.megatron import residency

from .aux_batch import install_worker as install_aux_batches
from .aux_batch import worker_microbatches
from .aux_loss import build_wrapper
from .device_ordinal import VisibleDeviceOrdinal
from .megatron import build_module, finish_optimizer_step, init_configs
from .online_prescore import install_worker as install_online_prescore
from .rollout import admit_generator
from .scoring import _find_isoexec_stage
from .weight_source import build_weight_source


class IsoExecMegatronStrategy(MegatronStrategy):
    def optimizer_step(self, optimizer, model, scheduler, name="model", **kwargs):
        result = super().optimizer_step(optimizer, model, scheduler, name=name, **kwargs)
        # Complete overlapped parameter gathers before the next consumer reads the parameters (weight
        # sync, offload, the next forward). The worker clears DDP gradient buffers after this returns.
        finish_optimizer_step(model)
        return result


class _IsoExecWorker:
    """The parts the policy and reference workers share: IsoExec's configs, its module and the offload."""

    _isoexec_section_name: str

    def init_configs(
        self,
        model_path,
        megatron_config,
        model_config_kwargs,
        transformer_config_kwargs,
        bf16=True,
        flash_attn=False,
        lora_config=None,
        enable_mtp=False,
        language_model_only=False,
        bridge_weights_path=None,
    ):
        init_configs(self, model_path, megatron_config)

    def make_megatron_module(
        self,
        wrap_with_ddp=True,
        ddp_config=None,
        lora_config=None,
        lora_type="lora",
        experts_shared_outer_loras=False,
        bf16=True,
    ):
        module = build_module(self, wrap_with_ddp=wrap_with_ddp, ddp_config=ddp_config, lora_config=lora_config, bf16=bf16)
        self._isoexec_stack_baseline = residency.baseline()
        admit_generator(self.cfg, _find_isoexec_stage(module))
        return module

    def offload_to_cpu(self, *args, **kwargs):
        result = super().offload_to_cpu(*args, **kwargs)  # type: ignore[misc]
        # A colocated engine generates next: give back the local memory the trainer's kernels grew.
        residency.release(getattr(self, "_isoexec_stack_baseline", None))
        return result


class IsoExecPolicyWorker(VisibleDeviceOrdinal, _IsoExecWorker, MegatronPolicyWorkerBase):
    strategy_cls = IsoExecMegatronStrategy
    _isoexec_section_name = "policy"

    def init_model(self, model_path, num_training_steps=1e9):
        super().init_model(model_path, num_training_steps)
        if not self.cfg.policy.inference_only_init:
            install_online_prescore(self)
            install_aux_batches(self)

    def forward_backward(self, *args, **kwargs):
        with worker_microbatches(self):
            return super().forward_backward(*args, **kwargs)

    def _build_optimizer(self):
        return self._isoexec_optimizer

    def _build_model_wrapper(self):
        return build_wrapper(
            megatron_config=self.cfg.policy.megatron_config,
            training=not self.cfg.policy.inference_only_init,
            config=self.cfg,
            actor_module=self.actor_module,
            actor_optimizer=self.optimizer,
            policy_loss_fn=self.policy_loss_fn,
            is_vlm=False,
        )

    # SkyRL infers only its built-in source types; dispatch consumes the shared
    # WeightSource protocol, which IsoExecWeightSource implements as well.
    def _build_weight_source(self, dtype, backend):  # pyright: ignore[reportIncompatibleMethodOverride]
        # Under the full comparison a colocated sync sends every Parameter twice for the engine to compare;
        # configure_isoexec_vllm_engine turns the engine's side on for the same runs.
        verify = self.cfg._isoexec_config.full_distribution_comparison and backend == "ipc"
        return build_weight_source(self.actor_module, dtype, backend, verify=verify)


class IsoExecRefWorker(VisibleDeviceOrdinal, _IsoExecWorker, MegatronRefWorkerBase):
    _isoexec_section_name = "ref"

    def _build_model_wrapper(self):
        return build_wrapper(
            megatron_config=self.cfg.ref.megatron_config,
            training=False,
            config=self.cfg,
            actor_module=self.actor_module,
            is_vlm=False,
        )


PolicyWorker = ray.remote(num_gpus=1)(IsoExecPolicyWorker)
RefWorker = ray.remote(num_gpus=1)(IsoExecRefWorker)
