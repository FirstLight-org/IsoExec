"""Native Megatron options SkyRL may set for an IsoExec model."""

import pytest

pytest.importorskip("skyrl.train.config")
pytest.importorskip("megatron.core")

from skyrl.train.config import SkyRLTrainConfig

from isoexec.integrations.skyrl.megatron import validate_model_options


def test_disk_optimizer_offload_allowed_with_fp32_grad_buffers():
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.megatron_config.optimizer_offload_to_disk.dir = "/tmp/unused-optimizer-state"
    cfg.trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32 = True
    validate_model_options(cfg.trainer.policy.megatron_config)


def test_production_precision_aware_cpu_adam_allowed_for_training():
    cfg = SkyRLTrainConfig.from_cli_overrides(
        {
            "trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload": True,
            "trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction": 1.0,
            "trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d": True,
            "trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer": True,
        }
    )
    validate_model_options(cfg.trainer.policy.megatron_config)


def test_optimizer_support_does_not_admit_mtp():
    cfg = SkyRLTrainConfig.from_cli_overrides({"trainer.policy.megatron_config.mtp_num_layers": 1})
    with pytest.raises(ValueError, match="mtp_num_layers"):
        validate_model_options(cfg.trainer.policy.megatron_config)


def test_sequence_parallel_is_a_request_for_isoexecs_own_layout():
    base = "trainer.policy.megatron_config."
    cfg = SkyRLTrainConfig.from_cli_overrides(
        {base + "tensor_model_parallel_size": 2, base + "transformer_config_kwargs.sequence_parallel": True}
    )
    validate_model_options(cfg.trainer.policy.megatron_config)
    off = SkyRLTrainConfig.from_cli_overrides({base + "transformer_config_kwargs.sequence_parallel": False})
    validate_model_options(off.trainer.policy.megatron_config)
    with pytest.raises(ValueError, match="tensor_model_parallel_size >= 2"):
        validate_model_options(
            SkyRLTrainConfig.from_cli_overrides(
                {base + "transformer_config_kwargs.sequence_parallel": True}
            ).trainer.policy.megatron_config
        )
    with pytest.raises(ValueError, match="must be a bool"):
        validate_model_options(
            SkyRLTrainConfig.from_cli_overrides(
                {base + "tensor_model_parallel_size": 2, base + "transformer_config_kwargs.sequence_parallel": "yes"}
            ).trainer.policy.megatron_config
        )
