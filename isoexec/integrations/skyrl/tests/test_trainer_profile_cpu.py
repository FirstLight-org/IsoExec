"""Actual SkyRL packing rules and early trainer profile admission; no native GPU proof."""

from typing import Any
from types import SimpleNamespace

import pytest

from isoexec.integrations.skyrl.config import resolve
from isoexec.integrations.skyrl.tests.test_config_cpu import config
from isoexec.integrations.skyrl.trainer_profile import trainer_capture_bound
from isoexec.compiler.source import ModelSource


def _profile_config() -> Any:
    cfg = config()
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.ref.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.ckpt_interval = cfg.trainer.hf_save_interval = -1
    cfg.trainer.resume_mode = "none"
    cfg.trainer.max_prompt_length = 512
    cfg.trainer.micro_train_batch_size_per_gpu = cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.max_tokens_per_microbatch = -1
    cfg.generator.batched = True
    cfg.generator.sampling_params.max_generate_length = 1024
    return cfg


def test_actual_skyrl_packer_bound_covers_sample_soft_budget_and_padding() -> None:
    packing = pytest.importorskip("skyrl.backends.skyrl_train.distributed.megatron.packing_utils")
    balanced = pytest.importorskip("skyrl.train.dataset.bin_packing").make_seq_packer
    cfg = _profile_config()
    trainer = cfg.trainer
    trainer.policy.megatron_config.tensor_model_parallel_size = 2
    trainer.policy.megatron_config.context_parallel_size = 2
    trainer.max_prompt_length = 6001
    cfg.generator.sampling_params.max_generate_length = 1000
    trainer.micro_train_batch_size_per_gpu = 2
    trainer.micro_forward_batch_size_per_gpu = 3
    align = packing.get_packed_seq_align_size(2, 2)
    padded_longest = ((7001 + align - 1) // align) * align
    assert trainer_capture_bound(cfg) == 3 * padded_longest > 4096
    trainer.max_tokens_per_microbatch = 17
    lengths = [7001, 1, 1, 1, 8, 9, 3, 5]
    plan = balanced("balanced", bin_capacity=17).pack(lengths)
    packed = [sum(((lengths[i] + align - 1) // align) * align for i in group) for group in plan]
    assert max(packed) <= trainer_capture_bound(cfg) == padded_longest
    trainer.max_tokens_per_microbatch = 10000
    assert trainer_capture_bound(cfg) == 10000 * align


def test_driver_resolves_bound_and_refuses_checkpoint_without_metadata_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("skyrl.backends.skyrl_train.distributed.megatron.packing_utils")
    cfg = _profile_config()
    monkeypatch.setattr("isoexec.integrations.skyrl.config.dependencies.check", lambda **kwargs: [])
    monkeypatch.setattr("isoexec.integrations.skyrl.config.prepare_checkpoints", lambda _trainer: None)
    source = ModelSource("isoexec.models.qwen2_moe")
    monkeypatch.setattr("isoexec.integrations.skyrl.config.source_config",
                        lambda *_args: (source, {"model_type": "qwen2_moe", "architectures": ["Qwen2MoeForCausalLM"]}))
    result = resolve(cfg)
    assert result is not None and result.trainer_capture_max_tokens == 1536
    assert cfg.trainer._isoexec_config is result
    for field, value in (("ckpt_interval", 10), ("hf_save_interval", 1), ("resume_mode", "latest")):
        original = getattr(cfg.trainer, field)
        setattr(cfg.trainer, field, value)
        with pytest.raises(ValueError, match="checkpoint|export|restore"):
            resolve(cfg)
        setattr(cfg.trainer, field, original)
    cfg.generator.batched = False
    with pytest.raises(ValueError, match="isoexec.trainer_capture_max_tokens"):
        resolve(cfg)
    cfg.isoexec["trainer_capture_max_tokens"] = 8192
    result = resolve(cfg)
    assert result is not None and result.trainer_capture_max_tokens == 8192
    for bad in (True, -1, 3, "8192"):
        cfg.isoexec["trainer_capture_max_tokens"] = bad
        with pytest.raises(ValueError, match="integer >=4"):
            resolve(cfg)


def test_actual_framework_config_exposes_bound_and_default_checkpoint_refuses() -> None:
    from skyrl.train.config import SkyRLTrainConfig

    cfg = SkyRLTrainConfig()
    assert "isoexec_capture_max_tokens" not in cfg.trainer.__dataclass_fields__
    with pytest.raises(ValueError, match="ckpt_interval.*10"):
        trainer_capture_bound(cfg)
    cfg.trainer.ckpt_interval = -1
    cfg.trainer.resume_mode = "none"
    cfg.isoexec = {"trainer_capture_max_tokens": 12000}
    assert trainer_capture_bound(cfg) == 12000


@pytest.mark.parametrize("kl_flag", ["use_kl_loss", "use_kl_in_reward"])
def test_reference_refresh_refuses_implicit_hf_export(kl_flag):
    cfg = _profile_config()
    cfg.trainer.update_ref_every_epoch = True
    setattr(cfg.trainer.algorithm, kl_flag, True)
    with pytest.raises(ValueError, match="update_ref_every_epoch.*HF export"):
        trainer_capture_bound(cfg)
    setattr(cfg.trainer.algorithm, kl_flag, False)
    cfg.isoexec = {"trainer_capture_max_tokens": 4096}
    assert trainer_capture_bound(cfg) == 4096


@pytest.mark.parametrize("kl_flag", ["use_kl_loss", "use_kl_in_reward"])
def test_reference_padding_and_forward_batch_are_covered_only_when_enabled(kl_flag: str) -> None:
    packing = pytest.importorskip("skyrl.backends.skyrl_train.distributed.megatron.packing_utils")
    cfg = _profile_config()
    trainer = cfg.trainer
    trainer.max_prompt_length = 5
    cfg.generator.sampling_params.max_generate_length = 4
    trainer.ref = SimpleNamespace(megatron_config=SimpleNamespace(
        tensor_model_parallel_size=2, context_parallel_size=1))
    assert trainer_capture_bound(cfg) == 9
    setattr(trainer.algorithm, kl_flag, True)
    assert trainer_capture_bound(cfg) == 10
    trainer.micro_train_batch_size_per_gpu = 4
    trainer.micro_forward_batch_size_per_gpu = 3
    # Policy's four 9-token sequences dominate reference's three 10-token
    # sequences; reference must not inherit the training microbatch count.
    assert trainer_capture_bound(cfg) == 36
    trainer.ref.megatron_config.context_parallel_size = 2
    alignment = packing.get_packed_seq_align_size(2, 2)
    ref_global = 3 * ((9 + alignment - 1) // alignment) * alignment
    assert trainer_capture_bound(cfg) == ref_global == 48
    trainer.max_tokens_per_microbatch = 17
    assert trainer_capture_bound(cfg) == 17 * alignment
    cfg.isoexec = {"trainer_capture_max_tokens": 123}
    assert trainer_capture_bound(cfg) == 123
