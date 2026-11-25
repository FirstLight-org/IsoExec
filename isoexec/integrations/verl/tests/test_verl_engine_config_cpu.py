"""CPU contracts for the IsoExec verl engine's configuration admission and verl-facing attributes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _engine_config(**overrides):
    values = dict(context_parallel_size=1, virtual_pipeline_model_parallel_size=None, dynamic_context_parallel=False,
                  use_fused_kernels=False, param_offload=False, optimizer_offload=False, use_dist_checkpointing=False,
                  use_remove_padding=True, use_megatron_fsdp=False, router_replay=SimpleNamespace(mode="disabled"),
                  pad_to_length=False, override_transformer_config={}, expert_tensor_parallel_size=1,
                  expert_model_parallel_size=4, tensor_model_parallel_size=2, use_dynamic_bsz=True,
                  max_token_len_per_gpu=4096, infer_max_token_len_per_gpu=None, dtype="bfloat16")
    values.update(overrides)
    return SimpleNamespace(**values)


def _model_config(**overrides):
    values = dict(lora_rank=0, lora={}, mtp=SimpleNamespace(enable=False), model_type="language_model",
                  share_embeddings_and_output_weights=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_engine_refuses_unimplemented_megatron_options():
    from isoexec.integrations.verl.engine import validate_engine_config

    validate_engine_config(_engine_config(), _model_config())
    for cp in (2, 4, 8):  # contiguous context parallelism is admitted at any degree
        validate_engine_config(_engine_config(context_parallel_size=cp), _model_config())
    for change in ({"context_parallel_size": 0}, {"dynamic_context_parallel": True}, {"use_fused_kernels": True},
                   {"expert_tensor_parallel_size": 2}, {"use_remove_padding": False}):
        with pytest.raises(ValueError):
            validate_engine_config(_engine_config(**change), _model_config())
    with pytest.raises(ValueError, match="LoRA"):
        validate_engine_config(_engine_config(), _model_config(lora_rank=8))


def test_capture_bound_covers_dynamic_and_static_batching_and_tp_padding():
    from isoexec.integrations.verl.engine import capture_bound

    assert capture_bound(_engine_config(infer_max_token_len_per_gpu=6144)) == 6144 * 2
    static = dict(use_dynamic_bsz=False, max_token_len_per_gpu=1536, micro_batch_size_per_gpu=4,
                  infer_max_token_len_per_gpu=1536, infer_micro_batch_size_per_gpu=8)
    assert capture_bound(_engine_config(**static)) == 8 * 1536 * 2
    with pytest.raises(ValueError):
        capture_bound(_engine_config(use_dynamic_bsz=False))  # no micro-batch size: no bound


def test_tf_config_stage_sets_the_attributes_verl_reads():
    from isoexec.integrations.verl.engine import IsoExecMegatronEngine

    engine = IsoExecMegatronEngine.__new__(IsoExecMegatronEngine)
    engine.model_config, engine.engine_config = _model_config(), _engine_config()
    engine._build_tf_config()
    import torch

    assert engine.param_dtype is torch.bfloat16 and engine.is_value_model is False
    assert engine.bridge is None and engine.provider is None and engine.tf_config is None
    engine.model_config = _model_config(model_type="value_model")
    with pytest.raises(ValueError, match="language-model"):
        engine._build_tf_config()
