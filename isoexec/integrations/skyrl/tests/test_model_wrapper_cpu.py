"""The actual SkyRL wrapper with IsoExec's finalizer, scorer and temperature, without CUDA."""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch

from isoexec.integrations.skyrl.tests.test_optimizer_update_cpu import framework  # noqa: F401

pytestmark = pytest.mark.parametrize("framework", [
    "skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper"
], indirect=True)


@pytest.mark.parametrize("enabled,installed", [(True, True), (True, False), (False, True), (False, False)])
def test_actual_wrapper_preserves_finalizer_and_native_setup_order(framework, monkeypatch, enabled, installed):  # noqa: F811
    native, _ = framework
    integration = importlib.import_module("isoexec.integrations.skyrl.model_wrapper")
    events = []
    actor = [torch.nn.Linear(1, 1)]

    def default(modules, tokens):
        assert modules is actor and tokens is None
        events.append("default")

    def supplied(modules, tokens):
        assert modules is actor and tokens is None
        events.append("installed")

    config = NS(finalize_model_grads_func=supplied if installed else None)
    monkeypatch.setattr(native, "get_model_config", lambda module: config)
    monkeypatch.setattr(integration, "get_model_config", lambda module: config)
    monkeypatch.setattr(native, "finalize_model_grads_with_expert_adapter_sync", default)

    def expert_sync(modules):
        assert modules is actor and config.finalize_model_grads_func.__name__ == "_defer_finalize_model_grads"
        events.append("expert_sync")

    monkeypatch.setattr(native, "enable_expert_parallel_grad_sync_in_finalize", expert_sync)
    monkeypatch.setattr(integration, "lm_head_log_probs", lambda wrapper: events.append("bind"))
    cls = integration.IsoExecModelWrapper if enabled else native.MegatronModelWrapper
    from skyrl.train.config import TrainerConfig

    trainer_config = TrainerConfig()
    trainer_config.remove_microbatch_padding = True
    wrapper = cls(config=trainer_config, actor_module=actor)
    assert events == (["expert_sync", "bind"] if enabled else ["expert_sync"])
    if enabled:
        assert wrapper._isoexec_finalize_model_grads is (supplied if installed else None)
    assert config.finalize_model_grads_func.__self__ is wrapper
    config.finalize_model_grads_func(actor)
    config.finalize_model_grads_func(actor)
    assert events == (["expert_sync", "bind"] if enabled else ["expert_sync"])
    wrapper.run_pending_grad_sync()
    assert events[-1] == ("installed" if enabled and installed else "default")
    assert wrapper._pending_grad_sync is None
    # A rank with no local microbatch must still participate in the collective.
    wrapper.run_pending_grad_sync()
    assert events[-2:] == ["installed" if enabled and installed else "default"] * 2


def _wrapper(integration, log_probs):
    wrapper = integration.IsoExecModelWrapper.__new__(integration.IsoExecModelWrapper)
    wrapper._isoexec_log_probs = log_probs
    wrapper.cfg = NS(logprobs_chunk_size=2)
    return wrapper


def test_logits_are_tempered_with_the_engines_fp32_tensor_divide(framework):  # noqa: F811
    integration = importlib.import_module("isoexec.integrations.skyrl.model_wrapper")
    wrapper = _wrapper(integration, None)
    logits = torch.randn(5, 64, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    assert wrapper._scale_logits(logits, 1.0) is logits
    tempered = wrapper._scale_logits(logits, 0.7)
    expected = logits.float() / torch.tensor(0.7, dtype=torch.float32)
    assert tempered.dtype == torch.float32
    assert torch.equal(tempered.view(torch.int32), expected.view(torch.int32))


def test_target_logprobs_come_from_the_model_head_in_chunks(framework):  # noqa: F811
    integration = importlib.import_module("isoexec.integrations.skyrl.model_wrapper")
    calls = []

    def log_probs(logits, targets, temperature):
        calls.append((tuple(logits.shape), logits.dtype, targets.tolist(), temperature))
        return logits.gather(1, targets[:, None]).squeeze(1)

    wrapper = _wrapper(integration, log_probs)
    # What SkyRL's packed scorer passes: one rank's logits [1, T, vocab] and shifted targets [1, T].
    logits = torch.arange(15, dtype=torch.bfloat16).reshape(1, 5, 3)
    targets = torch.tensor([[0, 1, 2, 0, 1]])
    scored = wrapper._target_logprob_fn()(logits, targets)
    assert scored.tolist() == [[0.0, 4.0, 8.0, 9.0, 13.0]]
    assert calls == [((2, 3), torch.float32, [0, 1], 1.0), ((2, 3), torch.float32, [2, 0], 1.0),
                     ((1, 3), torch.float32, [1], 1.0)]
    # A stage without the LM head leaves SkyRL's own scorer in place.
    assert _wrapper(integration, None)._target_logprob_fn() is None


def test_finalizer_error_propagates_after_clearing_pending_state(framework):  # noqa: F811
    integration = importlib.import_module("isoexec.integrations.skyrl.model_wrapper")
    wrapper = integration.IsoExecModelWrapper.__new__(integration.IsoExecModelWrapper)
    wrapper.actor_module = []
    wrapper._pending_grad_sync = {"num_tokens": None}

    def fail(*args):
        raise RuntimeError("collective failed")

    wrapper._isoexec_finalize_model_grads = fail
    with pytest.raises(RuntimeError, match="collective failed"):
        wrapper.run_pending_grad_sync()
    assert wrapper._pending_grad_sync is None


@pytest.mark.parametrize("method", ["forward", "forward_backward_mini_batch"])
def test_wrapper_carries_rollout_requests_into_each_forward(framework, monkeypatch, method):  # noqa: F811
    native, _ = framework
    integration = importlib.import_module("isoexec.integrations.skyrl.model_wrapper")
    from isoexec.integrations.skyrl.tests.test_rollout_requests_cpu import packed_cu, skyrl_batch
    from isoexec.schema import rollout

    stage = NS(reads_rollout_requests=True, forward_context=None)
    monkeypatch.setattr(integration, "_find_isoexec_stage", lambda modules: stage)
    batches = [skyrl_batch([[1] * p], [[2] * g]) for p, g in ((5, 2), (3, 4))]
    seen = []

    def schedule(self, micro_batches, *args, **kwargs):
        for batch in micro_batches:
            psp = NS(cu_seqlens_q_padded=torch.tensor(packed_cu(batch["attention_mask"]), dtype=torch.int32))
            with stage.forward_context(psp):
                seen.append(rollout.current().requests)
        return "done"

    monkeypatch.setattr(native.MegatronModelWrapper, method, schedule)
    wrapper = integration.IsoExecModelWrapper.__new__(integration.IsoExecModelWrapper)
    wrapper.actor_module = [torch.nn.Linear(1, 1)]
    assert getattr(wrapper, method)(batches, 8, 1) == "done"
    assert seen == [(((5, 2),),), (((3, 4),),)] and stage.forward_context is None
