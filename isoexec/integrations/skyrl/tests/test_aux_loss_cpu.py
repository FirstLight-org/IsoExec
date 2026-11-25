"""Actual Qwen programs on native MegatronModel stages and SkyRL loss callbacks; no CUDA/scheduler qualification."""

import importlib
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

from isoexec.integrations.skyrl.aux_loss import bind, microbatch_aux_loss
from isoexec.integrations.skyrl.tests.test_optimizer_update_cpu import framework  # noqa: F401
from isoexec.compiler.source import ModelSource
from isoexec.models.qwen2_moe.tests.test_qwen1_5_moe_a2_7b_cpu import checkpoint_state, configuration
from isoexec.ops.table import OpTable
from isoexec.schema import Axis

pytestmark = pytest.mark.parametrize("framework", [
    "skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper"
], indirect=True)

SOURCE = ModelSource("isoexec.models.qwen2_moe")


def _settings(coefficient=0.001, mode="aux_loss"):
    return SimpleNamespace(moe_aux_loss_coeff=coefficient, moe_router_load_balancing_type=mode)


def _native_scale(monkeypatch, scale=1.0):
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", torch.tensor(scale))


@pytest.fixture
def world(tmp_path):
    torch.distributed.init_process_group("gloo", init_method=(tmp_path / "group").as_uri(), rank=0, world_size=1,
                                         timeout=timedelta(seconds=10))
    try:
        yield torch.distributed.group.WORLD
    finally:
        torch.distributed.destroy_process_group()


def _packed(cu):
    from megatron.core.packed_seq_params import PackedSeqParams

    longest = int((cu[1:] - cu[:-1]).max())
    return PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, cu_seqlens_q_padded=cu,
                           cu_seqlens_kv_padded=cu, max_seqlen_q=longest, max_seqlen_kv=longest)


def _hosts(world, monkeypatch, raw=None):
    """Two native PP1 MegatronModel stages loaded from one seeded checkpoint, bound to reference ops on CPU."""
    from megatron.core.process_groups_config import ProcessGroupCollection

    from isoexec.compiler.capture.capture import capture
    from isoexec.compiler.partition import define_stage
    from isoexec.compiler.text import text_inputs
    from isoexec.runtimes.megatron import _binding
    from isoexec.runtimes.megatron import model as native
    from isoexec.runtimes.megatron._config import make_transformer_config

    raw = configuration() if raw is None else raw
    reference = OpTable(mode="reference")
    monkeypatch.setattr(_binding, "OpTable", lambda: reference)
    monkeypatch.setattr(_binding, "bind_row_parallel",
                        lambda *, tp_group, site, sp: reference.bind("collectives.row_parallel", site=site))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    allocate = native.allocate_tensors
    monkeypatch.setattr(native, "allocate_tensors", lambda graph, **kwargs: allocate(graph, device=torch.device("cpu")))
    author = SOURCE.build_meta(raw).train(True)
    examples = text_inputs(author, device="cpu", max_tokens=16, profile="trainer")
    captured = capture(author, examples.args, examples.kwargs, dynamic_shapes=examples.dynamic_shapes,
                       root_layouts=examples.root_layouts, seq=examples.seq)
    # The same settings factory.build_model passes for a PP1 trainer.
    config = make_transformer_config(
        captured, tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1,
        expert_model_parallel_size=1, expert_tensor_parallel_size=1, sequence_parallel=False,
        bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16, gradient_accumulation_fusion=False,
        variable_seq_lengths=True, calculate_per_token_loss=False, perform_initialization=False,
    )
    groups = ProcessGroupCollection()
    for field in ("tp", "pp", "ep", "expt_tp", "dp", "cp"):
        setattr(groups, field, world)
    definition = define_stage(captured, ((0, len(captured.layers)),), pp_rank=0)
    generator = torch.Generator().manual_seed(71)
    weights = {name: (torch.randn(tensor.shape, generator=generator) * 0.1).bfloat16()
               for name, tensor in checkpoint_state(raw).items()}
    for name, tensor in weights.items():
        if "norm.weight" in name:
            tensor.fill_(1)
    hosts = []
    for _ in range(2):
        host = native.MegatronModel(definition,
                                    source=SOURCE, hf_config=SOURCE.build_config(raw), config=config, pg_collection=groups)
        host.load_weights(weights.items())
        hosts.append(host)
    return hosts


def _forward_with_aux(host, ids, packed):
    """Return logits and the auxiliary outputs of one native forward, leaving the loss unchanged."""
    aux = {}

    def keep(logits, outputs, _packed):
        aux.update(outputs)
        return logits

    host.aux_loss_injector = keep
    try:
        return host(ids, None, None, packed_seq_params=packed), aux
    finally:
        host.aux_loss_injector = None


def test_actual_qwen_aux_loss_padding_gradients_and_two_updates(framework, world, monkeypatch):
    _native_scale(monkeypatch)
    actual, expected = _hosts(world, monkeypatch)
    helper = bind(actual, _settings())
    expected_helper = bind(expected, _settings())
    assert helper is not None and expected_helper is not None
    # Two sequences, lengths three and one, packed into aligned four/two rows.
    ids = torch.tensor([[1, 2, 3, 0, 4, 0]])
    packed = _packed(torch.tensor([0, 4, 6], dtype=torch.int32))
    data = {"attention_mask": torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]]),
            "packed_seq_params": packed, "num_microbatches": 1, "num_real_microbatches": 1}
    mask = torch.tensor([True, True, True, False, True, False])
    optimizers = [torch.optim.SGD(host.parameters(), lr=0.1) for host in (actual, expected)]
    for _ in range(2):
        observations = []
        for host, optimizer in zip((actual, expected), optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            if host is actual:
                with microbatch_aux_loss(host, helper, [data]) as reports:
                    logits = host(ids, None, None, packed_seq_params=packed)
                    logits.float().square().mean().backward()
                assert reports[0]["moe_aux_loss"] > 0
                assert reports[0]["moe_aux_loss_coeff"] == 0.001
                assert host.last_aux == {}
                assert host.aux_loss_injector is None
            else:
                logits, aux = _forward_with_aux(host, ids, packed)
                weighted, _ = expected_helper(aux, mask, packed.cu_seqlens_q_padded)
                (logits.float().square().mean() + weighted).backward()
            gradients = {name: p.grad.clone() for name, p in host.named_parameters() if p.grad is not None}
            assert any(grad.abs().sum() > 0 for name, grad in gradients.items() if name.endswith("router.weight"))
            observations.append((logits.detach(), gradients))
            optimizer.step()
        assert torch.equal(observations[0][0], observations[1][0])
        assert observations[0][1].keys() == observations[1][1].keys()
        for name in observations[0][1]:
            torch.testing.assert_close(observations[0][1][name], observations[1][1][name], rtol=0, atol=0)
        for left, right in zip(actual.parameters(), expected.parameters(), strict=True):
            assert torch.equal(left, right)


@pytest.mark.parametrize("probabilities,top_k,expected_loss,expected_gradients", [
    ([[0.9, 0.1], [0.1, 0.9]], 1, 3.6, [[0.18, -0.18], [-0.18, 0.18]]),
    ([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]], 2, 2.8,
     [[0.24, 0.18, -0.28, -0.14], [-0.14, -0.28, 0.18, 0.24]]),
])
def test_aux_loss_balances_each_layer(framework, world, monkeypatch,
                                     probabilities, top_k, expected_loss, expected_gradients):
    raw = {**configuration(), "num_experts": len(probabilities[0]), "num_experts_per_tok": top_k}
    host, _ = _hosts(world, monkeypatch, raw)
    helper = bind(host, _settings(1.0))
    assert helper is not None
    # Each row is one layer's token. Pooling layers would hide their imbalance
    # and cancel the router-logit gradients; these expectations are hand-derived.
    logits = torch.tensor(probabilities).log().requires_grad_()
    scores = tuple(layer.unsqueeze(0) for layer in logits.softmax(-1).unbind(0))
    names = host.aux_outputs
    actual, _ = helper(dict(zip(names, scores, strict=True)), torch.ones(1, dtype=torch.bool), torch.tensor([0, 1]))
    torch.testing.assert_close(actual, torch.tensor(expected_loss))
    actual.backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, torch.tensor(expected_gradients))


def test_aux_loss_matches_masked_probability_formula(framework, world, monkeypatch):
    host, _ = _hosts(world, monkeypatch)
    helper = bind(host, _settings(1.0))
    assert helper is not None
    logits = torch.tensor([[2., 0., -1., -2.], [0., 3., -1., -2.], [9., -1., 0., -2.]], requires_grad=True)
    scores = logits.softmax(-1)
    aux = {name: scores for name in host.aux_outputs}
    actual, _ = helper(aux, torch.tensor([True, True, False]), torch.tensor([0, 3]))
    # In each of two layers, both valid tokens select experts 0 and 1, each
    # taking half the assignments: 2 layers * 4 experts * 1/2 assignment share.
    expected = 4 * scores[:2, :2].mean(0).sum()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad[2], torch.zeros(4))
    assert logits.grad[:2].abs().sum() > 0


def test_microbatch_grouping_defines_the_objective(framework, world, monkeypatch):
    raw = {**configuration(), "num_experts": 2, "num_experts_per_tok": 1}
    host, _ = _hosts(world, monkeypatch, raw)
    helper = bind(host, _settings(1.0))
    assert helper is not None
    scores = torch.tensor([[0.9, 0.1], [0.1, 0.9]], requires_grad=True)
    names = host.aux_outputs
    separate = [helper({name: row[None] for name in names}, torch.ones(1), torch.tensor([0, 1]))[0]
                for row in scores]
    merged = helper({name: scores for name in names}, torch.ones(2), torch.tensor([0, 2]))[0]
    # Two layers: per-layer losses are 1.8 for separate one-token batches and
    # 1.0 for the combined batch. Averaging must retain the chosen grouping.
    torch.testing.assert_close(torch.stack(separate).mean(), torch.tensor(3.6))
    torch.testing.assert_close(merged, torch.tensor(2.0))


def test_synthetic_rows_are_excluded_without_dropping_real_loss_masked_rows(framework, world, monkeypatch):
    _native_scale(monkeypatch)
    host, _ = _hosts(world, monkeypatch)
    helper = bind(host, _settings(1.0))
    assert helper is not None
    logits = torch.tensor([[2., 1., 0., -1.]] * 4 + [[0., -1., 8., 7.]] * 2, requires_grad=True)
    scores = logits.softmax(-1)
    aux = {name: scores for name in host.aux_outputs}
    packed = SimpleNamespace(cu_seqlens_q_padded=torch.tensor([0, 2, 4, 6]))
    batch = {"attention_mask": torch.ones(3, 2), "loss_mask": torch.zeros(3, 1),
             "isoexec_aux_valid_rows": torch.tensor([True, True, False]),
             "num_microbatches": 1, "num_real_microbatches": 1}
    with microbatch_aux_loss(host, helper, [batch]) as reports:
        assert host.aux_loss_injector is not None
        output = host.aux_loss_injector(torch.zeros((), requires_grad=True), aux, packed)
    output.backward()
    reference = logits.detach().clone().requires_grad_()
    expected = reference.softmax(-1)[:4, :2].sum()  # Two layers, four valid tokens, top-2 of four experts.
    expected.backward()
    assert reports[0]["moe_aux_loss"] == pytest.approx(float(expected.detach()))
    assert logits.grad is not None and reference.grad is not None
    torch.testing.assert_close(logits.grad, reference.grad)
    assert logits.grad[:4].abs().sum() > 0
    assert torch.count_nonzero(logits.grad[4:]) == 0


def test_aux_metric_matches_real_microbatch_and_dp_averaging(framework, world, monkeypatch):
    from skyrl.backends.skyrl_train.workers.worker_utils import reduce_metrics

    from isoexec.integrations.skyrl.aux_loss import _merge_metrics

    host, _ = _hosts(world, monkeypatch)
    raw = [{"moe_aux_loss": 1.8, "moe_aux_loss_coeff": 0.01}] * 2
    reports = _merge_metrics(replace(host.placement, dp=Axis(2, 0)), raw, real_count=2)
    worker_metrics = reduce_metrics({key: [report[key] for report in reports] for key in raw[0]},
                                    sum_loss_metrics=True)
    assert 2 * worker_metrics["moe_aux_loss"] == pytest.approx(1.8)  # DP SUM of two identical ranks.
    assert worker_metrics["moe_aux_loss_coeff"] == 0.01


def test_invalid_native_options_are_checked_before_model_build(framework):
    from skyrl.train.config.config import MegatronConfig

    _, integration = framework
    for invalid in (-1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="finite nonnegative"):
            integration.validate_model_options(MegatronConfig(moe_aux_loss_coeff=invalid))
    for mode in ("seq_aux_loss", "global_aux_loss", "sinkhorn"):
        with pytest.raises(ValueError, match="supports only"):
            integration.validate_model_options(MegatronConfig(moe_router_load_balancing_type=mode))


def test_disabled_and_distributed_refusal(framework, world, monkeypatch):
    host, _ = _hosts(world, monkeypatch)
    assert bind(object(), _settings(0, "none")) is None
    assert bind(host, _settings(0)) is None
    assert bind(host, _settings(0.001, "none")) is None
    assert bind(host, _settings()) is not None
    with pytest.raises(ValueError, match="native MegatronModel"):
        bind(object(), _settings())
    # One hidden tensor crosses each PP cut, so router scores stay local only at PP1.
    groups = host.pg_collection
    monkeypatch.setattr(host, "pg_collection", SimpleNamespace(pp=SimpleNamespace(size=lambda: 2)))
    with pytest.raises(ValueError, match="requires PP=1"):
        bind(host, _settings())
    monkeypatch.setattr(host, "pg_collection", groups)


def test_masked_padding_and_subsequence_layout(framework, world, monkeypatch):
    _native_scale(monkeypatch)
    host, _ = _hosts(world, monkeypatch)
    helper = bind(host, _settings())
    assert helper is not None
    names = host.aux_outputs
    logits = torch.tensor([[1., 0., -1., 2.], [0., 1., -2., 2.], [8., -1., 0., 0.],
                           [1., 3., 0., 2.], [2., 0., 4., 1.], [9., -2., 0., 1.]], requires_grad=True)
    probabilities = logits.softmax(-1)
    data = {"attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0]]),
            "packed_seq_params": SimpleNamespace(cu_seqlens_q_padded=torch.tensor([0, 3, 6])),
            "sub_seq_lengths": [torch.tensor([2, 2])], "num_microbatches": 1, "num_real_microbatches": 1}
    with microbatch_aux_loss(host, helper, [data]) as reports:
        assert host.aux_loss_injector is not None
        loss = host.aux_loss_injector(torch.zeros((), requires_grad=True),
            {name: probabilities for name in names}, data["packed_seq_params"])
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[[2, 5]]) == 0
    assert logits.grad[[0, 1, 3, 4]].abs().sum() > 0
    data["attention_mask"] = torch.zeros(1, 6)
    data.pop("sub_seq_lengths")
    data["packed_seq_params"] = SimpleNamespace(cu_seqlens_q_padded=torch.tensor([0, 6]))
    with microbatch_aux_loss(host, helper, [data]) as reports:
        assert host.aux_loss_injector is not None
        zero = host.aux_loss_injector(torch.zeros((), requires_grad=True),
            {name: probabilities for name in names}, data["packed_seq_params"])
    assert float(zero.detach()) == reports[0]["moe_aux_loss"] == 0.0
    # Schedule padding still has attended tokens, but contributes no aux loss.
    data["attention_mask"] = torch.ones(1, 6)
    data["is_padding_batch"] = True
    data["num_real_microbatches"] = 0
    with microbatch_aux_loss(host, helper, [data]) as reports:
        assert host.aux_loss_injector is not None
        zero = host.aux_loss_injector(torch.zeros((), requires_grad=True),
            {name: probabilities for name in names}, data["packed_seq_params"])
    assert float(zero.detach()) == reports[0]["moe_aux_loss"] == 0.0


@pytest.mark.parametrize("loss_name", ["ppo", "cross_entropy"])
def test_native_worker_wires_aux_into_real_skyrl_loss_and_updates(framework, world, tmp_path, monkeypatch, loss_name):
    """Normal imports and actual loss closure; CPU scheduling analogue only."""
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
    from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
    from skyrl.backends.skyrl_train.workers.worker_utils import BaseBatchIterator
    from skyrl.train.config.config import TrainerConfig

    from isoexec.integrations.skyrl.aux_batch import install_worker, worker_microbatches
    from isoexec.integrations.skyrl.aux_loss import build_wrapper

    _native_scale(monkeypatch)

    workers = importlib.import_module("isoexec.integrations.skyrl.workers")
    wrapper_module = importlib.import_module("skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper")
    mpu = wrapper_module.mpu
    group = world
    for name in ("get_tensor_model_parallel_world_size", "get_data_parallel_world_size",
                 "get_context_parallel_world_size", "get_pipeline_model_parallel_world_size"):
        monkeypatch.setattr(mpu, name, lambda *args, **kwargs: 1)
    for name in ("get_tensor_model_parallel_rank", "get_context_parallel_rank",
                 "get_pipeline_model_parallel_last_rank"):
        monkeypatch.setattr(mpu, name, lambda *args, **kwargs: 0)
    for name in ("get_tensor_model_parallel_group", "get_context_parallel_group",
                 "get_pipeline_model_parallel_group"):
        monkeypatch.setattr(mpu, name, lambda *args, **kwargs: group)
    for name in ("is_pipeline_first_stage", "is_pipeline_last_stage"):
        monkeypatch.setattr(mpu, name, lambda *args, **kwargs: True)
    actual, expected = _hosts(world, monkeypatch)
    # SkyRL moves each microbatch to the current device before its forward step.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    cfg = TrainerConfig()
    cfg.remove_microbatch_padding = True
    cfg.policy.model.path = str(tmp_path)
    cfg.policy.megatron_config.moe_router_load_balancing_type = "aux_loss"
    cfg.policy.megatron_config.moe_aux_loss_coeff = 0.001
    cfg.algorithm.policy_loss_type = loss_name
    cfg.algorithm.use_kl_loss = cfg.algorithm.use_entropy_loss = False
    cfg.algorithm.enable_sample_support_replay = False
    # A zero task loss makes a disconnected auxiliary branch fail the test.
    worker = object.__new__(workers.IsoExecPolicyWorker)
    worker.cfg = cfg
    worker.strategy = SimpleNamespace()
    worker.policy_loss_fn = lambda scores, *args, **kwargs: (scores.sum() * 0, {})
    worker.actor_module = [actual]
    worker.optimizer = None
    # The wrapper the policy worker builds for training (workers.IsoExecPolicyWorker._build_model_wrapper).
    worker.model = build_wrapper(megatron_config=cfg.policy.megatron_config, training=True, config=cfg,
                                 actor_module=worker.actor_module, actor_optimizer=worker.optimizer,
                                 policy_loss_fn=worker.policy_loss_fn, is_vlm=False)
    install_worker(worker)
    expected_helper = bind(expected, cfg.policy.megatron_config)
    assert expected_helper is not None
    assert not hasattr(worker.model, "_aux_loss_fn")
    assert type(worker.model).__module__ == "isoexec.integrations.skyrl.aux_loss"

    def schedule(*, forward_step_func, data_iterator, model, num_microbatches, **kwargs):
        metrics = []
        MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(1.0 / num_microbatches))
        for _ in range(num_microbatches):
            outputs, loss_fn = forward_step_func(data_iterator, model[0])
            loss, report = loss_fn(outputs)
            # Megatron's two-output loss callback uses this exact PP1 scaling.
            (loss / num_microbatches).backward()
            metrics.append(report)
        return metrics

    monkeypatch.setattr(wrapper_module, "get_forward_backward_func", lambda: schedule)

    def batch(padding=False):
        return {"sequences": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "position_ids": torch.arange(4).unsqueeze(0), "num_actions": 3,
                "old_action_log_probs": torch.zeros(1, 3), "base_action_log_probs": torch.zeros(1, 3),
                "advantages": torch.ones(1, 3), "loss_mask": torch.ones(1, 3),
                "rollout_action_logprobs": torch.zeros(1, 3), "rollout_expert_indices": None,
                "num_microbatches": 2, "num_real_microbatches": 1, "is_padding_batch": padding}

    optimizers = [torch.optim.SGD(host.parameters(), lr=0.1) for host in (actual, expected)]
    for _ in range(2):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        micro_batches = [batch(), batch(True)]
        with worker_microbatches(worker):
            for item in micro_batches:
                native_batch = TrainingInputBatch({"sequences": item["sequences"],
                                                   "attention_mask": item["attention_mask"]})
                native_batch.metadata = {"response_length": 3, "is_padding_batch": item["is_padding_batch"]}
                BaseBatchIterator.batch_to_experience(native_batch)
            reports = worker.model.forward_backward_mini_batch(micro_batches, 4, 1,
                                                                temperature=1.0, return_per_token_outputs=False)
        assert reports[0]["moe_aux_loss"] > 0
        assert reports[1]["moe_aux_loss"] == 0
        assert actual.last_aux == {}
        assert actual.aux_loss_injector is None
        _, aux = _forward_with_aux(expected, batch()["sequences"], _packed(torch.tensor([0, 4], dtype=torch.int32)))
        auxiliary, _ = expected_helper(aux, torch.ones(4, dtype=torch.bool), torch.tensor([0, 4]))
        auxiliary.backward()
        left, right = dict(actual.named_parameters()), dict(expected.named_parameters())
        for name in left:
            if name.endswith("router.weight"):
                assert left[name].grad is not None and left[name].grad.abs().sum() > 0
                torch.testing.assert_close(left[name].grad, right[name].grad, rtol=0, atol=0)
        for optimizer in optimizers:
            optimizer.step()
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
