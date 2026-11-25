"""Real Tinker conversion/dispatch order, with GPU work replaced at its boundary."""

from types import SimpleNamespace as NS
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("skyrl.backends.skyrl_train_backend")

from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train_backend import MegatronBackendOverrides, SkyRLTrainBackend
from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.engine import TinkerEngine, prepare_model_pass_batch, prepare_sample_batch
from skyrl.train.config import SkyRLTrainConfig

from isoexec.checks import Refusal
from isoexec.integrations.skyrl import audit, tinker
from isoexec.integrations.skyrl.tinker_state import SampleRows, TinkerState


def config(full=True):
    return {"model_package": "isoexec.models.qwen3_5", "full_logprob_comparison": full}


def prepared(*, first_token=1, score=-1.0):
    datum = types.Datum(
        model_input=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[first_token, 2, 3])]),
        loss_fn_inputs=types.LossFnInputs(
            target_tokens=types.TensorData(data=[2, 3, 4]), weights=types.TensorData(data=[0.0, 1.0, 1.0]),
            advantages=types.TensorData(data=[0.0, 1.0, 1.0]),
            logprobs=types.TensorData(data=[0.0, score, score]),
        ),
    )
    return prepare_model_pass_batch({"request": ("policy", types.ForwardBackwardInput(data=[datum], loss_fn="ppo"))})


def backend(monkeypatch, *, full=True, fail_backward=False):
    events = []
    obj = cast(Any, object.__new__(tinker.IsoExecTinkerBackend))
    obj.config = MegatronBackendOverrides.model_validate({"isoexec": config(full)})
    obj._cfg = SkyRLTrainConfig.from_cli_overrides({"trainer.strategy": "megatron"})
    obj._cfg.trainer.micro_train_batch_size_per_gpu = obj._cfg.trainer.micro_forward_batch_size_per_gpu = 1
    obj._tokenizer = NS(pad_token_id=0)
    obj._max_sequence_length = 8
    obj._vocab_size = 5
    obj._isoexec_state = TinkerState()
    obj._model_ids_to_role = {"policy": "policy"}
    obj._inference_engines_initialized = False
    obj._renderer = None
    obj._isoexec_state.acknowledge_sync("w0")
    rows = SampleRows("policy", 0, (1, 2), (3, 4), np.full((2, 5), -1, dtype=np.float32),
                      np.full(2, -1, dtype=np.float32))
    with obj._isoexec_state.reserve(rows.nbytes):
        obj._isoexec_state.commit([rows], rows.nbytes)

    class Dispatch:
        def get_lcm_dp_size(self):
            return 1

        def forward(self, role, batch, **kwargs):
            if "rollout_full_logprobs" in batch:
                events.append("prescore")
                return WorkerOutput(loss_fn_outputs=[{"logprobs": row.tolist()} for row in batch["rollout_logprobs"]])
            events.append("forward")
            width = batch.metadata["response_length"]
            return WorkerOutput(loss_fn_outputs=[{"logprobs": [-1.0] * width} for _ in range(len(batch))])

        def forward_backward(self, role, batch, **kwargs):
            events.append("backward")
            assert "rollout_full_logprobs" not in batch
            if fail_backward:
                raise RuntimeError("partial backward failure")
            return WorkerOutput(loss_fn_outputs=[{"logprobs": [-1.0] * 3} for _ in range(len(batch))],
                                metrics={"loss": 1.0})

        def set_optimizer_hparams(self, *args, **kwargs):
            events.append("set_optimizer_hparams")

        def optim_step(self, *args, **kwargs):
            events.append("optimizer")
            return 1.0

    obj._dispatch = Dispatch()
    # Without Ray the plan is not registered with a collector, and the verdict counts the compared rows.
    monkeypatch.setattr("ray.is_initialized", lambda: False)
    begin, finish = audit.begin_online_prescore, audit.finish_online_prescore
    monkeypatch.setattr(audit, "begin_online_prescore",
                        lambda *args: (events.append("begin_coverage"), begin(*args))[1])
    monkeypatch.setattr(audit, "finish_online_prescore",
                        lambda batch: (events.append("finish_coverage"), finish(batch))[1])
    return obj, events


def step():
    return types.OptimStepInput(adam_params=types.AdamParams(
        learning_rate=1e-6, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0))


def test_tinker_engine_selects_package_backend_before_construction(monkeypatch):
    selected = []

    class Backend:
        def __init__(self, model, cfg):
            selected.append((model, cfg.model_extra["isoexec"]))

    original = tinker.backend_class_for
    monkeypatch.setattr(tinker, "backend_class_for", lambda cfg: (selected.append(original(cfg)) or Backend))
    TinkerEngine(EngineConfig(base_model="checkpoint", backend="megatron", database_url="sqlite:///:memory:",
                             backend_config={"isoexec": config()}))
    assert selected == [tinker.IsoExecTinkerBackend, ("checkpoint", config())]


@pytest.mark.parametrize("overrides,match", [
    ({"backend": "fsdp"}, "megatron"),
    ({"external_inference_url": "http://other"}, "managed"),
    ({"runtime_role": "trainer"}, "combined"),
])
def test_backend_selection_refuses_incompatible_runtime(overrides, match):
    values = dict(base_model="checkpoint", backend="megatron", backend_config={"isoexec": config()})
    values.update(overrides)
    with pytest.raises(ValueError, match=match):
        tinker.backend_class_for(EngineConfig.model_validate(values))


def test_full_comparison_finishes_before_backward_and_optimizer(monkeypatch):
    obj, events = backend(monkeypatch)
    result = obj.forward_backward(prepared())
    assert events == ["begin_coverage", "prescore", "finish_coverage", "backward"]
    assert result["request"].metrics["isoexec/full_logprobs_verified_rows:sum"] == 2
    obj.optim_step("policy", step())
    assert events[-2:] == ["set_optimizer_hparams", "optimizer"]
    assert obj._isoexec_state.version == 1 and not obj._isoexec_state._rows


def test_accumulated_backwards_get_distinct_plans_at_the_same_weights(monkeypatch):
    from isoexec.integrations.skyrl.online_prescore import _plan

    obj, events = backend(monkeypatch)
    plans = []
    begin = audit.begin_online_prescore

    def planned(batch, boundaries, global_step):
        plans.append(_plan(batch, boundaries, global_step))
        return begin(batch, boundaries, global_step)

    monkeypatch.setattr(audit, "begin_online_prescore", planned)
    obj.forward_backward(prepared())
    obj.forward_backward(prepared())
    assert len(plans) == 2 and plans[0].id != plans[1].id
    assert plans[0].step == plans[1].step == 1
    obj.optim_step("policy", step())
    assert events.count("backward") == 2 and events.count("optimizer") == 1
    assert obj._isoexec_state.version == 1 and obj._isoexec_state.completed_backward_calls == 0


def test_model_execution_without_full_comparison_does_not_require_rollout(monkeypatch):
    obj, events = backend(monkeypatch, full=False)
    obj._isoexec_state.clear_rows()
    obj.forward_backward(prepared(first_token=9))
    assert events == ["backward"]
    obj.optim_step("policy", step())
    assert events[-1] == "optimizer"


def test_tokenizer_and_worker_config_share_the_resolved_checkpoint(monkeypatch):
    observed = []

    def resolve(cfg):
        observed.append(cfg.trainer.policy.model.path)
        cfg.trainer.policy.model.path = "/frozen/model.safetensors"
        cfg.trainer._isoexec_config = NS(tokenizer_path="/frozen")

    monkeypatch.setattr(tinker, "resolve", resolve)
    monkeypatch.setattr(tinker, "source_config", lambda *args: (None, {"vocab_size": 5}))
    monkeypatch.setattr(tinker, "get_tokenizer", lambda path: NS(path=path, pad_token_id=0))
    overrides = MegatronBackendOverrides.model_validate({
        "isoexec": {**config(), "trainer_capture_max_tokens": 64},
        "generator.inference_engine.engine_init_kwargs": {"max_model_len": 8},
        "trainer.placement.policy_num_gpus_per_node": 1,
        "generator.inference_engine.tensor_parallel_size": 1,
        "generator.inference_engine.num_engines": 1,
        "trainer.algorithm.temperature": 1.0,
    })
    obj = tinker.IsoExecTinkerBackend("hub/model", overrides)
    cfg = obj._build_config(types.LoraConfig(rank=0, alpha=16, seed=0))
    assert observed == ["hub/model", "/frozen/model.safetensors"]
    assert getattr(obj._tokenizer, "path") == "/frozen"
    assert cfg.trainer.policy.model.path == obj.base_model == "/frozen/model.safetensors"
    assert "trainer.ckpt_interval" not in (overrides.model_extra or {})


@pytest.mark.parametrize("resume", [None, "latest"])
def test_native_resume_is_off_unless_explicitly_requested_which_is_still_refused(monkeypatch, resume):
    from isoexec.integrations.skyrl.trainer_profile import trainer_capture_bound

    def resolve(cfg):
        trainer_capture_bound(cfg)  # the native-checkpoint checks the real resolve starts with
        cfg.trainer._isoexec_config = NS(tokenizer_path="/frozen")

    monkeypatch.setattr(tinker, "resolve", resolve)
    monkeypatch.setattr(tinker, "source_config", lambda *args: (None, {"vocab_size": 5}))
    monkeypatch.setattr(tinker, "get_tokenizer", lambda path: NS(path=path, pad_token_id=0))
    overrides = MegatronBackendOverrides.model_validate({
        "isoexec": {**config(), "trainer_capture_max_tokens": 64},
        "generator.inference_engine.engine_init_kwargs": {"max_model_len": 8},
        **({} if resume is None else {"trainer.resume_mode": resume}),
    })
    if resume is None:
        assert tinker.IsoExecTinkerBackend("hub/model", overrides)._build_config().trainer.resume_mode == "none"
    else:
        with pytest.raises(ValueError, match="checkpoint restore is unsupported"):
            tinker.IsoExecTinkerBackend("hub/model", overrides)


def test_backend_process_takes_the_actor_environment_that_decides_engine_arguments(monkeypatch):
    # The value a process outside the job's Ray runtime environment holds; resolve's selection must replace it
    # before configure_isoexec_vllm_engine reads the GDN composition in this process.
    monkeypatch.setenv("ISOEXEC_GDN_KERNEL", "recurrent")

    def resolve(cfg):
        cfg.trainer._isoexec_config = NS(tokenizer_path="/frozen")
        cfg.trainer.extra_runtime_env_vars = {"ISOEXEC_GDN_KERNEL": "cpr"}

    monkeypatch.setattr(tinker, "resolve", resolve)
    monkeypatch.setattr(tinker, "source_config", lambda *args: (None, {"vocab_size": 5}))
    monkeypatch.setattr(tinker, "get_tokenizer", lambda path: NS(path=path, pad_token_id=0))
    overrides = MegatronBackendOverrides.model_validate({
        "isoexec": {**config(), "trainer_capture_max_tokens": 64},
        "generator.inference_engine.engine_init_kwargs": {"max_model_len": 8},
    })
    tinker.IsoExecTinkerBackend("hub/model", overrides)
    assert tinker.os.environ["ISOEXEC_GDN_KERNEL"] == "cpr"


def test_failed_weight_sync_does_not_acknowledge_sampler_version(monkeypatch):
    obj, _ = backend(monkeypatch)

    def failed(*args, **kwargs):
        raise RuntimeError("receiver failed")

    monkeypatch.setattr(SkyRLTrainBackend, "save_sampler_checkpoint", failed)
    with pytest.raises(RuntimeError, match="receiver failed"):
        obj.save_sampler_checkpoint("/tmp/w1.tar.gz", "policy", persist=False)
    with pytest.raises(Refusal, match="acknowledged"):
        obj._isoexec_state.require_sampler("w0")


def test_unload_releases_comparison_actor_before_ray_teardown(monkeypatch):
    obj, events = backend(monkeypatch)
    monkeypatch.setattr(tinker, "close_online_prescore", lambda: events.append("close_collector"))
    monkeypatch.setattr(SkyRLTrainBackend, "delete_model", lambda *args: events.append("native_teardown"))
    with pytest.raises(ValueError, match="not found"):
        obj.delete_model("unknown")
    assert not events and obj._isoexec_state._rows
    obj.delete_model("policy")
    assert events == ["close_collector", "native_teardown"]
    assert not obj._isoexec_state._rows


def test_collector_handle_does_not_survive_a_new_ray_lifecycle(monkeypatch):
    import ray
    from isoexec.integrations.skyrl import online_prescore

    actor = object()
    killed = []
    monkeypatch.setattr(online_prescore, "_DRIVER_COLLECTOR", actor)
    monkeypatch.setattr(online_prescore, "_REMOTE_COLLECTOR", object())
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "kill", lambda value, **kwargs: killed.append(value))
    online_prescore.close_online_prescore()
    online_prescore.close_online_prescore()
    assert killed == [actor]
    assert online_prescore._DRIVER_COLLECTOR is None and online_prescore._REMOTE_COLLECTOR is None


def test_wrong_prefix_or_sampled_values_never_reach_backward(monkeypatch):
    obj, events = backend(monkeypatch)
    with pytest.raises(Refusal, match="no matching"):
        obj.forward_backward(prepared(first_token=9))
    assert not events
    with pytest.raises(Refusal, match="disagrees"):
        obj.forward_backward(prepared(score=-2.0))
    assert not events
    assert not obj._isoexec_state.failed
    with pytest.raises(Refusal):
        obj.optim_step("policy", step())
    assert "optimizer" not in events
    obj.forward_backward(prepared())
    obj.optim_step("policy", step())
    assert events[-1] == "optimizer"


def test_prompt_loss_refusal_leaves_the_model_usable(monkeypatch):
    obj, events = backend(monkeypatch)
    batch = prepared()
    batch.all_token_weights[0][0] = 1.0
    with pytest.raises(Refusal, match="outside"):
        obj.forward_backward(batch)
    assert not obj._isoexec_state.failed and not events
    obj.forward_backward(prepared())
    assert events[-1] == "backward"


def test_incomplete_receipt_cannot_authorize_backward(monkeypatch):
    obj, events = backend(monkeypatch)
    monkeypatch.setattr(audit, "finish_online_prescore", lambda batch: {"rows": 1})
    with pytest.raises(Refusal, match="does not cover"):
        obj.forward_backward(prepared())
    assert "backward" not in events and obj._isoexec_state.failed


def test_absent_comparison_cannot_reach_backward(monkeypatch):
    obj, events = backend(monkeypatch)
    monkeypatch.setattr(obj, "_compare_batch", lambda *args, **kwargs: None)
    with pytest.raises(Refusal, match="completed full comparison"):
        obj.forward_backward(prepared())
    assert not events and obj._isoexec_state.completed_backward_calls == 0


def test_padding_does_not_inflate_verified_rows(monkeypatch):
    obj, events = backend(monkeypatch)
    obj._cfg.trainer.micro_train_batch_size_per_gpu = 2
    obj._cfg.trainer.micro_forward_batch_size_per_gpu = 2
    from isoexec.integrations.skyrl.online_prescore import _plan

    registered = audit.begin_online_prescore

    def begin(batch, boundaries, global_step):
        assert len(batch) == 2
        assert batch["loss_mask"].shape[1] == 2  # Only response columns, not the three-token loss window.
        plan = _plan(batch, boundaries, global_step)
        assert len(plan.calls) == 1 and len(plan.calls[0]) == 2
        return registered(batch, boundaries, global_step)

    monkeypatch.setattr(audit, "begin_online_prescore", begin)
    result = obj.forward_backward(prepared())
    assert result["request"].metrics["isoexec/full_logprobs_verified_rows:sum"] == 2
    assert result["request"].metrics["isoexec/full_logprobs_compared_elements:sum"] == 10


def test_forward_scores_prompt_tokens_and_compares_only_captured_response(monkeypatch):
    obj, events = backend(monkeypatch)
    batch = prepared()
    batch.all_token_weights[0][0] = 1.0
    result = obj.forward(batch)
    assert result["request"].loss_fn_outputs[0]["logprobs"]["data"] == [-1.0] * 3
    assert events == ["begin_coverage", "prescore", "finish_coverage", "forward"]
    assert obj._isoexec_state.completed_backward_calls == 0 and not obj._isoexec_state.failed


def test_forward_mixed_sampled_and_arbitrary_text_keeps_the_same_check(monkeypatch):
    obj, events = backend(monkeypatch)
    batch, other = prepared(), prepared(first_token=9)
    batch.all_token_weights[0] = [1.0]
    batch.all_sampling_logprobs[0] = [-1.0]
    batch.all_advantages[0] = [1.0]
    for field in type(batch).model_fields:
        if field != "request_batch_slices":
            getattr(batch, field).extend(getattr(other, field))
    batch.request_batch_slices = [("request", "policy", 0, 2)]
    from isoexec.integrations.skyrl.online_prescore import _plan

    registered = audit.begin_online_prescore

    def begin(comparison, boundaries, global_step):
        plan = _plan(comparison, boundaries, global_step)
        assert len(plan.calls[0]) == 1 and comparison.metadata["response_length"] == 1
        return registered(comparison, boundaries, global_step)

    monkeypatch.setattr(audit, "begin_online_prescore", begin)
    result = obj.forward(batch)
    assert len(result["request"].loss_fn_outputs) == 2
    assert len(result["request"].loss_fn_outputs[0]["logprobs"]["data"]) == 1
    assert events == ["begin_coverage", "prescore", "finish_coverage", "forward"]
    assert not obj._isoexec_state.failed


def test_comparison_memory_refusal_precedes_model_work_and_is_retryable(monkeypatch):
    obj, events = backend(monkeypatch)
    with monkeypatch.context() as small:
        small.setattr(tinker, "MAX_PENDING_FULL_LOGPROB_BYTES", 1)
        with pytest.raises(Refusal, match="reduce the model-pass batch"):
            obj.forward_backward(prepared())
    assert not events and not obj._isoexec_state.failed
    obj.forward_backward(prepared())
    obj.optim_step("policy", step())
    assert events[-1] == "optimizer"


def test_sequence_bound_refusal_without_full_mode_is_retryable(monkeypatch):
    obj, events = backend(monkeypatch, full=False)
    obj._max_sequence_length = 3
    with pytest.raises(ValueError, match="bounds"):
        obj.forward_backward(prepared())
    assert not events and not obj._isoexec_state.failed
    obj._max_sequence_length = 8
    obj.forward_backward(prepared())
    assert events == ["backward"]


def test_invalid_native_loss_request_is_retryable(monkeypatch):
    obj, events = backend(monkeypatch)
    batch = prepared()
    batch.all_loss_fns[0] = "ppo_critic"
    with pytest.raises(ValueError):
        obj.forward_backward(batch)
    assert not events and not obj._isoexec_state.failed
    obj.forward_backward(prepared())
    assert events[-1] == "backward"


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_loss_contributions_cannot_escape_the_comparison_mask(monkeypatch, weight):
    obj, events = backend(monkeypatch)
    batch = prepared()
    batch.all_token_weights[0][1] = weight
    with pytest.raises(ValueError, match="finite nonnegative"):
        obj.forward_backward(batch)
    assert not events


def test_failed_backward_blocks_optimizer_despite_earlier_receipt(monkeypatch):
    obj, events = backend(monkeypatch, fail_backward=True)
    obj._isoexec_state.completed_backward_calls = 1
    with pytest.raises(RuntimeError, match="partial backward"):
        obj.forward_backward(prepared())
    with pytest.raises(Refusal, match="unload and recreate"):
        obj.optim_step("policy", step())
    assert "optimizer" not in events and obj._isoexec_state.failed


@pytest.mark.parametrize("full", [True, False])
def test_response_failure_after_backward_prevents_duplicate_accumulation(monkeypatch, full):
    obj, events = backend(monkeypatch, full=full)

    def fail_response(metrics):
        raise RuntimeError("cannot format worker metrics")

    monkeypatch.setattr(obj, "_extract_metrics", fail_response)
    with pytest.raises(RuntimeError, match="format worker metrics"):
        obj.forward_backward(prepared())
    assert events[-1] == "backward"
    with pytest.raises(Refusal, match="unload and recreate"):
        obj.forward_backward(prepared())
    with pytest.raises(Refusal, match="unload and recreate"):
        obj.optim_step("policy", step())
    assert events.count("backward") == 1 and "optimizer" not in events


@pytest.mark.asyncio
@pytest.mark.parametrize("top_p,top_k,stops", [(1.0, -1, None), (0.9, 5, ["END"])])
async def test_full_sampling_keeps_rows_server_side_and_returns_native_tinker_output(monkeypatch, top_p, top_k, stops):
    obj, _ = backend(monkeypatch)
    obj._isoexec_state.clear_rows()
    seen = []

    class Client:
        async def generate(self, **kwargs):
            seen.append(kwargs)
            return NS(response_ids=[3, 4], response_logprobs=[-1.0, -1.0], stop_reason="length",
                      full_logprobs=np.full((2, 5), -1, dtype=np.float32))

    obj._inference_engine_client = NS(_get_generate_client=lambda: Client())
    request = types.SampleInput(
        prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2])]),
        sampling_params=types.SamplingParams(temperature=1.0, max_tokens=2, seed=0,
                                            top_p=top_p, top_k=top_k, stop_strings=stops),
        num_samples=1, checkpoint_id="w0", prompt_logprobs=False,
    )
    batch = prepare_sample_batch({"sample": ("policy", request)})
    result = await obj.sample_batch_async(batch)
    assert isinstance(result["sample"], types.SampleOutput)
    assert result["sample"].sequences[0].tokens == [3, 4]
    assert "full_logprobs" not in result["sample"].model_dump_json()
    assert seen[0]["return_full_logprobs"] is True
    assert seen[0]["sampling_params"]["top_p"] == top_p
    assert seen[0]["sampling_params"]["top_k"] == top_k
    from isoexec.integrations.skyrl.inference import prepare_full_logprobs_capture

    wire = dict(seen[0]["sampling_params"])
    prepare_full_logprobs_capture({}, wire, NS(logprobs_mode="raw_logprobs"), None)
    assert wire["logprobs"] == -1 and ("detokenize" not in wire) == bool(stops)
    if stops:
        assert wire["stop"] == stops
    assert obj._isoexec_state.find("policy", [1, 2, 3, 4], required=True) is not None
    assert obj._isoexec_state._reserved == 0


def test_lora_rejected_before_any_native_model_allocation(monkeypatch):
    obj, _ = backend(monkeypatch)
    monkeypatch.setattr(SkyRLTrainBackend, "create_model", lambda *args, **kwargs: pytest.fail("native allocation"))
    with pytest.raises(ValueError, match="rank=0"):
        obj.create_model("new", types.LoraConfig(rank=8, alpha=16, seed=0))
