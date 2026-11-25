"""IsoExec models inside SkyRL's native Tinker backend and request lifecycle."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from skyrl.backends.skyrl_train_backend import SkyRLTrainBackend, _build_skyrl_train_config
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.tinker import types
from skyrl.train.dataset.preprocess import collate_rollout_full_logprobs
from skyrl.utils.tok import get_tokenizer

from isoexec.checks import Refusal
from isoexec.integrations.full_distribution import full_comparison_enabled
from .audit import after_policy_forward, before_policy_forward
from .config import SkyRLConfig, resolve
from .full_distribution_config import validate_full_distribution_runtime
from .model_folder import source_config, validate_model_config
from .online_prescore import close_online_prescore
from .tinker_state import MAX_PENDING_FULL_LOGPROB_BYTES, SampleRows, TinkerState


def _tokens(model_input: types.ModelInput) -> tuple[int, ...]:
    tokens = []
    for chunk in model_input.chunks:
        if not isinstance(chunk, types.EncodedTextChunk):
            raise ValueError("IsoExec Tinker accepts encoded text inputs only")
        tokens.extend(chunk.tokens)
    return tuple(tokens)


def backend_class_for(config: Any) -> type[SkyRLTrainBackend]:
    """Select the package backend before Tinker constructs any model resources."""
    selected = config.backend_config.get("isoexec")
    validate_model_config(selected)
    if config.backend != "megatron":
        raise ValueError("IsoExec Tinker requires backend=megatron")
    if config.external_inference_url is not None:
        raise ValueError("IsoExec Tinker requires its managed vLLM engines, not external_inference_url")
    if config.runtime_role == "inference":
        raise ValueError("IsoExec Tinker supports full-parameter trainer or combined runtimes")
    if full_comparison_enabled(selected) and config.runtime_role != "combined":
        raise ValueError("Tinker full logprob comparison requires runtime_role=combined")
    return IsoExecTinkerBackend


class IsoExecTinkerBackend(SkyRLTrainBackend):
    """Reuse native dispatch, optimizer accumulation and acknowledged weight sync."""

    def __init__(self, base_model: str, config: Any):
        self._isoexec_state = TinkerState()
        # Tinker owns checkpoint requests; the native RL loop's periodic save
        # interval and its resume at start (SkyRL defaults resume_mode to
        # "latest"; its Tinker backend never reads it) are inapplicable.
        # Explicit incompatible overrides still fail.
        config = config.model_copy(deep=True)
        assert config.model_extra is not None
        config.model_extra.setdefault("trainer.ckpt_interval", -1)
        config.model_extra.setdefault("trainer.resume_mode", "none")
        config.model_extra.setdefault("trainer.remove_microbatch_padding", True)
        super().__init__(base_model, config)

    def _build_config(self, lora_config: types.LoraConfig | None = None):
        if lora_config is not None and lora_config.rank != 0:
            raise ValueError("IsoExec Tinker supports full-parameter training only (rank=0); LoRA is unsupported")
        model_path = self.base_model
        if not isinstance(model_path, str) or not model_path:
            raise ValueError("IsoExec Tinker requires a base model/checkpoint path")
        cfg = _build_skyrl_train_config(model_path, self.config, lora_config)
        engine = cfg.generator.inference_engine
        if cfg.trainer.algorithm.enable_sample_support_replay or cfg.trainer.algorithm.enable_score_centering:
            raise ValueError("IsoExec Tinker requests do not carry sample-support replay/score-centering fields")
        if engine.external_proxy_url is not None or engine.external_server_urls is not None:
            raise ValueError("IsoExec Tinker requires managed inference engines")
        limit = cfg.trainer.algorithm.max_seq_len
        if limit is None:
            limit = engine.engine_init_kwargs.get("max_model_len")
        if type(limit) is not int or limit < 2:
            raise ValueError("IsoExec Tinker needs a finite trainer.algorithm.max_seq_len or "
                             "generator.inference_engine.engine_init_kwargs.max_model_len")
        self._max_sequence_length = limit
        selected = validate_model_config(cfg.isoexec)
        if selected is None:
            raise ValueError("IsoExec Tinker requires an isoexec model configuration")
        if selected.get("trainer_capture_max_tokens") is None:
            from skyrl.backends.skyrl_train.distributed.megatron.packing_utils import get_packed_seq_align_size

            mc = cfg.trainer.policy.megatron_config
            alignment = int(get_packed_seq_align_size(mc.tensor_model_parallel_size, mc.context_parallel_size,
                                                    fp8_enabled=False))
            longest = ((limit + alignment - 1) // alignment) * alignment
            count = max(cfg.trainer.micro_train_batch_size_per_gpu, cfg.trainer.micro_forward_batch_size_per_gpu)
            budget = cfg.trainer.max_tokens_per_microbatch
            bound = max(budget * alignment, longest) if budget > 0 else count * longest
            cfg.isoexec = {**selected, "trainer_capture_max_tokens": max(4, bound)}
        if full_comparison_enabled(cfg.isoexec):
            if self.config.runtime_role != "combined":
                raise ValueError("Tinker full comparison needs a combined training/inference runtime")
            if cfg.trainer.micro_forward_batch_size_per_gpu != cfg.trainer.micro_train_batch_size_per_gpu:
                raise ValueError("Tinker full comparison requires equal forward and training microbatch sizes")
            validate_full_distribution_runtime(cfg)
        resolve(cfg)
        # SkyRL's Tinker engine runs the trainer driver in this process, not in a Ray task that starts with
        # the job's runtime environment as main_base's skyrl_entrypoint does. configure_isoexec_vllm_engine
        # decides the engine's arguments here, chunked prefill and prefix caching from the GDN composition
        # in the environment, so this process takes the environment every actor of the job starts with.
        os.environ.update(cfg.trainer.extra_runtime_env_vars)
        return cfg

    def _load_tokenizer(self):
        cfg = self._build_config()
        # Preserve the immutable checkpoint and its resolved source JSON when
        # the native backend rebuilds its worker config for create_model.
        checkpoint = cfg.trainer.policy.model.path
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("IsoExec did not resolve a policy checkpoint")
        self.base_model = checkpoint
        assert self.config.model_extra is not None
        self.config.model_extra["isoexec"] = cfg.isoexec
        raw = source_config(cfg.isoexec, cfg.trainer, checkpoint)[1]
        vocab = raw.get("text_config", raw).get("vocab_size")
        if type(vocab) is not int or vocab < 1:
            raise ValueError("IsoExec Tinker requires a text configuration with a positive vocab_size")
        self._vocab_size = vocab
        options = cast(SkyRLConfig, getattr(cfg.trainer, "_isoexec_config"))
        return get_tokenizer(options.tokenizer_path)

    def _build_policy(self, PolicyWorker, model_id: str):
        from .workers import PolicyWorker as IsoExecPolicyWorker

        return super()._build_policy(IsoExecPolicyWorker, model_id)

    def create_model(self, model_id: str, lora_config: types.LoraConfig, model_role: str = "policy") -> None:
        if model_role != "policy" or lora_config is None or lora_config.rank != 0:
            raise ValueError("IsoExec Tinker supports one full-parameter policy model (rank=0)")
        if self._model_ids_to_role:
            raise ValueError("IsoExec Tinker supports one full-parameter model per backend")
        super().create_model(model_id, lora_config, model_role)
        self._isoexec_state = TinkerState()

    def _full_comparison(self) -> bool:
        return full_comparison_enabled((self.config.model_extra or {}).get("isoexec"))

    def _sequences(self, prepared: types.PreparedModelPassBatch) -> list[tuple[int, ...]]:
        sequences = []
        for model_input, targets, weights in zip(prepared.all_model_inputs, prepared.all_targets,
                                                 prepared.all_token_weights, strict=True):
            inputs = _tokens(model_input)
            if not inputs or len(targets) != len(inputs) or tuple(targets[:-1]) != inputs[1:]:
                raise ValueError("IsoExec Tinker requires next-token targets aligned with the complete input sequence")
            tokens = inputs + (targets[-1],)
            if len(tokens) > self._max_sequence_length or not 0 < len(weights) <= len(targets):
                raise ValueError("Tinker sequence/token weights exceed the configured training bounds")
            sequences.append(tokens)
        return sequences

    def _request_rows(self, prepared: types.PreparedModelPassBatch, *, required: bool):
        supplied_fields = {
            "action_log_probs": prepared.all_sampling_logprobs,
            "rollout_logprobs": prepared.all_rollout_logprobs,
        }
        for name, values in supplied_fields.items():
            if values and (len(values) != len(prepared.all_model_inputs)
                           or (any(values) and not all(values))):
                raise ValueError(f"Tinker {name} must be provided consistently across the batch")
        records = []
        for index, (model_id, tokens, weights) in enumerate(zip(
            prepared.all_model_ids, self._sequences(prepared), prepared.all_token_weights, strict=True,
        )):
            if any(not np.isfinite(weight) or weight < 0 for weight in weights):
                raise ValueError("Tinker full comparison requires finite nonnegative token weights")
            record = self._isoexec_state.find(model_id, tokens, required=required)
            if record is not None:
                count = min(len(weights), len(record.response))
                if any(weight > 0 for weight in weights[:len(weights) - count]):
                    raise Refusal("Tinker loss includes tokens outside the independently captured response rows")
                expected = np.zeros(len(weights), dtype=np.float32)
                if count:
                    expected[-count:] = record.sampled[-count:]
                active = np.asarray(weights) > 0
                for name, values in supplied_fields.items():
                    if not values or not values[index]:
                        continue
                    supplied = np.asarray(values[index], dtype=np.float32)
                    if supplied.shape != expected.shape or not np.isfinite(supplied).all():
                        raise ValueError(f"Tinker {name} must be finite and align with token weights")
                    if not np.array_equal(supplied[active].view(np.uint32), expected[active].view(np.uint32)):
                        raise Refusal(f"Tinker {name} disagrees with the captured sampled-token values")
            records.append(record)
        return records

    def _to_training_batch(self, prepared_batch: types.PreparedModelPassBatch, role: str):
        self._sequences(prepared_batch)
        return super()._to_training_batch(prepared_batch, role)

    def _compare_batch(self, batch: TrainingInputBatch, model_id: str, *, backward: bool):
        """Compare the captured response window without changing the training batch.

        The comparison is a policy forward over its own batch, built and checked like the one
        SkyRL's trainer runs under IsoExec (``RayPPOTrainer._isoexec_policy_forward``).
        """
        from skyrl.backends.skyrl_train.distributed.dispatch import loss_fn_outputs_to_tensor

        assert batch.metadata is not None and self._dispatch is not None
        sequences = cast(torch.Tensor, batch["sequences"])
        attention = cast(torch.Tensor, batch["attention_mask"])
        loss_mask = cast(torch.Tensor, batch["loss_mask"])
        response_mask = cast(torch.Tensor, batch["response_mask"])
        real_rows = len(batch) - int(batch.metadata.get("pad_size", 0))
        records = [self._isoexec_state.find(
            model_id, sequences[index][attention[index].bool()].tolist(),
            required=backward,
        ) for index in range(real_rows)] + [None] * (len(batch) - real_rows)
        counts = [min(int(response_mask[index].sum()), len(record.response)) if record is not None else 0
                  for index, record in enumerate(records)]
        width = max(counts, default=0)
        if width == 0:
            if backward:
                raise Refusal("Tinker full comparison requires trainable captured response tokens")
            return None
        array_bytes = len(batch) * width * self._vocab_size * 4
        if array_bytes > MAX_PENDING_FULL_LOGPROB_BYTES:
            raise Refusal("Tinker prescore full rows exceed the 8 GiB diagnostic bound; reduce the model-pass batch")
        comparison_mask = torch.zeros((len(batch), width), dtype=torch.float32)
        rollout = torch.zeros_like(comparison_mask)
        arrays = []
        for index, (record, count) in enumerate(zip(records, counts, strict=True)):
            if record is None or count == 0:
                arrays.append(np.empty((0, self._vocab_size), dtype=np.float32))
                continue
            if backward and bool((loss_mask[index, :-count] > 0).any()):
                raise Refusal("Tinker loss includes tokens outside the captured response window")
            comparison_mask[index, -count:] = loss_mask[index, -count:] if backward else 1.0
            rollout[index, -count:] = torch.from_numpy(record.sampled[-count:].copy())
            arrays.append(record.rows[-count:])
        if not bool((comparison_mask > 0).any()):
            if backward:
                raise Refusal("Tinker full comparison requires trainable response tokens")
            return None
        comparison = TrainingInputBatch({
            "sequences": sequences, "attention_mask": attention, "loss_mask": comparison_mask,
            "rollout_logprobs": rollout, "rollout_full_logprobs": collate_rollout_full_logprobs(arrays, width),
            # A worker gets these rows split across DP ranks and regrouped into micro-batches.
            "sample_indices": torch.arange(len(batch)),
        })
        comparison.metadata = {"response_length": width}
        isoexec = (self.config.model_extra or {}).get("isoexec")
        try:
            before_policy_forward(isoexec, comparison, comparison, global_step=self._isoexec_state.version + 1)
            output = self._dispatch.forward("policy", comparison)
            actual = loss_fn_outputs_to_tensor(output.loss_fn_outputs, key="logprobs")
            verified_rows = after_policy_forward(comparison, comparison, actual)["policy/full_logprobs_verified_rows"]
            row_counts = tuple(int(count) for count in (comparison_mask > 0).sum(dim=1).tolist())
            if verified_rows != sum(row_counts):
                raise Refusal("Tinker comparison receipt does not cover the active training rows")
            return row_counts
        except Exception:
            self._isoexec_state.fail()
            raise

    def _execute_model_pass(self, role: str, batch: TrainingInputBatch, *, backward: bool = False, **kwargs):
        self._isoexec_state.require_healthy()
        full = self._full_comparison()
        compared = self._compare_batch(batch, kwargs["model_id"], backward=backward) if full else None
        if backward and full and compared is None:
            raise Refusal("Tinker backward requires a completed full comparison")
        try:
            result = super()._execute_model_pass(role, batch, backward=backward, **kwargs)
        except Exception:
            self._isoexec_state.fail()
            raise
        if backward:
            if full:
                self._isoexec_state.compared_rows_per_sample = compared
            self._isoexec_state.completed_backward_calls += 1
        return result

    def _forward_backward_single_model_batch(self, prepared_batch: types.PreparedModelPassBatch):
        full = self._full_comparison()
        self._isoexec_state.require_healthy()
        self._isoexec_state.compared_rows_per_sample = None
        if full:
            self._request_rows(prepared_batch, required=True)
        previous_calls = self._isoexec_state.completed_backward_calls
        try:
            results = super()._forward_backward_single_model_batch(prepared_batch)
        except Exception:
            # Input preparation errors leave gradients untouched. A response
            # formatting failure after completed backward must still stop updates.
            if self._isoexec_state.completed_backward_calls > previous_calls:
                self._isoexec_state.fail()
            raise
        if full:
            if any(isinstance(result, types.ErrorResponse) for result in results.values()):
                self._isoexec_state.fail()
                return results
            row_counts = self._isoexec_state.compared_rows_per_sample
            if row_counts is None:
                self._isoexec_state.fail()
                raise Refusal("Tinker backward completed without its full-comparison receipt")
            for request_id, _, start, stop in prepared_batch.request_batch_slices:
                result = results[request_id]
                assert isinstance(result, types.ForwardBackwardOutput)
                rows = sum(row_counts[start:stop])
                result.metrics["isoexec/full_logprobs_verified_rows:sum"] = rows
                result.metrics["isoexec/full_logprobs_compared_elements:sum"] = rows * self._vocab_size
            self._isoexec_state.compared_rows_per_sample = None
        return results

    def optim_step(self, model_id: str, request_data: types.OptimStepInput):
        if self._full_comparison():
            self._isoexec_state.require_update()
        else:
            self._isoexec_state.require_healthy()
        try:
            result = super().optim_step(model_id, request_data)
        except Exception:
            self._isoexec_state.fail()
            raise
        self._isoexec_state.updated()
        return result

    def save_sampler_checkpoint(self, output_path, model_id: str, persist: bool = True) -> None:
        if persist:
            raise NotImplementedError("IsoExec does not support native HF checkpoint export; "
                                      "use Tinker's ephemeral sampling client (persist=False)")
        self._isoexec_state.require_healthy()
        # A failed transfer must not leave an old acknowledgement authoritative.
        self._isoexec_state.synced_version = None
        super().save_sampler_checkpoint(output_path, model_id, persist=False)
        checkpoint_id = Path(str(output_path)).name.removesuffix(".tar.gz")
        self._isoexec_state.acknowledge_sync(checkpoint_id)

    def save_checkpoint(self, output_path, model_id: str) -> None:
        raise NotImplementedError("IsoExec native training-checkpoint export is not supported")

    def load_checkpoint(self, checkpoint_path, model_id: str, load_optimizer: bool) -> None:
        raise NotImplementedError("IsoExec native training-checkpoint restore is not supported")

    def delete_model(self, model_id: str) -> None:
        self._get_role(model_id)
        try:
            try:
                close_online_prescore()
            finally:
                super().delete_model(model_id)
        except Exception:
            self._isoexec_state.fail()
            raise
        finally:
            self._isoexec_state.clear_rows()

    async def _sample_with_remote_client_async(self, prepared_batch, close_client):
        self._isoexec_state.require_healthy()
        if not self._full_comparison():
            return await super()._sample_with_remote_client_async(prepared_batch, close_client)
        inference_client, cfg = self._inference_engine_client, self._cfg
        if inference_client is None or cfg is None:
            raise RuntimeError("Create the Tinker policy and synchronize sampler weights before sampling")
        client = inference_client._get_generate_client()
        model_name = resolve_policy_model_name(cfg)

        async def sample_request(request_id, model_id, start, stop, prompt_logprobs, topk):
            try:
                if not model_id or self._get_role(model_id) != "policy":
                    raise Refusal("Full comparison requires the registered Tinker policy sampling client")
                if prompt_logprobs or topk:
                    raise ValueError("Tinker full comparison captures response rows; prompt-logprob requests are unsupported")
                requests = []
                reservation = 0
                for index in range(start, stop):
                    version = self._isoexec_state.require_sampler(prepared_batch.all_checkpoint_ids[index])
                    model_input = prepared_batch.all_model_inputs[index]
                    prompt = _tokens(model_input)
                    params = prepared_batch.all_sampling_params[index]
                    if params.temperature != 1.0:
                        raise ValueError("Tinker full comparison requires temperature=1")
                    if not prompt or params.max_tokens < 1 or len(prompt) + params.max_tokens > self._max_sequence_length:
                        raise ValueError("Tinker prompt plus max_tokens exceeds the configured model/training limit")
                    reservation += params.max_tokens * (self._vocab_size + 1) * 4
                    requests.append((index, version, prompt, params))
                with self._isoexec_state.reserve(reservation):
                    async def generate(index, version, prompt, params):
                        sampling = {"temperature": 1.0, "top_p": params.top_p, "top_k": params.top_k,
                                    "max_tokens": params.max_tokens, "seed": params.seed, "n": 1, "output_kind": 2,
                                    "stop_token_ids": params.stop_tokens, "stop": params.stop_strings}
                        result = await client.generate(
                            prompt_token_ids=list(prompt), sampling_params=sampling,
                            session_id=prepared_batch.all_session_ids[index], model=model_name,
                            return_full_logprobs=True,
                        )
                        if result.full_logprobs is None or result.response_logprobs is None:
                            raise Refusal("Tinker inference omitted full FP32 rows or sampled logprobs")
                        if result.full_logprobs.shape[1] != self._vocab_size:
                            raise Refusal("Tinker engine vocabulary differs from the resolved model configuration")
                        record = SampleRows(model_id, version, prompt, tuple(result.response_ids),
                                            result.full_logprobs, np.asarray(result.response_logprobs, dtype=np.float32))
                        sequence = types.GeneratedSequence(
                            tokens=result.response_ids, logprobs=result.response_logprobs,
                            stop_reason="stop" if result.stop_reason in ("stop", "stop_token") else "length",
                        )
                        return record, sequence

                    generated = await asyncio.gather(*(generate(*request) for request in requests),
                                                     return_exceptions=True)
                    records, sequences = [], []
                    for result in generated:
                        if isinstance(result, BaseException):
                            raise result
                        record, sequence = result
                        records.append(record)
                        sequences.append(sequence)
                    self._isoexec_state.commit(records, reservation)
                    return request_id, types.SampleOutput(sequences=sequences)
            except Exception as error:
                return request_id, types.ErrorResponse(error=str(error), status="error")

        try:
            return dict(await asyncio.gather(*(sample_request(*part) for part in prepared_batch.request_batch_slices)))
        finally:
            if close_client:
                await inference_client.aclose()
