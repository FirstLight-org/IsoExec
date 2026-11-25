"""verl's native Megatron engine around IsoExec's host-owned ``MegatronModel``.

verl keeps its worker, batch dispatch, micro-batching, Megatron schedule, loss and LR scheduler.
This engine replaces only what the model owns: construction from the IsoExec model package, DDP
and the native optimizer (``runtimes.megatron.factory``), the model's own log-softmax for the
sampled-token log probabilities, and the weight stream sent to the rollout engines
(``IsoExecWeightSource``). While scoring a rollout it also compares every response row with the
engine's published complete-vocabulary row.

Selected with ``VERL_ENGINE_VENDOR=isoexec`` after this module is imported in each worker
(``actor_rollout_ref.model.external_lib=isoexec.integrations.verl.engine``).
"""

from __future__ import annotations

import hashlib
import os
from functools import partial
from typing import Any, cast

import torch
from megatron.core import parallel_state as mpu
from verl.utils import tensordict_utils as tu
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

from isoexec.checks import Refusal

VENDOR = "isoexec"
#: Packed source codes: ``sample * SAMPLE_STRIDE + position + 1``; zero marks THD padding.
SAMPLE_STRIDE = 1 << 24


def _unwrap(module: Any):
    from megatron.core.utils import unwrap_model

    from isoexec.runtimes.megatron.model import MegatronModel

    models = unwrap_model(list(module) if isinstance(module, (list, tuple)) else [module])
    if len(models) != 1 or not isinstance(models[0], MegatronModel):
        raise Refusal("IsoExec verl engine requires exactly one native MegatronModel chunk")
    return models[0]


def validate_engine_config(engine_config: Any, model_config: Any) -> None:
    """Refuse verl Megatron options this engine does not implement instead of ignoring them."""
    checks = {
        "virtual_pipeline_model_parallel_size": (engine_config.virtual_pipeline_model_parallel_size, None),
        "dynamic_context_parallel": (bool(engine_config.dynamic_context_parallel), False),
        "use_fused_kernels": (bool(engine_config.use_fused_kernels), False),
        "use_dist_checkpointing": (bool(engine_config.use_dist_checkpointing), False),
        "use_remove_padding": (bool(engine_config.use_remove_padding), True),
        "use_megatron_fsdp": (bool(getattr(engine_config, "use_megatron_fsdp", False)), False),
        "router_replay.mode": (engine_config.router_replay.mode, "disabled"),
    }
    for name, (value, expected) in checks.items():
        if value != expected:
            raise ValueError(f"IsoExec verl engine requires megatron.{name}={expected!r}, got {value!r}")
    cp = engine_config.context_parallel_size
    if type(cp) is not int or cp < 1:
        raise ValueError(f"IsoExec verl engine requires a positive int megatron.context_parallel_size, got {cp!r}")
    if engine_config.pad_to_length:
        raise ValueError("IsoExec verl engine does not support megatron.pad_to_length")
    if model_config.lora_rank or model_config.lora.get("rank", 0):
        raise ValueError("IsoExec verl engine does not support LoRA")
    if model_config.mtp.enable:
        raise ValueError("IsoExec verl engine does not support MTP")
    if engine_config.override_transformer_config.get("fp8") or engine_config.override_transformer_config.get("fp4"):
        raise ValueError("IsoExec verl engine does not support low-precision training recipes")
    etp = engine_config.expert_tensor_parallel_size
    ep, tp = engine_config.expert_model_parallel_size, engine_config.tensor_model_parallel_size
    allowed = (1, tp) if ep == 1 else (1,)
    if etp is not None and etp not in allowed:
        raise ValueError(f"expert_tensor_parallel_size={etp} must be one of {allowed} for expert_model_parallel_size={ep}")


def capture_bound(engine_config: Any) -> int:
    """Packed-token bound for trainer capture under verl's dynamic batching.

    verl packs training (and inference) micro-batches to at most ``max_token_len_per_gpu``
    (``infer_max_token_len_per_gpu``) real tokens and pads each
    sequence to a multiple of TP*CP, so a micro-batch holds at most that many tokens times TP*CP
    globally, and each CP rank holds a contiguous 1/CP of it: at most that many tokens times TP.

    Static micro-batching (``use_dynamic_bsz=false``) is bounded by micro-batch size times the
    per-sequence cap ``max_token_len_per_gpu`` (verl requires prompt+response to fit it); a batch
    past the bound is refused by the capture's dynamic range, never run silently.
    """
    pairs = [(engine_config.max_token_len_per_gpu, getattr(engine_config, "micro_batch_size_per_gpu", None)),
             (engine_config.infer_max_token_len_per_gpu, getattr(engine_config, "infer_micro_batch_size_per_gpu", None))]
    pairs = [(tokens, micro) for tokens, micro in pairs if tokens is not None]
    if not pairs or any(type(tokens) is not int or tokens < 1 for tokens, _ in pairs):
        raise ValueError(f"max_token_len_per_gpu / infer_max_token_len_per_gpu must be positive ints, got {pairs!r}")
    if engine_config.use_dynamic_bsz:
        budgets = [tokens for tokens, _ in pairs]
    else:
        if any(type(micro) is not int or micro < 1 for _, micro in pairs):
            raise ValueError(f"static micro-batching needs positive int micro_batch_size_per_gpu, got {pairs!r}")
        budgets = [tokens * micro for tokens, micro in pairs]
    return max(budgets) * engine_config.tensor_model_parallel_size


#: Re-digest the parameters before every optimizer update instead of reusing the previous "after".
DIGEST_BEFORE_ENV = "ISOEXEC_OPTIMIZER_DIGEST_BEFORE"


def parameter_fingerprint(module: Any) -> str:
    """Digest of every local registered parameter, in name order (W0/W1 receipt; see ``digest``)."""
    from .digest import parameter_digest

    return parameter_digest(_unwrap(module).graph_module.named_parameters())


@EngineRegistry.register(model_type="language_model", backend="megatron", vendor=VENDOR)
class IsoExecMegatronEngine(MegatronEngineWithLMHead):
    """verl Megatron engine whose model, DDP and optimizer come from IsoExec."""

    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config):
        validate_engine_config(engine_config, model_config)
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        self._isoexec_optimizer = None
        self._isoexec_optimizer_steps = 0
        self._isoexec_head = None
        self._isoexec_last_digest = None

    # ---------------------------------------------------------------- construction
    def _build_tf_config(self):
        # The model package, not mbridge, defines the model; the TransformerConfig comes from capture.
        # Keep the attributes verl's own builder sets that the rest of the engine reads.
        from verl.utils.torch_dtypes import PrecisionType

        self.is_value_model = self.model_config.model_type == "value_model"
        if self.is_value_model:
            raise ValueError("IsoExec verl engine supports language-model policies only")
        self.share_embeddings_and_output_weights = self.model_config.share_embeddings_and_output_weights
        self.param_dtype = PrecisionType.to_dtype(self.engine_config.dtype)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        self.provider = None
        self.bridge = None
        self.peft_cls = None
        self.tf_config = None

    def _build_megatron_module(self):
        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.process_groups_config import ProcessGroupCollection
        from verl.utils.megatron.optimizer import init_megatron_optim_config

        from isoexec.runtimes.megatron.factory import build_model, wrap_ddp_and_optimizer
        from isoexec.runtimes.trainer_installs import install_trainer_process

        from .model import checkpoint_directory, source_config

        if self.param_dtype != torch.bfloat16:
            raise ValueError("IsoExec verl engine requires bf16 parameters")
        path = str(checkpoint_directory(self.model_config.local_path))
        source, raw = source_config(path)
        cfg = self.engine_config
        _validate_expert_data_parallel(cfg)
        recompute = cfg.override_transformer_config.get("recompute_granularity") is not None
        deterministic = bool(cfg.override_transformer_config.get("deterministic_mode", False))
        sequence_parallel = bool(cfg.sequence_parallel)
        if sequence_parallel and cfg.tensor_model_parallel_size < 2:
            raise ValueError("sequence_parallel=true needs tensor_model_parallel_size >= 2")
        install_trainer_process(path, source=source, raw_config=raw, deterministic_attention_backward=deterministic)
        training = not cfg.forward_only
        model = build_model(
            source, raw, checkpoint_path=path, training=training,
            max_tokens=capture_bound(cfg),
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            transformer_config_kwargs={
                "tensor_model_parallel_size": cfg.tensor_model_parallel_size,
                "pipeline_model_parallel_size": cfg.pipeline_model_parallel_size,
                "context_parallel_size": cfg.context_parallel_size,
                "expert_model_parallel_size": cfg.expert_model_parallel_size,
                "expert_tensor_parallel_size": cfg.expert_tensor_parallel_size,
                "sequence_parallel": sequence_parallel,
                "deterministic_mode": deterministic,
                "recompute_granularity": "full" if recompute else None,
                "recompute_method": "uniform" if recompute else None,
                "recompute_num_layers": cfg.override_transformer_config.get("recompute_num_layers", 1) if recompute else None,
            },
        )
        self.tf_config = model.config
        if not training:
            return [model]
        ddp_options: dict[str, Any] = {"use_distributed_optimizer": bool(cfg.use_distributed_optimizer)}
        ddp_options.update(self._resolve_override_ddp_config())
        optim_config = init_megatron_optim_config(
            self.optimizer_config, use_distributed_optimizer=cfg.use_distributed_optimizer, fp16=False, bf16=True,
        )
        ddp, optimizer = wrap_ddp_and_optimizer(
            model, ddp_config=DistributedDataParallelConfig(**ddp_options), optimizer_config=optim_config,
        )
        self._isoexec_optimizer = optimizer
        return [ddp]

    def _build_optimizer(self):
        if self._isoexec_optimizer is None:
            raise RuntimeError("IsoExec optimizer is built with the model; forward-only engines have none")
        # The factory installed the model's own gradient finalizer and DDP no-sync callback.
        self.tf_config.grad_scale_func = self._isoexec_optimizer.scale_loss
        return self._isoexec_optimizer

    def initialize(self):
        self._hf_export_tasks = None
        self._build_tf_config()
        self.module = self._build_megatron_module()
        from isoexec.core.local_memory import stack_limit

        # The per-thread stack limit at model build; offload lowers it back (runtimes/megatron/residency).
        self._isoexec_stack_limit = stack_limit()
        if self.engine_config.forward_only:
            self.optimizer = None
            self.lr_scheduler = None
            # As verl's MegatronEngine.initialize: offload a forward-only model right after build.
            self.to(device="cpu", model=self._is_offload_param, optimizer=False, grad=False)
            return
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()
        # Native checkpoint save/restore is unsupported for IsoExec models.
        self.checkpoint_mananager = None
        # As verl's MegatronEngine.initialize: offload per the native config before the engine starts.
        self.to(device="cpu", model=self._is_offload_param, optimizer=self._is_offload_optimizer,
                grad=self._is_offload_param)

    def save_checkpoint(self, *args, **kwargs):
        raise NotImplementedError("IsoExec verl engine does not support native checkpoint save")

    def load_checkpoint(self, *args, **kwargs):
        raise NotImplementedError("IsoExec verl engine does not support native checkpoint restore")

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        # Native verl/Megatron offload: DDP param/grad buffers and optimizer state move by storage, so the
        # IsoExec module's parameter views (and any binding that reads them) stay valid. No trainer graphs.
        result = super().to(device=device, model=model, optimizer=optimizer, grad=grad)
        if device == "cpu" and model:
            from isoexec.core.local_memory import release_local_memory

            limit = getattr(self, "_isoexec_stack_limit", None)
            if limit is not None:
                release_local_memory(limit, reason="after verl trainer offload")
        return result

    # ------------------------------------------------------------------ optimizer
    def optimizer_step(self):
        from isoexec.runtimes.megatron.optimizer import finish_native_optimizer_step

        from .full_distribution import append_receipt, run_root

        # One digest per weight version: the weights change only here (native checkpoint restore is refused
        # above), so the previous update's "after" IS this update's "before". ISOEXEC_OPTIMIZER_DIGEST_BEFORE=1
        # re-digests anyway (catches an out-of-band write between updates).
        reused = self._isoexec_last_digest is not None and os.environ.get(DIGEST_BEFORE_ENV, "0") != "1"
        before = self._isoexec_last_digest if reused else parameter_fingerprint(self.module)
        # Return the backward's cached activation blocks to the device before every step: NCCL allocates its
        # communicator buffers (first grad-norm all-reduce) through cuMem, outside PyTorch's cache, and the
        # step's own allocations otherwise compete with the backward's cached blocks.
        torch.cuda.empty_cache()
        update_successful, grad_norm, num_zeros = self.optimizer.step()
        if not update_successful:
            raise Refusal("IsoExec verl engine: the native optimizer step reported no update")
        finish_native_optimizer_step(self.module)
        grad = float(grad_norm) if grad_norm is not None else float("nan")
        if not torch.isfinite(torch.tensor(grad)):
            raise Refusal(f"IsoExec verl engine: non-finite gradient norm {grad}")
        self._isoexec_optimizer_steps += 1
        after = parameter_fingerprint(self.module)
        self._isoexec_last_digest = after
        append_receipt(run_root(), f"optimizer-rank{torch.distributed.get_rank():04d}.jsonl", {
            "event": "optimizer_step", "step": self._isoexec_optimizer_steps, "grad_norm": grad,
            "num_zeros_in_grad": None if num_zeros is None else int(num_zeros),
            "parameter_sha256_before": before, "parameter_sha256_after": after,
            "changed": before != after, "before_reused": reused, "rank": torch.distributed.get_rank(),
            "tp_rank": mpu.get_tensor_model_parallel_rank(), "pp_rank": mpu.get_pipeline_model_parallel_rank(),
            "dp_rank": mpu.get_data_parallel_rank(), "ep_rank": mpu.get_expert_model_parallel_rank(),
        })
        return grad_norm

    @property
    def weight_version(self) -> int:
        return self._isoexec_optimizer_steps

    # ---------------------------------------------------------------- weight sync
    def get_per_tensor_param(self, base_sync_done=False, **kwargs):
        from isoexec.sync.weight_source import IsoExecWeightSource

        if self._is_offload_param:
            # As verl's MegatronEngine.get_per_tensor_param: an offloaded model streams from the device;
            # the worker offloads it again after the sync. NCCL gathers from host storage are illegal.
            from verl.utils.megatron_utils import load_megatron_model_to_gpu

            load_megatron_model_to_gpu(self.module, load_grad=False)
        source = IsoExecWeightSource(self.module, torch.bfloat16)
        return source.iter_full(), None

    # -------------------------------------------------------------------- forward
    def _lm_head(self):
        if self._isoexec_head is None:
            from isoexec.nn import LMHead

            model = _unwrap(self.module)
            self._isoexec_head = next(
                (module for module in model.graph_module.modules() if isinstance(module, LMHead)), None
            )
            if self._isoexec_head is None:
                raise Refusal("IsoExec verl engine: the final pipeline stage has no LMHead")
        return self._isoexec_head

    def forward_step(self, batch_iter, model, logits_processor_func, postprocess_micro_batch_func):
        from verl.models.mcore.model_forward import gptmodel_forward_model_engine
        from verl.utils.device import get_device_id
        from verl.utils.dataset.dataset_utils import DatasetPadMode

        batch = next(batch_iter).to(get_device_id())
        for key in ("distillation_use_topk", "distillation_only", "calculate_sum_pi_squared", "record_r2_routes"):
            if tu.get_non_tensor_data(batch, key=key, default=False):
                raise NotImplementedError(f"IsoExec verl engine does not support {key}")
        if tu.get_non_tensor_data(batch, key="pad_mode", default=DatasetPadMode.NO_PADDING) != DatasetPadMode.NO_PADDING:
            raise NotImplementedError("IsoExec verl engine requires pad_mode=no_padding")
        calculate_entropy = bool(tu.get_non_tensor_data(batch, key="calculate_entropy", default=False))
        input_ids = batch["input_ids"]
        if not input_ids.is_nested:
            raise NotImplementedError("IsoExec verl engine requires nested (no-padding) input_ids")
        lengths = input_ids.offsets().diff().tolist()
        temperature = batch["temperature"]
        values = temperature if isinstance(temperature, torch.Tensor) else torch.tensor([temperature])
        values = values.values() if values.is_nested else values
        unique = torch.unique(values.float())
        if unique.numel() != 1 or not bool(unique[0] > 0):
            raise NotImplementedError("IsoExec verl engine requires one positive temperature per batch")
        scalar_temperature = float(unique[0].item())
        source = torch.nested.nested_tensor(
            [torch.arange(n, dtype=torch.int64, device=input_ids.device) + index * SAMPLE_STRIDE + 1
             for index, n in enumerate(lengths)],
            layout=torch.jagged,
        )
        compare = (not torch.is_grad_enabled()) and not self.engine_config.forward_only
        processor = partial(
            self._isoexec_logits_processor, temperature_value=scalar_temperature,
            calculate_entropy=calculate_entropy, batch=batch, lengths=lengths, compare=compare,
        )
        args = {"label": input_ids.clone(), "isoexec_source": source}
        if _unwrap(model).cp_layout == "contiguous":
            # IsoExec packs contiguous CP rows itself: verl's packer refuses contiguous CP without
            # PackedSeqParams.cp_partition_mode, which the shipped megatron-core lacks and the model
            # does not need (it chose its layout at construction).
            from .thd import forward_contiguous

            output = forward_contiguous(model, input_ids, logits_processor=processor, logits_processor_args=args)
        else:
            output = gptmodel_forward_model_engine(
                model, input_ids, {},
                logits_processor=processor,
                logits_processor_args=args,
                vision_model=False,
                pad_token_id=self.model_config.tokenizer.pad_token_id,
                data_format="thd",
                position_ids=None,
                cp_layout=_unwrap(model).cp_layout or "zigzag",
            )
        return output, partial(postprocess_micro_batch_func, data=batch, local_cp_size=None)

    def _isoexec_logits_processor(self, logits, label, isoexec_source, *, temperature_value, calculate_entropy,
                                  batch, lengths, compare):
        head = self._lm_head()
        if logits.ndim != 3 or logits.shape[0] != 1 or label.shape != logits.shape[:2]:
            raise Refusal(f"IsoExec verl engine: unexpected packed logits {tuple(logits.shape)} / labels {tuple(label.shape)}")
        packed = logits[0].float()
        if compare:
            from .full_distribution import full_rows_enabled

            if full_rows_enabled():
                self._compare_full_rows(packed, isoexec_source[0], batch, lengths, temperature_value)
        ret = {"log_probs": head.log_probs(packed.contiguous(), label[0], temperature_value).unsqueeze(0)}
        if calculate_entropy:
            ret["entropy"] = self._host_entropy(packed, temperature_value)
        return ret

    def _host_entropy(self, packed: torch.Tensor, temperature: float) -> torch.Tensor:
        """verl's own vocab-parallel entropy of ``logits / T``, as verl's native Megatron engine computes it.

        Entropy is a loss term the engine never reports, so it is not IsoExec numerics: verl's function
        (and its ``entropy_from_logits_with_chunking`` knob) computes it over Megatron's TP group. Only the
        layout is adapted: IsoExec masks the trimmed vocabulary padding to ``-inf`` while verl's function
        expects finite columns, so each rank passes its real columns only (the function reduces per row,
        so shard widths may differ; the padding gets zero gradient). The scaled copy is verl's own
        ``logits_bak`` split: its in-place backward never touches the logits the log-probs read.
        """
        from verl.utils.megatron.tensor_parallel import (
            vocab_parallel_entropy,
            vocab_parallel_entropy_with_chunking,
        )

        real = _real_vocab_columns(self._lm_head(), int(packed.shape[-1]))
        scaled = (packed[:, :real] / temperature).unsqueeze(0)
        if self.engine_config.entropy_from_logits_with_chunking:
            chunk = self.engine_config.entropy_from_logits_chunk_size
            return vocab_parallel_entropy_with_chunking(scaled, chunk_size=chunk)
        return vocab_parallel_entropy(scaled)

    @torch.no_grad()
    def _compare_full_rows(self, packed_logits, source, batch, lengths, temperature):
        from . import full_distribution as fd

        root = fd.run_root()
        # The trainer names the weight version it rescored (one sync per training step, which may hold
        # several optimizer updates); without it, one update per step makes the two counts equal.
        version = tu.get_non_tensor_data(batch, key="isoexec_weight_version", default=None)
        version = self._isoexec_optimizer_steps if version is None else int(version)
        input_ids = batch["input_ids"]
        response_lengths = batch["loss_mask"].offsets().diff().tolist() if batch["loss_mask"].is_nested else None
        if response_lengths is None:
            raise Refusal("full_distribution: verl loss_mask must be nested per sample")
        slots, sample_of, column_of = select_response_slots(source, lengths, response_lengths)
        if slots.numel() == 0:
            return
        tp = mpu.get_tensor_model_parallel_group()
        tp_size, tp_rank = mpu.get_tensor_model_parallel_world_size(), mpu.get_tensor_model_parallel_rank()
        width = int(packed_logits.shape[1])
        vocab_start, vocab_end = tp_rank * width, (tp_rank + 1) * width
        offsets = input_ids.offsets().tolist()
        flat = input_ids.values()
        rollout = batch.get("rollout_log_probs", None)
        loss_mask = batch["loss_mask"]
        rank = torch.distributed.get_rank()
        samples = sample_of.tolist()
        columns = column_of.tolist()
        by_sample: dict[int, list[tuple[int, int]]] = {}
        for slot, index, column in zip(slots.tolist(), samples, columns, strict=True):
            by_sample.setdefault(index, []).append((slot, column))
        for index, pairs in sorted(by_sample.items()):
            ids = flat[offsets[index]:offsets[index + 1]].tolist()
            prompt, completion = ids[: len(ids) - response_lengths[index]], ids[len(ids) - response_lengths[index]:]
            mask = loss_mask[index].tolist()
            pairs = [(slot, column) for slot, column in pairs if mask[column]]
            if not pairs:
                continue
            rows = fd.read_rows(root, version=version, prompt_ids=prompt, response_ids=completion)
            if rows.shape[1] < vocab_end:
                raise Refusal(f"full_distribution: trainer vocabulary slice [{vocab_start}, {vocab_end}) exceeds "
                              f"engine vocabulary {rows.shape[1]}")
            slot_index = torch.tensor([slot for slot, _ in pairs], device=packed_logits.device)
            column_index = [column for _, column in pairs]
            engine = torch.from_numpy(rows[column_index, vocab_start:vocab_end].copy())
            if rollout is not None:
                sampled = torch.from_numpy(rows[column_index, [completion[c] for c in column_index]].copy())
                reported = rollout[index].detach().float().cpu()[column_index]
                if not torch.equal(sampled.view(torch.int32), reported.contiguous().view(torch.int32)):
                    raise Refusal(f"full_distribution: rollout sampled logprobs differ from their full rows "
                                  f"for sample {fd.sample_key(prompt, completion)}")
            selected_logits = packed_logits[slot_index].contiguous()
            trainer = _trainer_full_rows(self.module, selected_logits, vocab_start, vocab_end,
                                         tp if tp_size > 1 else None, temperature)
            result = fd.compare_slices(trainer, engine, vocab_start=vocab_start)
            if result.first_bad_row is None:
                trainer_stats = fd.row_stats(trainer, group=tp if tp_size > 1 else None)
                engine_stats = (fd.row_stats(torch.from_numpy(rows[column_index].copy()).to(trainer.device))
                                if tp_rank == 0 else None)
            else:
                # Record the mismatch first; the validity statistics are irrelevant to a refused batch.
                trainer_stats, engine_stats = None, None
            fd.append_receipt(root, f"compare-rank{rank:04d}.jsonl", {
                "event": "compare", "version": version, "key": fd.sample_key(prompt, completion),
                "response_columns": column_index, "vocab_slice": [vocab_start, vocab_end],
                "engine_vocab": int(rows.shape[1]), "rows": result.rows,
                "first_bad_row": result.first_bad_row, "first_token": result.first_token,
                "trainer_u32": result.trainer_u32, "engine_u32": result.engine_u32,
                "mismatch_count": result.mismatch_count, "mismatched_rows": result.mismatched_rows,
                "max_abs_diff": result.max_abs_diff,
                "trainer_stats": None if trainer_stats is None else vars(trainer_stats),
                "engine_stats": None if engine_stats is None else vars(engine_stats),
                "rank": rank, "tp_rank": tp_rank, "tp_size": tp_size,
                "pp_rank": mpu.get_pipeline_model_parallel_rank(), "dp_rank": mpu.get_data_parallel_rank(),
            })
            if result.first_bad_row is not None:
                raise Refusal(
                    f"full_distribution.bits: version={version} sample={fd.sample_key(prompt, completion)} "
                    f"row={column_index[result.first_bad_row]} token={result.first_token} "
                    f"trainer=0x{result.trainer_u32:08x} engine=0x{result.engine_u32:08x} "
                    f"mismatch_count={result.mismatch_count} mismatched_rows={result.mismatched_rows}/{result.rows} "
                    f"max_abs_diff={result.max_abs_diff:.3e}"
                )


def select_response_slots(source: torch.Tensor, lengths: list[int], response_lengths: list[int]):
    """Packed logit slots that predict a response token, with that token's sample and response column.

    ``source`` holds ``sample * SAMPLE_STRIDE + position + 1`` per packed slot (zero for THD
    padding), packed exactly like the input IDs. Slot ``p`` of a sample at position ``i`` predicts
    position ``i + 1``; it is selected when that position lies in the sample's response.
    """
    if source.ndim != 1 or source.dtype != torch.int64:
        raise Refusal("full_distribution: packed source codes must be a 1-D int64 tensor")
    valid = source > 0
    code = torch.where(valid, source - 1, torch.zeros_like(source))
    sample = torch.div(code, SAMPLE_STRIDE, rounding_mode="floor")
    position = torch.remainder(code, SAMPLE_STRIDE)
    if len(lengths) != len(response_lengths) or (valid.any() and int(sample[valid].max()) >= len(lengths)):
        raise Refusal("full_distribution: packed source codes name a sample outside the micro-batch")
    length = torch.tensor(lengths, dtype=torch.int64, device=source.device)[sample]
    response = torch.tensor(response_lengths, dtype=torch.int64, device=source.device)[sample]
    target = position + 1
    selected = valid & (target < length) & (target >= length - response)
    slots = selected.nonzero(as_tuple=True)[0]
    return slots, sample[slots], target[slots] - (length[slots] - response[slots])


def _trainer_full_rows(module, logits, vocab_start, vocab_end, group, temperature):
    """Trainer rows at the rollout temperature: the engine scores logits / T (score_temperature)."""
    model = _unwrap(module)
    shard_check = getattr(model, "compute_full_logprobs_for_check_shard", None)
    if shard_check is not None:
        rows, _temperature, bounds = shard_check(logits, temperature)
        if tuple(bounds) != (vocab_start, vocab_end):
            raise Refusal(f"full_distribution: model vocabulary slice {bounds} != ({vocab_start}, {vocab_end})")
        return rows
    if group is not None:
        from isoexec.runtimes.trainer_installs import compute_full_logprobs_for_check_shard

        rows, _ = compute_full_logprobs_for_check_shard(
            logits, temperature, vocab_start_index=vocab_start, vocab_end_index=vocab_end, group=group,
        )
        return rows
    from isoexec.runtimes.trainer_installs import compute_full_logprobs_for_check

    rows, _ = compute_full_logprobs_for_check(logits, temperature)
    return rows


def _real_vocab_columns(head, width: int) -> int:
    """How many leading columns of this rank's ``width``-wide logits shard are real vocabulary.

    The shard is either the whole trimmed vocabulary (one rank) or rank ``r``'s equal slice
    ``[r * width, (r + 1) * width)`` of the padded head over Megatron's TP group, with padding last.
    """
    tp, rank = mpu.get_tensor_model_parallel_world_size(), mpu.get_tensor_model_parallel_rank()
    if width == head.vocab and tp == 1:
        return width
    if width * tp != head.padded_vocab:
        raise Refusal(f"IsoExec verl engine: logits shard width {width} x TP {tp} is not the head's padded "
                      f"vocabulary {head.padded_vocab}; verl's vocab-parallel entropy needs Megatron TP shards")
    real = min(width, head.vocab - rank * width)
    if real < 1:
        raise Refusal(f"IsoExec verl engine: TP rank {rank} holds only vocabulary padding ({width} columns); "
                      f"verl's vocab-parallel entropy needs at least one real column per rank")
    return real


def _validate_expert_data_parallel(cfg) -> None:
    """Refuse a live expert mesh that differs from the configured one (EP | TP x CP x DP, ETP=1)."""
    ep = cfg.expert_model_parallel_size
    if ep <= 1:
        return
    from isoexec.runtimes.parallel import expert_data_parallel_degree, expert_data_parallel_refusal

    expert_dp = expert_data_parallel_degree(
        cfg.tensor_model_parallel_size, mpu.get_data_parallel_world_size(with_context_parallel=False), ep,
        mpu.get_context_parallel_world_size())
    refusal = expert_data_parallel_refusal(expert_dp, int(mpu.get_expert_data_parallel_world_size()))
    if refusal:
        raise ValueError(refusal)


__all__ = ["VENDOR", "IsoExecMegatronEngine", "capture_bound", "parameter_fingerprint", "select_response_slots",
           "validate_engine_config"]
