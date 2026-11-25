"""SkyRL microbatch metadata for native Megatron aux backward, owned by IsoExec."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, cast

import torch
import torch.distributed as dist

from isoexec.integrations.skyrl.aux_batch import VALID_ROWS, annotate_microbatches
from isoexec.compiler.loss import AuxiliaryLoss
from isoexec.nn import MoE
from isoexec.nn.inspect import executed_modules
from isoexec.runtimes.megatron.model import MegatronModel
from isoexec.schema import Placement


def bind(stage: Any, megatron_config: Any) -> AuxiliaryLoss | None:
    """Select Megatron aux_loss after validate_model_options has checked the settings."""
    coefficient = (float(megatron_config.moe_aux_loss_coeff)
                   if megatron_config.moe_router_load_balancing_type == "aux_loss" else 0.0)
    if not coefficient:
        return None
    if not isinstance(stage, MegatronModel):
        raise ValueError("Megatron auxiliary loss requires a native MegatronModel stage")
    return _bind_native(stage, coefficient)


def _bind_native(stage: MegatronModel, coefficient: float) -> AuxiliaryLoss:
    """Select Megatron aux_loss for a native PP1 stage from its MoE components."""
    if stage.pg_collection.pp.size() != 1:
        raise ValueError("Native Megatron auxiliary loss currently requires PP=1")
    if getattr(stage.config, "num_moe_experts", None) is None or getattr(stage.config, "calculate_per_token_loss", False):
        raise ValueError("Native aux backward requires a MoE carrier with calculate_per_token_loss=False")
    layers = []
    # Checkpoint regions share the registered outer modules. A second walk of
    # their programs would count each MoE objective twice.
    for site, module in executed_modules(stage.graph_module):
        if isinstance(module, MoE):
            if module.routing not in ("softmax", "softmax_renorm"):
                raise ValueError("Megatron auxiliary loss requires declared full softmax router probabilities")
            layers.append((f"{site}.router.scores_full", module.num_experts, module.top_k))
    if not layers or not {name for name, _, _ in layers} <= set(stage.aux_outputs):
        raise ValueError("Megatron auxiliary loss needs explicit scores_full outputs for every local MoE layer")
    stage.config.moe_aux_loss_coeff = coefficient
    stage.config.moe_router_load_balancing_type = "aux_loss"
    return _native_aux_loss(layers, cast(Placement, stage.placement), coefficient)


def _token_groups(placement: Placement) -> list[Any]:
    """The groups whose ranks hold disjoint slices of one microbatch's tokens: the CP cut, and the TP
    group under SP. Megatron's router reduces over the same TP x CP ranks (``tp_cp_group``)."""
    groups = []
    if placement.cp.size > 1:
        groups.append(placement.cp.group)
    if placement.sp:
        groups.append(placement.tp.group)
    return groups


def _native_aux_loss(layers: Sequence[tuple[str, int, int]], placement: Placement,
                     coefficient: float) -> AuxiliaryLoss:
    """Compute native per-layer objectives from local token slices."""
    from megatron.core.transformer.moe.moe_utils import switch_load_balancing_loss_func

    groups = _token_groups(placement)

    def loss(aux: Mapping[str, torch.Tensor], token_mask: torch.Tensor,
             cu_seqlens: torch.Tensor) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        losses = []
        metrics: dict[str, torch.Tensor | float] = {"moe_aux_loss_coeff": coefficient}
        for name, experts, top_k in layers:
            scores = aux[name]
            if tuple(scores.shape) != (token_mask.numel(), experts):
                raise ValueError(f"Router output {name} differs from its declared token/expert geometry")
            probabilities = scores.float()
            valid = token_mask.reshape(-1).to(device=scores.device, dtype=torch.float32)
            choices = torch.topk(probabilities, top_k, dim=-1).indices
            selected = torch.zeros_like(probabilities).scatter_(1, choices, 1)
            counts = (selected * valid[:, None]).sum(dim=0)
            statistics = torch.cat((counts, valid.sum().reshape(1)))
            for group in groups:
                # Explicit MoE outputs follow this rank's CP rows and SP slice. Counts and
                # the denominator are global; probabilities retain local autograd.
                dist.all_reduce(statistics, group=group)
            losses.append(switch_load_balancing_loss_func(
                probs=probabilities, tokens_per_expert=statistics[:-1],
                # Keep the count on device; all-padding batches have zero numerator.
                total_num_tokens=cast(Any, statistics[-1].clamp_min(1)), topk=top_k,
                num_experts=experts, moe_aux_loss_coeff=coefficient, fused=False, padding_mask=valid,
            ))
            logged = losses[-1].detach().clone() / coefficient
            for group in groups:
                dist.all_reduce(logged, group=group)
            metrics[f"moe_aux/{name}"] = logged
        if not losses:
            # A stage without aux outputs is skipped by the injector. A stage
            # exposing unrelated aux tensors still needs a differentiable zero.
            zero = next(iter(aux.values())).sum() * 0
            return zero, {"moe_aux_loss": zero.detach(), **metrics}
        auxiliary = torch.stack(losses).sum()
        metrics["moe_aux_loss"] = torch.stack([cast(torch.Tensor, value) for key, value in metrics.items()
                                              if key.startswith("moe_aux/")]).sum()
        return auxiliary, metrics

    return loss


@contextmanager
def microbatch_aux_loss(stage: MegatronModel, helper: AuxiliaryLoss,
                        micro_batches: Sequence[dict[str, Any]]) -> Iterator[list[dict[str, float]]]:
    """Pair a stage's forwards with their original, unpacked microbatch metadata."""
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
    from skyrl.backends.skyrl_train.workers.megatron import megatron_model_wrapper

    pack_values = getattr(megatron_model_wrapper, "_pack_sequence_values", None)

    if stage.aux_loss_injector is not None:
        raise RuntimeError("Auxiliary loss cannot overlap another microbatch schedule on the same stage")
    count = micro_batches[0]["num_microbatches"]
    real_count = micro_batches[0]["num_real_microbatches"]
    index = 0
    reports: list[dict[str, float]] = []

    def inject(output: torch.Tensor, aux: dict[str, torch.Tensor], packed: Any) -> torch.Tensor:
        nonlocal index
        batch = micro_batches[index]
        index += 1
        if not stage.aux_outputs:
            reports.append({})
            return output
        # SkyRL copies its batch before packing; derived fields on that copy
        # are not visible here. Convert the original TensorList in the same way.
        lengths = batch.get("sub_seq_lengths")
        sub_seq_lengths = [value.tolist() for value in lengths] if lengths is not None else None
        attention_mask = batch["attention_mask"].to(device=output.device)
        if pack_values is not None:
            mask = pack_values(torch.ones_like(attention_mask, dtype=torch.float32), attention_mask, packed,
                               sub_seq_lengths=sub_seq_lengths)
        else:
            # Older SkyRL revisions expose the same packing layout through a mask-only helper.
            mask = getattr(megatron_model_wrapper, "_build_packed_valid_mask")(
                attention_mask, packed, sub_seq_lengths=sub_seq_lengths)
        mask = mask.reshape(-1)
        if batch.get("is_padding_batch", False):
            mask.zero_()
        boundaries = packed.cu_seqlens_q_padded.to(device=mask.device)
        valid_rows = batch.get(VALID_ROWS)
        if valid_rows is not None:
            # Row validity is supplied by the batching adapter, independently
            # of policy/response loss masks (e.g. DAPO overlong filtering).
            row_validity = valid_rows.to(device=mask.device, dtype=torch.bool)
            if sub_seq_lengths is not None:
                segments = torch.tensor([len(row) for row in sub_seq_lengths], device=mask.device)
                row_validity = row_validity.repeat_interleave(segments)
            if row_validity.numel() != boundaries.numel() - 1:
                raise ValueError("Aux row validity does not match the packed sequence boundaries")
            mask *= row_validity.repeat_interleave(boundaries[1:] - boundaries[:-1]).to(mask.dtype)
        placement = cast(Placement, stage.placement)
        if placement.cp.size > 1:
            # Router scores are this rank's zigzag rows: 2 of every sequence's 2 * CP equal chunks. SkyRL
            # builds its model with cp_layout="zigzag" (megatron.build_module), which this selection assumes.
            if getattr(stage, "cp_layout", "zigzag") != "zigzag":
                raise ValueError(f"SkyRL aux loss selects zigzag CP rows; the stage binds {stage.cp_layout!r}")
            from isoexec.ops.collectives.cp_gather import layout_for

            mask = mask.index_select(0, layout_for(boundaries, placement.cp.size, placement.cp.rank).mine)
            boundaries = torch.div(boundaries, placement.cp.size, rounding_mode="floor")
        if placement.sp:
            if mask.numel() % placement.tp.size:
                raise ValueError("Aux packed token count must divide over the SP group")
            length = mask.numel() // placement.tp.size
            start = placement.tp.rank * length
            mask = mask.narrow(0, start, length)
            boundaries = (boundaries - start).clamp(0, length)
        auxiliary, metrics = helper(aux, mask, boundaries)
        if not isinstance(auxiliary, torch.Tensor) or auxiliary.ndim != 0:
            raise ValueError("The auxiliary loss helper must return a scalar Tensor")
        if not auxiliary.requires_grad:
            raise ValueError("The auxiliary loss must remain attached to autograd")
        reports.append({name: float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                        for name, value in metrics.items()})
        # Native scheduling divides aux gradients by all scheduled microbatches.
        # Match SkyRL's normalization over real microbatches, excluding padding.
        return MoEAuxLossAutoScaler.apply(output, auxiliary * count / max(1, real_count))

    stage.aux_loss_injector = inject
    try:
        yield reports
        if index != len(micro_batches):
            raise RuntimeError("Auxiliary loss did not observe every scheduled microbatch")
    finally:
        stage.aux_loss_injector = None
        stage.last_aux = {}


def _merge_metrics(placement: Placement, reports: list[dict[str, float]], real_count: int) -> list[dict[str, float]]:
    """Sum stage contributions after the PP schedule, then follow SkyRL's reductions."""
    from skyrl.backends.skyrl_train.workers.worker_utils import MEAN_LOSS_METRICS

    stage_reports = [reports]
    if placement.pp.size > 1:
        gathered: list[Any] = [None] * placement.pp.size
        dist.all_gather_object(gathered, reports, group=placement.pp.group)
        if any(len(peer) != len(reports) for peer in gathered):
            raise RuntimeError("PP stages observed different auxiliary microbatch counts")
        stage_reports = gathered
    merged = []
    for index in range(len(reports)):
        values: dict[str, float] = {}
        for peer in stage_reports:
            for name, value in peer[index].items():
                if name == "moe_aux_loss_coeff":
                    if name in values and values[name] != value:
                        raise ValueError("PP stages configured different auxiliary loss coefficients")
                    values[name] = value
                else:
                    values[name] = values.get(name, 0.0) + value
        for name in values:
            if (name == "loss" or name.endswith("_loss")) and name not in MEAN_LOSS_METRICS:
                values[name] /= max(1, real_count) * placement.dp.size
        merged.append(values)
    return merged


def build_wrapper(*, config: Any, actor_module: Any, megatron_config: Any, training: bool, **kwargs: Any) -> Any:
    """Select auxiliary loss and construct the matching SkyRL wrapper together."""
    from isoexec.integrations.skyrl.model_wrapper import IsoExecModelWrapper
    from isoexec.integrations.skyrl.scoring import _find_isoexec_stage

    stage = _find_isoexec_stage(actor_module)
    helper = bind(stage, megatron_config) if training else None
    if helper is None:
        return IsoExecModelWrapper(config=config, actor_module=actor_module, **kwargs)

    class AuxiliaryLossModelWrapper(IsoExecModelWrapper):
        def forward_backward_mini_batch(
            self, micro_batches, seq_len, micro_batch_size, temperature=1.0,
            loss_fn=None, loss_fn_config=None, forward_only=False, return_per_token_outputs=True,
        ):
            if forward_only:
                return super().forward_backward_mini_batch(
                    micro_batches, seq_len, micro_batch_size, temperature, loss_fn,
                    loss_fn_config, forward_only, return_per_token_outputs)
            micro_batches = annotate_microbatches(micro_batches, required=True)
            with microbatch_aux_loss(stage, helper, micro_batches) as auxiliary_metrics:
                reports = super().forward_backward_mini_batch(
                    micro_batches, seq_len, micro_batch_size, temperature, loss_fn,
                    loss_fn_config, forward_only, return_per_token_outputs)
            metrics = _merge_metrics(cast(Placement, stage.placement), auxiliary_metrics,
                                     micro_batches[0]["num_real_microbatches"])
            for report, values in zip(reports, metrics, strict=True):
                report.update(values)
            return reports

    return AuxiliaryLossModelWrapper(config=config, actor_module=actor_module, **kwargs)
