"""The model-owned logprob operation SkyRL's packed scorer runs for an IsoExec model."""

import torch

from isoexec.checks import Refusal
from isoexec.nn import LMHead


def _find_isoexec_stage(actor_module):
    chunks = actor_module if isinstance(actor_module, (list, tuple)) else [actor_module]
    candidates = []
    for chunk in chunks:
        inner = chunk
        seen = set()
        while id(inner) not in seen:
            seen.add(id(inner))
            candidates.append(inner)
            if hasattr(inner, "module"):
                inner = inner.module
            else:
                break
    from isoexec.runtimes.megatron.model import MegatronModel

    for inner in candidates:
        if isinstance(inner, MegatronModel):
            return inner
    return None


def lm_head_log_probs(wrapper):
    """``log_probs`` of the LM head registered in the wrapper's IsoExec stage; ``None`` on a stage without the head."""
    stage = _find_isoexec_stage(wrapper.actor_module)
    if stage is None:
        raise RuntimeError("IsoExec was configured but no IsoExec Megatron stage was built")
    if not wrapper.remove_microbatch_padding:
        raise ValueError("IsoExec requires trainer.remove_microbatch_padding=true")
    head = next((module for module in stage.graph_module.modules() if isinstance(module, LMHead)), None)
    if head is None:
        if stage.post_process:
            raise RuntimeError("The IsoExec stage that produces logits has no registered LMHead")
        return None
    return head.log_probs


def target_logprobs(compute_logprobs, logits: torch.Tensor, targets: torch.Tensor, *,
                    chunk_size: int | None = None) -> torch.Tensor:
    """Each target token's logprob from its logits row, computed by the model's LM head.

    SkyRL's packed scorer finds the targets and scatters the results; it calls this in between with one
    rank's logits ``[1, T, vocab]`` and shifted targets ``[1, T]``, and gets ``[1, T]`` back.
    ``chunk_size`` bounds how many rows the head scores at once.
    """
    logits, targets = logits.squeeze(0), targets.squeeze(0)
    rows = int(logits.shape[0])
    if rows != int(targets.shape[0]):
        raise ValueError(f"packed scoring: {rows} local logit rows for {int(targets.shape[0])} local targets")
    step = int(chunk_size) if chunk_size is not None and int(chunk_size) < rows else rows
    parts = [compute_logprobs(lg.float().contiguous(), tg, 1.0)
             for lg, tg in zip(torch.split(logits, step, dim=0), torch.split(targets, step, dim=0))]
    probs = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    if probs.dim() != 1:
        raise ValueError(f"expected 1D per-token logprobs from the LM head, got shape {tuple(probs.shape)}")
    return probs.unsqueeze(0)


def _evidence_context(data) -> dict:
    """JSON-safe scalars SkyRL forwards beside the rows (absent keys stay None)."""
    context = {}
    for key in ("global_step", "weight_version", "num_actions"):
        value = data.get(key)
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else None
        context[key] = value if isinstance(value, (int, float, str)) or value is None else str(value)
    return context


@torch.no_grad()
def verify_full_distribution(wrapper, logits, data) -> int:
    """Compare trainable response rows using IsoExec's trainer producer and comparator."""
    from isoexec.integrations.skyrl.evidence import FullDistributionEvidence

    evidence = FullDistributionEvidence.from_env()
    try:
        return _verify_full_distribution(wrapper, logits, data, evidence)
    except Refusal as exc:
        if evidence is not None:
            evidence.record_refusal(str(exc), context=_evidence_context(data))
        raise


def _verify_full_distribution(wrapper, logits, data, evidence) -> int:
    from isoexec.checks.full_distribution import compare_full_distribution

    packed_seq_params = data.get("packed_seq_params")
    stage = _find_isoexec_stage(wrapper.actor_module)
    if stage is None:
        raise Refusal("full_distribution: no IsoExec Megatron stage is active")
    if packed_seq_params is None:
        raise Refusal("full_distribution: the local SkyRL integration requires packed THD scoring")
    if data.get("sub_seq_lengths_list") is not None:
        raise Refusal("full_distribution: controller sub-sequence packing is not supported")

    engine_rows = data.get("rollout_full_logprobs")
    mask = data.get("loss_mask")
    rollout_logprobs = data.get("rollout_logprobs")
    sequences = data["sequences"]
    attention_mask = data["attention_mask"].to(dtype=torch.bool)
    if engine_rows is None or mask is None or rollout_logprobs is None:
        raise Refusal("full_distribution: rollout rows, sampled logprobs, and loss mask are required")
    if (
        engine_rows.dtype is not torch.float32
        or engine_rows.ndim != 3
        or mask.shape != engine_rows.shape[:2]
        or rollout_logprobs.shape != mask.shape
    ):
        raise Refusal(
            "full_distribution: expected float32 [batch,response,vocab] rows with aligned mask and sampled scores"
        )

    selected = mask > 0
    coordinates = selected.nonzero(as_tuple=False)
    if coordinates.numel() == 0:
        return 0
    batch_size, response_length, vocab_size = engine_rows.shape
    if sequences.shape[0] != batch_size or response_length <= 0 or vocab_size <= 0:
        raise Refusal("full_distribution: response rows do not align with trainer sequences")

    row_indices = coordinates[:, 0]
    response_columns = coordinates[:, 1]
    target_columns = sequences.shape[1] - response_length + response_columns
    if bool((target_columns <= 0).any()) or not bool(attention_mask[row_indices, target_columns].all()):
        raise Refusal("full_distribution: selected response token has no real predecessor")

    token_ordinals = attention_mask.to(torch.long).cumsum(dim=1) - 1
    target_ordinals = token_ordinals[row_indices, target_columns]
    cu_seqlens = packed_seq_params.cu_seqlens_q_padded.to(device=logits.device, dtype=torch.long)
    if cu_seqlens.numel() != batch_size + 1:
        raise Refusal("full_distribution: packed sequence ledger does not match the SkyRL batch")
    predictor_indices = cu_seqlens[row_indices] + target_ordinals - 1
    packed_logits = logits.squeeze(0)
    cp_group = getattr(getattr(stage, "pg_collection", None), "cp", None)
    cp_size = int(cp_group.size()) if cp_group is not None else 1
    if cp_size > 1:
        # Map global predictor positions onto this rank's context-parallel logits.
        from isoexec.ops.collectives.cp_gather import layout_for

        cp_rank = int(cp_group.rank())
        layout = layout_for(packed_seq_params.cu_seqlens_q_padded, cp_size, cp_rank)
        if int(cu_seqlens[-1]) != layout.total or packed_logits.ndim != 2 or packed_logits.shape[0] != layout.local:
            raise Refusal(
                f"full_distribution: context-parallel layout ({layout.total} global / {layout.local} local tokens) "
                f"does not match the packed ledger ({int(cu_seqlens[-1])}) or the logits {tuple(packed_logits.shape)}"
            )
        # ``perm[g]`` is the slot of GLOBAL token g in the rank-ordered gather (rank * local + local index).
        slot = layout.perm.to(predictor_indices.device)[predictor_indices]
        owned = torch.div(slot, layout.local, rounding_mode="floor") == cp_rank
        if not bool(owned.any()):
            return 0
        coordinates, row_indices = coordinates[owned], row_indices[owned]
        response_columns, target_columns = response_columns[owned], target_columns[owned]
        predictor_indices = torch.remainder(slot[owned], layout.local)
        selected = torch.zeros_like(selected)
        selected[row_indices, response_columns] = True
    if packed_logits.ndim != 2 or bool((predictor_indices >= packed_logits.shape[0]).any()):
        raise Refusal("full_distribution: response-to-packed-logit mapping is out of bounds")

    shard_check = getattr(stage, "compute_full_logprobs_for_check_shard", None)
    if shard_check is None and not getattr(stage, "post_process", True):
        raise Refusal("full_distribution: only the last PP stage holds the head's logits")
    selected_logits = packed_logits[predictor_indices].float().contiguous()
    # Native models carry their TP group; at TP>1 the logits are this rank's vocabulary columns.
    tp = getattr(getattr(stage, "pg_collection", None), "tp", None)
    if shard_check is not None:
        trainer_rows, _temperature, (vocab_start, vocab_end) = shard_check(selected_logits, 1.0)
    elif tp is not None and tp.size() > 1:
        from isoexec.runtimes.trainer_installs import compute_full_logprobs_for_check_shard

        width = int(selected_logits.shape[1])
        vocab_start, vocab_end = tp.rank() * width, (tp.rank() + 1) * width
        trainer_rows, _temperature = compute_full_logprobs_for_check_shard(
            selected_logits, 1.0, vocab_start_index=vocab_start, vocab_end_index=vocab_end, group=tp,
        )
    else:
        from isoexec.runtimes.trainer_installs import compute_full_logprobs_for_check

        trainer_rows, _temperature = compute_full_logprobs_for_check(selected_logits, 1.0)
        vocab_start, vocab_end = 0, int(trainer_rows.shape[1])
    if not (0 <= vocab_start < vocab_end <= vocab_size) or vocab_end - vocab_start != int(trainer_rows.shape[1]):
        raise Refusal(
            f"full_distribution: trainer vocabulary slice [{vocab_start}, {vocab_end}) does not fit "
            f"the engine vocabulary {vocab_size} or the trainer rows {tuple(trainer_rows.shape)}"
        )
    # Compare each TP vocabulary slice against the matching engine columns.
    selected_engine_rows = engine_rows[selected][:, vocab_start:vocab_end].contiguous()
    if trainer_rows.shape != selected_engine_rows.shape:
        raise Refusal(
            f"full_distribution: trainer rows {tuple(trainer_rows.shape)} do not match "
            f"engine rows {tuple(selected_engine_rows.shape)} for vocabulary slice [{vocab_start}, {vocab_end})"
        )

    if trainer_rows.device != selected_engine_rows.device:
        raise Refusal(
            f"full_distribution: trainer rows are on {trainer_rows.device} but engine rows are on "
            f"{selected_engine_rows.device}; comparison requires one device"
        )
    device_type = trainer_rows.device.type
    if device_type not in ("cpu", "cuda"):
        raise Refusal(f"full_distribution: comparison requires CPU or CUDA rows, got {trainer_rows.device}")

    cuda_bad_row = None
    if device_type == "cuda" and evidence is None:
        cuda_bad_row = _compare_full_distribution_cuda(trainer_rows, selected_engine_rows)
    use_cuda_comparison = cuda_bad_row is not None

    selected_count = int(trainer_rows.shape[0])
    first_mismatch = None
    trainer_cpu = None
    engine_cpu = None
    if use_cuda_comparison:
        if cuda_bad_row != selected_count:
            trainer_row = trainer_rows[cuda_bad_row].detach().to(device="cpu", copy=True).contiguous()
            engine_row = selected_engine_rows[cuda_bad_row].detach().to(device="cpu", copy=True).contiguous()
            mismatch = compare_full_distribution(trainer_row, engine_row)
            if mismatch is None:
                raise Refusal(f"full_distribution: CUDA selected row={cuda_bad_row} but the CPU comparator passed it")
            first_mismatch = (cuda_bad_row, mismatch)
    else:
        trainer_cpu = trainer_rows.detach().to(device="cpu", copy=True).contiguous()
        engine_cpu = selected_engine_rows.detach().to(device="cpu", copy=True).contiguous()
        for selected_index, (trainer_row, engine_row) in enumerate(zip(trainer_cpu, engine_cpu, strict=True)):
            mismatch = compare_full_distribution(trainer_row, engine_row)
            if mismatch is not None:
                first_mismatch = (selected_index, mismatch)
                break
    if evidence is not None:
        from isoexec.integrations.skyrl.evidence import build_row_keys

        keys = build_row_keys(
            sequences=sequences,
            attention_mask=attention_mask,
            row_indices=row_indices,
            response_columns=response_columns,
            target_columns=target_columns,
            rollout_logprobs=rollout_logprobs,
            engine_rows=engine_rows[selected].detach().to(device="cpu", copy=True).contiguous(),
            trainer_rows=trainer_cpu,
            sample_indices=data.get("sample_indices"),
            vocab_start=vocab_start,
            vocab_end=vocab_end,
        )
        evidence.record_compare(
            keys=keys,
            trainer_rows=trainer_cpu,
            engine_rows=engine_cpu,
            mismatch=first_mismatch,
            context=_evidence_context(data),
        )
    if first_mismatch is not None:
        selected_index, mismatch = first_mismatch
        raise Refusal(
            "full_distribution.bits: "
            f"row={selected_index} token={vocab_start + mismatch.first_token} "
            f"trainer=0x{mismatch.trainer_u32:08x} engine=0x{mismatch.engine_u32:08x} "
            f"mismatch_count={mismatch.mismatch_count}"
        )
    from isoexec.integrations.skyrl.online_prescore import record_compared_rows

    record_compared_rows(
        sequences=sequences,
        attention_mask=attention_mask,
        row_indices=row_indices,
        response_columns=response_columns,
        target_columns=target_columns,
        sample_indices=data.get("sample_indices"),
        vocab_start=vocab_start,
        vocab_end=vocab_end,
    )
    return int(coordinates.shape[0])


def _compare_full_distribution_cuda(trainer_rows: torch.Tensor, engine_rows: torch.Tensor) -> int:
    from isoexec.checks.full_distribution import compare_full_distribution_cuda

    selected_count = int(trainer_rows.shape[0])
    cuda_bad_row = int(compare_full_distribution_cuda(trainer_rows, engine_rows).item())
    if not 0 <= cuda_bad_row <= selected_count:
        raise Refusal(f"full_distribution: CUDA comparator returned row={cuda_bad_row}; expected 0..{selected_count}")
    return cuda_bad_row
