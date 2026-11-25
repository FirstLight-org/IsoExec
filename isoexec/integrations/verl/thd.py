"""Contiguous context-parallel THD packing for the IsoExec verl engine, owned here.

verl's ``preprocess_thd_engine(cp_layout="contiguous")`` cuts the rows correctly but refuses to run
unless Megatron-core's ``PackedSeqParams`` has a ``cp_partition_mode`` field, which the shipped
megatron-core (0.20) lacks. ``MegatronModel`` does not read that field: it chose its CP layout at
construction (``model.cp_layout``). So this adapter packs the rows itself, with exactly verl's
contiguous semantics, and hands Megatron a plain ``PackedSeqParams`` with the GLOBAL padded
boundaries:

* every sequence is padded to a multiple of ``tp * cp``; the padded packed stream has ``T`` rows;
* rank ``r`` holds rows ``[r*T/cp, (r+1)*T/cp)`` of it; padding rows are zero ids at position 0;
* positions are in-sequence positions; ``need_roll`` shifts each sequence left by one (labels).

verl's ``postprocess_thd_engine(cp_layout="contiguous")`` gathers the per-rank outputs back; it has no
such check.
"""

from __future__ import annotations

import torch


def contiguous_thd(values: torch.Tensor, *, tp: int, cp: int, cp_rank: int, need_roll: bool = False):
    """``(rows [1, T/cp, ...], PackedSeqParams, positions [1, T/cp])`` for a nested ``[B, j, ...]`` tensor."""
    from megatron.core.packed_seq_params import PackedSeqParams

    if not values.is_nested:
        raise ValueError("contiguous_thd packs a nested (jagged) tensor")
    device = values.device
    lengths = values.offsets().diff().tolist()
    align = tp * cp
    padded = [n + (-n) % align for n in lengths]
    starts = [0]
    for n in padded:
        starts.append(starts[-1] + n)
    total = starts[-1]
    if total % cp:
        raise ValueError(f"the padded stream of {total} rows does not split over cp={cp}")
    local = total // cp
    lo, hi = local * cp_rank, local * (cp_rank + 1)
    shape = [local, *values.shape[2:]]
    rows = torch.zeros(shape, dtype=values.dtype, device=device)
    positions = torch.zeros(local, dtype=torch.long, device=device)
    for index, sequence in enumerate(values.unbind()):
        seq_start, valid_end = starts[index], starts[index] + lengths[index]
        first, last = max(lo, seq_start), min(hi, valid_end)
        if first >= last:
            continue
        source = torch.arange(first - seq_start, last - seq_start, device=device)
        if need_roll:
            source = (source + 1) % lengths[index]
        rows[first - lo:last - lo] = sequence.index_select(0, source)
        positions[first - lo:last - lo] = source
    cu = torch.tensor(starts, dtype=torch.int32, device=device)
    params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, max_seqlen_q=max(padded), cu_seqlens_kv=cu,
                             max_seqlen_kv=max(padded), cu_seqlens_q_padded=cu, cu_seqlens_kv_padded=cu)
    return rows.unsqueeze(0), params, positions.unsqueeze(0)


def forward_contiguous(model, input_ids, *, logits_processor, logits_processor_args: dict):
    """The thd branch of verl's ``gptmodel_forward_model_engine`` under contiguous CP (text only, no MTP)."""
    from megatron.core import parallel_state as mpu
    from verl.models.mcore.util import postprocess_thd_engine
    from verl.utils.megatron_utils import unwrap_model

    tp, cp = mpu.get_tensor_model_parallel_world_size(), mpu.get_context_parallel_world_size()
    rank = mpu.get_context_parallel_rank()
    post_process = unwrap_model(model).post_process
    rows, params, positions = contiguous_thd(input_ids, tp=tp, cp=cp, cp_rank=rank)
    output = model(input_ids=rows.contiguous(), attention_mask=None, position_ids=positions,
                   packed_seq_params=params)
    batch = input_ids.shape[0]
    if post_process and logits_processor is not None:
        args = {key: contiguous_thd(value, tp=tp, cp=cp, cp_rank=rank, need_roll=(key == "label"))[0]
                for key, value in logits_processor_args.items()}
        processed = logits_processor(output, **args)
        return {key: postprocess_thd_engine(value, params, input_ids, batch, post_process=True,
                                            cp_layout="contiguous")
                for key, value in processed.items()}
    return postprocess_thd_engine(output, params, input_ids, batch, post_process=post_process,
                                  cp_layout="contiguous")


__all__ = ["contiguous_thd", "forward_contiguous"]
