"""Convert native inference inputs to the graph's sequence context."""

from collections.abc import Mapping

import torch

from isoexec.schema import SequenceContext, make_sequence_context


def select_sequence_metadata(metadata: Mapping):
    """Prefer host query lengths supplied by softmax attention in hybrid models.

    GDN metadata omits max_query_len for mixed batches. Selecting it when an
    attention entry is available would force a device read of the same lengths.
    GDN-only models retain their existing context construction fallback.
    """
    fallback = None
    for value in metadata.values():
        if value is None:
            continue
        if fallback is None:
            fallback = value
        if getattr(value, "max_query_len", None) is not None and getattr(value, "query_start_loc", None) is not None:
            return value
    return fallback


def engine_sequence_context(
    input_ids: torch.Tensor | None, positions: torch.Tensor, metadata,
    *, max_model_len: int, mrope: bool,
) -> SequenceContext:
    """Build one forward's context from native positions and request boundaries.

    Positions retain their values; one-dimensional text positions expand to three
    rows for MRoPE. GDN metadata uses max_model_len as the KV upper bound.
    Metadata-free profiling represents one request covering all input tokens.
    """
    if input_ids is not None and (input_ids.ndim != 1 or input_ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("VllmModel input_ids must be a one-dimensional integer tensor")
    if (positions.dtype not in (torch.int32, torch.int64)
            or (input_ids is not None and positions.device != input_ids.device)):
        raise ValueError("VllmModel positions must be integer positions on the input_ids device")
    count = input_ids.numel() if input_ids is not None else positions.shape[-1]
    if positions.shape != (count,) and positions.shape != (3, count):
        raise ValueError("VllmModel positions must have shape [T] or [3, T] matching input_ids")
    ids = input_ids.to(dtype=torch.int64) if input_ids is not None else None
    pos = positions.to(dtype=torch.int64)
    if mrope and pos.ndim == 1:
        pos = pos.expand(3, -1)
    pos = pos.contiguous()

    if metadata is None:
        cu = torch.tensor([0, count], dtype=torch.int32, device=pos.device)
        max_q = max_kv = count
    else:
        cu = getattr(metadata, "query_start_loc", None)
        if cu is None:
            cu = getattr(metadata, "non_spec_query_start_loc", None)
        if not isinstance(cu, torch.Tensor) or cu.ndim != 1 or cu.numel() < 2:
            raise ValueError("VllmModel native metadata must provide one-dimensional request boundaries")
        max_q = getattr(metadata, "max_query_len", None)
        if max_q is None:
            if getattr(metadata, "num_prefills", None) == 0 and getattr(metadata, "num_decodes", 0) > 0:
                max_q = 1
            else:
                if pos.is_cuda and torch.cuda.is_current_stream_capturing():
                    raise ValueError("VllmModel native metadata must supply max_query_len during CUDA graph capture")
                max_q = max(1, int(torch.diff(cu).max().item()))
        max_kv = getattr(metadata, "max_seq_len", None)
        if max_kv is None:
            max_kv = max_model_len
        cu = cu.to(device=pos.device, dtype=torch.int32)
    return make_sequence_context(pos, cu, max_q=int(max_q), max_kv=int(max_kv), input_ids=ids)
