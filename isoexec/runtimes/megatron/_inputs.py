"""Convert native packed batch fields to graph inputs."""

from itertools import pairwise

import torch

from isoexec.ops.collectives.cp_gather import build_layout, register_layout
from isoexec.schema import Axis, SequenceContext, make_sequence_context


def packed_sequence_context(
    input_ids: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    *,
    max_q: int,
    max_kv: int | None,
    mrope: bool,
    cp: Axis | None = None,
) -> SequenceContext:
    """Build one text-only packed context, resetting positions at each boundary.

    First-stage input_ids are [1, T]; later stages provide no token IDs.
    Supplied positions are preserved; absent
    positions start at zero for each sequence. The context belongs to this
    forward and retains host boundaries for trainer kernels.

    Under context parallelism (``cp.size > 1``) input IDs and positions are this rank's zigzag rows, while
    the boundaries and maximum lengths describe the global pack; the zigzag layout is registered on the
    boundary tensor for the attention and mixer seams of this forward.
    """
    if input_ids is not None and (input_ids.ndim != 2 or input_ids.shape[0] != 1
                                  or input_ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Megatron packed input_ids must be an integer tensor with shape [1, T]")
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32:
        raise ValueError("Packed sequence boundaries must be a one-dimensional int32 tensor")
    ids = input_ids[0].to(dtype=torch.int64) if input_ids is not None else None
    cu_cpu = cu_seqlens.detach().to(device="cpu")
    boundaries = cu_cpu.tolist()
    if (len(boundaries) < 2 or boundaries[0] != 0
            or any(stop <= start or stop - start > max_q for start, stop in pairwise(boundaries))):
        raise ValueError("Packed boundaries must cover input_ids with positive lengths bounded by max_seqlen_q")
    device = ids.device if ids is not None else cu_seqlens.device
    cu = cu_seqlens.to(device=device)
    layout = build_layout(boundaries, cp.size, cp.rank, device) if cp is not None and cp.size > 1 else None
    count = boundaries[-1] if layout is None else layout.local
    if ids is not None and ids.numel() != count:
        raise ValueError(f"Packed boundaries must cover the {count} local input_ids, got {ids.numel()}")
    if position_ids is None and layout is not None:
        positions = layout.positions
    elif position_ids is None:
        starts = torch.repeat_interleave(cu[:-1], cu[1:] - cu[:-1], output_size=count)
        positions = torch.arange(count, device=device) - starts
    else:
        if position_ids.dtype not in (torch.int32, torch.int64) or position_ids.device != device:
            raise ValueError("position_ids must be integer positions on the input_ids device")
        positions = position_ids.to(dtype=torch.int64)
        if positions.shape == (1, count):
            positions = positions[0]
        if positions.shape != (count,) and not (mrope and positions.shape == (3, count)):
            raise ValueError("position_ids must match the packed token count and the model's rotary format")
    if mrope and positions.ndim == 1:
        positions = positions.expand(3, -1)
    # Standard text capture declares contiguous position rows.
    positions = positions.contiguous()
    if layout is not None:
        register_layout(cu, layout)
    return make_sequence_context(positions, cu, max_q=max_q, max_kv=max_q if max_kv is None else max_kv,
                                 input_ids=ids, cu_seqlens_cpu=cu_cpu)
