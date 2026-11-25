"""The op-table face of gqa_sm90: implementation identities and the two host cores.

``trainer_core`` is what ``attention.core.gqa`` resolves to on the trainer (packed THD rows, FA3
forward and backward). ``paged_core`` is the engine's call over its paged KV cache; the vLLM
adapter (runtimes/vllm/bindings.py) supplies the cache views and metadata. ``install`` builds and
loads the library once per process and returns the receipt the trainer records.

Both cores run the segment split (spec.SEG_RECIPE). The engine core never allocates inside a call: it
passes a per-device fp32 workspace (``reserve_paged_workspace``, sized from the engine's limits when the
CUSTOM backend's metadata builder is created), so CUDA-graph capture (FULL_DECODE_ONLY, FULL, piecewise)
records fixed addresses. A workspace is never freed once handed out, because a captured graph may still
read it; growing (eagerly, outside capture) retires the old one. One forward stream per device is assumed
(vLLM's dual-batch overlap would race on the shared workspace).
"""

from __future__ import annotations

from typing import Any

import torch

from isoexec.ops.attention.gqa_sm90 import spec
from isoexec.ops.attention.gqa_sm90.spec import PAGED_IMPL, TRAINER_IMPL

__all__ = ["PAGED_IMPL", "TRAINER_IMPL", "core_seg_tokens", "install", "paged_core", "paged_workspace",
           "reserve_paged_workspace", "trainer_core", "trainer_core_qk"]

_WORKSPACE: dict[int, torch.Tensor] = {}
_RETIRED: list[torch.Tensor] = []  # never freed: a captured graph may hold their addresses


def _num_sm(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _workspace(device: torch.device, floats: int) -> torch.Tensor:
    index = device.index if device.index is not None else torch.cuda.current_device()
    ws = _WORKSPACE.get(index)
    if ws is not None and ws.numel() >= floats:
        return ws
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"gqa_sm90.paged_core: the engine workspace ({0 if ws is None else ws.numel()} floats) is smaller than "
            f"this captured call needs ({floats}); reserve it before capture (reserve_paged_workspace, called by "
            f"the CUSTOM backend's metadata builder from max_num_batched_tokens / max_num_seqs)")
    if ws is not None:
        _RETIRED.append(ws)
    ws = torch.empty(floats, dtype=torch.float32, device=torch.device("cuda", index))
    _WORKSPACE[index] = ws
    return ws


def reserve_paged_workspace(*, device, max_tokens: int, max_seqs: int, num_heads: int, head_dim_v: int) -> int:
    """Allocate (or keep) the engine core's workspace for calls of up to ``max_tokens`` query rows and
    ``max_seqs`` sequences, plus ``spec.PAGED_PARTIAL_FLOATS`` for split partials; return its size."""
    device = torch.device(device)
    floats = paged_workspace_floats(max_tokens, num_heads, max_seqs, head_dim_v, _num_sm(device))
    return _workspace(device, floats).numel()


def paged_workspace_floats(total_q: int, num_heads: int, num_seqs: int, head_dim_v: int, num_sm: int) -> int:
    from isoexec.ops.attention.gqa_sm90.ops import workspace_floats

    return workspace_floats(total_q, num_heads, num_seqs, head_dim_v, num_sm, partial_floats=spec.PAGED_PARTIAL_FLOATS)


def paged_workspace(device) -> torch.Tensor | None:
    """The engine core's current workspace on ``device`` (None before the first reservation)."""
    device = torch.device(device)
    return _WORKSPACE.get(device.index if device.index is not None else torch.cuda.current_device())


def core_seg_tokens() -> int | None:
    """The KV reduction both cores run: ``spec.SEG_TOKENS`` (segment split) or None (one pass, ns1)."""
    return spec.SEG_TOKENS if spec.CORE_USES_SEGMENT_SPLIT else None


def trainer_core(q, k, v, cu_seqlens, max_seqlen: int, scale: float) -> torch.Tensor:
    """Packed causal self-attention over ``cu_seqlens``: ``q`` [T, Hq, D], ``k``/``v`` [T, Hk, D] -> [T, Hq, Dv]."""
    from isoexec.ops.attention.gqa_sm90.ops import varlen_attention

    cu = cu_seqlens.to(device=q.device, dtype=torch.int32)
    return varlen_attention(q.contiguous(), k.contiguous(), v.contiguous(), cu, cu, max_seqlen, max_seqlen, scale,
                            seg_tokens=core_seg_tokens())


def trainer_core_qk(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q: int, max_seqlen_k: int,
                    scale: float) -> torch.Tensor:
    """:func:`trainer_core` with separate query/key boundaries (queries right-aligned to their keys):
    the context-parallel form, where a rank's first sequence carries a key prefix from lower ranks.
    Same kernel and segment split, so every query row has the bits it has in :func:`trainer_core`."""
    from isoexec.ops.attention.gqa_sm90.ops import varlen_attention

    return varlen_attention(q.contiguous(), k.contiguous(), v.contiguous(),
                            cu_seqlens_q.to(device=q.device, dtype=torch.int32),
                            cu_seqlens_k.to(device=q.device, dtype=torch.int32), max_seqlen_q, max_seqlen_k, scale,
                            seg_tokens=core_seg_tokens())


def paged_core(q, k_cache, v_cache, *, cu_seqlens_q, seqused_k, block_table, max_seqlen_q: int, max_seqlen_k: int,
               scale: float, out: torch.Tensor) -> None:
    """Write causal attention of ``q`` [T, Hq, D] over the paged cache into ``out`` [T, Hq, Dv]."""
    from isoexec.ops.attention.gqa_sm90.ops import paged_fwd, workspace_floats

    num_seqs, num_sm = cu_seqlens_q.shape[0] - 1, _num_sm(q.device)
    ws = paged_workspace(q.device)
    if ws is None or ws.numel() < workspace_floats(q.shape[0], q.shape[1], num_seqs, v_cache.shape[-1], num_sm):
        # Grow to this call plus the partial budget (refused inside a capture: reserve first).
        ws = _workspace(q.device, paged_workspace_floats(q.shape[0], q.shape[1], num_seqs, v_cache.shape[-1], num_sm))
    paged_fwd(q, k_cache, v_cache, cu_seqlens_q, seqused_k, block_table, max_seqlen_q, max_seqlen_k, scale, out=out,
              seg_tokens=core_seg_tokens(), workspace=ws)


def install(*, deterministic_backward: bool = False) -> dict[str, Any]:
    """Build/load the library on this SM90 worker and set the backward policy; return the receipt."""
    from isoexec.ops.attention.gqa_sm90 import _extension, ops

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("gqa_sm90 attention requires an SM90 CUDA device")
    _extension.extension()
    ops.require_deterministic_backward(deterministic_backward)
    return {
        "package": "attention.gqa_sm90",
        "impl": TRAINER_IMPL,
        "recipe": spec.SEG_RECIPE if spec.CORE_USES_SEGMENT_SPLIT else "sm90_fa3_ns1",
        "version": spec.RECIPE_VERSION,
        "forward": "FA3",
        "backward": {h: ("FA3-deterministic" if h <= spec.DETERMINISTIC_BWD_MAX_HEAD_DIM
                         else "FA2-deterministic" if deterministic_backward else "FA3-atomic")
                     for h in spec.HEAD_DIMS},
        "fa3_commit": spec.FA3_COMMIT,
        "num_splits": spec.NUM_SPLITS,
        "seg_tokens": core_seg_tokens(),
        "deterministic_backward_required": deterministic_backward,
    }
