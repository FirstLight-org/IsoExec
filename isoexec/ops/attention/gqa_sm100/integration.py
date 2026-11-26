"""The op-table face of gqa_sm100: implementation identities, the two host cores and the cache contract.

``trainer_core`` is what ``attention.core.gqa`` resolves to on the trainer (packed THD rows, this
package's forward and FA4's SM100 backward). ``paged_core`` is the engine's call over its paged KV
cache; the vLLM adapter (runtimes/vllm/bindings.py) supplies the cache views and metadata, and the
CUSTOM backend (ops/attention/varlen_backend.py) reads ``PAGE_MULTIPLE`` and ``KV_CACHE_LAYOUTS`` so
the engine allocates pages this kernel can read. ``install`` builds and loads the extension once per
process and returns the receipt the trainer records.
"""

from __future__ import annotations

from typing import Any

import torch

from isoexec.ops.attention.gqa_sm100 import spec
from isoexec.ops.attention.gqa_sm100.spec import PAGED_IMPL, TRAINER_IMPL

#: A page must hold whole 128-key tiles.
PAGE_MULTIPLE = spec.PAGE_MULTIPLE

#: vLLM KV-cache layouts (``vllm.v1.kv_cache_layout.KVCacheLayout`` names) whose per-layer K/V view
#: this kernel reads in place: token-major pages, so a block is ``page`` token strides long.
KV_CACHE_LAYOUTS = ("LBNHC",)

__all__ = ["KV_CACHE_LAYOUTS", "PAGED_IMPL", "PAGE_MULTIPLE", "TRAINER_IMPL", "install", "paged_core",
           "trainer_core"]


def trainer_core(q, k, v, cu_seqlens, max_seqlen: int, scale: float) -> torch.Tensor:
    """Packed causal self-attention over ``cu_seqlens``: ``q`` [T, Hq, D], ``k``/``v`` [T, Hk, D] -> [T, Hq, D]."""
    from isoexec.ops.attention.gqa_sm100.ops import varlen_attention

    cu = cu_seqlens.to(device=q.device, dtype=torch.int32)
    return varlen_attention(q.contiguous(), k.contiguous(), v.contiguous(), cu, max_seqlen, scale)


def paged_core(q, k_cache, v_cache, *, cu_seqlens_q, seqused_k, block_table, max_seqlen_q: int, max_seqlen_k: int,
               scale: float, out: torch.Tensor) -> None:
    """Write causal attention of ``q`` [T, Hq, D] over the paged cache into ``out`` [T, Hq, D]."""
    from isoexec.ops.attention.gqa_sm100.ops import paged_fwd

    paged_fwd(q, k_cache, v_cache, cu_seqlens_q, seqused_k, block_table, max_seqlen_q, max_seqlen_k, scale, out=out)


def install(*, deterministic_backward: bool = False) -> dict[str, Any]:
    """Build/load the extension on this SM100 worker and return the receipt.

    The backward is FA4's at every head dim and is deterministic per layout whatever
    ``deterministic_backward`` says, so the flag is recorded and changes nothing.
    """
    from isoexec.ops.attention.gqa_sm100 import _extension, ops

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("gqa_sm100 attention requires an SM100 CUDA device")
    _extension.extension()
    cute_dsl = ops.require_consistent_cute_dsl()
    return {
        "package": "attention.gqa_sm100",
        "impl": TRAINER_IMPL,
        "recipe": spec.RECIPE,
        "forward": "gqa_sm100 segment tree",
        "seg_tokens": {h: ops.segment_span(h) for h in spec.HEAD_DIMS},
        "target_batch": spec.TARGET_BATCH,  # None: no target batch declared (one segment per row)
        "backward": {h: ops.backward_kernel(h) for h in spec.HEAD_DIMS},
        "fa4_commit": spec.FA4_COMMIT,
        "cute_dsl": cute_dsl,
        "num_splits": spec.NUM_SPLITS,
        "deterministic_backward_required": deterministic_backward,
    }
