"""Run attention against vLLM's paged KV cache."""

from __future__ import annotations

import torch

from isoexec.ops.attention.paged_kv import split_paged_kv_cache
from isoexec.runtimes.engine_ops import BindingError


def _gqa_integration():
    """The arch-selected GQA package's integration (gqa_sm90 on SM90, gqa_sm100 on SM100); refuses an
    arch without one."""
    from isoexec.ops._dispatch import backend_module

    return backend_module("gqa", "integration")  # not the facade: no reference fallback on an engine


class PagedAttentionBinding:
    """The engine's ``attention.core.gqa`` implementation over a native vLLM ``Attention`` layer.

    The kernel is the arch-selected GQA package's paged core (``ops/attention/gqa_sm90``: FA3 under the
    fixed segment split ``sm90_fa3_segsplit_v1``; ``ops/attention/gqa_sm100``: the segment-tree forward),
    the same numerical program the trainer's packed core runs.
    """

    #: Report material only: the impl declares no inputs, so none of this enters the digest.
    pins = {
        "kernel": "ops.attention.gqa (arch-selected).paged_core",
        "backend": "CUSTOM",
        "window": "(-1, 0)",
    }

    @staticmethod
    def impl_id() -> str:
        return _gqa_integration().PAGED_IMPL

    @staticmethod
    def run(attn, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, out: torch.Tensor) -> None:
        """Write attention outputs using the native layer's current paged KV cache."""
        from vllm.model_executor.layers.attention.attention import (
            _encode_layer_name,
            get_attention_context,
            unified_kv_cache_update,
        )

        T = q.shape[0]
        query = q.reshape(T, attn.num_heads, attn.head_size)
        key = k.reshape(T, attn.num_kv_heads, attn.head_size)
        value = v.reshape(T, attn.num_kv_heads, attn.head_size_v)
        output = out.view(T, attn.num_heads, attn.head_size_v)
        if getattr(attn, "kv_sharing_target_layer_name", None) is not None:
            raise BindingError("cross-layer KV sharing is not supported by the native model")
        if getattr(attn, "calculate_kv_scales", False):
            raise BindingError("kv-scale calibration (fp8 KV) contradicts the bf16 kv_dtype pin (A36)")
        settings = getattr(attn, "_isoexec_cp_settings", None)
        if settings is not None:
            # PCP/DCP: gather K/V rows for the sharded cache write and the complete KV for the core.
            from .context_parallel import run as run_context_parallel

            run_context_parallel(attn, query, key, value, output, paged_core=_gqa_integration().paged_core,
                                 settings=settings)
            return
        # 1. the KV-cache write, through vLLM's own seam (same op Attention.forward uses)
        if not attn.attn_backend.forward_includes_kv_cache_update:
            if attn.use_direct_call:
                unified_kv_cache_update(key, value, attn.layer_name)
            else:
                torch.ops.vllm.unified_kv_cache_update(key, value, _encode_layer_name(attn.layer_name))
        # 2. the trainer's kernel over the paged cache
        md, _layer, kv_cache, _slot = get_attention_context(attn.layer_name)
        if md is None:  # vLLM's metadata-less dummy run: shapes only
            output.zero_()
            return
        if not getattr(md, "causal", True) or getattr(md, "use_cascade", False):
            raise BindingError("non-causal / cascade attention is outside this binding")
        if str(getattr(attn, "kv_cache_dtype", "auto")).startswith("fp8"):
            raise BindingError("fp8 KV contradicts the bf16 kv_dtype pin (A36)")
        n = int(md.num_actual_tokens)
        key_cache, value_cache = split_paged_kv_cache(
            kv_cache,
            num_kv_heads=attn.num_kv_heads,
            head_size=attn.head_size,
        )
        _gqa_integration().paged_core(
            query[:n],
            key_cache,
            value_cache,
            cu_seqlens_q=md.query_start_loc,
            seqused_k=md.seq_lens,
            block_table=md.block_table,
            max_seqlen_q=int(md.max_query_len),
            max_seqlen_k=int(md.max_seq_len),
            scale=float(attn.impl.scale),
            out=output[:n],
        )
        if n < T:
            output[n:].zero_()


__all__ = ["PagedAttentionBinding"]
