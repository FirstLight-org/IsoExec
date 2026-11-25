# Derived from torchtitan, torchtitan/experiments/rl/models/attention.py
# (https://github.com/pytorch/torchtitan), modified for IsoExec:
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Used under the BSD 3-Clause License reproduced in NOTICE.
"""vLLM CUSTOM attention backend: vLLM's FlashAttention cache layout and metadata, the arch-selected GQA
package's paged kernel.

The native model calls the kernel through ``runtimes/vllm/bindings.PagedAttentionBinding``; this
backend supplies the KV cache layout, the cache-update op and the metadata builder, and its own
``forward`` runs the same paged core (gqa_sm90: FA3 under the fixed segment split
``sm90_fa3_segsplit_v1``; gqa_sm100: the segment-tree forward) for any caller of vLLM's
``Attention.forward``.

A package whose paged core needs a persistent workspace (gqa_sm90: the lse, scheduler metadata and
segment-split scratch, so CUDA-graph capture allocates nothing) exposes ``reserve_paged_workspace``; the
metadata builder reserves it from the engine's limits when it is created, before any capture.

A package whose kernel constrains the paged cache says so in its integration module:
``PAGE_MULTIPLE`` (the kernel block size must be a multiple of it) and ``KV_CACHE_LAYOUTS`` (the
vLLM layouts whose per-layer K/V views it reads in place). Without them the backend keeps
FlashAttention's declarations. They are read for the host's arch without creating a CUDA context,
because vLLM asks for them while it is still configuring the engine.
"""

import importlib
from typing import cast

import torch
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType, MultipleOf
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from isoexec.ops.attention.paged_kv import split_paged_kv_cache


def paged_cache_contract():
    """The arch-selected GQA package's integration module on this host, or ``None`` without one."""
    from isoexec.core.arch import ARCH
    from isoexec.ops._dispatch import FAMILIES

    package = FAMILIES["gqa"].archs.get(ARCH)
    return None if package is None else importlib.import_module(f"isoexec.ops.{package}.integration")


def _page_multiple():
    return getattr(paged_cache_contract(), "PAGE_MULTIPLE", None)


class PyTorchVarlenAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    """Reuse FlashAttention metadata while restricting CUDA graph capture to supported modes."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Under PCP a rank holds up to two chunks per prefill (vLLM PCPManager: 2 * max_num_seqs local
        # requests), so FlashAttention's DCP context-length buffer, sized for max_num_seqs, would overflow.
        # The IsoExec binding does not read these DCP fields; they only have to fit the local batch.
        if self.dcp_world_size > 1 and vllm_config.parallel_config.prefill_context_parallel_size > 1:
            self._dcp_context_kv_lens = torch.zeros(2 * vllm_config.scheduler_config.max_num_seqs,
                                                    dtype=torch.int32, device=device)
        _reserve_paged_workspace(kv_cache_spec, vllm_config, device)


def _reserve_paged_workspace(kv_cache_spec, vllm_config, device) -> None:
    """Size the paged core's workspace for the largest step this engine can schedule (no-op for a package
    without one). Under PCP a rank holds up to two chunks per prefill, so the sequence bound doubles."""
    reserve = getattr(paged_cache_contract(), "reserve_paged_workspace", None)
    if reserve is None or torch.device(device).type != "cuda":
        return
    sched, model, parallel = vllm_config.scheduler_config, vllm_config.model_config, vllm_config.parallel_config
    pcp = getattr(parallel, "prefill_context_parallel_size", 1) or 1
    head_dim_v = getattr(kv_cache_spec, "head_size_v", None) or getattr(kv_cache_spec, "head_size", None) \
        or model.get_head_size()
    reserve(device=device, max_tokens=int(sched.max_num_batched_tokens),
            max_seqs=int(sched.max_num_seqs) * (2 if pcp > 1 else 1),
            num_heads=int(model.get_num_attention_heads(parallel)), head_dim_v=int(head_dim_v))


@register_backend(AttentionBackendEnum.CUSTOM)
class PyTorchVarlenAttentionBackend(FlashAttentionBackend):
    """Inherit vLLM's cache-update and batch-invariance declarations so incompatible changes remain visible."""

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return PyTorchVarlenAttentionImpl

    @staticmethod
    def get_builder_cls():
        return PyTorchVarlenAttentionMetadataBuilder

    @classmethod
    def get_supported_kernel_block_sizes(cls, kv_cache_spec: KVCacheSpec | None = None) -> list[int | MultipleOf]:
        multiple = _page_multiple()
        return super().get_supported_kernel_block_sizes(kv_cache_spec) if multiple is None else [MultipleOf(multiple)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        multiple = _page_multiple()
        if multiple is None:
            return super().get_preferred_block_size(default_block_size)
        return -(-max(default_block_size, multiple) // multiple) * multiple

    @classmethod
    def supported_kv_cache_layouts(cls):
        names = getattr(paged_cache_contract(), "KV_CACHE_LAYOUTS", None)
        return super().supported_kv_cache_layouts() if names is None else tuple(KVCacheLayout[n] for n in names)


class PyTorchVarlenAttentionImpl(FlashAttentionImpl):
    # vLLM admits PCP only for backends that declare it. Under PCP/DCP the IsoExec binding
    # (runtimes/vllm/context_parallel.py) gathers the complete KV and runs the attention itself, so
    # this impl's forward is never the CP path and refuses CP below.
    supports_pcp = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.out_transform = None  # no epilogue

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for FlashAttentionImpl")

        # Re-read live per-layer metadata + kv_cache from forward context
        attn_metadata, _, kv_cache, _ = get_attention_context(cast(str, layer.layer_name))
        if attn_metadata is None:
            return output.fill_(0)
        assert self.attn_type not in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER), (
            "Encoder-only attention not supported yet."
        )
        assert not self.kv_cache_dtype.startswith("fp8"), "FP8 KV cache not supported."
        assert not getattr(attn_metadata, "use_cascade", False), "Cascade not supported yet."
        assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
            "PCP/DCP attention runs in the IsoExec binding, not in this impl")
        assert self.alibi_slopes is None, "Alibi slopes not supported yet."
        if not attn_metadata.causal:
            raise RuntimeError("Non-causal attention not supported yet.")
        if self.sliding_window != (-1, -1):
            raise RuntimeError("sliding-window attention is outside the GQA core's (-1, 0) causal pin")

        key_cache, value_cache = split_paged_kv_cache(kv_cache, num_kv_heads=self.num_kv_heads,
                                                      head_size=self.head_size)
        from isoexec.ops._dispatch import backend_module

        n = attn_metadata.num_actual_tokens
        backend_module("gqa", "integration").paged_core(
            query[:n], key_cache, value_cache,
            cu_seqlens_q=attn_metadata.query_start_loc, seqused_k=attn_metadata.seq_lens,
            block_table=attn_metadata.block_table, max_seqlen_q=int(attn_metadata.max_query_len),
            max_seqlen_k=int(attn_metadata.max_seq_len), scale=self.scale, out=output[:n])
        return output
