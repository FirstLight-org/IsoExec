"""Native vLLM GDN state: the selected provider's pool, or vLLM's own mamba pages for a native-cache provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator, MambaStateShapeCalculator
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from isoexec.nn import GDN
from isoexec.ops import gdn

from . import gdn_engine_patch, graph_breaks

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _selected_needs_native_cache() -> bool:
    from isoexec.ops._dispatch import backend_module, selected

    return selected("gdn") is not None and bool(getattr(backend_module("gdn", ""), "needs_native_cache", False))


class GDNStateLayer(nn.Module, MambaBase):
    """Own a CPR pool with its tensors registered as non-persistent buffers.

    Construction borrows allocated GDN weights and registers the layer in the
    native forward context. The pool publishes allocated or replaced tensors
    through its registration callback. vLLM pages supply state slot IDs.
    """

    def __init__(self, module: GDN, *, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.cache_config = vllm_config.cache_config
        self.kv_cache = (torch.tensor([]), torch.tensor([]))
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        self._pool_kwargs = dict(
            max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
            chunk_size=int(gdn.fla_chunk_size()),
            on_alloc=self._register_pool_tensor,
            conv_weight=module.conv_w.squeeze(1), conv_bias=None,
            A_log=module.A_log, dt_bias=module.dt_bias,
            num_k_heads=module.key_heads_local, head_k_dim=module.key_head_dim,
            num_v_heads=module.value_heads_local, head_v_dim=module.value_head_dim,
            activation="silu", dtype=module.conv_w.dtype, device=module.conv_w.device,
        )
        self._geometry = (module.key_heads_local, module.value_heads_local, module.key_head_dim,
                          module.value_head_dim, module.conv_w.shape[-1], module.conv_w.dtype)
        # A provider whose state is vLLM's own mamba cache builds it when vLLM binds the pages.
        self._native_cache = _selected_needs_native_cache()
        self._pool = None if self._native_cache else gdn.build_state_pool(**self._pool_kwargs)
        if self._pool is not None:
            # Dummy batches (warmup, CUDA-graph capture) put every request on vLLM's null block.
            self._pool.null_slot = gdn_engine_patch.NULL_BLOCK_ID
        _join_decode_group(self._pool, context)
        context[prefix] = self

    def _register_pool_tensor(self, name: str, tensor: torch.Tensor) -> None:
        self.register_buffer(name, tensor, persistent=False)

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Unpack vLLM's pages (``MambaBase``); a native-cache provider then builds its state on them."""
        super().bind_kv_cache(kv_cache)
        if not self._native_cache:
            return
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

        conv, ssm = self.kv_cache[0], self.kv_cache[1]
        # (num_blocks, D, W-1), as vLLM's own GDN layer orients it for the conv kernels.
        conv = conv if is_conv_state_dim_first() else conv.transpose(-1, -2)
        self._pool = gdn.build_state_pool(**self._pool_kwargs, ssm_state=ssm, conv_state=conv)

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        """The native backend supplying this layer's GDN metadata."""
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return the native page shapes for the pool's local geometry."""
        if gdn.gdn_cpr_min_pages() and not self._native_cache:
            return gdn.GDN_CPR_MIN_STATE_SHAPES
        hk, hv, dk, dv, width, _ = self._geometry
        return MambaStateShapeCalculator.gated_delta_net_state_shape(1, hk, hv, dk, dv, width)

    def get_prefill_checkpoint_alignment(self) -> int | None:
        """The chunk of a private (CPR) pool: in vLLM's align mode (prefix caching) the platform makes the
        mamba block size a multiple of it, so every block boundary vLLM caches state at is a chunk boundary,
        where the pool's state is a boundary checkpoint (runtimes/vllm/cpr_apc.py). None for a provider whose
        state is vLLM's own cache. vLLM reads it only to size blocks: this layer's spec declares no
        in-prefill checkpoints."""
        if self._native_cache:
            return None
        return int(self._pool_kwargs["chunk_size"])

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        """Return native page dtypes using the current cache configuration."""
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self._geometry[5],
            self.cache_config.mamba_cache_dtype, self.cache_config.mamba_ssm_cache_dtype,
        )

    @torch.no_grad()
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        alpha: torch.Tensor, beta: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor,
        cu_seqlens: torch.Tensor, *, conv_weight: torch.Tensor,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the pool with native metadata and the supplied current weights.

        Metadata-free profiling returns zeros. Native padding rows are zeroed. While a breakable
        PIECEWISE graph is recorded, the whole core is an eager break writing into an output this
        segment allocates, so every replay reruns it on that step's own metadata.
        """
        del cu_seqlens, cu_seqlens_cpu  # Native metadata supplies the sequence boundaries.
        if graph_breaks.piecewise_capturing():
            out = v.new_empty(v.shape)
            graph_breaks.gdn_core_into(self, q, k, v, alpha, beta, A_log, dt_bias, conv_weight, out)
            return out
        return self.run_core(q, k, v, alpha, beta, A_log, dt_bias, conv_weight=conv_weight)

    @torch.no_grad()
    def run_core(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        alpha: torch.Tensor, beta: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor,
        *, conv_weight: torch.Tensor,
    ) -> torch.Tensor:
        """The batch-composition-dependent part of the layer: everything that reads native metadata."""
        metadata = gdn_engine_patch.gdn_metadata(self.prefix)
        if metadata is None or self._pool is None:
            return torch.zeros_like(v)
        self._pool.conv_weight = conv_weight.squeeze(1)
        self._pool.A_log, self._pool.dt_bias = A_log, dt_bias
        mixed = _adjacent_qkv(q, k, v) if _reads_strided_qkv(self._pool, metadata) else None
        if mixed is None:
            mixed = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1)
        output = gdn_engine_patch.gdn_layer_core(self._pool, metadata, mixed, alpha, beta)
        if output.shape[0] == v.shape[0]:
            return output
        padded = torch.zeros_like(v)
        padded[:output.shape[0]].copy_(output)
        return padded


#: One decode group per model (keyed by its forward context): every GDN layer of a model decodes on the
#: same slot ids each step, so a state that supports it resolves all layers' rows in one launch.
_DECODE_GROUPS: dict[int, object] = {}


def _join_decode_group(pool, context) -> None:
    if pool is None or not hasattr(pool, "decode_group"):
        return
    from isoexec.ops.gdn.shared.gdn_decode_group import DecodeGroup

    group = _DECODE_GROUPS.get(id(context))
    if group is None or getattr(group, "_context", None) is not context:
        group = DecodeGroup()
        group._context = context
        _DECODE_GROUPS[id(context)] = group
    group.add(pool)


def _reads_strided_qkv(pool, metadata) -> bool:
    """A pure-decode step on a state whose decode reads a row-strided pre-conv q|k|v in place.

    The recurrent state on vLLM's native kernels does (conv_sm90 ``update`` takes row strides and the
    fused split reads any row stride), and so does a state declaring ``reads_strided_decode_qkv``
    (gdn_cpr_sm100: its conv update takes row strides and its core reads q/k/v strided). Prefill keeps
    the contiguous copy its varlen conv expects.
    """
    from isoexec.ops.gdn.shared.gdn_recurrent_state import RecurrentGDN

    return (
        isinstance(pool, RecurrentGDN)
        and int(getattr(metadata, "num_prefills", 1)) == 0
        and bool(getattr(pool, "_native_kernels", False) or getattr(pool, "reads_strided_decode_qkv", False))
    )


def _adjacent_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor | None:
    """``cat(q, k, v)`` as a VIEW when q, k and v are adjacent column windows of one row-major tensor
    (the in_proj GEMM output); ``None`` otherwise. Same values in the same order, no copy."""
    q2, k2, v2 = q.flatten(1), k.flatten(1), v.flatten(1)
    if not (q2.dim() == k2.dim() == v2.dim() == 2 and q2.stride(1) == k2.stride(1) == v2.stride(1) == 1):
        return None
    # The row stride is read off the un-flattened views. A single row has no meaningful row stride (views
    # restate it), so it is presented as packed; the conv then updates the q|k|v columns in place, which
    # are dead after the conv (vLLM's own GDN layer convolves its projection in place the same way). The
    # live columns of the GEMM output (z, beta, alpha) are never written either way.
    rows, stride = q2.shape[0], q.stride(0)
    if k2.shape[0] != rows or v2.shape[0] != rows:
        return None
    if rows > 1 and (k.stride(0) != stride or v.stride(0) != stride):
        return None
    if q2.dtype != k2.dtype or q2.dtype != v2.dtype or q2.untyped_storage().data_ptr() != v2.untyped_storage().data_ptr():
        return None
    size = q2.element_size()
    if k2.data_ptr() != q2.data_ptr() + q2.shape[1] * size or v2.data_ptr() != k2.data_ptr() + k2.shape[1] * size:
        return None
    width = q2.shape[1] + k2.shape[1] + v2.shape[1]
    if rows == 1:
        stride = width
    elif stride < width:
        return None
    return q2.as_strided((rows, width), (stride, 1))

