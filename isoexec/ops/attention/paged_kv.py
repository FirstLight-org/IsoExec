"""Normalize vLLM paged-KV cache layouts across supported runtime versions."""

from __future__ import annotations

import torch


def _canonicalize_singleton_dim_strides(tensor: torch.Tensor) -> torch.Tensor:
    """Return canonical size-one strides without copying storage."""
    if 1 not in tensor.shape:
        return tensor
    strides = list(tensor.stride())
    previous = 1
    changed = False
    for index in range(tensor.ndim - 1, -1, -1):
        if tensor.shape[index] == 1 and strides[index] != previous:
            strides[index] = previous
            changed = True
        previous = strides[index] * tensor.shape[index]
    return tensor.as_strided(tensor.shape, strides) if changed else tensor


def split_paged_kv_cache(
    kv_cache: torch.Tensor,
    *,
    num_kv_heads: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FA3 `(B, N, H, D)` key/value views for old and new vLLM."""
    if kv_cache.ndim == 5 and kv_cache.shape[1] == 2:
        key_cache, value_cache = kv_cache.unbind(1)
    elif (
        kv_cache.ndim == 4
        and kv_cache.shape[1] == num_kv_heads
        and kv_cache.shape[-1] == 2 * head_size
    ):
        key_cache, value_cache = kv_cache.transpose(1, 2).split(head_size, dim=-1)
    else:
        raise ValueError(
            "unsupported paged KV cache layout: "
            f"shape={tuple(kv_cache.shape)}, num_kv_heads={num_kv_heads}, head_size={head_size}; "
            "expected legacy (B,2,N,H,D) or packed (B,H,N,2D)"
        )
    return _canonicalize_singleton_dim_strides(key_cache), _canonicalize_singleton_dim_strides(value_cache)


__all__ = ["split_paged_kv_cache"]
