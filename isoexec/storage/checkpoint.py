"""Read checkpoint tensors without requiring the whole checkpoint in memory."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch

from isoexec.schema import ShardError


class CheckpointStore:
    """Read a safetensors file/directory or an in-memory mapping of tensors."""

    def __init__(self, weights: str | Mapping[str, torch.Tensor]) -> None:
        self._handles: list[Any] = []
        self._by_key: dict[str, Any] = {}
        self._tensors: Mapping[str, torch.Tensor] | None = None
        if isinstance(weights, str):
            from safetensors import safe_open

            files = (
                sorted(os.path.join(weights, f) for f in os.listdir(weights) if f.endswith(".safetensors"))
                if os.path.isdir(weights)
                else [weights]
            )
            if not files:
                raise FileNotFoundError(f"no .safetensors under {weights}")
            try:
                for path in files:
                    handle = safe_open(path, "pt", device="cpu")
                    handle.__enter__()
                    self._handles.append(handle)
                    for key in handle.keys():
                        if key in self._by_key:
                            raise ShardError(f"HF key {key!r} present in two safetensors files")
                        self._by_key[key] = handle
            except BaseException:
                self.close()
                raise
        else:
            self._tensors = weights

    def keys(self) -> list[str]:
        return sorted(self._by_key) if self._tensors is None else sorted(self._tensors)

    def shape(self, key: str) -> tuple[int, ...]:
        if self._tensors is not None:
            return tuple(self._tensors[key].shape)
        return tuple(self._by_key[key].get_slice(key).get_shape())

    def dtype(self, key: str) -> torch.dtype | str:
        """Read source encoding metadata without materializing tensor values."""
        if self._tensors is not None:
            return self._tensors[key].dtype
        return str(self._by_key[key].get_slice(key).get_dtype())

    def reader(self, key: str) -> Any:
        if self._tensors is not None:
            return self._tensors[key]
        return self._by_key[key].get_slice(key)

    def close(self) -> None:
        for handle in self._handles:
            handle.__exit__(None, None, None)
        self._handles = []
