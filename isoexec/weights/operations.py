"""Model-independent Hugging Face ``ConversionOps`` extensions."""

from __future__ import annotations

from typing import Any

import torch
from transformers.core_model_loading import ConversionOps


def _one_tensor(input_dict: dict[str, Any], operation: str) -> torch.Tensor:
    if len(input_dict) != 1:
        raise ValueError(f"{operation} requires exactly one source pattern")
    value = next(iter(input_dict.values()))
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"{operation} requires exactly one source tensor")
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{operation} requires a Tensor source")
    return value


class Deinterleave(ConversionOps):
    """Reorder [group, part, width] into [part, group, width] on a flattened axis.

    This changes layout only. HF Split subsequently separates the target tensors
    using their full parameter shapes.

    Example with groups=2, parts=2, width=2 (dim=0, shape [8, ...]):

        packed checkpoint layout (group-major):
            [q_h0 | gate_h0 | q_h1 | gate_h1]
             0 1    2  3      4 5    6   7

        after Deinterleave (part-major, still one tensor):
            [q_h0 | q_h1 | gate_h0 | gate_h1]
             0 1    4 5    2  3      6   7

        HF Split then cuts that into two targets by parameter size:
            q    = [q_h0 | q_h1]
            gate = [gate_h0 | gate_h1]
    """

    def __init__(self, *, groups: int, parts: int, dim: int = 0, inverse: bool = False) -> None:
        if any(type(size) is not int or size < 1 for size in (groups, parts)):
            raise ValueError("Deinterleave groups and parts must be positive integers")
        if type(dim) is not int:
            raise TypeError("Deinterleave dim must be an integer")
        self.groups = groups
        self.parts = parts
        self.dim = dim
        self.inverse = inverse

    @torch.no_grad()
    def convert(
        self, input_dict: dict[str, Any], source_patterns: list[str], target_patterns: list[str], **kwargs: Any
    ) -> Any:
        tensor = _one_tensor(input_dict, type(self).__name__)
        dim = self.dim if self.dim >= 0 else tensor.ndim + self.dim
        if not 0 <= dim < tensor.ndim:
            raise ValueError(f"Deinterleave dim {self.dim} is invalid for rank-{tensor.ndim} source")
        if not target_patterns:
            raise ValueError("Deinterleave requires a target pattern")
        # The inverse turns [part, group, width] back into [group, part, width].
        outer, inner = (self.parts, self.groups) if self.inverse else (self.groups, self.parts)
        extent = tensor.shape[dim]
        if extent % (outer * inner):
            raise ValueError("Deinterleave source dimension must divide into complete groups and parts")
        grouped = tensor.unflatten(dim, (outer, inner, extent // (outer * inner)))
        value = grouped.transpose(dim, dim + 1).flatten(dim, dim + 2).contiguous()
        return {target_patterns[0]: value}

    @property
    def reverse_op(self) -> ConversionOps:
        return Deinterleave(groups=self.groups, parts=self.parts, dim=self.dim, inverse=not self.inverse)


class AddScalar(ConversionOps):
    """Add one scalar in an optional compute dtype, with an additive inverse.

    Example for Qwen3.5 linear-attn RMSNorm stored as (weight + 1):

        checkpoint  (bf16):  [1.0, 1.5]
        AddScalar(-1, dtype=float32):
            cast to fp32 → [1.0, 1.5]
            add -1       → [0.0, 0.5]   # IsoExec mixer.out_norm

        reverse_op AddScalar(+1) restores [1.0, 1.5].
    """

    def __init__(self, value: float, *, dtype: torch.dtype | None = None) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("AddScalar value must be numeric")
        self.value = float(value)
        self.dtype = dtype

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, Any],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> Any:
        if len(target_patterns) != 1:
            raise ValueError("AddScalar requires exactly one target pattern")
        tensor = _one_tensor(input_dict, type(self).__name__)
        if self.dtype is not None:
            tensor = tensor.to(self.dtype)
        return {target_patterns[0]: tensor + self.value}

    @property
    def reverse_op(self) -> ConversionOps:
        return AddScalar(-self.value, dtype=self.dtype)
