"""Primitives for allocating registered parameters and buffers."""

from __future__ import annotations

import torch
from torch import nn


def allocate_tensors(module: nn.Module, *, device: torch.device) -> None:
    """Allocate registered parameters and buffers using their current shapes.

    Registered names sharing one tensor still share one physical tensor. Dtype,
    trainability, buffer values, and per-name persistence are preserved. Note
    that this function does not constrain later registry changes.

    Megatron's ``to_empty_if_meta_device`` delegates to ``Module._apply``. This
    materializes a tied embedding/head separately across submodules. See:
    https://github.com/NVIDIA/Megatron-LM/blob/1ac85e91/megatron/training/utils/common_utils.py#L678-L703

    Allocation failure may leave the target module partially modified. Caller
    must discard it and reconstruct from the stage definition.

    Args:
        module: Module to allocate tensors.
        device: Device where the tensor is allocated.
    """
    if any(not parameter.is_meta for parameter in module.parameters()):
        raise ValueError("Tensor allocation requires meta parameters")

    allocated: dict[torch.Tensor, torch.Tensor] = {}
    for child in module.modules():
        for name, meta_param in child._parameters.items():
            if meta_param is None:
                continue
            # Use meta parameter as the hash key to handle sharing; support
            # only embedding/head tie now.
            tensor = allocated.get(meta_param)
            if tensor is None:
                tensor = nn.Parameter(torch.empty_like(meta_param, device=device), requires_grad=meta_param.requires_grad)
                allocated[meta_param] = tensor
            child._parameters[name] = tensor

    for child in module.modules():
        for name, buffer in child._buffers.items():
            if buffer is None:
                continue
            tensor = allocated.get(buffer)
            if tensor is None:
                # TODO(jiangjiawei1103): Move meta buffer checks to top if all buffers must be meta.
                tensor = torch.empty_like(buffer, device=device) if buffer.is_meta else buffer.to(device=device)
                allocated[buffer] = tensor
            child._buffers[name] = tensor
