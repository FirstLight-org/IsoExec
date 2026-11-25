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


def take_host_resident_buffers(module: nn.Module) -> list[tuple[nn.Module, str, torch.Tensor]]:
    """Detach the meta buffers a module keeps in host memory (``HOST_RESIDENT_BUFFERS``: tables too large
    for the device that kernels read through mapped host memory) before ``allocate_tensors``."""
    taken = []
    for child in module.modules():
        for name in getattr(child, "HOST_RESIDENT_BUFFERS", ()):
            buffer = child._buffers.get(name)
            if buffer is not None and buffer.is_meta:
                taken.append((child, name, buffer))
                child._buffers[name] = None
    return taken


def place_host_resident_buffers(taken: list[tuple[nn.Module, str, torch.Tensor]]) -> None:
    """Allocate each taken buffer as page-aligned host memory (an anonymous mapping the loader fills)."""
    import mmap

    for child, name, meta in taken:
        nbytes = meta.numel() * meta.element_size()
        mapping = mmap.mmap(-1, max(nbytes, mmap.PAGESIZE), flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        tensor = torch.frombuffer(mapping, dtype=torch.uint8, count=nbytes).view(meta.dtype).view(meta.shape)
        child.__dict__.setdefault("_host_mappings", {})[name] = mapping
        child._buffers[name] = tensor
