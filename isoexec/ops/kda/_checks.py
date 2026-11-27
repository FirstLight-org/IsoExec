"""Host-side checks shared by the two KDA reducers' entry points (no torch import at package import)."""

from __future__ import annotations


def slots(who: str, name: str, x, n: int, device, capacity: int, null_slot: int | None = None, *, padding=True):
    """``x`` (slot indices: a list, a CPU tensor or a CUDA tensor) as a contiguous int32 ``[n]`` tensor on ``device``.

    Host data (a list or a CPU tensor) is range-checked against the pool's ``capacity``: with ``padding``, negative
    and ``null_slot`` lanes are padding and every other value must be a slot of the pool; without, every value must
    be. A CUDA tensor's values are not read: that would be a host synchronization on the decode path, which is
    CUDA-graph captured (the engine's slot tensors are its own)."""
    import torch

    if isinstance(x, torch.Tensor) and (x.is_floating_point() or x.is_complex() or x.dtype == torch.bool):
        raise ValueError(f"{who}: {name} must be integer slot indices, got {x.dtype}")
    if isinstance(x, torch.Tensor) and x.is_cuda:
        t = torch.as_tensor(x, dtype=torch.int32, device=device)
        if t.numel() != n:
            raise ValueError(f"{who}: {name} must hold {n} slot indices, got {t.numel()}")
        return t.reshape(n).contiguous()
    t = torch.as_tensor(x, dtype=torch.int32).reshape(-1)
    if t.numel() != n:
        raise ValueError(f"{who}: {name} must hold {n} slot indices, got {t.numel()}")
    bad = (t >= capacity) if padding else ((t < 0) | (t >= capacity))
    if padding and null_slot is not None:
        bad &= t != null_slot
    if bool(bad.any()):
        raise ValueError(f"{who}: {name} {t[bad][:4].tolist()} out of range for a pool of {capacity} slots"
                         + ("" if padding else " (no padding lanes here)"))
    return t.to(device).contiguous()
