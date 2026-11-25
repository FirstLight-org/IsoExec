"""Torch-facing surface of the sm90 projection-backward kernels (wgrad accumulate, column assembly).

``accumulate(dy, x, main_grad)`` runs ``main_grad += dy^T @ x`` in place (``dy[T, N_out]``,
``x[T, K_in]`` bf16; ``main_grad[N_out, K_in]`` fp32, unit inner stride, any 16-byte row pitch) as
one CUTLASS launch: an fp32 WGMMA chain over T per output tile and one fp32 add of the existing
value in the epilogue. Capture-safe: no host sync, no allocation.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from ._extension import extension

__all__ = ["accumulate", "assemble_columns", "grouped_accumulate", "grouped_supported", "assemble_supported", "available", "choose_config", "supported"]


class UnsupportedShape(ValueError):
    """The sm90 wgrad kernel declines these operands; the caller takes another path."""


@lru_cache(None)
def available() -> bool:
    """A Hopper device and a loadable extension. Never raises."""
    try:
        from isoexec.core.arch import live_device_arch

        if live_device_arch() != "sm90":
            return False
        extension()
        return True
    except Exception:  # noqa: BLE001 -- unavailability is an answer here
        return False


@lru_cache(None)
def _ids() -> dict[str, int]:
    return {name: i for i, name in enumerate(extension().config_names())}


def _aligned(t: torch.Tensor, pitch: int) -> bool:
    return t.dim() == 2 and t.stride(1) == 1 and t.stride(0) % pitch == 0 and t.data_ptr() % 16 == 0


def supported(dy: torch.Tensor, x: torch.Tensor, main_grad: torch.Tensor) -> bool:
    """Whether :func:`accumulate` serves these operands (dtypes, shapes, 16-byte strides/bases)."""
    if dy.dtype is not torch.bfloat16 or x.dtype is not torch.bfloat16 or main_grad.dtype is not torch.float32:
        return False
    if not (dy.is_cuda and x.is_cuda and main_grad.is_cuda) or dy.device != main_grad.device:
        return False
    if dy.dim() != 2 or x.dim() != 2 or main_grad.dim() != 2 or dy.shape[0] != x.shape[0]:
        return False
    if tuple(main_grad.shape) != (dy.shape[1], x.shape[1]):
        return False
    return _aligned(dy, 8) and _aligned(x, 8) and _aligned(main_grad, 4)


def choose_config(n_out: int, k_in: int, t: int) -> tuple[str, int, int]:
    """(config, swizzle, raster). Speed only: every configuration computes the same element program."""
    tiles = -(-n_out // 128) * -(-k_in // 256)
    if tiles < 132:
        return "co_128x128", 1, 0
    return "co_128x256_c2", 1, 0


_FAST: dict = {}


def accumulate(dy: torch.Tensor, x: torch.Tensor, main_grad: torch.Tensor, *, config: str | None = None,
               swizzle: int | None = None, raster: int | None = None) -> torch.Tensor:
    """``main_grad += dy^T @ x`` in place; returns ``main_grad``."""
    if not supported(dy, x, main_grad):
        raise UnsupportedShape(f"mm_bwd_sm90.accumulate declines dy={tuple(dy.shape)}/{dy.dtype}/{dy.stride()} "
                               f"x={tuple(x.shape)}/{x.dtype}/{x.stride()} "
                               f"main_grad={tuple(main_grad.shape)}/{main_grad.dtype}/{main_grad.stride()}")
    if config is None:
        key = (main_grad.shape[0], main_grad.shape[1], dy.shape[0])
        hit = _FAST.get(key)
        if hit is None:
            hit = _FAST[key] = choose_config(*key)
        config, swz, ras = hit
    else:
        swz, ras = 1, 0
    swz = swz if swizzle is None else swizzle
    ras = ras if raster is None else raster
    extension().accumulate(_ids()[config], dy, x, main_grad, swz, ras)
    return main_grad


def assemble_supported(grads) -> bool:
    """Whether :func:`assemble_columns` serves these blocks (1..8 bf16 CUDA [T, w_i], unit column stride)."""
    if not 1 <= len(grads) <= 8:
        return False
    t = grads[0].shape[0] if grads[0].dim() == 2 else -1
    return all(isinstance(g, torch.Tensor) and g.is_cuda and g.dtype is torch.bfloat16 and g.dim() == 2
               and g.shape[0] == t and (g.stride(1) == 1 or g.shape[1] <= 1) and g.device == grads[0].device
               for g in grads)


def assemble_columns(grads) -> torch.Tensor:
    """``cat([g + 0 for g in grads], dim=1)`` in one launch (``+ 0`` maps -0 to +0), contiguous bf16."""
    return extension().assemble_columns(list(grads))


_GROUP_META: dict = {}


def grouped_supported(dy: torch.Tensor, b: torch.Tensor) -> bool:
    return (dy.dtype is torch.bfloat16 and b.dtype is torch.bfloat16 and dy.is_cuda and b.is_cuda
            and dy.dim() == 2 and b.dim() == 2 and dy.shape[0] == b.shape[0] and _aligned(dy, 8) and _aligned(b, 8))


def grouped_accumulate(dy: torch.Tensor, b: torch.Tensor, ends: torch.Tensor, groups) -> None:
    """For each ``(expert, col, main_grad)`` in ``groups``: ``main_grad += dy[r, col:col+M]^T @ b[r]`` over
    the expert's token rows ``r`` (``ends``: cumulative per-expert token ends, on the device). One launch
    (plus a one-block prep kernel); main_grads are fp32 ``[M, b.shape[1]]`` with 16-byte row pitch."""
    key = tuple((e, c, g.data_ptr(), g.shape[0], g.stride(0)) for e, c, g in groups)
    cached = _GROUP_META.get(key)
    if cached is None:
        for e, c, g in groups:
            if (g.dtype is not torch.float32 or g.dim() != 2 or g.shape[1] != b.shape[1] or g.stride(1) != 1
                    or g.stride(0) % 4 or g.data_ptr() % 16 or c % 8 or c + g.shape[0] > dy.shape[1]):
                raise UnsupportedShape(f"grouped wgrad declines group (expert {e}, col {c}, {tuple(g.shape)})")
        meta = torch.tensor([[e, c, g.shape[0], g.stride(0)] for e, c, g in groups], dtype=torch.int64)
        mg = torch.tensor([g.data_ptr() for _, _, g in groups], dtype=torch.int64)
        cached = (meta.to(dy.device), mg.to(dy.device))
        if len(_GROUP_META) > 4096:
            _GROUP_META.clear()
        _GROUP_META[key] = cached
    extension().grouped_accumulate(dy, b, ends, *cached)
