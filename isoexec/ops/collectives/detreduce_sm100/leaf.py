"""The BF16-leaf fold of the V4.1 ``wo_b`` program (local form).

``leaf_fold(planes)`` = ``BF16(tree(planes))``: ``planes`` BF16 ``[P, ...]`` in leaf order (P a power of two, at most
16), each widened to FP32 exactly, the balanced adjacent-pair FP32 tree (lower index left), one RNE at the root
(``csrc/leaf_fold.cu``). With the 8 ``wo_b`` leaves (one per output group, each one BF16-rounded S = 1 chain over its
1024 K) every TP degree dividing 8 evaluates the same expression: a rank's planes are its leaves, in rank order. This
is the arithmetic of the BF16-leaf symmetric-memory transport and of its NCCL realization (all-gather of leaf planes,
then this fold); the FP32-root reducer (``ops.TreeReducer``) is unchanged.
"""
from __future__ import annotations

import torch

from ._extension import extension


def reference(planes: torch.Tensor) -> torch.Tensor:
    """The same expression in torch (FP32 elementwise adds are IEEE RNE): the bit oracle."""
    t = [p.float() for p in planes]
    while len(t) > 1:
        t = [t[2 * i] + t[2 * i + 1] for i in range(len(t) // 2)]
    return t[0].to(torch.bfloat16)


def leaf_fold(planes: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    P = planes.shape[0]
    if P < 1 or P > 16 or P & (P - 1):
        raise ValueError("leaf planes: a power of two, at most 16")
    flat = planes.reshape(P, -1)
    if out is None:
        out = torch.empty(planes.shape[1:], dtype=torch.bfloat16, device=planes.device)
    if not planes.is_cuda or flat.shape[1] % 8:
        out.copy_(reference(planes))
        return out
    extension().leaf_fold_bf16(flat.contiguous() if flat.stride(-1) != 1 else flat, out.view(-1))
    return out


__all__ = ["leaf_fold", "reference"]
