"""Checkpoint tensors retained for loading, sharding and synchronization without execution.
Unexecuted subtree roots are skipped by operation placement and inspection, while their held
tensors still narrow to rank ownership.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn

import torch
from torch import nn

from isoexec.schema import ShardAxis, torch_dtype

#: Family policy across the fleet, as everywhere else in this tier: a held tensor is bf16.
HELD_DTYPE = "bf16"

#: The method a subtree root defines to say that nothing on either host executes it. Named here,
#: beside the single-tensor case, because it is the same statement about a bigger thing.
UNEXECUTED_SUBTREE = "unexecuted_subtree"


def is_unexecuted_subtree(module: nn.Module) -> bool:
    """Return whether a module declares its entire subtree unexecuted. Execution walks skip the
    root and descendants; weight-facing walks retain them.
    """
    declares = getattr(module, UNEXECUTED_SUBTREE, None)
    return bool(declares()) if callable(declares) else False


class Unexecuted(nn.Module):
    """Hold one parameter named ``weight`` with a global ``shape`` and shard ``axis``. ``dtype``
    defaults to bf16 and ``device`` to meta. The tensor remains in weight inventory and
    synchronization, but calling the module raises an error.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        axis: ShardAxis,
        *,
        dtype: str = HELD_DTYPE,
        device: torch.device | str = "meta",
    ) -> None:
        shape = tuple(shape)
        if not shape or any(d < 1 for d in shape):
            raise ValueError(f"Unexecuted: shape must be non-empty with positive dims, got {shape}")
        super().__init__()
        self.axis = axis
        self.dtype_name = dtype
        self.weight = nn.Parameter(torch.empty(shape, dtype=torch_dtype(dtype), device=device))

    def tensor_axes(self) -> Mapping[str, ShardAxis]:
        """The one declaration this module makes. ``inventory`` duck-types on this method."""
        return {"weight": self.axis}

    def forward(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Refuses. Being unexecuted is what this module is, so a call is the mistake, by name."""
        del args, kwargs
        raise TypeError(
            f"Unexecuted{tuple(self.weight.shape)} is held and not executed: it is loaded, sharded and "
            f"synced, and no op in this model computes with it. A tensor that should run belongs in a "
            f"component, which resolves the op it runs and reports it in the identity report."
        )

    def extra_repr(self) -> str:
        return f"shape={tuple(self.weight.shape)}, shard={self.axis.kind}, dtype={self.dtype_name}, executed=False"


__all__ = ["HELD_DTYPE", "UNEXECUTED_SUBTREE", "Unexecuted", "is_unexecuted_subtree"]
