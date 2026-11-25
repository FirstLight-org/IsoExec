"""Framework-independent inputs and results for an authored auxiliary loss."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor


class AuxiliaryLoss(Protocol):
    """Compute a weighted scalar loss for one packed microbatch.

    ``aux`` contains the current forward's named tensors, still attached to
    autograd. ``token_mask`` is [T], excluding alignment and schedule padding.
    ``cu_seqlens`` contains the padded [R+1] sequence boundaries in those same
    rows, so a helper can compute per-sequence statistics without framework
    objects. An all-zero mask must produce a finite, differentiable zero.

    The helper owns its formula, coefficients, and token/layer/sequence
    reductions. The trainer owns microbatch accumulation, backward, and optimizer
    updates. Returned metrics are scalar tensors or floats; only the adapter
    detaches them for logging. This interface does not schedule state updates.
    """

    def __call__(self, aux: Mapping[str, Tensor], token_mask: Tensor,
                 cu_seqlens: Tensor, /) -> tuple[Tensor, Mapping[str, Tensor | float]]: ...
