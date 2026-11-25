"""Attach model-local packing hooks after parameter materialization."""

from __future__ import annotations

from typing import Any, Callable, cast

from torch import nn


def attach_packed_experts(model: nn.Module) -> Any:
    from isoexec.ops.moe import attach

    return cast(Callable[[nn.Module], Any], attach)(model)


def attach_packed_proj(model: nn.Module) -> Any:
    """Attach column-parallel projection packing hooks before the first forward."""
    from isoexec.ops.mm.packed_proj import attach

    return attach(model)


def refresh_packed_storage(model: nn.Module) -> None:
    """Refresh only packing owned by this tree at a quiescent update boundary."""
    from isoexec.ops.moe import param_storage

    param_storage.refresh_packed_for(model)


def prepare_packed_storage(model: nn.Module) -> None:
    """Finish owner-local lazy packing before recording native storage identity.

    Providers retain their foreign-storage fallback after DDP takes ownership.
    Call this outside forwards and CUDA graph capture, after host storage wrapping.
    """
    from isoexec.ops.moe import param_storage

    param_storage.prepare_packed_for(model)


def release_packing_to_storage_owner(model: nn.Module) -> None:
    """Retire this owner's packing aliases before native DDP/FSDP wrapping.

    Providers keep their own unpacking mechanics. Resolve experts through the
    existing architecture facade; do not select an accelerator in the adapter.
    The caller must hold the owner's quiescent storage-update boundary.
    """
    from isoexec.ops.mm.packed_proj import release_to_storage_owner as release_projections
    from isoexec.ops.moe import release_to_storage_owner as release_experts

    release_projections(model)
    cast(Callable[[nn.Module], None], release_experts)(model)
