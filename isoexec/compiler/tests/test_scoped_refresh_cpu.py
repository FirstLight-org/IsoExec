"""Owner-scoped sync refresh repairs owned aliases and leaves foreign models alone."""

from __future__ import annotations

from typing import Callable, cast

import pytest
import torch
from torch import nn

from isoexec.ops.mm import packed_proj as projections
from isoexec.ops.moe.shared import packed_experts as experts
from isoexec.ops.moe.shared import param_storage


class _Projection(nn.Module):
    packed_proj_name = "projection_packed"

    def __init__(self) -> None:
        super().__init__()
        self.ns = "same.decoder.projection"
        self.q = nn.Parameter(torch.ones(2, 3))
        self.k = nn.Parameter(torch.full((2, 3), 2.0))
        self.packed_proj: Callable[[], torch.Tensor | None] | None = None

    def packed_proj_roles(self) -> tuple[str, ...]:
        return ("q", "k")

    def operand(self) -> torch.Tensor | None:
        hook = self.packed_proj
        assert hook is not None
        return hook()


class _Expert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.ones(2, 3))
        self.up = nn.Parameter(torch.full((2, 3), 2.0))
        self.down = nn.Parameter(torch.full((3, 2), 3.0))


class _Experts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ns = "same.decoder.experts"
        self.experts = nn.ModuleList([_Expert(), _Expert()])
        self.ep_range = (0, 2)
        self.expert_views: Callable[[], tuple[torch.Tensor, torch.Tensor] | None] | None = None

    def held_experts(self) -> list[_Expert]:
        return [cast(_Expert, expert) for expert in self.experts]

    def operands(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        hook = self.expert_views
        assert hook is not None
        return hook()


class _Owner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = _Projection()
        self.mixture = _Experts()


def test_refresh_repairs_only_owned_sites_and_preserves_foreign_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real projection and expert buffers detect both isolation and missing refresh."""
    monkeypatch.setenv("ISOEXEC_MOE_PACKED_EXPERTS", "1")
    a, b, c = _Owner(), _Owner(), _Owner()
    for owner in (a, b, c):
        assert projections.attach(owner) == 1
        assert experts.attach(owner) == 1
    try:
        a_projection, b_projection = a.projection.operand(), b.projection.operand()
        a_experts, b_experts = a.mixture.operands(), b.mixture.operands()
        assert a_projection is not None and b_projection is not None
        assert a_experts is not None and b_experts is not None
        addresses = (a_projection.data_ptr(), *(buffer.data_ptr() for buffer in a_experts))
        epoch = param_storage.SYNC_EPOCH

        # Both models have identical names. B's new storage belongs to a bucket;
        # its hooks must keep taking the fallback until B receives its own sync.
        projection_bucket, expert_bucket = torch.full((4, 3), 9.0), torch.full((4, 3), 8.0)
        b.projection.q.data = projection_bucket[:2]
        b.mixture.held_experts()[0].gate.data = expert_bucket[:2]
        assert b.projection.operand() is None and b.mixture.operands() is None
        b_projection_ptr = b.projection.q.data_ptr()
        b_expert_ptr = b.mixture.held_experts()[0].gate.data_ptr()
        attempts = (projections.stats()["repacks"], experts.stats()["repacks"])

        # A really needs refresh: replacement parameters are off its old buffers.
        # Successful refresh must copy the fresh bytes back to those same buffers.
        a.projection.q = nn.Parameter(torch.full((2, 3), 4.0))
        a.mixture.held_experts()[0].gate = nn.Parameter(torch.full((2, 3), 5.0))
        allocation_counts = (projections.stats()["buffer_allocs"], experts.stats()["buffer_allocs"])
        projection_pack, expert_pack = projections.pack_site, experts.pack_layer

        # Deterministically interleave C's valid first allocation inside A's
        # refresh. Global allocation statistics cannot decide A's buffer safety.
        def interleaved_projection(mod: nn.Module, *, force: bool = False) -> torch.Tensor | None:
            if mod is a.projection and force:
                assert c.projection.operand() is not None
            return projection_pack(mod, force=force)

        def interleaved_experts(layer: nn.Module, *, force: bool = False) -> dict[str, torch.Tensor] | None:
            if layer is a.mixture and force:
                assert c.mixture.operands() is not None
            return expert_pack(layer, force=force)

        with monkeypatch.context() as interleaving:
            interleaving.setattr(projections, "pack_site", interleaved_projection)
            interleaving.setattr(experts, "pack_layer", interleaved_experts)
            param_storage.refresh_packed_for(a)
        assert projections.stats()["buffer_allocs"] == allocation_counts[0] + 1
        assert experts.stats()["buffer_allocs"] == allocation_counts[1] + 2
        refreshed_projection, refreshed_experts = a.projection.operand(), a.mixture.operands()
        assert refreshed_projection is not None and refreshed_experts is not None
        assert addresses == (
            refreshed_projection.data_ptr(), *(buffer.data_ptr() for buffer in refreshed_experts)
        )
        assert torch.equal(refreshed_projection[:2], torch.full((2, 3), 4.0))
        assert torch.equal(refreshed_experts[0][0, :2], torch.full((2, 3), 5.0))
        assert projections.check_module(a.projection) == (True, "")
        assert experts.check_layer(a.mixture) == (True, "")
        assert param_storage.SYNC_EPOCH == epoch

        assert b.projection.operand() is None and b.mixture.operands() is None
        assert (attempts[0] + 1, attempts[1] + 1) == (
            projections.stats()["repacks"], experts.stats()["repacks"]
        )
        assert b.projection.q.data_ptr() == b_projection_ptr
        assert b.mixture.held_experts()[0].gate.data_ptr() == b_expert_ptr
        assert b_projection.data_ptr() != b_projection_ptr and b_experts[0].data_ptr() != b_expert_ptr

        # Isolation must not swallow this owner's own refresh failure.
        a_bucket = torch.full((4, 3), 6.0)
        a.projection.q.data = a_bucket[:2]
        with pytest.raises(RuntimeError, match="refresh refused.*projection_packed"):
            param_storage.refresh_packed_for(a)
        assert param_storage.SYNC_EPOCH == epoch
        assert a.projection.operand() is None

        # An allocation by this owner still invalidates a captured buffer.
        projections.unpack(a.projection)
        a.projection.q = nn.Parameter(torch.ones(2, 3))
        assert a.projection.operand() is not None
        a.projection.q = nn.Parameter(torch.ones(4, 3))
        a.projection.k = nn.Parameter(torch.ones(4, 3))
        with pytest.raises(RuntimeError, match="GRAPH-UNSAFE.*REALLOCATED"):
            projections.refresh_packed_for(a)
        for expert in a.mixture.held_experts():
            expert.gate = nn.Parameter(torch.ones(3, 3))
            expert.up = nn.Parameter(torch.ones(3, 3))
        with pytest.raises(RuntimeError, match="GRAPH-UNSAFE.*REALLOCATED"):
            experts.refresh_packed_for(a)
    finally:
        for owner in (a, b, c):
            projections.release_to_storage_owner(owner)
            experts.release_to_storage_owner(owner)
