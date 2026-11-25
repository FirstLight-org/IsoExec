"""Resolve attention and MLP residual joins to the shared eager add implementation.
The two operation IDs allow fused stages to record ownership separately."""

from __future__ import annotations

from typing import Any

from isoexec.ops._resolve import refuse, resolution

#: The two stages of the S=1 residual seam, and the impl id ``spec.py`` declares for each.
STAGES: dict[str, str] = {"residual.add.attn": "eager", "residual.add.mlp": "eager"}


def resolve(op: str, *, site: str, **unused: Any):
    """The block residual add on the live arch."""
    if unused:
        raise refuse(op, site=site, asked=unused, on_offer=(),
                     why="the residual add takes no caller-supplied fact.")
    impl = STAGES.get(op)
    if impl is None:
        raise refuse(op, site=site, asked={}, on_offer=tuple(STAGES),
                     why="residual.hc_mix (S>1 hyper-connections) is reserved and unregistered.")
    from isoexec.ops.residual.eager import residual_add_megatron

    return resolution(op, impl, residual_add_megatron, provider="ops.residual", site=site)


__all__ = ["STAGES", "resolve"]
