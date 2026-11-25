"""Resolve the supported silu_mul activation, refusing reserved activation kinds."""

from __future__ import annotations

from typing import Any

from isoexec.ops._resolve import refuse, resolution

ACTIVATIONS: dict[str, str] = {"silu_mul": "eager"}

_FUSED_ENV = "ISOEXEC_MLP_FUSED_ACT"


def fused_act_enabled() -> bool:
    """sm90 fused ``mlp.act`` (default on). ``0`` binds the eager three-launch statement (same bits)."""
    import os

    return os.environ.get(_FUSED_ENV, "1").strip().lower() not in ("", "0", "false", "no")


def resolve(op: str, *, site: str, activation: str = "silu_mul", **unused: Any):
    """The dense MLP activation on the live arch."""
    if unused:
        raise refuse(op, site=site, asked=unused, on_offer=("activation",))
    if op != "mlp.act":
        raise refuse(op, site=site, asked={}, on_offer=("mlp.act",), why="the mlp family states one op id.")
    impl = ACTIVATIONS.get(activation)
    if impl is None:
        raise refuse(op, site=site, asked={"activation": activation}, on_offer=tuple(ACTIVATIONS),
                     why="an activation with no registered impl refuses the build rather than guessing.")
    from isoexec.ops._dispatch import current_arch
    if current_arch() == "sm100":
        from isoexec.ops.mlp.mlp_sm100.ops import silu_mul_fused
        return resolution(op, "sm100_fused_v1", silu_mul_fused, provider="ops.mlp.mlp_sm100", site=site)
    if current_arch() == "sm90" and fused_act_enabled():
        # One launch for silu, the +0.0 and the multiply; bit-identical to the eager statement below.
        from isoexec.ops.mlp.mlp_sm90.ops import silu_mul_fused
        from isoexec.ops.mlp.mlp_sm90.spec import ACT_IMPL
        return resolution(op, ACT_IMPL, silu_mul_fused, provider="ops.mlp.mlp_sm90", site=site)
    from isoexec.ops.mlp.eager import silu_mul_megatron

    return resolution(op, impl, silu_mul_megatron, provider="ops.mlp", site=site)


__all__ = ["ACTIVATIONS", "resolve"]
