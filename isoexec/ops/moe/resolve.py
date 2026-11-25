"""Resolve MoE implementations from routing kind, execution side and the selected expert package."""

from __future__ import annotations

from typing import Any

from isoexec.ops._resolve import integration, kernel_package, provider_of, refuse, resolution

#: The router mechanics this family implements, as ``routing -> (impl id, where the callable is)``.
#: ``softmax_renorm``'s callable is the trainer's own; ``sigmoid_noaux_bias_select``'s is the shared
#: routing function ``ops/moe/routing.py`` states and both hosts bind by the same name.
MECHANICS: dict[str, str] = {
    "softmax": "deterministic_softmax_topk",
    "softmax_renorm": "deterministic_sorted_topk",
    "sigmoid_noaux_bias_select": "sigmoid_noaux_bias_select",
}

#: The impl ids of the four stages with no arch seam of their own.
SCORE_IMPL = "deterministic_fp32_gemm"
COMBINE_IMPL = "pik_leaf_tree"
SHARED_IMPL = "eager"

#: The side whose forward-only recipes this package also ships. Spelled once; ``schema/run.SIDES``
#: is the vocabulary and ``nn/moe.py`` reads the value off the ``Placement``.
ENGINE = "engine"

#: The routing whose selection function the fused O2 router does not implement. The sigmoid/no-aux
#: family is a class of its own in ``moe_sm90/spec.py``, so there is no twin to fall back to and no
#: claim to make: the engine runs the same shared routing function the trainer does.
_NO_FUSED_ROUTER = "sigmoid_noaux_bias_select"


def resolve(op: str, *, site: str, routing: str | None = None, side: str | None = None,
            weight_after_fc2: bool = False, **unused: Any):
    """The mixture kernel serving ``op`` on the live arch, for this ``routing`` and this ``side``.
    ``moe.route`` / ``moe.route.backward`` are the SM100 fused routers, selected by routing name."""
    if op in ("moe.route", "moe.route.backward"):
        from isoexec.ops.moe.router_sm100.resolve import resolve as router

        return router(op, site=site, routing=routing, **unused)
    if unused:
        raise refuse(op, site=site, asked=unused, on_offer=("routing", "side"),
                     why="the mixture selects on the router mechanics and on the side, and on nothing else.")
    from isoexec.ops._dispatch import current_arch
    if current_arch() == "sm100":
        from isoexec.ops.moe.sm100 import resolve_sm100
        return resolve_sm100(op, site=site, routing=routing, side=side, weight_after_fc2=weight_after_fc2)
    engine = side == ENGINE

    if op == "moe.route_score":
        from isoexec.ops.moe.eager import route_score_fp32

        return resolution(op, SCORE_IMPL, route_score_fp32, provider="ops.moe", site=site)

    if op == "moe.route_mechanics":
        impl = MECHANICS.get(routing) if routing is not None else None
        if impl is None:
            raise refuse(op, site=site, asked={"routing": routing}, on_offer=tuple(MECHANICS),
                         why="two routings are two selection functions, so neither substitutes for the other.")
        if routing == _NO_FUSED_ROUTER:
            from isoexec.ops.moe.routing import sigmoid_noaux_bias_select

            return resolution(op, impl, sigmoid_noaux_bias_select, provider="ops.moe", site=site)
        if routing == "softmax":
            from isoexec.ops.moe.eager import route_mechanics_softmax

            return resolution(op, impl, route_mechanics_softmax, provider="ops.moe", site=site)
        from isoexec.ops.moe.eager import route_mechanics_megatron

        if engine:
            package = kernel_package("moe", op=op, site=site)
            integ = integration(package)
            return resolution(
                op, integ.ENGINE_MECHANICS_IMPL, integ.engine_route_mechanics(route_mechanics_megatron),
                provider=provider_of(package), site=site,
            )
        return resolution(op, impl, route_mechanics_megatron, provider="ops.moe", site=site)

    if op == "moe.expert_mlp":
        package = kernel_package("moe", op=op, site=site)
        integ = integration(package)
        return resolution(
            op, integ.AFTER_FC2_IMPL if weight_after_fc2 else integ.IMPL, integ.expert_mlp,
            provider=provider_of(package), site=site,
            # The one declared input whose VALUE is the kernel's; see the module docstring.
            inputs={"act": integ.ACT},
        )

    if op == "moe.combine":
        from isoexec.ops.moe.eager import combine_fixed_order

        if engine:
            package = kernel_package("moe", op=op, site=site)
            integ = integration(package)
            # The id is the eager stage's, because the function is: see the module docstring.
            return resolution(
                op, COMBINE_IMPL, integ.engine_combine(combine_fixed_order),
                provider=provider_of(package), site=site,
            )
        return resolution(op, COMBINE_IMPL, combine_fixed_order, provider="ops.moe", site=site)

    if op == "moe.shared_expert":
        from isoexec.ops.moe.eager import shared_expert_megatron

        if engine:
            package = kernel_package("moe", op=op, site=site)
            integ = integration(package)
            return resolution(
                op, SHARED_IMPL, integ.engine_shared_expert(shared_expert_megatron),
                provider=provider_of(package), site=site,
            )
        return resolution(op, SHARED_IMPL, shared_expert_megatron, provider="ops.moe", site=site)

    raise refuse(op, site=site, asked={},
                 on_offer=("moe.route_score", "moe.route_mechanics", "moe.expert_mlp", "moe.combine",
                           "moe.shared_expert"),
                 why="the moe family states these op ids and no others.")


__all__ = ["COMBINE_IMPL", "ENGINE", "MECHANICS", "SCORE_IMPL", "SHARED_IMPL", "resolve"]
