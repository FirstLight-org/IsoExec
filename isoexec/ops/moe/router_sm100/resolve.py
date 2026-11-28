"""Bind the fused router without exposing numerical recipe overrides at execution."""

from isoexec.ops._configured import configured
from isoexec.ops._resolve import refuse
from . import spec


def resolve(op, *, site, routing, top_k, n_group=1, topk_group=1, num_shared=0, routed_scale=1.0, **unused):
    if unused or routing not in (spec.GROUPED, spec.SQRTSOFTPLUS):
        raise refuse(op, site=site, asked=dict(unused, routing=routing), on_offer=(spec.GROUPED, spec.SQRTSOFTPLUS))
    if type(top_k) is not int or top_k <= 0 or not 1 <= topk_group <= n_group or num_shared < 0:
        raise refuse(op, site=site, asked={"top_k": top_k, "n_group": n_group, "topk_group": topk_group,
                                         "num_shared": num_shared}, on_offer="positive top-k and valid groups")
    from isoexec.ops._dispatch import current_arch

    if current_arch() != spec.ARCH:
        raise refuse(op, site=site, asked={"routing": routing}, on_offer=(),
                     why=f"the fused routers are proven on {spec.ARCH} only (live arch {current_arch()!r}).")
    settings = dict(routing=routing, top_k=top_k, n_group=n_group, topk_group=topk_group,
                    num_shared=num_shared, routed_scale=routed_scale)
    if op == "moe.route":
        impl = spec.IMPL_GROUPED if routing == spec.GROUPED else spec.IMPL_SQRTSOFTPLUS
        symbol = "route_fused"
    else:
        impl, symbol = spec.IMPL_BACKWARD, "route_backward"
    return configured(op, impl, site=site, module="isoexec.ops.moe.router_sm100.ops", symbol=symbol,
                      inputs=settings, kwargs=settings, forbidden=("segments", "seg_t"))
