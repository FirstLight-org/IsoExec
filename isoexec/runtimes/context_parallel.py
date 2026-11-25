"""Bind trainer components for contiguous context parallelism (host-independent).

``bind_cp_attention`` and ``bind_cp_gdn`` take a component's resolved bindings and return their
context-parallel forms (``ops/collectives/cp_attention.py``, ``ops/gdn/cp.py``). The numerical
identity of every kernel binding is unchanged -- CP=k computes the same bits as CP=1 -- so the core's
op/impl/version record stays the CP=1 record, and the movement is recorded on its own
(``collectives.cp_gather`` for attention, ``collectives.cp_state`` owned by the GDN core and conv).

The forward of a CP>1 model needs a :class:`~isoexec.ops.collectives.cp_contiguous.ContiguousPlan`
registered on its local boundary tensor (the host does this per microbatch, see
``runtimes/megatron/model.py``).
"""

from __future__ import annotations

from typing import Callable

from isoexec.core.bindings import COLLECTIVES_PROVIDER, BoundOp

#: One transport per CP process group, shared by every layer (one outstanding-send queue).
_TRANSPORTS: dict[int, object] = {}


class ContextParallelBindingError(ValueError):
    """A component's resolved kernel has no context-parallel form."""


def dist_transport_factory(group) -> Callable[[], object]:
    """A lazily built, per-group shared :class:`~isoexec.ops.collectives.cp_transport.DistTransport`."""

    def factory():
        key = id(group)
        transport = _TRANSPORTS.get(key)
        if transport is None or transport.group is not group:
            from isoexec.ops.collectives.cp_transport import DistTransport

            transport = _TRANSPORTS[key] = DistTransport(group)
        return transport

    return factory


def _movement(op: str, impl: str, cp: int) -> BoundOp:
    from isoexec.nn._seams import _declared_version

    return BoundOp(op, impl, _declared_version(op, impl), COLLECTIVES_PROVIDER, {"cp": cp}, _no_call(op))


def _no_call(op: str):
    def refuse(*_a, **_k):
        raise RuntimeError(f"{op}: the context-parallel movement runs inside its core; this record is not callable")

    return refuse


def attention_core_qk(core: BoundOp, *, site: str) -> Callable:
    """The varlen form (separate query/key boundaries) of a resolved ``attention.core.gqa`` binding."""
    if core.impl == "ref.attention":
        from isoexec.ops.attention.reference import ref_attention_core_qk

        return ref_attention_core_qk
    from isoexec.ops._resolve import integration, kernel_package

    package = kernel_package("gqa", op=core.op, site=site)
    integ = integration(package)
    qk = getattr(integ, "trainer_core_qk", None)
    if getattr(integ, "TRAINER_IMPL", None) != core.impl or qk is None:
        raise ContextParallelBindingError(
            f"{site}: attention core {core.impl!r} ({package}) exposes no trainer_core_qk (queries right-aligned to "
            "a key prefix), which contiguous context parallelism runs on each rank's own rows"
        )
    return qk


def gdn_stateful_core(core: BoundOp, *, site: str) -> Callable:
    """The state hand-off form ``(…, cu_seqlens, lengths, h0, want_last) -> (o, h_last)`` of a ``gdn.core``."""
    if core.impl == "ref.gdn_recurrent":
        from isoexec.ops.gdn.reference import ref_gdn_core_stateful

        return ref_gdn_core_stateful
    from isoexec.ops._resolve import integration, kernel_package

    package = kernel_package("gdn", op=core.op, site=site)
    integ = integration(package)
    stateful = getattr(integ, "gdn_core_stateful", None)
    if getattr(integ, "IMPL", None) != core.impl or stateful is None:
        raise ContextParallelBindingError(
            f"{site}: GDN core {core.impl!r} ({package}) has no state hand-off. Contiguous context parallelism "
            "runs the native recurrent composition (ISOEXEC_GDN_KERNEL=recurrent), whose rank r starts from the "
            "fp32 state rank r-1 ended with; a chunk-aligned (CPR) hand-off would plug into the same "
            "ops/gdn/cp.make_cp_gdn_core interface but is not implemented"
        )
    return stateful()


def _has(seam: str, op: str, attribute: str) -> bool:
    from isoexec.ops._resolve import integration, kernel_package

    try:
        return getattr(integration(kernel_package(seam, op=op, site="select_cp_layout")), attribute, None) is not None
    except Exception:  # noqa: BLE001 - no package for this arch: no contiguous form
        return False


def select_cp_layout(graph_module, *, cp: int, required: str | None = None) -> str:
    """The one CP row layout a model binds and its host shards with.

    ``"contiguous"`` (any degree, ``runtimes/context_parallel.py``) when every attention core has a
    varlen form with separate query/key boundaries (``trainer_core_qk``) and every GDN core a state
    hand-off (``gdn_core_stateful``): today sm90. Otherwise ``"zigzag"`` at CP=2, the gathered / owner-tile
    attention core and GDN's head-wise all-to-all (#190; the gqa_sm100 path). Any other degree is refused.

    ``required`` is the layout the HOST shards its packed rows with, when the host cannot follow the
    model's choice: SkyRL packs Megatron's zigzag cut itself (``_get_tokens_on_this_cp_rank``), so it
    requires ``"zigzag"``. Both layouts hold the same number of rows per rank, so a host that shards one
    way while the model binds the other would run without a shape error on the wrong tokens; the
    layout is therefore declared, never inferred. ``"zigzag"`` is admitted at CP=2 on every arch (its
    attention core gathers K/V; GDN re-cuts by heads), ``"contiguous"`` only where every core has the
    contiguous form.
    """
    if required not in (None, "contiguous", "zigzag"):
        raise ContextParallelBindingError(f"cp_layout must be 'contiguous' or 'zigzag', got {required!r}")
    if required == "zigzag":
        if cp != 2:
            raise ContextParallelBindingError(
                f"context parallelism cp={cp}: the host shards zigzag rows, which are qualified at cp=2 only")
        return "zigzag"
    from isoexec.nn import GDN, Attention, Component
    from isoexec.nn.inspect import executed_modules

    contiguous = True
    for _site, module in executed_modules(graph_module):
        if isinstance(module, Attention):
            contiguous &= _has("gqa", "attention.core.gqa", "trainer_core_qk")
        elif isinstance(module, GDN):
            contiguous &= _has("gdn", "gdn.core", "gdn_core_stateful")
        elif isinstance(module, Component) and module.CONTEXT_PARALLEL is None:
            contiguous = False
    if contiguous:
        return "contiguous"
    if required == "contiguous":
        raise ContextParallelBindingError(
            f"context parallelism cp={cp}: the host shards contiguous rows, but this arch's attention/GDN kernels "
            "have no contiguous CP form (trainer_core_qk / gdn_core_stateful)")
    if cp == 2:
        return "zigzag"
    raise ContextParallelBindingError(
        f"context parallelism cp={cp}: this arch's attention/GDN kernels have no contiguous CP form "
        "(trainer_core_qk / gdn_core_stateful), and the zigzag form is qualified at cp=2 only"
    )


def bind_cp_attention(core: BoundOp, *, group, site: str,
                      transport_factory: Callable[[], object] | None = None) -> tuple[BoundOp, BoundOp]:
    """``(cp_gather record, wrapped core)`` for one attention site."""
    from isoexec.ops.collectives.cp_attention import CP_ATTENTION_IMPL, make_cp_attention_core

    cp, rank = group.size(), group.rank()
    factory = transport_factory or dist_transport_factory(group)
    fn = make_cp_attention_core(attention_core_qk(core, site=site), transport_factory=factory, cp=cp, rank=rank,
                                site=site)
    gather = _movement("collectives.cp_gather", CP_ATTENTION_IMPL, cp)
    owns = {**core.owns, core.op: core.impl, gather.op: gather.impl}
    return gather, BoundOp(core.op, core.impl, core.version, core.provider, core.inputs, fn, owns=owns)


def bind_cp_gdn(core: BoundOp, conv: BoundOp | None, *, group, site: str,
                transport_factory: Callable[[], object] | None = None) -> tuple[BoundOp, BoundOp | None]:
    """``(wrapped core, wrapped conv)`` for one GDN site; the conv must be a separate binding."""
    from isoexec.ops.gdn.cp import CP_STATE_IMPL, make_cp_conv, make_cp_gdn_core

    if conv is None or "conv.short" in core.owns:
        raise ContextParallelBindingError(
            f"{site}: the GDN core {core.impl!r} owns the convolution; the conv-tail hand-off needs a separate "
            "conv.short binding"
        )
    cp, rank = group.size(), group.rank()
    factory = transport_factory or dist_transport_factory(group)
    movement = {"collectives.cp_state": CP_STATE_IMPL}
    _movement("collectives.cp_state", CP_STATE_IMPL, cp)  # declared-version check
    core_fn = make_cp_gdn_core(gdn_stateful_core(core, site=site), transport_factory=factory, cp=cp, rank=rank,
                               site=f"{site}.core")
    conv_fn = make_cp_conv(conv.fn, transport_factory=factory, cp=cp, rank=rank, site=f"{site}.conv")
    wrapped_core = BoundOp(core.op, core.impl, core.version, core.provider, core.inputs, core_fn,
                           owns={**core.owns, **movement})
    wrapped_conv = BoundOp(conv.op, conv.impl, conv.version, conv.provider, conv.inputs, conv_fn,
                           owns={**conv.owns, conv.op: conv.impl, **movement})
    return wrapped_core, wrapped_conv


__all__ = [
    "ContextParallelBindingError",
    "attention_core_qk",
    "bind_cp_attention",
    "bind_cp_gdn",
    "dist_transport_factory",
    "gdn_stateful_core",
    "select_cp_layout",
]
