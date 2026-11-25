"""Residual join + RMSNorm fusion over a bound stage graph, shared by the trainer and the engine.

The authored decoder writes ``h = h + sublayer(norm(h))``; capture records the join as a bf16
``aten.add.Tensor`` whose result feeds the NEXT pre-norm (the pre-MLP norm of the same layer, the
pre-attention norm of the next layer, or the final norm). This pass rewrites each such
``add -> RMSNorm`` pair into one call of the RMSNorm with the residual passed in, bound to the fused
kernel (``norms.rms.*:sm90_add_rms_v1``; on sm100 ``sm100_add_rms_v1/v2``, one per RMS recipe) that returns ``(norm(h), h)``. The fused kernel is the same
numerical program as the pair (one fp32 add + RNE round, then the unchanged RMS recipe on the
rounded sum), and its autograd backward is the pair's backward, so the rewrite changes launches, not
bits. The binding records the absorbed join in ``owns``, so a peer comparison sees it.

Both runtimes call :func:`fuse_residual_norms` after binding and before any graph wrapping
(activation checkpoint regions, CUDA graph capture). A pair the pass cannot prove is a plain
two-operand join (an ``alpha``, a non-node operand, a norm called with extra arguments) or whose norm
has no fused binding on this arch is left exactly as captured.
"""

from __future__ import annotations

import operator
import os

import torch
from torch import fx

_ENV = "ISOEXEC_FUSED_RESIDUAL_NORM"


def fused_residual_norm_enabled() -> bool:
    """Default on. A launch-count change only; ``0`` keeps the captured add + norm pairs."""
    return os.environ.get(_ENV, "1").strip().lower() not in ("", "0", "false", "no")


def _plain_join(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops.aten.add.Tensor
        and len(node.args) == 2
        and not node.kwargs
        and all(isinstance(a, fx.Node) for a in node.args)
    )


def fuse_residual_norms(graph_module: fx.GraphModule, *, op_table=None) -> int:
    """Rewrite every ``aten.add -> RMSNorm`` pair into a fused call. Returns the number rewritten.

    Requires bound RMSNorm components (``module.site`` set). Idempotent: a fused call site no longer
    matches. Node metadata (logical layer) of the join is carried by the fused node and its results.
    """
    if not fused_residual_norm_enabled():
        return 0
    from isoexec.nn.norm import RMSNorm
    from isoexec.ops.table import OpResolutionError, OpTable

    table = OpTable() if op_table is None else op_table
    graph = graph_module.graph
    fused = 0
    for node in tuple(graph.nodes):
        if node.op != "call_module" or len(node.args) != 1 or node.kwargs:
            continue
        module = graph_module.get_submodule(str(node.target))
        if type(module) is not RMSNorm or module.site is None:
            continue
        join = node.args[0]
        if not isinstance(join, fx.Node) or not _plain_join(join):
            continue
        absorbed = RMSNorm.ABSORBED_JOIN[module.role]
        try:
            binding = table.bind(
                module.slots.op_of("add_norm"), site=f"{module.site}.add_norm", inputs={"eps": module.eps},
                convention=module.convention, facts={"width": module.hidden, "residual": absorbed},
            )
        except OpResolutionError:
            continue  # no fused kernel for this stage on this arch: keep the authored pair
        module.slots.add_norm = binding
        residual, x = join.args  # aten.add(h, y): h is the stream, y the sublayer output
        with graph.inserting_after(join):
            call = graph.call_module(str(node.target), (x,), {"residual": residual})
            call.meta.update(join.meta)
            call.meta.pop("val", None)
            call.meta.pop("tensor_meta", None)
        with graph.inserting_after(call):
            normed = graph.call_function(operator.getitem, (call, 0))
            normed.meta.update(node.meta)
        with graph.inserting_after(normed):
            stream = graph.call_function(operator.getitem, (call, 1))
            stream.meta.update(join.meta)
        for key in ("isoexec_logical_layer",):
            if key in join.meta:
                normed.meta[key] = join.meta[key]
        node.replace_all_uses_with(normed)
        join.replace_all_uses_with(stream)
        graph.erase_node(node)
        graph.erase_node(join)
        fused += 1
    if fused:
        graph.lint()
        graph_module.recompile()
    pair_row_parallel_norms(graph_module)
    return fused


_PAIR_ENV = "ISOEXEC_PIK_OWNER_ARNORM"


def _sublayer_of(graph_module: fx.GraphModule, x: fx.Node):
    """The sublayer module whose output ``x`` is (directly or through capture's metadata-only output check),
    provided nothing else reads that output; else None."""
    node = x
    if node.op == "call_function" and getattr(node.target, "__name__", "") == "_checked_output":
        if len(node.users) != 1:
            return None
        node = node.args[0]
        if not isinstance(node, fx.Node):
            return None
    if node.op != "call_module" or len(node.users) != 1:
        return None
    module = graph_module.get_submodule(str(node.target))
    return module if getattr(type(module), "ROW_PARALLEL_OUTPUT", False) else None


def pair_row_parallel_norms(graph_module: fx.GraphModule) -> int:
    """Hand each fused pre-norm the row-parallel reduction of the sublayer output it absorbs.

    For every fused ``norm(x, residual=h)`` whose ``x`` is the unshared output of a sublayer that returns its
    ``row_parallel`` slot unmodified (``ROW_PARALLEL_OUTPUT``), the sublayer's row-parallel binding defers its
    tree reduction and the norm's add+norm binding runs it with the residual join and the norm fused into the
    reduction kernel (pik owner ``RootHandoff``; one launch at the decode site instead of two). Implementation
    ids, versions and inputs are unchanged: the fused kernel is the same program (bit-compared on live data,
    agreed across the TP group, before first use), and outside its domain (grad, tp=1, owner path off, sizes
    beyond one Lamport generation) both bindings run exactly as before. Returns the number of pairs wired.
    A sublayer or norm module called at more than one site of the graph is never paired (the rewrite is per
    module), nor a norm whose bound program the fused epilogue does not reproduce (``pik.owner._NORM_IMPLS``).
    Idempotent. ``ISOEXEC_PIK_OWNER_ARNORM=0`` wires none."""
    if os.environ.get(_PAIR_ENV, "1").strip().lower() in ("", "0", "false", "no", "off"):
        return 0
    from dataclasses import replace

    from isoexec.nn.norm import RMSNorm

    # The rewrite below is on the MODULES (their slots), not on the graph node: a module called at a second
    # site (a shared or tied block) would defer there too, and that site's consumer is not the paired norm.
    # Only modules called exactly once in this graph are paired.
    calls: dict[int, int] = {}
    for n in graph_module.graph.nodes:
        if n.op == "call_module":
            key = id(graph_module.get_submodule(str(n.target)))
            calls[key] = calls.get(key, 0) + 1
    paired = 0
    for node in graph_module.graph.nodes:
        if node.op != "call_module" or "residual" not in node.kwargs or len(node.args) != 1:
            continue
        norm = graph_module.get_submodule(str(node.target))
        if type(norm) is not RMSNorm or norm.site is None or not isinstance(node.args[0], fx.Node):
            continue
        sub = _sublayer_of(graph_module, node.args[0])
        if sub is None or "row_parallel" not in sub.slots:
            continue
        if calls.get(id(sub), 0) != 1 or calls.get(id(norm), 0) != 1:
            continue
        rp = sub.slots.resolved().get("row_parallel")
        an = norm.slots.resolved().get("add_norm")
        if rp is None or an is None or getattr(an.fn, "root_handoff", None) is not None:
            continue
        if getattr(rp.fn, "root_handoff", None) is not None:
            continue  # this sublayer already defers to another norm (paired through another graph)
        with_handoff = getattr(rp.fn, "with_handoff", None)
        if with_handoff is None:
            continue
        from isoexec.ops.collectives.pik_bootstrap import ensure_pik

        ensure_pik()
        from pik.owner import RootHandoff, add_norm_deferred, fuses_norm_impl  # type: ignore

        if not fuses_norm_impl(an.impl):
            continue  # the fused epilogue reproduces only these norm programs: the pair would never fuse

        handoff = RootHandoff(site=f"{sub.site}.row_parallel -> {norm.site}.add_norm")
        unfused, impl = an.fn, an.impl
        norm_mode = 2 if norm.convention == "rms_zero_centered" else 1

        def add_norm(x, residual, w, eps, _u=unfused, _h=handoff, _m=norm_mode, _i=impl):
            return add_norm_deferred(x, residual, w, eps, norm_mode=_m, impl=_i, unfused=_u, handoff=_h)

        add_norm.root_handoff = handoff  # type: ignore[attr-defined]
        deferring = with_handoff(handoff)
        deferring.root_handoff = handoff  # type: ignore[attr-defined]
        sub.slots.row_parallel = replace(rp, fn=deferring)
        norm.slots.add_norm = replace(an, fn=add_norm)
        paired += 1
    return paired


__all__ = ["fuse_residual_norms", "fused_residual_norm_enabled", "pair_row_parallel_norms"]
