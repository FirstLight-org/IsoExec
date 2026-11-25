"""The bitwise decomposition table: graph-level repair of sign-sensitive lowerings."""

from __future__ import annotations

from typing import Callable, Sequence


def _rewrite_neg(node, graph) -> None:
    """aten.neg(x) -> aten.mul(x, -1.0): sign-preserving at zero."""
    import torch

    node.target = torch.ops.aten.mul.Tensor
    node.args = (node.args[0], -1.0)


#: Overload-packet base name -> rewrite, which mutates the node in place.
BITWISE_DECOMPS: dict[str, Callable] = {
    "neg": _rewrite_neg,
}


def decomp_names(table: dict | None) -> list[str]:
    return sorted(table.keys()) if table else []


def apply_bitwise_decomps(gm, table: dict | None):
    """Rewrite every table-matched node of an aten fx graph in place, returning ``(gm, applied_names)``.

    Each rewrite is exactly equal in real arithmetic (including signed zeros and NaN payloads); what
    it changes is which lowering inductor picks.
    """
    if not table:
        return gm, []
    applied: list[str] = []
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        packet = getattr(node.target, "overloadpacket", None)
        base = getattr(packet, "__name__", None) or str(node.target)
        base = base.split(".", 1)[0]
        rewrite = table.get(base)
        if rewrite is not None:
            rewrite(node, gm.graph)
            applied.append(base)
    if applied:
        gm.graph.lint()
        gm.recompile()
    return gm, sorted(set(applied))


def compile_rewritten(fn: Callable, example_inputs: Sequence, table: dict | None, *, dynamic: bool):
    """Trace ``fn`` to aten, apply the table, and return ``(compiled_callable, applied_names)``.

    The compile target is the rewritten GraphModule, so a dynamo retrace re-traces the repaired graph
    and the table cannot be lost. An empty table compiles the original function directly.
    """
    import torch

    if not table:
        return torch.compile(fn, dynamic=dynamic), []
    from .region_gate import _trace_aten

    gm = _trace_aten(fn, example_inputs)
    gm, applied = apply_bitwise_decomps(gm, table)
    return torch.compile(gm, dynamic=dynamic), applied
