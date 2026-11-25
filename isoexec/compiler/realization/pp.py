"""Primitives for adapting stage graphs to native pipeline-parallel (PP) calls."""

from __future__ import annotations

import torch
from torch import fx


def prepare_pipeline_graph(module: fx.GraphModule) -> None:
    """Use native roots and one hidden tensor; derive crossed sizes from its shape."""
    original = module.graph
    inputs = [node for node in original.nodes if node.op == "placeholder"]
    carried = [node for node in inputs if "isoexec_input_type" not in node.meta]
    tensors = [node for node in carried if isinstance(node.meta.get("val"), torch.Tensor)]
    if len(tensors) > 1:
        raise ValueError("Native PP requires one hidden tensor across each cut")
    graph = fx.Graph()
    roots = {name: graph.placeholder(name) for name in ("input_ids", "seq", "hidden_states")}
    env = {}
    for node in inputs:
        if node not in carried:
            if node.name not in ("input_ids", "seq"):
                value = node.meta.get("val")
                if type(value) in (int, float, bool, str) or value is None and node.meta.get("isoexec_input_type") is type(None):
                    env[node] = value
                    continue
                raise ValueError(f"Native PP does not provide root {node.name!r}")
            env[node] = roots[node.name]
            env[node].meta.update(node.meta)
        elif node in tensors:
            env[node] = roots["hidden_states"]
            env[node].meta.update(node.meta)
        else:
            value = node.meta.get("val")
            if type(value) in (int, float, bool):
                env[node] = value
                continue
            shape = tensors[0].meta["val"].shape if tensors else ()
            dimension = next((i for i, size in enumerate(shape)
                              if isinstance(size, torch.SymInt) and isinstance(value, torch.SymInt)
                              and size.node.expr == value.node.expr), None)
            if dimension is None:
                raise ValueError(f"{node.name}: native PP cannot derive this scalar from the hidden tensor")
            env[node] = graph.call_function(torch.ops.aten.sym_size.int, (roots["hidden_states"], dimension))
    for node in original.nodes:
        if node.op == "placeholder":
            continue
        if node.op == "output":
            result = node.args[0]
            if isinstance(result, tuple):
                values = [value for value in result if isinstance(value, fx.Node)
                          and isinstance(value.meta.get("val"), torch.Tensor)]
                if len(values) != 1:
                    raise ValueError("Native PP requires one hidden tensor across each cut")
                result = values[0]
            graph.output(fx.map_arg(result, lambda value: env[value]))
        else:
            env[node] = graph.node_copy(node, lambda value: env[value])
    module.graph = graph
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()


def uses_mrope(module: fx.GraphModule) -> bool:
    """Read the captured position rank, including stages without Attention."""
    from isoexec.nn import Attention

    for node in module.graph.nodes:
        if node.op == "placeholder" and node.name == "seq":
            positions = node.meta.get("isoexec_input_contract", {}).get("positions", {})
            if "rank" in positions:
                return positions["rank"] == 2
    return any(isinstance(child, Attention) and child.mrope_section is not None for child in module.modules())
