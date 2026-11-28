"""Primitives for adapting stage graphs to native pipeline-parallel (PP) calls."""

from __future__ import annotations

import operator
from dataclasses import dataclass

import torch
from torch import fx


def prepare_pipeline_graph(module: fx.GraphModule, *, named_carries: bool = False, final: bool = False):
    """Use native roots and one hidden tensor; derive crossed sizes from its shape.

    With ``named_carries`` (vLLM stages whose cut crosses several tensors) the stage instead takes and returns
    every crossing tensor under a name (``carry_names``) in the cut's order, token-leading on the wire (see
    ``_prepare_named_carries``; ``final``: the last stage, whose output is the body's hidden tensor), and the
    call returns the carry specs; otherwise it returns None and the single-tensor contract above is unchanged."""
    if named_carries:
        return _prepare_named_carries(module, final=final)
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


@dataclass(frozen=True)
class Carry:
    """One tensor crossing a PP cut: its IntermediateTensors key, the per-token shape after the token axis, its
    dtype, and how many leading size-1 axes the captured value has before the token axis (dropped on the wire,
    restored on receipt, so vLLM's ``[:num_tokens]`` slicing applies to the token axis)."""
    key: str
    tail: tuple[int, ...]
    dtype: torch.dtype
    lead: int


def crosses_several_tensors(module: fx.GraphModule) -> bool:
    """Whether a stage's cut, in or out, crosses more than one tensor (the named-carry form is then needed;
    single-tensor cuts keep the original form)."""
    nodes = list(module.graph.nodes)
    incoming = [node for node in nodes if node.op == "placeholder" and "isoexec_input_type" not in node.meta
                and isinstance(node.meta.get("val"), torch.Tensor)]
    result = next(node for node in nodes if node.op == "output").args[0]
    outgoing = [value for value in (result if isinstance(result, tuple) else (result,))
                if isinstance(value, fx.Node) and isinstance(value.meta.get("val"), torch.Tensor)]
    return len(incoming) > 1 or len(outgoing) > 1


def carry_names(count: int) -> tuple[str, ...]:
    """Keys of the tensors crossing a cut, in the cut's order: ``hidden_states`` first (vLLM's own key)."""
    return ("hidden_states", *(f"carry_{i}" for i in range(1, count)))


def _carry_of(key: str, value: torch.Tensor, where: str) -> Carry:
    shape = tuple(value.shape)
    token = next((i for i, size in enumerate(shape) if isinstance(size, torch.SymInt)), None)
    if token is None or any(size != 1 for size in shape[:token]) or any(
            not isinstance(size, int) for size in shape[token + 1:]):
        raise ValueError(f"{where}: a PP carry needs one symbolic token axis after size-1 axes and static "
                         f"per-token axes, not {shape}")
    return Carry(key, tuple(int(size) for size in shape[token + 1:]), value.dtype, token)


def _prepare_named_carries(module: fx.GraphModule, *, final: bool) -> tuple[tuple[Carry, ...], tuple[Carry, ...]]:
    """Several tensors across a cut: every incoming crossing tensor becomes a root named by ``carry_names``
    (token-leading ``[T, *tail]``, reshaped to its captured layout inside the stage), every outgoing tensor
    leaves token-leading in the cut's order (``define_stage`` orders a cut's outputs and the next stage's inputs
    identically). Crossed sizes derive from the first incoming tensor. Returns ``(incoming, outgoing)`` specs
    (outgoing empty on the final stage, whose output is unchanged). Pure moves: no value changes."""
    original = module.graph
    inputs = [node for node in original.nodes if node.op == "placeholder"]
    carried = [node for node in inputs if "isoexec_input_type" not in node.meta]
    tensors = [node for node in carried if isinstance(node.meta.get("val"), torch.Tensor)]
    incoming = tuple(_carry_of(key, node.meta["val"], node.name) for key, node in zip(carry_names(len(tensors)), tensors))
    graph = fx.Graph()
    roots = {name: graph.placeholder(name) for name in ("input_ids", "seq")}
    received = {}
    for spec, node in zip(incoming, tensors):
        root = graph.placeholder(spec.key)
        root.meta.update(node.meta)
        value = root
        for _ in range(spec.lead):
            value = graph.call_function(torch.ops.aten.unsqueeze.default, (value, 0))
        received[node] = value
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
        elif node in received:
            env[node] = received[node]
        else:
            value = node.meta.get("val")
            if type(value) in (int, float, bool):
                env[node] = value
                continue
            # a crossed size: an integer multiple of one symbolic axis of the first carried tensor
            shape = tensors[0].meta["val"].shape if tensors else ()
            coeff, factor = value.node.expr.as_coeff_Mul() if isinstance(value, torch.SymInt) else (None, None)
            dimension = next((i for i, size in enumerate(shape)
                              if isinstance(size, torch.SymInt) and factor is not None
                              and size.node.expr == factor and coeff.is_Integer and coeff > 0), None)
            if dimension is None:
                raise ValueError(f"{node.name}: native PP cannot derive this scalar from the carried tensors")
            size = graph.call_function(torch.ops.aten.sym_size.int, (received[tensors[0]], dimension))
            env[node] = size if coeff == 1 else graph.call_function(operator.mul, (size, int(coeff)))
    outgoing: tuple[Carry, ...] = ()
    for node in original.nodes:
        if node.op == "placeholder":
            continue
        if node.op == "output":
            result = node.args[0]
            values = [value for value in (result if isinstance(result, tuple) else (result,))
                      if isinstance(value, fx.Node) and isinstance(value.meta.get("val"), torch.Tensor)]
            if final:
                if len(values) != 1:
                    raise ValueError("The final PP stage returns one hidden tensor")
                graph.output(env[values[0]])
            else:
                outgoing = tuple(_carry_of(key, value.meta["val"], value.name)
                                 for key, value in zip(carry_names(len(values)), values))
                sent = []
                for spec, value in zip(outgoing, values):
                    wire = env[value]
                    for _ in range(spec.lead):
                        wire = graph.call_function(torch.ops.aten.squeeze.dim, (wire, 0))
                    sent.append(wire)
                graph.output(tuple(sent))
        else:
            env[node] = graph.node_copy(node, lambda value: env[value])
    module.graph = graph
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    return incoming, outgoing


def uses_mrope(module: fx.GraphModule) -> bool:
    """Read the captured position rank, including stages without Attention."""
    from isoexec.nn import Attention

    for node in module.graph.nodes:
        if node.op == "placeholder" and node.name == "seq":
            positions = node.meta.get("isoexec_input_contract", {}).get("positions", {})
            if "rank" in positions:
                return positions["rank"] == 2
    return any(isinstance(child, Attention) and child.mrope_section is not None for child in module.modules())
