"""Primitives for adapting stage graphs to native trainer sequence parallelism (SP).

Under SP the residual stream between components holds this TP rank's equal, rank-ordered token
chunk. The first stage scatters the Embedding output once; component seams gather tokens
and reduce-scatter their outputs, and the LMHead gathers the stream before the head.
"""

from __future__ import annotations

import operator
from collections.abc import Callable

import torch
from torch import fx

from isoexec.nn import GDN, MLP, Attention, Embedding, LMHead, MoE, RMSNorm

# Components whose bound seams read and return this rank's token chunk.
_CHUNKED = (RMSNorm, Attention, GDN, MLP, MoE)


def prepare_sequence_parallel_graph(
    module: fx.GraphModule, *, tp_size: int, scatter: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Carry this rank's token chunk between the stage's components and adjust captured checks.

    The first PP stage scatters its Embedding output once; a later stage receives the chunk as
    ``hidden_states`` as a ``[T // TP, hidden]`` stream. From there to the LMHead,
    the stream may pass only through RMSNorm, Attention, GDN, MLP, and MoE calls, their tuple
    outputs, and ``aten.add.Tensor`` of two chunks; other operations on it are refused. Checks of
    chunked outputs expect ``T // tp_size`` rows and the gathered LMHead output expects ``T``. A
    stage with an LMHead must return its output; a stage without one returns the chunk.
    """
    from isoexec.compiler.capture.export_normalize import _checked_output
    from isoexec.compiler.outputs import model_outputs

    graph = module.graph
    owners = {node: module.get_submodule(str(node.target)) for node in graph.nodes if node.op == "call_module"}
    embeddings = [node for node, owner in owners.items() if isinstance(owner, Embedding)]
    if len(embeddings) > 1:
        raise ValueError(f"Native SP requires at most one Embedding call in the stage, got {len(embeddings)}")
    if embeddings:
        stream = next(iter(embeddings[0].users))
        if len(embeddings[0].users) != 1 or stream.target is not _checked_output:
            raise ValueError("Native SP requires the Embedding output to be one captured component output")
        tokens = stream.args[3][0]
        if not isinstance(tokens, fx.Node):
            raise ValueError("Native SP requires a symbolic captured token count")
        with graph.inserting_after(stream):
            chunk = graph.call_function(scatter, (stream,))
        stream.replace_all_uses_with(chunk, delete_user_cb=lambda user: user is not chunk)
        with graph.inserting_after(chunk):
            local = start = graph.call_function(operator.floordiv, (tokens, tp_size))
    else:
        chunk = next((node for node in graph.nodes if node.op == "placeholder" and node.name == "hidden_states"), None)
        # prepare_pipeline_graph derives the token count from the received rows, which are this rank's chunk.
        local = next((node for node in graph.nodes if node.target is torch.ops.aten.sym_size.int
                      and node.args == (chunk, 0)), None)
        if chunk is None or local is None:
            raise ValueError("Native SP requires an Embedding or a hidden_states input sized by its rows")
        with graph.inserting_after(local):
            tokens = start = graph.call_function(operator.mul, (local, tp_size))

    has_head = any(isinstance(owner, LMHead) for owner in owners.values())
    chunked = {chunk}
    nodes = list(graph.nodes)
    for node in nodes[nodes.index(start) + 1:]:
        inputs = [value for value in node.all_input_nodes if value in chunked]
        if not inputs:
            continue
        if node.op == "call_module":
            owner = owners[node]
            if not isinstance(owner, (*_CHUNKED, LMHead)) or inputs != [node.args[0]]:
                raise ValueError(f"{node.name}: native SP cannot pass the token chunk to {type(owner).__name__}")
            if not isinstance(owner, LMHead):
                chunked.add(node)
        elif node.op == "call_function" and node.target is _checked_output:
            shape = list(node.args[3])
            if shape[0] is not tokens and shape[0] is not local:
                raise ValueError(f"{node.args[2]}: native SP requires the captured token count as dim 0")
            shape[0] = local if node.args[0] in chunked else tokens
            node.args = (*node.args[:3], shape, *node.args[4:])
            if node.args[0] in chunked:
                chunked.add(node)
        elif node.op == "output":
            if has_head and model_outputs(node.args[0])[0] in chunked:
                raise ValueError("Native SP requires a stage with an LMHead to return its gathered output")
        elif node.op == "call_function" and node.target in (operator.getitem, getattr):
            chunked.add(node)
        elif node.op == "call_function" and node.target is torch.ops.aten.add.Tensor \
                and len(inputs) == len(node.all_input_nodes) == 2:
            chunked.add(node)
        else:
            raise ValueError(f"{node.name}: native SP does not support {node.target} on the token chunk")
    graph.lint()
    module.recompile()
