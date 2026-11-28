"""Activation checkpoints over compiled native layers without changing state names."""

from __future__ import annotations

import operator
from dataclasses import replace

import torch
from torch import fx

from isoexec.ops.collectives.cp_transport import cp_region
from isoexec.schema import SequenceContext, rollout


def checkpoint_layers(graph_module: fx.GraphModule, *, span: int = 1) -> tuple:
    """Compile complete authored decoder-layer regions, preserving registered state.

    Regions share the existing components and their Parameters. Their compiled
    GraphModules live in callable closures, outside the parent's registered
    module tree, so DDP, loading and export retain their original names/objects.
    No interpreter or per-node dispatch runs during forward or recomputation.
    """
    if type(span) is not int or span < 1:
        raise ValueError("Native uniform recompute requires a positive integer layer span")
    graph = graph_module.graph
    layers = dict.fromkeys(node.meta["isoexec_logical_layer"] for node in graph.nodes
                           if "isoexec_logical_layer" in node.meta and node.op not in ("placeholder", "get_attr", "output"))
    if not layers:
        raise ValueError("Native full recompute requires captured logical decoder-layer boundaries")
    boundaries = list(layers)
    regions = []
    for start in range(0, len(boundaries), span):
        group = tuple(boundaries[start:start + span])
        nodes = [node for node in graph.nodes if node.meta.get("isoexec_logical_layer") in group
                 and node.op not in ("placeholder", "get_attr", "output")]
        if any(getattr(getattr(node.target, "_schema", None), "is_mutable", False) for node in nodes):
            raise ValueError("Native full recompute requires functional layer operations")
        owned = set(nodes)
        external = {source for node in nodes for source in node.all_input_nodes
                    if source not in owned and source.op != "get_attr"}
        inputs = [node for node in graph.nodes if node in external]
        order = {node: index for index, node in enumerate(graph.nodes)}
        if any(order[node] >= order[nodes[0]] for node in inputs):
            raise ValueError("Native full recompute requires layer inputs before their region")
        outputs = [node for node in nodes if any(user not in owned for user in node.users)]
        if not outputs:
            raise ValueError("Native full recompute found a layer region without live outputs")
        region_graph = fx.Graph()
        values = {node: region_graph.placeholder(node.name) for node in inputs}

        def value(node, values=values, region_graph=region_graph):
            if node not in values:
                if node.op != "get_attr":
                    raise ValueError(f"Unresolved checkpoint region input: {node.name}")
                values[node] = region_graph.get_attr(str(node.target))
            return values[node]

        for node in nodes:
            values[node] = region_graph.node_copy(node, value)
        region_graph.output(tuple(values[node] for node in outputs))
        region_graph.lint()
        program = fx.GraphModule(graph_module, region_graph)
        program.training = graph_module.training
        # Only an output some external expression may write in place needs a private copy.
        # Members of any checkpoint region are functional by the same construction rule this region obeys.
        mutated = tuple(any(_may_mutate(user, node) for user in node.users
                            if user not in owned and user.meta.get("isoexec_logical_layer") not in layers)
                        for node in outputs)
        run = _layer_forward(program, mutated)
        run.__name__ = run.__qualname__ = f"_checkpoint_layers_{start}"
        run.layer_paths = tuple(path for _, path in group)
        run.node_names = tuple(node.name for node in nodes)
        with graph.inserting_before(nodes[0]):
            call = graph.call_function(run, tuple(inputs))
            for index, node in enumerate(outputs):
                selected = graph.call_function(operator.getitem, (call, index))
                selected.meta.update(node.meta)
                # This node is the external result, not another region member.
                selected.meta.pop("isoexec_logical_layer", None)
                node.replace_all_uses_with(selected)
        for node in reversed(nodes):
            graph.erase_node(node)
        regions.append(run)
    for node in tuple(graph.nodes):
        if node.op == "get_attr" and not node.users:
            graph.erase_node(node)
    graph.lint()
    graph_module.recompile()
    return tuple(regions)


def _may_mutate(user: fx.Node, value: fx.Node) -> bool:
    """Whether ``user`` may write ``value`` in place. Conservative: anything not provably functional.

    Functional: a schema'd op whose argument holding ``value`` is not written (``alias_info.is_write``),
    ``operator.getitem``, and an already-built checkpoint region (regions refuse mutable ops at
    construction). Members of a region still to be built are excluded by the caller for the same reason.
    """
    if user.op == "output":
        return False
    if user.op == "call_function":
        if user.target is operator.getitem:
            return False
        if getattr(user.target, "program", None) is not None and getattr(user.target, "layer_paths", None):
            return False  # another checkpoint region: functional by construction
        schema = getattr(user.target, "_schema", None)
        if schema is not None:
            flat = list(user.args) + [user.kwargs.get(a.name) for a in schema.arguments[len(user.args):]]
            for arg, spec in zip(flat, schema.arguments):
                if spec.alias_info is not None and spec.alias_info.is_write and _refers(arg, value):
                    return True
            return False
    return True


def _refers(arg, value: fx.Node) -> bool:
    if arg is value:
        return True
    if isinstance(arg, (list, tuple)):
        return any(_refers(a, value) for a in arg)
    return False


def _layer_forward(program: fx.GraphModule, mutated: tuple[bool, ...] | None = None):
    def run(*values):
        def invoke_values(arguments):
            contexts = {}
            fresh = []
            for value in arguments:
                if isinstance(value, SequenceContext):
                    value = contexts.setdefault(id(value), replace(value, rope_freqs=None, rope_cos_sin=None))
                fresh.append(value)
            return program(*fresh)

        if not run.enabled or not torch.is_grad_enabled():
            return invoke_values(values)
        indices = [index for index, value in enumerate(values) if isinstance(value, torch.Tensor)]
        tensors = tuple(values[index] for index in indices)
        if not tensors:
            raise ValueError("Native full recompute requires a tensor activation input")
        anchor = torch.empty((), device=tensors[0].device, dtype=torch.float32, requires_grad=True)
        metadata = [None if isinstance(value, torch.Tensor) else value for value in values]

        # Backward replays this microbatch after its SkyRL forward context has exited.
        requests = rollout.current()

        def invoke(*saved):
            arguments = list(metadata)
            for index, tensor in zip(indices, saved[:-1], strict=True):
                arguments[index] = tensor
            # The first pass runs without grad and records what context-parallel peers hand this rank;
            # the recompute (inside the backward, with grad) replays it instead of re-running the
            # forward hand-offs, which would serialize the CP ranks (ops/collectives/cp_transport.py).
            with rollout.carry(requests), cp_region("replay" if torch.is_grad_enabled() else "record"):
                return invoke_values(arguments)

        from megatron.core import tensor_parallel

        result = tensor_parallel.checkpoint(invoke, False, *tensors, anchor)
        # A region output that an external expression may modify in place gets a private copy; so does one
        # that aliases a region input or another output (a pass-through), whose in-place modification would
        # reach storage the checkpoint still reads. Every other output is returned as produced.
        seen: set = {t.untyped_storage().data_ptr() for t in tensors}
        out = []
        for index, value in enumerate(result):
            if isinstance(value, torch.Tensor):
                ptr = value.untyped_storage().data_ptr()
                if mutated is None or mutated[index] or ptr in seen:
                    value = value.clone()
                else:
                    seen.add(ptr)
            out.append(value)
        return tuple(out)

    run.enabled = True
    run.program = program
    run.mutated_outputs = mutated
    return run
