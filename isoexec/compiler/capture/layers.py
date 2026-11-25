"""Resolve native logical-layer indices from module references and captured calls."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import cast

from torch import fx, nn

from isoexec.nn import Embedding, LMHead, RMSNorm

from .graph import LogicalLayer, ModuleCall, PipelineError, get_callable_nodes


def resolve_layers(
    model: nn.Module, graph: fx.Graph, calls: tuple[ModuleCall, ...], head: fx.Node,
) -> tuple[tuple[LogicalLayer, ...], str]:
    """Require an explicit mapping unless one serial ModuleList level is unique.

    An explicit mapping may describe a DAG: native boundary/layout admission is
    later. Automatic inference additionally requires a serial dependency chain
    and coverage of all non-embedding/norm/head component calls.
    """
    model_name = f"{type(model).__module__}.{type(model).__qualname__}"
    nodes = {node.name: node for node in graph.nodes}
    callable_nodes = get_callable_nodes(graph.nodes)
    order = {name: index for index, name in enumerate(nodes)}
    paths: defaultdict[int, list[str]] = defaultdict(list)
    by_path: defaultdict[str, list[ModuleCall]] = defaultdict(list)
    for path, module in model.named_modules(remove_duplicate=False):
        paths[id(module)].append(path)
    for call in calls:
        by_path[call.path].append(call)

    def refusal(
        location: str,
        reason: str,
        *,
        code: str = "PP_LAYERS_INVALID",
        context: Mapping[str, object] | None = None,
    ) -> PipelineError:
        return PipelineError(
            code, phase="layer_mapping", model=model_name, location=location, reason=reason,
            required_information="Ordered existing nn.Module instances for native logical layers.",
            action="Define Model.pipeline_layers(), for example: return self.blocks. "
                   "Select one enclosing module per layer in forward invocation order.",
            context=context,
        )

    def ancestors(name: str) -> set[str]:
        found: set[str] = set()
        pending = list(nodes[name].all_input_nodes)
        while pending:
            node = pending.pop()
            if node.name not in found:
                found.add(node.name)
                pending.extend(node.all_input_nodes)
        return found

    live = ancestors(head.name)

    def validate(entries: object) -> tuple[LogicalLayer, ...]:
        if not isinstance(entries, (nn.ModuleList, list, tuple)) or not len(entries):
            raise refusal("pipeline_layers()", "Expected a nonempty ModuleList, list, or tuple of module references.")
        layers: list[LogicalLayer] = []
        previous_end = -1
        used: set[str] = set()
        for index, module in enumerate(entries):
            location = f"pipeline_layers()[{index}]"
            if not isinstance(module, nn.Module):
                raise refusal(location, f"Expected an nn.Module, got {type(module).__name__}.")
            names = paths.get(id(module), [])
            if len(names) != 1 or not names[0]:
                raise refusal(location, f"Layer must have one registered non-root path; found {names}.")
            path = names[0]
            invocations = by_path[path]
            if len(invocations) != 1:
                raise refusal(location, f"{path!r} was invoked {len(invocations)} times; expected once.")
            call = invocations[0]
            work = tuple(node.name for node in get_callable_nodes(nodes[name] for name in call.nodes))
            if not work or not live.intersection(work):
                raise refusal(location, f"{path!r} has no captured computation contributing to the terminal head.")
            if head.name in call.nodes:
                raise refusal(location, f"{path!r} includes LMHead; select decoder layers before the terminal head.")
            if used.intersection(call.nodes):
                raise refusal(location, f"{path!r} overlaps an earlier logical layer's invocation scope.")
            start, end = order[work[0]], order[work[-1]]
            if start <= previous_end:
                raise refusal(location, f"{path!r} is out of forward invocation order.")
            # Calls are synchronous, but explicitly validate that their work is contiguous.
            outside = [n.name for n in callable_nodes if start <= order[n.name] <= end
                       and n.name not in call.nodes]
            if outside:
                raise refusal(location, f"{path!r} has interleaved operations: {outside}.")
            layers.append(LogicalLayer(index, path, call.index, call.nodes))
            used.update(call.nodes)
            previous_end = end
        return tuple(layers)

    if hasattr(model, "pipeline_layers"):
        provider = model.pipeline_layers
        if not callable(provider):
            raise refusal("pipeline_layers", "The explicit annotation must be a callable method.")
        try:
            entries = provider()
        except Exception as exc:
            raise refusal("pipeline_layers()", f"The annotation raised {type(exc).__name__}: {exc}") from exc
        return validate(entries), "explicit"

    candidates: dict[str, tuple[LogicalLayer, ...]] = {}
    complete: dict[str, tuple[LogicalLayer, ...]] = {}
    rejected: dict[str, str] = {}
    body = {
        node.name for node in graph.nodes if node.op == "call_module"
        and not node.meta.get("isoexec_guard")
        and not isinstance(model.get_submodule(cast(str, node.target)), (Embedding, RMSNorm, LMHead))
    }
    for path, module in model.named_modules():
        if not isinstance(module, nn.ModuleList):
            continue
        try:
            layers = validate(module)
        except PipelineError as exc:
            rejected[path] = exc.reason
            continue
        # Independent branches have no uniquely implied logical-layer order.
        if any(not any((set(left.nodes) & live).intersection(ancestors(name))
                       for name in right.nodes if name in live)
               for left, right in zip(layers, layers[1:])):
            rejected[path] = "Module calls do not form one serial dependency chain."
            continue
        candidates[path] = layers
        if body.issubset({name for layer in layers for name in layer.nodes}):
            complete[path] = layers

    # Even a locally valid nested list exposes another possible layer granularity.
    if len(complete) == 1 and len(candidates) == 1:
        return next(iter(complete.values())), "inferred"
    raise refusal(
        "forward()", "Cannot uniquely infer PP logical layers from captured module calls and dependencies.",
        code="PP_LAYERS_REQUIRED",
        context={"candidates": {path: [layer.path for layer in layers] for path, layers in candidates.items()},
                 "rejected_candidates": rejected},
    )
