"""Map native logical-layer ranges to executable stage definitions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from torch import fx, nn

from isoexec.nn.component import Component
from isoexec.nn.embed import Embedding
from isoexec.nn.head import LMHead

from isoexec.compiler.capture.graph import ExecutionGraph, PipelineError, get_callable_nodes
from .outputs import model_outputs


@dataclass(frozen=True)
class StageDefinition:
    """Describe a local stage for native pipeline-parallel construction.

    The ``module`` reuses upstream submodules and registered tensors. All its
    attributes (e.g., graph, submodules, parameters, buffers) must be treated
    as read-only and immutable during lowering.

    Attributes:
        module: FX graph module defining the local computation.
        pp_shared_parameter: qualified name of the local parameter involved in
            a cross-PP embedding/head tie; None if the stage has no such tie.
        remote_parameters: global parameter names that only other PP stages
            register.
        parameter_aliases: global parameter names this stage registers under
            another local name, e.g., ``embed.weight`` to a final stage's
            ``head.weight``.
    """

    module: fx.GraphModule
    pp_shared_parameter: str | None
    # TODO(jiangjiawei1103): After MoE support, review PP weight-name ownership.
    # Remove these fields and their native-model copies once loading/update boundaries
    # supply this infor, preserving unknown-name rejection and cross-PP tied-weight loading.
    remote_parameters: frozenset[str] = frozenset()
    parameter_aliases: Mapping[str, str] = field(default_factory=dict)


def _error(captured: ExecutionGraph, reason: str, *, code: str = "PP_ASSIGNMENT_INVALID",
           location: str = "assignment", context: Mapping[str, object] | None = None) -> PipelineError:
    root = next((call for call in captured.calls if not call.path and call.parent is None), None)
    model_type = type(captured.program) if root is None else root.module_type
    return PipelineError(
        code, phase="partition", model=f"{model_type.__module__}.{model_type.__qualname__}", location=location, reason=reason,
        action=("Provide native-assigned, ordered half-open layer ranges covering every logical layer once."
                if code == "PP_ASSIGNMENT_INVALID" else
                "Use a partition with representable dependencies and local state ownership; "
                "resolve the named missing boundary capability before native execution."),
        context=context,
    )


def _validate_pp_ranges(captured: ExecutionGraph,
                        assignments: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Validate PP layer ranges and return a contiguous cover of all logical layers."""
    if not isinstance(assignments, (tuple, list)) or not assignments:
        raise _error(captured, "Expected one ordered (start, stop) layer range per stage")
    result: list[tuple[int, int]] = []
    cursor, count = 0, len(captured.layers)
    for index, pair in enumerate(assignments):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise _error(captured, f"Stage {index} must specify (start, stop)")
        start, stop = pair
        if type(start) is not int or type(stop) is not int or start != cursor or not start <= stop <= count:
            raise _error(captured, f"Stage {index} range {pair!r} is not an ordered contiguous cover of {count} layers")
        if start == stop:
            raise _error(captured, f"Stage {index} must contain at least one logical layer")
        result.append((start, stop))
        cursor = stop
    if cursor != count:
        raise _error(captured, f"Ranges cover {cursor} of {count} logical layers")
    return tuple(result)


def _map_callable_nodes_to_pp_ranks(captured: ExecutionGraph, assignments: tuple[tuple[int, int], ...]) -> dict[fx.Node, int]:
    nodes = tuple(captured.program.graph.nodes)
    callable_nodes = get_callable_nodes(nodes)
    by_name = {node.name: node for node in nodes}
    order = {node: index for index, node in enumerate(nodes)}
    layer_stage = {layer: stage for stage, (start, stop) in enumerate(assignments) for layer in range(start, stop)}
    if not captured.layers:
        raise _error(captured, "Captured model has no logical layers")
    scopes = [get_callable_nodes(by_name[name] for name in layer.nodes)
              for layer in captured.layers]
    scoped = {node: layer_stage[index] for index, scope in enumerate(scopes) for node in scope}
    first, last = order[scopes[0][0]], order[scopes[-1][-1]]
    owners: dict[fx.Node, int] = {}
    for node in callable_nodes:
        if node.meta.get("isoexec_guard"):
            owners[node] = 0
        elif node in scoped:
            owners[node] = scoped[node]
        elif order[node] < first:
            owners[node] = 0
        elif order[node] > last:
            owners[node] = len(assignments) - 1
        else:
            # Inter-layer work remains with the preceding layer, in original order.
            previous = max(i for i, scope in enumerate(scopes) if order[scope[-1]] < order[node])
            owners[node] = layer_stage[previous]
    for node, stage in owners.items():
        for dependency in node.all_input_nodes:
            if dependency in owners and owners[dependency] > stage:
                raise _error(captured, f"{node.name} depends on later-stage value {dependency.name}",
                             code="PP_BOUNDARY_UNSUPPORTED", location=node.name)
    return owners


def _get_nodes_crossing_pp_cut(
    callable_nodes: Sequence[fx.Node],
    pp_rank_by_callable_node: Mapping[fx.Node, int],
    *,
    after_pp_rank: int,
) -> tuple[fx.Node, ...]:
    """Return nodes computed before the PP cut and consumed after it.

    A node computed at PP rank <= ``after_pp_rank`` and used at PP rank
    > ``after_pp_rank`` crosses the cut; its value must be passed across it.
    """
    crossing_nodes = []
    # Iterate over all callable nodes since a value may cross a stage that doesn't use it.
    for node in callable_nodes:
        if pp_rank_by_callable_node[node] > after_pp_rank:
            continue
        for user in node.users:
            # The output node has no PP rank; handle None PP rank below.
            user_pp_rank = pp_rank_by_callable_node.get(user)
            if user_pp_rank is not None and user_pp_rank > after_pp_rank:
                crossing_nodes.append(node)
                break
    return tuple(crossing_nodes)


def _build_local_graph(
    selected: Sequence[fx.Node],
    inputs: Sequence[fx.Node],
    outputs: Sequence[fx.Node],
    *,
    final: bool,
    aux: Mapping[str, fx.Node] | None = None,
) -> fx.Graph:
    graph = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}
    for node in inputs:
        copied = graph.placeholder(node.name, type_expr=node.type)
        copied.meta.update(node.meta)
        env[node] = copied

    def resolve(node: fx.Node) -> fx.Node:
        if node not in env:
            if node.op != "get_attr":
                raise ValueError(f"Missing stage input for {node.name}")
            env[node] = graph.node_copy(node, resolve)
        return env[node]

    for node in selected:
        env[node] = graph.node_copy(node, resolve)
    result = tuple(resolve(node) for node in outputs)
    primary = result[0] if final else result
    graph.output((primary, {name: resolve(node) for name, node in aux.items()}) if aux else primary)
    return graph


def _validate_state_sharing(captured: ExecutionGraph) -> None:
    """Reject multiple StateRefs for the same state ID."""
    seen_state_ids = set()
    for ref in captured.contract.state_refs:
        if ref.state_id in seen_state_ids:
            raise _error(captured, f"Persistent state {ref.state_id!r} has multiple StateRefs; state sharing is unsupported",
                         code="PP_BOUNDARY_UNSUPPORTED", location=ref.module_path)
        seen_state_ids.add(ref.state_id)


def _pp_shared_parameter(
    captured: ExecutionGraph,
    pp_rank_by_callable_node: Mapping[fx.Node, int],
    world_size: int,
    pp_rank: int,
) -> str | None:
    """Find this stage's qualified name for shared parameter.

    Support only cross-PP embed/head tie now. Sharing confined to one stage
    needs no cross-PP name.
    """
    gm = captured.program
    nodes = gm.graph.nodes
    head = next(node for node in nodes if node.name == captured.head)
    head_module = gm.get_submodule(str(head.target)) if head.op == "call_module" else None
    head_weight = id(head_module.weight) if isinstance(head_module, LMHead) else None

    param_refs_by_id = defaultdict(dict)
    embeds: dict[int, str] = {}
    # Current embed/head ties use call_module; skip get_attr parameter uses.
    for node in nodes:
        if node.op != "call_module":
            continue

        module = gm.get_submodule(str(node.target))
        node_pp_rank = pp_rank_by_callable_node[node]
        for name, parameter in module.named_parameters(remove_duplicate=False):
            param_refs = param_refs_by_id[id(parameter)]
            if node_pp_rank not in param_refs:
                param_refs[node_pp_rank] = f"{node.target}.{name}"
        if node_pp_rank == 0 and isinstance(module, Embedding) and id(module.weight) not in embeds:
            embeds[id(module.weight)] = f"{node.target}.weight"
    shared = None
    for param_id, refs in param_refs_by_id.items():
        if len(refs) == 1:
            # Embed/head tie within the same stage doesn't need cross-PP sharing.
            continue

        if param_id != head_weight or param_id not in embeds or set(refs) != {0, world_size - 1}:
            param_name = next(iter(refs.values()))
            raise _error(captured, f"Parameter {param_name!r} is shared across stages {tuple(refs)}; "
                         "only the first/last embedding and LMHead tie is supported",
                         code="PP_BOUNDARY_UNSUPPORTED", location=param_name)
        if pp_rank == 0:
            shared = embeds[param_id]
        elif pp_rank == world_size - 1:
            shared = f"{head.target}.weight"
    return shared


def define_stage(
    captured: ExecutionGraph,
    assignments: Sequence[tuple[int, int]],
    *,
    pp_rank: int,
) -> StageDefinition:
    """Define the requested PP stage with local inputs and outputs.

    Referenced submodules and registered tensors retain their identities and
    global names. Returned module must be treated as read-only and immutable
    during lowering.

    Args:
        captured: Unplaced captured computation.
        assignments: Half-open logical-layer ranges, one per PP rank.
        pp_rank: PP rank index.

    Returns:
        A stage definition whose module returns ``LMHead`` logits on the final
        stage, or an ordered tuple of values needed by later stages. With local
        auxiliary outputs, it returns ``(logits, aux)`` on the final stage or
        ``(outputs, aux)`` otherwise. Auxiliary values cross stages only if a
        later stage consumes them.
    """
    pp_ranges = _validate_pp_ranges(captured, assignments)
    if type(pp_rank) is not int or not 0 <= pp_rank < len(pp_ranges):
        raise _error(captured, f"pp_rank={pp_rank!r} is outside the {len(pp_ranges)} PP stages")
    _validate_state_sharing(captured)

    src_gm = captured.program
    src_gm.graph.lint()
    if any(isinstance(module, Component) and module.placed for module in src_gm.modules()):
        raise _error(captured, "Partition before component placement and local parameter sharding",
                     code="PP_BOUNDARY_UNSUPPORTED")
    nodes = tuple(src_gm.graph.nodes)
    callable_nodes = get_callable_nodes(nodes)
    pp_rank_by_callable_node = _map_callable_nodes_to_pp_ranks(captured, pp_ranges)

    output = next(node for node in nodes if node.op == "output")
    returned, aux = model_outputs(output.args[0])
    if not isinstance(returned, fx.Node) or returned not in callable_nodes:
        raise _error(captured, "Expected one computed terminal output", code="PP_BOUNDARY_UNSUPPORTED")
    if any(node not in callable_nodes for node in aux.values()):
        raise _error(captured, "Auxiliary outputs must be produced by captured operations",
                     code="PP_BOUNDARY_UNSUPPORTED")

    if len(pp_ranges) == 1:
        # TODO(jiangjiawei1103): Remove recompile if we make sure graph isn't edited.
        src_gm.recompile()
        return StageDefinition(src_gm, None)

    local_nodes = tuple(node for node in nodes if pp_rank_by_callable_node.get(node) == pp_rank)
    local_roots = {
        input_node
        for node in local_nodes
        for input_node in node.all_input_nodes
        # Source placeholders represent forward function inputs.
        if input_node.op == "placeholder"
    }
    # _build_local_graph creates placeholders by inputs order so we preserve positional
    # arg order for local forward.
    inputs = tuple(node for node in nodes if node in local_roots)
    if pp_rank != 0:
        inputs += _get_nodes_crossing_pp_cut(
            callable_nodes, pp_rank_by_callable_node, after_pp_rank=pp_rank - 1,
        )
    final_stage = pp_rank == len(pp_ranges) - 1
    if final_stage:
        outputs = (returned,)
    else:
        outputs = _get_nodes_crossing_pp_cut(
            callable_nodes, pp_rank_by_callable_node, after_pp_rank=pp_rank,
        )
    local_aux = {name: node for name, node in aux.items() if node in local_nodes}
    shared = _pp_shared_parameter(captured, pp_rank_by_callable_node, len(pp_ranges), pp_rank)
    graph = _build_local_graph(local_nodes, inputs, outputs, final=final_stage, aux=local_aux)
    stage_gm = fx.GraphModule(src_gm, graph)

    # PyTorch <= 2.13 re-registers get_attr buffers as persistent.
    # https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/torch/fx/graph_module.py#L252-L256
    # 2.14's _copy_attr keeps the source's persistence; remove this once 2.13 leaves checks/dependencies.py.
    for node in graph.nodes:
        if node.op == "get_attr":
            parent, _, name = str(node.target).rpartition(".")
            if name in src_gm.get_submodule(parent)._non_persistent_buffers_set:
                stage_gm.get_submodule(parent)._non_persistent_buffers_set.add(name)

    graph.lint()
    remote, aliases = _stage_parameter_names(src_gm, stage_gm)
    return StageDefinition(stage_gm, shared, remote, aliases)


def _stage_parameter_names(source: fx.GraphModule, stage: fx.GraphModule) -> tuple[frozenset[str], dict[str, str]]:
    """Name source Parameters that ``stage`` registers under another name, and those other stages own.

    ``stage`` shares Parameter identity with ``source``.
    """
    local = dict(stage.named_parameters(remove_duplicate=False))
    owners: dict[nn.Parameter, str] = {}
    for name, parameter in local.items():
        owners.setdefault(parameter, name)
    remote, aliases = set(), {}
    for name, parameter in source.named_parameters(remove_duplicate=False):
        if name in local:
            continue
        if parameter in owners:
            aliases[name] = owners[parameter]
        else:
            remote.add(name)
    # Checkpoint buffers (e.g. DeepSeek V4.1's frozen Engram tables) that only other stages register are remote
    # too, so their checkpoint names are skipped on this stage like remote Parameters. A buffer this stage
    # registers (under any name) is neither remote nor aliased: unchanged.
    local_buffers = dict(stage.named_buffers(remove_duplicate=False))
    held = {id(buffer) for buffer in local_buffers.values()}
    for name, buffer in source.named_buffers(remove_duplicate=False):
        if name not in local_buffers and id(buffer) not in held:
            remote.add(name)
    return frozenset(remote), aliases


__all__ = ["StageDefinition", "define_stage"]
