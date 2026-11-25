"""Normalize a functional ExportedProgram back to original component calls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import inspect
import operator
from typing import Any, Callable, cast

import torch
from torch import fx, nn
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind, OutputKind
from torch.utils import _pytree as pytree

from .export_boundary import Argument, CaptureSession, Invocation, MARKER_OP
from .export_inputs import ExportInputGuard, graph_value_at_path
from .graph import ModuleCall
from .input_policy import _normalize_token_ids, logical_input_contract
from isoexec.compiler.outputs import model_outputs


def _original_path(path: str) -> str:
    if path == "model":
        return ""
    if path.startswith("model."):
        return path[len("model."):]
    raise ValueError(f"Export scope {path!r} is outside the authored model")


def _decode(spec: Argument, tensors: Sequence[Any], roots: Sequence[fx.Node]) -> Any:
    if spec.kind == "tensor":
        return tensors[spec.value]
    if spec.kind == "root":
        return roots[spec.value]
    if spec.kind == "constant":
        return spec.value
    if spec.kind in ("tuple", "list"):
        values = [_decode(item, tensors, roots) for item in spec.value]
        return tuple(values) if spec.kind == "tuple" else values
    if spec.kind == "dict":
        return {key: _decode(value, tensors, roots) for key, value in spec.value.items()}
    raise ValueError(f"Unknown captured argument kind {spec.kind!r}")


def _checked_output(
    value: torch.Tensor, dtype: torch.dtype, path: str,
    shape: Sequence[int], strides: Sequence[int], device: torch.device,
    inputs: Sequence[torch.Tensor],
) -> torch.Tensor:
    expected = (tuple(shape), dtype, device, tuple(strides))
    actual = (tuple(value.shape), value.dtype, value.device, value.stride())
    if actual != expected or value.layout != torch.strided or value.storage_offset() != 0:
        raise RuntimeError(
            f"{path}: component output metadata {actual} differs from captured {expected}; "
            "the runtime binding must satisfy the captured component boundary"
        )
    is_alias = cast(Callable[[torch.Tensor, torch.Tensor], bool], getattr(torch._C, "_is_alias_of"))
    if value.is_conj() or value.is_neg() or any(is_alias(value, source) for source in inputs):
        raise RuntimeError(f"{path}: component output violates the captured fresh-output storage contract")
    return value


def _scope_ids(node: fx.Node, invocations: Sequence[Invocation]) -> tuple[int, ...]:
    lookup = {(call.path, call.occurrence): call.index for call in invocations}
    stack = cast(Mapping[str, tuple[str, type]], node.meta.get("nn_module_stack", {}))
    result: list[int] = []
    for key, (path, _) in stack.items():
        if path != "model" and not path.startswith("model."):
            continue
        suffix = key.rsplit("@", 1)[-1] if "@" in key else "0"
        occurrence = int(suffix) if suffix.isdecimal() else 0
        index = lookup.get((_original_path(path), occurrence))
        if index is not None:
            result.append(index)
    return tuple(result)


def _ancestors(call: Invocation, invocations: Sequence[Invocation]) -> tuple[int, ...]:
    result = [call.index]
    parent = call.parent
    while parent is not None:
        result.append(parent)
        parent = invocations[parent].parent
    return tuple(reversed(result))


def normalize(
    model: nn.Module, exported: ExportedProgram, session: CaptureSession,
) -> tuple[fx.GraphModule, tuple[ModuleCall, ...]]:
    unsupported = [spec.kind.name for spec in exported.graph_signature.output_specs
                   if spec.kind != OutputKind.USER_OUTPUT]
    if unsupported:
        raise ValueError(
            f"Exported persistent/input effects {unsupported} need a supported writeback contract; "
            "only activation mutation is normalized in this capture envelope"
        )
    graph = fx.Graph()
    roots = [graph.placeholder(name) for name in session.names]
    for node, value in zip(roots, session.example_values, strict=True):
        node.meta["isoexec_input_type"] = type(value)
        node.meta["isoexec_input_contract"] = logical_input_contract(value, str(node.target))
    root_objects: dict[str, Any] = {path: module for path, module in model.named_modules() if path}
    guard_name = "_isoexec_export_input_guard"
    if hasattr(model, guard_name):
        raise ValueError(f"{guard_name} is reserved for the export input contract")
    root_objects[guard_name] = ExportInputGuard(exported, session)
    guard = graph.call_module(guard_name, (tuple(roots),))
    guard.meta["isoexec_guard"] = True
    guard.meta["isoexec_calls"] = ()
    env: dict[fx.Node, Any] = {}
    user_index = 0
    specs = iter(exported.graph_signature.input_specs)
    seen_markers: set[int] = set()
    components: dict[int, fx.Node] = {}
    output_meta: dict[int, list[Any]] = {call.index: [None] * len(call.output_paths) for call in session.calls}
    for node in exported.graph.nodes:
        if node.target == MARKER_OP:
            arguments = cast(Sequence[Any], node.args)
            output_meta[arguments[4]][arguments[7] if len(arguments) > 7 else 0] = node.meta.get("val")

    for node in exported.graph.nodes:
        if node.op == "placeholder":
            spec = next(specs)
            if spec.kind == InputKind.USER_INPUT:
                env[node] = graph_value_at_path(graph, roots, session.leaf_paths[user_index])
                user_index += 1
            elif spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
                path = _original_path(cast(str, spec.target))
                root_objects[path] = model.get_parameter(path) if spec.kind == InputKind.PARAMETER else model.get_buffer(path)
                env[node] = graph.get_attr(path)
            elif spec.kind == InputKind.CONSTANT_TENSOR:
                name = f"_export_constant_{node.name}"
                root_objects[name] = exported.constants[cast(str, spec.target)]
                env[node] = graph.get_attr(name)
            else:
                raise ValueError(f"Unsupported exported input kind {spec.kind}")
            # Lifted parameters, buffers and constants remain tensor operands.
            # Preserve their shape/dtype for generic layout and gradient rules.
            env[node].meta.update(node.meta)
            continue

        if node.op == "output":
            flat = cast(tuple[Any, ...], fx.map_arg(node.args[0], lambda arg: env[arg]))
            assert exported.call_spec.out_spec is not None
            returned = pytree.tree_unflatten(list(flat), exported.call_spec.out_spec)
            logits, aux = model_outputs(returned)
            output = graph.output((logits, aux) if aux else logits)
            output.meta.update(node.meta)
            continue

        if node.op == "call_function" and node.target == MARKER_OP:
            arguments = cast(Sequence[Any], node.args)
            call_id = cast(int, arguments[4])
            if call_id >= len(session.calls):
                raise ValueError("Export changed a component invocation's marker identity")
            seen_markers.add(call_id)
            call = session.calls[call_id]
            tensors = cast(Sequence[Any], fx.map_arg(arguments[0], lambda arg: env[arg]))
            args = tuple(_decode(arg, tensors, roots) for arg in call.args)
            kwargs = {name: _decode(value, tensors, roots) for name, value in call.kwargs.items()}
            if call_id not in components:
                original = graph.call_module(call.path, args, kwargs)
                original.meta.update(node.meta)
                original.meta["val"] = pytree.tree_unflatten(output_meta[call_id], call.output_spec)
                original.meta["isoexec_calls"] = _ancestors(call, session.calls)
                components[call_id] = original
            original = components[call_id]
            selected = original
            metadata: Any = original.meta.get("val")
            output_index = arguments[7] if len(arguments) > 7 else 0
            for key in call.output_paths[output_index]:
                if isinstance(key, pytree.GetAttrKey):
                    selected = graph.call_function(getattr, (selected, key.name))
                else:
                    index = key.idx if isinstance(key, pytree.SequenceKey) else key.key
                    selected = graph.call_function(operator.getitem, (selected, index))
                metadata = key.get(metadata)
                selected.meta.update(original.meta)
                selected.meta["val"] = metadata
                selected.meta["isoexec_component"] = original.name
            shape = fx.map_arg(arguments[2], lambda arg: env[arg])
            strides = fx.map_arg(arguments[5], lambda arg: env[arg])
            checked = graph.call_function(
                _checked_output,
                (selected, arguments[3], call.path, shape, strides, arguments[6], tuple(tensors)),
            )
            checked.meta.update(selected.meta)
            checked.meta["isoexec_component"] = original.name
            checked.meta["isoexec_validation"] = True
            env[node] = checked
        else:
            if node.op != "call_function":
                raise ValueError(f"Unsupported export node {node.op}: {node.target}")
            if node.target == torch.ops.aten.copy.default:
                raise ValueError("aten.copy from slice mutation has no backward in PyTorch 2.11; "
                                 "use a functional slice expression supported by autograd")
            schema = getattr(node.target, "_schema", None)
            if schema is not None and schema.is_mutable:
                raise ValueError(f"Functionalization left a mutating operator {node.target}")
            if isinstance(node.target, torch._ops.HigherOrderOperator):
                raise ValueError(f"Export higher-order region {node.target} needs a component/state region contract")
            copied = graph.node_copy(node, lambda arg: env[arg])
            copied.meta["isoexec_calls"] = _scope_ids(node, session.calls)
            env[node] = copied

    expected = {call.index for call in session.calls if call.output_dtype is not None}
    if seen_markers != expected:
        raise ValueError(f"Export removed component calls {sorted(expected - seen_markers)}; effects cannot be discarded")
    # Lifted weights used only by markers are now held by the original components.
    program = fx.GraphModule(root_objects, graph)
    program.training = model.training
    graph.eliminate_dead_code()
    graph.lint()
    program.recompile()
    program.delete_all_unused_submodules()

    # Runtime binding needs no example tensors or ExportedProgram ownership.
    signature, names = session.signature, session.names

    def canonicalize(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        normalized = _normalize_token_ids(bound.arguments)
        return tuple(normalized[name] for name in names), {}

    program.register_forward_pre_hook(canonicalize, with_kwargs=True)
    program.meta["isoexec_input_signature"] = signature
    program.meta["isoexec_root_defaults"] = {
        name: parameter.default for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    nodes_by_call: dict[int, list[str]] = {call.index: [] for call in session.calls}
    for node in graph.nodes:
        for call_id in node.meta.setdefault("isoexec_calls", ()):
            nodes_by_call[call_id].append(node.name)
    calls = tuple(
        ModuleCall(call.index, call.path, call.module_type, call.parent,
                   tuple(nodes_by_call[call.index]))
        for call in session.calls
    )
    return program, calls
