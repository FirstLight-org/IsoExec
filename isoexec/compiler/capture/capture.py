"""Capture fixed authored dataflow while retaining existing component calls."""

from __future__ import annotations

import inspect
import traceback
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import torch
from torch import fx, nn
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

import isoexec.nn as inn
from isoexec.core.registry_build import build_registry
from isoexec.compiler.contract.inference import OP_FACTS_BY_OP
from isoexec.compiler.contract.sema import build_model_contract
from isoexec.nn.capture_state import module_tree_state
from isoexec.nn.inspect import executed_modules, inventory
from isoexec.nn.unexecuted import Unexecuted, is_unexecuted_subtree
from isoexec.schema import SequenceContext
from isoexec.state.inference import STATE_SPECS_BY_OP

from .capture_context import capture_defaults
from .component_capture import is_builtin_capture_component, is_capture_component
from .export_boundary import CaptureSession, ExportRoot, project_export_meta
from .export_normalize import normalize
from .graph import ExecutionGraph, PipelineError, get_callable_nodes
from .layers import resolve_layers
from isoexec.compiler.outputs import model_outputs
from .rootspec import RootTensorLayout, bind_root_layouts


def _name(model: nn.Module) -> str:
    return f"{type(model).__module__}.{type(model).__qualname__}"


def _location(model: nn.Module, exc: Exception | None = None) -> str:
    frames = traceback.extract_tb(exc.__traceback__) if exc is not None else traceback.extract_stack()
    authored = [frame for frame in frames if frame.name in ("forward", "pipeline_layers")]
    if authored:
        frame = authored[-1]
        return f"{frame.filename}:{frame.lineno} ({frame.name})"
    try:
        return f"{inspect.getsourcefile(type(model).forward)}:{inspect.getsourcelines(type(model).forward)[1]}"
    except (OSError, TypeError):
        return f"{_name(model)}.forward"


def _error(model: nn.Module, reason: str, *, location: str | None = None,
           code: str = "PP_CAPTURE_UNSUPPORTED", context: dict[str, Any] | None = None,
           action: str | None = None) -> PipelineError:
    return PipelineError(
        code, phase="capture", model=_name(model), location=location or _location(model), reason=reason,
        action=action or "Use fixed authored dataflow and existing isoexec.nn components. Activation mutation must be functionalizable. "
                        "Invoke each stateful module once. A layer annotation cannot repair capture.",
        context=context,
    )


def _snapshot(model: nn.Module) -> tuple[Any, ...]:
    """Detect registration/tensor/mode/scalar mutation during author callbacks.

    This is a consistency check, not a sandbox for arbitrary Python side effects.
    Configuration and external Python state must remain immutable during capture.
    """
    return module_tree_state(model, objects=True)


def _execution_modes() -> tuple[Any, ...]:
    return (
        torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
        tuple((device, torch.is_autocast_enabled(device), str(torch.get_autocast_dtype(device)))
              for device in ("cpu", "cuda")),
    )


def _preflight(model: nn.Module) -> None:
    state_paths: dict[int, list[str]] = defaultdict(list)
    module_paths: dict[int, list[str]] = defaultdict(list)
    layer_members = {id(child) for module in model.modules() if isinstance(module, nn.ModuleList) for child in module}
    for path, module in model.named_modules(remove_duplicate=False):
        module_paths[id(module)].append(path)
    for identity in layer_members:
        if len(module_paths[identity]) > 1:
            raise _error(model, 'Pipeline layer modules require one registered non-root path',
                         code='PP_LAYERS_INVALID', context={'paths': module_paths[identity]})
    for path, module in executed_modules(model, remove_duplicate=False):
        hooks = (module._forward_hooks, module._forward_pre_hooks, module._backward_hooks, module._backward_pre_hooks)
        if any(hooks):
            raise _error(model, "Module hooks are not captured; "
                         "put the computation in the authored forward method.", location=path or "<root>")
        if any(getattr(tensor, "_backward_hooks", None) or getattr(tensor, "_post_accumulate_grad_hooks", None)
               for tensor in (*module.parameters(recurse=False), *module.buffers(recurse=False))):
            raise _error(model, "Tensor backward hooks are not part of the captured storage contract; "
                         "compose supported tensor operations or an existing Component backward instead.",
                         location=path or "<root>")
        if isinstance(module, inn.Component) and module.placed:
            raise _error(model, "Capture must precede component placement and local sharding.", location=path)
        if is_capture_component(module):
            if not is_builtin_capture_component(module):
                for child_path, child in executed_modules(module):
                    if child is not module and hasattr(child, "stateful_op_id"):
                        raise _error(model, "A registered opaque Component contains another stateful invocation; "
                                     "compose stateful components in an ordinary nn.Module outside the boundary.",
                                     location=f"{path}.{child_path}")
            for child_path, child in module.named_modules():
                if not is_capture_component(child) and type(child).forward is not nn.Module.forward:
                    raise _error(model, "An opaque component contains an authored executable child; "
                                 "compose it in an ordinary nn.Module outside the component.", location=f"{path}.{child_path}")
        if isinstance(module, Unexecuted) or is_unexecuted_subtree(module):
            continue
        if hasattr(module, "stateful_op_id"):
            state_paths[id(module)].append(path)
    for paths in state_paths.values():
        if len(paths) > 1:
            raise _error(model, f"One stateful module has multiple registered paths: {paths}.",
                         location=paths[0], code="PP_STATEFUL_INVOCATION", context={"paths": paths})


def _terminal_head(model: nn.Module, graph: fx.Graph) -> fx.Node:
    heads = [node for node in graph.nodes if node.op == "call_module" and not node.meta.get("isoexec_guard")
             and isinstance(model.get_submodule(str(node.target)), inn.LMHead)]
    output = next(node for node in graph.nodes if node.op == "output")
    work = [node for node in get_callable_nodes(graph.nodes)
            if not node.meta.get("isoexec_guard") and not node.meta.get("isoexec_validation")]
    returned, _ = model_outputs(output.args[0])
    if isinstance(returned, fx.Node) and returned.meta.get("isoexec_validation"):
        returned = returned.args[0]
    if len(heads) != 1 or returned is not heads[0] or work[-1] is not heads[0]:
        raise _error(model, "forward() must return the logits of one terminal isoexec.nn.LMHead call, optionally with named aux tensors; "
                     "post-head operations and extra head calls are outside this contract.", location=output.name,
                     context={"source": output.stack_trace or (work[-1].stack_trace if work else None)})
    return heads[0]


def _state_nodes(model: nn.Module, graph: fx.Graph, contract: Any) -> dict[str, str]:
    result = {}
    for ref in contract.state_refs:
        owners = [node for node in graph.nodes if node.op == "call_module"
                  and (ref.module_path == node.target or ref.module_path.startswith(str(node.target) + "."))]
        if len(owners) != 1:
            raise _error(model,
                         f"Stateful module {ref.module_path!r} has {len(owners)} captured invocations; expected one.",
                         location=ref.module_path, code="PP_STATEFUL_INVOCATION",
                         context={"nodes": [node.name for node in owners],
                                  "sources": [node.stack_trace for node in owners]})
        result[ref.module_path] = owners[0].name
    return result


@capture_defaults()
def capture(
    model: nn.Module, example_args: tuple[Any, ...], example_kwargs: Mapping[str, Any] | None = None,
    *, dynamic_shapes: tuple[Any, ...] | None = None, root_layouts: tuple[RootTensorLayout, ...] = (),
    seq: SequenceContext | None = None,
) -> ExecutionGraph:
    """Export and functionalize an unplaced authored model for later PP planning.

    Samples describe authored forward parameters (including defaults). An internal
    seq is appended to the canonical inputs; dynamic_shapes must describe it too.
    Export's guards remain executable. The normalized graph
    retains original components, parameters and OpTable dispatch; no markers run.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("capture() requires an authored nn.Module")
    _preflight(model)
    # Resolve logical output policy before capture.
    for _, module in executed_modules(model):
        if isinstance(module, inn.Component):
            module.prepare_output_contract()
    before, modes = _snapshot(model), _execution_modes()
    try:
        session = CaptureSession(model, example_args, example_kwargs, seq=seq)
        tensors = inventory(model)
        contract = build_model_contract(
            model, op_ids=build_registry().ops,
            inference_registry=STATE_SPECS_BY_OP, op_facts_registry=OP_FACTS_BY_OP,
        )
        root = ExportRoot(model, session)
        export_shapes = session.export_shapes(dynamic_shapes)
        tensor_inputs = [value for value in session.export_values if isinstance(value, torch.Tensor)]
        if not tensor_inputs:
            raise ValueError("Capture needs a Tensor example to declare its logical compute device")
        devices = {value.device for value in tensor_inputs if value.device.type not in ("cpu", "meta")}
        if len(devices) > 1:
            raise ValueError("One worker-local capture needs a single logical compute device")
        logical_device = next(iter(devices)) if devices else tensor_inputs[0].device
        existing_modes = {value.fake_mode for value in tensor_inputs if isinstance(value, FakeTensor)}
        if len(existing_modes) > 1:
            raise ValueError("Export inputs must share one FakeTensorMode")
        mode = next(iter(existing_modes)) if existing_modes else FakeTensorMode(shape_env=ShapeEnv())
        with mode:
            project_export_meta(root, logical_device)
            fake_values = tuple(mode.from_tensor(value, static_shapes=False) if isinstance(value, torch.Tensor)
                                else value for value in session.export_values)
            exported = torch.export.export(
                root, fake_values, strict=False,
                dynamic_shapes=None if export_shapes is None else {"values": export_shapes},
            )
            # Functionalization can delete unused draws, changing later RNG values.
            # Reject their effects in the raw graph, before any decomposition/DCE.
            for node in exported.graph.nodes:
                if torch.Tag.nondeterministic_seeded in getattr(node.target, "tags", ()):
                    raise _error(
                        model, f"Authored stochastic operation {node.target} requires an explicit RNG-effect contract.",
                        location=node.stack_trace or str(node.target),
                        action="Remove authored stochastic operations until IsoExec models RNG seed/counter and native scheduling effects.",
                        context={"operation": str(node.target), "node": node.name},
                    )
            functional = exported.run_decompositions(decomp_table={})
        program, calls = normalize(model, functional, session)
        bind_root_layouts(program, session, root_layouts)
        graph = program.graph
    except PipelineError:
        raise
    except ImportError as exc:
        raise _error(model, f"Capture dependency is unavailable: {exc}", location=_location(model, exc),
                     action="Install the named dependency in the worker environment.") from exc
    except Exception as exc:
        raise _error(model, f"Export capture failed: {type(exc).__name__}: {exc}",
                     location=_location(model, exc)) from exc
    if _snapshot(model) != before or _execution_modes() != modes:
        raise _error(model, "Capture changed registered modules/tensors, training mode, or scalar configuration. "
                     "Rebuild the model and remove persistent mutations from forward().")
    head = _terminal_head(model, graph)
    state_nodes = _state_nodes(model, graph, contract)
    layers, layer_source = resolve_layers(model, graph, calls, head)
    # Native recomputation follows the same authored layer boundaries as PP.
    # Immutable node labels survive cloning and native TP/SP/PP lowering.
    nodes_by_name = {node.name: node for node in graph.nodes}
    for layer in layers:
        for name in layer.nodes:
            nodes_by_name[name].meta["isoexec_logical_layer"] = (layer.index, layer.path)
    if _snapshot(model) != before or _execution_modes() != modes:
        raise PipelineError(
            "PP_LAYERS_INVALID", phase="layer_mapping", model=_name(model), location="pipeline_layers()",
            reason="The annotation changed registered modules/tensors, training mode, or scalar configuration.",
            action="Rebuild the model and return existing module references without mutating the model.",
        )
    return ExecutionGraph(
        program, calls, layers, layer_source, head.name,
        tensors, contract, state_nodes, tuple((path, module.training) for path, module in model.named_modules()), modes,
    )
