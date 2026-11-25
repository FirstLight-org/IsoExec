"""One capture session and export-only boundaries for original component calls."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch import nn
from torch.utils import _pytree as pytree

import isoexec.nn as inn
from isoexec.schema import SequenceContext
from isoexec.schema.sequence import register_pytrees
from isoexec.nn.unexecuted import Unexecuted, is_unexecuted_subtree

from .export_inputs import _shape_leaves
from .input_policy import _normalize_token_ids
from .component_capture import is_capture_component, supplies_sequence_context
from .capture_effects import capture_effects, execution_modes
from isoexec.nn.abstract import _OwnedWrites, _tensor_leaves
from isoexec.nn.capture_state import StateCopy, module_state, scalar_identity, tensor_state


@torch.library.custom_op("isoexec_capture::component", mutates_args=())
def component_marker(
    inputs: list[torch.Tensor], parameters: list[torch.Tensor],
    shape: list[int], dtype: torch.dtype, call_id: int, strides: list[int], device: torch.device,
    output_index: int = 0,
) -> torch.Tensor:
    raise RuntimeError("An export-only component marker reached execution; normalize it to an original component call")


@component_marker.register_fake
def _fake_component(
    inputs: list[torch.Tensor], parameters: list[torch.Tensor],
    shape: list[int], dtype: torch.dtype, call_id: int, strides: list[int], device: torch.device,
    output_index: int = 0,
) -> torch.Tensor:
    return torch.empty_strided(shape, strides, dtype=dtype, device=device)


MARKER_OP = torch.ops.isoexec_capture.component.default


@dataclass(frozen=True)
class Argument:
    kind: str
    value: Any


@dataclass
class Invocation:
    index: int
    path: str
    module_type: type[nn.Module]
    occurrence: int
    parent: int | None
    args: tuple[Argument, ...] = ()
    kwargs: dict[str, Argument] = field(default_factory=dict)
    output_dtype: torch.dtype | None = None
    output_spec: Any = None
    output_paths: tuple[Any, ...] = ()


class _CaptureSequence(SequenceContext):
    """Keep cache effects opaque while allowing ordinary readonly sequence fields."""

    def __setattr__(self, name: str, value: Any) -> None:
        raise ValueError(f"Writing SequenceContext.{name} in authored dataflow needs an explicit effect contract")

    def __getattribute__(self, name: str) -> Any:
        if name in ("rope_freqs", "rope_cos_sin"):
            raise ValueError(
                f"SequenceContext.{name} is runtime-owned mutable state; consume it inside an existing component"
            )
        return super().__getattribute__(name)


def _modes() -> tuple[Any, ...]:
    return execution_modes()


def _module_state(module: nn.Module) -> tuple[Any, ...]:
    return module_state(module, exclude=frozenset({'description'}) if isinstance(module, _Boundary) else frozenset(),
                        objects=True)


def _input_state(values: tuple[Any, ...]) -> tuple[Any, ...]:
    leaves, spec = pytree.tree_flatten(values)
    return spec, tuple(tensor_state(value) if isinstance(value, torch.Tensor)
                       else scalar_identity(value) for value in leaves)


class CaptureSession:
    """Keep input structure and transient call records together for one capture.

    Input descriptions are initialized once. Each export trace resets only the
    call records in begin(). Runtime graphs copy the metadata they need and do
    not retain this session or its example tensors.
    """

    def __init__(
        self, model: nn.Module, args: tuple[Any, ...], kwargs: Mapping[str, Any] | None = None,
        *, seq: SequenceContext | None = None,
    ) -> None:
        register_pytrees()
        self.authored_signature = inspect.signature(model.forward)
        bound = self.authored_signature.bind(*args, **({} if kwargs is None else kwargs))
        bound.apply_defaults()
        self.signature = self.authored_signature
        self.names = tuple(self.authored_signature.parameters)
        arguments = dict(bound.arguments)
        self.sequence_index: int | None = None
        if seq is not None:
            if not isinstance(seq, SequenceContext):
                raise TypeError("Internal seq must be a SequenceContext")
            if "seq" in self.names or kwargs is not None and "seq" in kwargs:
                raise ValueError("Internal seq conflicts with an authored seq argument")
            parameters = list(self.authored_signature.parameters.values())
            keyword_only = any(parameter.default is not inspect.Parameter.empty or parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD,
            ) for parameter in parameters)
            parameter = inspect.Parameter("seq", inspect.Parameter.KEYWORD_ONLY if keyword_only
                                          else inspect.Parameter.POSITIONAL_OR_KEYWORD)
            index = next((index for index, value in enumerate(parameters)
                          if value.kind == inspect.Parameter.VAR_KEYWORD), len(parameters))
            parameters.insert(index, parameter)
            self.signature = self.authored_signature.replace(parameters=parameters)
            # Canonical roots append seq even when Python places it before **kwargs.
            self.sequence_index = len(self.names)
            self.names += ("seq",)
            arguments["seq"] = seq
        normalized = _normalize_token_ids(arguments)
        self.example_values = tuple(normalized[name] for name in self.names)
        leaves, self.tree_spec = pytree.tree_flatten_with_path(self.example_values)
        self.leaf_paths = tuple(path for path, _ in leaves)

        self.calls: list[Invocation] = []
        self.active: list[int] = []
        self.counts: dict[str, int] = {}
        self.module_states: list[tuple[Any, ...]] = []
        self.roots: dict[int, int] = {}
        self.sequence: SequenceContext | None = None
        self.modes = _modes()

    @property
    def export_values(self) -> tuple[Any, ...]:
        """Flatten inputs to avoid nested-name collisions in PyTorch's guard code."""
        return tuple(self.tree_spec.flatten_up_to(self.example_values))

    def export_shapes(self, dynamic_shapes: tuple[Any, ...] | None) -> tuple[Any, ...] | None:
        if dynamic_shapes is None:
            return None
        if len(dynamic_shapes) != len(self.example_values):
            raise ValueError("dynamic_shapes must describe every canonical program input, including internal seq")
        return tuple(shape for value, shapes in zip(self.example_values, dynamic_shapes, strict=True)
                     for shape in _shape_leaves(value, shapes))

    def from_export_values(self, values: Sequence[Any]) -> tuple[Any, ...]:
        return self.tree_spec.unflatten(list(values))

    def invoke(self, model: nn.Module, canonical_values: Sequence[Any]) -> Any:
        """Call the original module, preserving its hooks and forward convention."""
        if len(canonical_values) != len(self.names):
            raise TypeError(f"Expected {len(self.names)} canonical arguments, got {len(canonical_values)}")
        positional: list[Any] = []
        keywords: dict[str, Any] = {}
        values = dict(zip(self.names, canonical_values, strict=True))
        for parameter in self.authored_signature.parameters.values():
            value = values[parameter.name]
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                positional.append(value)
            elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                if not isinstance(value, tuple):
                    raise TypeError(f"Canonical *{parameter.name} must be a tuple")
                positional.extend(value)
            elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
                keywords[parameter.name] = value
            else:
                if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
                    raise TypeError(f"Canonical **{parameter.name} must be a mapping with string keys")
                keywords.update(value)
        return model(*positional, **keywords)

    @property
    def authored(self) -> bool:
        return bool(self.active)

    @property
    def opaque(self) -> bool:
        return bool(self.active) and issubclass(self.calls[self.active[-1]].module_type, inn.Component)

    def begin(self, values: tuple[Any, ...]) -> tuple[Any, ...]:
        """Start a fresh trace without changing its input description."""
        self.calls.clear()
        self.active.clear()
        self.counts.clear()
        self.module_states.clear()
        self.roots.clear()
        self.sequence = None
        self.modes = _modes()
        wrapped: list[Any] = []
        for index, value in enumerate(values):
            if isinstance(value, SequenceContext):
                proxy = _CaptureSequence.__new__(_CaptureSequence)
                vars(proxy).update(vars(value))
                self.roots[id(proxy)] = index
                wrapped.append(proxy)
                if index == self.sequence_index:
                    self.sequence = proxy
            else:
                wrapped.append(value)
        return tuple(wrapped)

    def enter(self, path: str, module_type: type[nn.Module], module: nn.Module) -> None:
        occurrence = self.counts.get(path, 0)
        self.counts[path] = occurrence + 1
        index = len(self.calls)
        self.calls.append(Invocation(index, path, module_type, occurrence, self.active[-1] if self.active else None))
        self.active.append(index)
        self.module_states.append(_module_state(module))

    def leave(self, module: nn.Module) -> None:
        index = self.active.pop()
        before = self.module_states.pop()
        if _module_state(module) != before:
            raise ValueError(f"{self.calls[index].path or '<root>'}: authored forward mutated module state; "
                             "keep persistent configuration immutable")

    def encode(self, value: Any, tensors: list[torch.Tensor]) -> Argument:
        if isinstance(value, torch.Tensor):
            index = len(tensors)
            tensors.append(value)
            return Argument("tensor", index)
        if id(value) in self.roots:
            return Argument("root", self.roots[id(value)])
        if isinstance(value, tuple):
            return Argument("tuple", tuple(self.encode(item, tensors) for item in value))
        if isinstance(value, list):
            return Argument("list", tuple(self.encode(item, tensors) for item in value))
        if isinstance(value, dict):
            return Argument("dict", {key: self.encode(item, tensors) for key, item in value.items()})
        if isinstance(value, (str, int, float, bool, type(None), torch.dtype, torch.device)):
            return Argument("constant", value)
        raise TypeError(f"Unsupported component argument {type(value).__name__}; pass runtime context as a root input")


class _Boundary(nn.Module):
    def __init__(self, original: nn.Module, session: CaptureSession) -> None:
        super().__init__()
        # Metadata holds an unregistered reference. Held tensors keep original names.
        self.description = (original, session)
        self.training = original.training

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        original, session = self.description
        if _modes() != session.modes:
            raise ValueError("Execution context changed inside authored forward; grad/inference/autocast regions need an explicit export contract")
        if session.sequence is not None and supplies_sequence_context(original):
            bound = inspect.signature(original.forward).bind_partial(*args, **kwargs)
            if "seq" not in bound.arguments:
                if bound.signature.parameters["seq"].kind == inspect.Parameter.POSITIONAL_ONLY:
                    # Positional-only arguments need every preceding default filled.
                    bound.apply_defaults()
                bound.arguments["seq"] = session.sequence
                args, kwargs = bound.args, bound.kwargs
        call = session.calls[session.active[-1]]
        tensors: list[torch.Tensor] = []
        call.args = tuple(session.encode(value, tensors) for value in args)
        call.kwargs = {name: session.encode(value, tensors) for name, value in kwargs.items()}
        if not isinstance(original, inn.Component):
            raise TypeError("An export boundary requires a Component abstract contract")
        output = original.abstract_output(*args, **kwargs)
        leaves, call.output_spec = pytree.tree_flatten_with_path(output)
        call.output_paths = tuple(path for path, _ in leaves)
        call.output_dtype = leaves[0][1].dtype
        marked = [component_marker(tensors, list(self.parameters()), list(value.shape), value.dtype, call.index,
                                   list(value.stride()), value.device, index)
                  for index, (_, value) in enumerate(leaves)]
        return pytree.tree_unflatten(marked, call.output_spec)


def clone_for_export(model: nn.Module, session: CaptureSession, *, path: str = "", copier: StateCopy | None = None) -> nn.Module:
    if copier is None:
        paths = {id(module): f'{path}.{name}' if path and name else name or path for name, module in model.named_modules()}
        registered = {id(value) for module in model.modules() for values in (module._parameters, module._buffers)
                      for value in values.values() if value is not None}
        def tensor_copy(value: torch.Tensor) -> torch.Tensor:
            if id(value) in registered:
                # Capture needs private storage metadata, not checkpoint data.
                # project_export_meta moves these clone-only registrations onto
                # the input's logical device under FakeTensorMode.
                copied = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device='meta')
            else:
                with torch.no_grad():
                    copied = value.detach().clone()
            return nn.Parameter(copied, requires_grad=value.requires_grad) if isinstance(value, nn.Parameter) else copied
        copier = StateCopy(tensor_copy, lambda child: clone_for_export(child, session, path=paths[id(child)], copier=copier))
    if id(model) in copier.memo:
        return copier.memo[id(model)]
    if is_capture_component(model):
        result: nn.Module = _Boundary(model, session)
        copier.memo[id(model)] = result
        result._parameters = copier.value(model._parameters)
        result._buffers = copier.value(model._buffers)
        result._modules = copier.value(model._modules)
    else:
        if isinstance(model, inn.Component):
            raise ValueError("Component export requires an unchanged exact class registered with register_capture_component()")
        result = copy(model)
        copier.memo[id(model)] = result
        result.__dict__ = {name: copier.value(value) for name, value in vars(model).items()}
    result._forward_hooks = copy(model._forward_hooks)
    result._forward_pre_hooks = copy(model._forward_pre_hooks)

    def enter(_module: nn.Module, _args: tuple[Any, ...]) -> None:
        if isinstance(model, Unexecuted) or is_unexecuted_subtree(model):
            raise ValueError("An explicitly unexecuted subtree was invoked")
        session.enter(path, type(model), _module)

    def leave(_module: nn.Module, _args: tuple[Any, ...], _output: Any) -> None:
        session.leave(_module)

    result.register_forward_pre_hook(enter)
    result.register_forward_hook(leave, always_call=True)
    return result


class ExportRoot(nn.Module):
    def __init__(self, model: nn.Module, session: CaptureSession) -> None:
        super().__init__()
        self.model = clone_for_export(model, session)
        self.session = session

    def forward(self, *values: Any) -> Any:
        canonical = self.session.from_export_values(values)
        before = _input_state(canonical)
        wrapped = self.session.begin(canonical)
        try:
            with capture_effects(self.session), _OwnedWrites(self.model, _tensor_leaves(canonical)):
                output = self.session.invoke(self.model, wrapped)
            if _input_state(canonical) != before:
                raise ValueError("Authored forward mutated Python input structure or fields; an explicit effect contract is required")
            return output
        finally:
            self.session.sequence = None


def project_export_meta(model: nn.Module, device: torch.device) -> None:
    """Project clone-only registered meta tensors onto one logical input device.

    Called under the export FakeTensorMode: empty allocations describe storage
    and never materialize checkpoint weights. Shared registrations retain identity.
    The original authored tree is restored by normalize(), not this clone.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch._guards import detect_fake_mode

    if not isinstance(detect_fake_mode(), FakeTensorMode):
        raise ValueError("Export meta projection requires an active FakeTensorMode")
    memo: dict[int, torch.Tensor] = {}
    for module in model.modules():
        for entries, parameter in ((module._parameters, True), (module._buffers, False)):
            for name, value in tuple(entries.items()):
                if value is None or not value.is_meta:
                    continue
                if id(value) not in memo:
                    projected = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype,
                                                    device=device, requires_grad=value.requires_grad)
                    memo[id(value)] = (nn.Parameter(projected, requires_grad=value.requires_grad)
                                       if parameter or isinstance(value, nn.Parameter) else projected)
                cast(dict[str, torch.Tensor | None], entries)[name] = memo[id(value)]
