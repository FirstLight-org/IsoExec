"""Input structure, specialization and parameter-free export guards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import operator
from typing import TYPE_CHECKING, Any

import torch
from torch import fx, nn
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.utils import _pytree as pytree

from isoexec.nn.capture_state import scalar_identity

if TYPE_CHECKING:
    from .export_boundary import CaptureSession


def _shape_leaves(value: Any, shape: Any) -> list[Any]:
    """Flatten a shape specification alongside its actual registered pytree.

    Tensor dimension mappings remain indivisible. PyTorch's custom-pytree shape
    convention describes children using a list or tuple in flatten order.
    """
    children, spec = pytree.tree_flatten(value, is_leaf=lambda child: child is not value)
    if spec.is_leaf():
        return [shape]
    if shape is None:
        child_shapes = [None] * len(children)
    elif isinstance(shape, (tuple, list)):
        child_shapes = list(shape)
    else:
        child_shapes = spec.flatten_up_to(shape)
    if len(children) != len(child_shapes):
        raise ValueError("dynamic_shapes does not match the registered input pytree's children")
    return [leaf for child, child_shape in zip(children, child_shapes, strict=True)
            for leaf in _shape_leaves(child, child_shape)]


def graph_value_at_path(graph: fx.Graph, roots: Sequence[fx.Node], path: pytree.KeyPath) -> fx.Node:
    """Read a canonical-input leaf using the same keyed pytree path as export."""
    if not path or not isinstance(path[0], pytree.SequenceKey):
        raise TypeError(f"An input leaf path must start with a canonical argument index: {path!r}")
    index = path[0].idx
    if not 0 <= index < len(roots):
        raise IndexError(f"Canonical argument index {index} is outside {len(roots)} roots")
    value = roots[index]
    for key in path[1:]:
        if isinstance(key, pytree.SequenceKey):
            value = graph.call_function(operator.getitem, (value, key.idx))
        elif isinstance(key, pytree.MappingKey):
            value = graph.call_function(operator.getitem, (value, key.key))
        elif isinstance(key, pytree.GetAttrKey):
            value = graph.call_function(getattr, (value, key.name))
        else:
            raise TypeError(f"Unsupported input pytree key {key!r}; register a keyed flatten function")
    return value


class _ExportConstraintAdapter:
    """PyTorch 2.11 fallback for exports without a generated ``_guards_fn``.

    This is the exact helper called by ExportedProgram._check_input_constraints.
    Snapshot only USER_INPUT placeholder metadata and ranges, rather than holding
    the ExportedProgram and its model state. Symbolic solving remains PyTorch's.
    """

    def __init__(self, exported: ExportedProgram) -> None:
        graph = fx.Graph()
        placeholders = [node for node in exported.graph.nodes if node.op == "placeholder"]
        self.placeholders = [
            graph.node_copy(node, lambda other: other)
            for node, spec in zip(placeholders, exported.graph_signature.input_specs, strict=True)
            if spec.kind == InputKind.USER_INPUT
        ]
        self.ranges = dict(exported.range_constraints)

    def __call__(self, flat_args_with_path: list[tuple[pytree.KeyPath, Any]]) -> None:
        from torch._export.utils import _check_input_constraints_for_graph

        _check_input_constraints_for_graph(self.placeholders, flat_args_with_path, self.ranges)


@dataclass(frozen=True)
class _SymbolicScalarMetadata:
    """Export owns the value constraints of a dynamic scalar input."""

    value_type: type


@dataclass(frozen=True)
class _TensorMetadata:
    """Only tensor metadata, never a retained real example tensor."""

    tensor_type: type[torch.Tensor]
    identity_group: int
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool
    layout: torch.layout
    storage_offset: int
    static_strides: tuple[int, ...] | None
    contiguous_formats: tuple[bool, ...]
    conjugate: bool
    negative: bool


_CONTIGUOUS_FORMATS = (torch.contiguous_format, torch.channels_last, torch.channels_last_3d)


def _contiguity(tensor: torch.Tensor) -> tuple[bool, ...]:
    return tuple(tensor.is_contiguous(memory_format=format_) for format_ in _CONTIGUOUS_FORMATS)


def _canonical_strides(tensor: torch.Tensor) -> tuple[int, ...]:
    # PyTorch supplies the layout calculation. The meta allocation has no data.
    return torch.empty(tensor.shape, dtype=tensor.dtype, device="meta").stride()


def _mapping_key_identity(key: Any) -> Any:
    if type(key) in (str, int, float, bool, bytes, type(None)):
        return scalar_identity(key)
    if type(key) is tuple:
        return tuple, tuple(_mapping_key_identity(item) for item in key)
    raise ValueError("Export input mapping keys require immutable scalar or tuple keys")


def _context_identity(value: Any) -> Any:
    """Freeze registered pytree context without retaining tensors or resources."""
    if isinstance(value, (str, int, float, bool, bytes, type(None), torch.dtype, torch.device, type)):
        return scalar_identity(value)
    if isinstance(value, (tuple, list)):
        return type(value), tuple(_context_identity(item) for item in value)
    if isinstance(value, Mapping):
        return type(value), tuple((_mapping_key_identity(key), _context_identity(item)) for key, item in value.items())
    if is_dataclass(value) and not isinstance(value, type):
        return type(value), tuple((field.name, _context_identity(getattr(value, field.name))) for field in fields(value))
    raise ValueError("Export input pytree context requires tensor-free scalar/container metadata; "
                     "put tensors in pytree leaves and runtime resources in explicit ports")


def _mapping_key_signature(value: Any) -> Any:
    children, spec = pytree.tree_flatten(value, is_leaf=lambda child: child is not value)
    if spec.is_leaf():
        return None
    keys = tuple(_mapping_key_identity(key) for key in value) if isinstance(value, Mapping) else None
    return spec.type, _context_identity(spec.context), keys, tuple(_mapping_key_signature(child) for child in children)


def _input_metadata(exported: ExportedProgram, values: list[Any]) -> tuple[Any, ...]:
    placeholders = [node for node in exported.graph.nodes if node.op == "placeholder"]
    user_nodes = [node for node, spec in zip(placeholders, exported.graph_signature.input_specs, strict=True)
                  if spec.kind == InputKind.USER_INPUT]
    identities: dict[int, int] = {}
    metadata: list[Any] = []
    for value, node in zip(values, user_nodes, strict=True):
        if not isinstance(value, torch.Tensor):
            if not isinstance(value, (int, float, bool, str, type(None))):
                raise ValueError(f"Unsupported exported input leaf {type(value).__name__}; register a tensor/scalar pytree")
            abstract = node.meta.get("val")
            scalar_types = ((torch.SymInt, int), (torch.SymFloat, float), (torch.SymBool, bool))
            symbolic = next((expected for symbolic, expected in scalar_types if isinstance(abstract, symbolic)), None)
            if symbolic is not None:
                if type(value) is not symbolic:
                    raise ValueError("Export symbolic scalar metadata disagrees with its example Python type")
                metadata.append(_SymbolicScalarMetadata(symbolic))
            else:
                metadata.append(scalar_identity(value))
            continue
        if value.layout != torch.strided:
            raise ValueError("Export input specialization currently supports strided tensors only")
        identity = identities.setdefault(id(value), len(identities))
        abstract = node.meta.get("val")
        if not isinstance(abstract, torch.Tensor):
            raise ValueError("Export tensor input lacks tensor metadata")
        dynamic = any(isinstance(size, torch.SymInt) for size in abstract.shape)
        if dynamic and (not value.is_contiguous() or value.stride() != _canonical_strides(value)):
            raise ValueError("Dynamic export inputs require canonical contiguous strides; noncontiguous layouts need a guard contract")
        metadata.append(_TensorMetadata(
            type(value), identity, value.dtype, value.device, value.requires_grad, value.layout,
            int(value.storage_offset()), None if dynamic else value.stride(), _contiguity(value),
            value.is_conj(), value.is_neg(),
        ))
    return tuple(metadata)


def _check_metadata(values: list[Any], metadata: tuple[Any, ...]) -> None:
    identities: dict[int, int] = {}
    for index, (value, expected) in enumerate(zip(values, metadata, strict=True)):
        if isinstance(expected, _SymbolicScalarMetadata):
            if type(value) is not expected.value_type:
                raise ValueError(f"Export input {index} changed its dynamic scalar Python type")
            continue
        if not isinstance(expected, _TensorMetadata):
            if scalar_identity(value) != expected:
                raise ValueError(f"Export input {index} changed a specialized scalar value or type")
            continue
        if not isinstance(value, torch.Tensor) or type(value) is not expected.tensor_type:
            raise ValueError(f"Export input {index} changed its tensor type")
        identity = identities.setdefault(id(value), len(identities))
        if identity != expected.identity_group:
            raise ValueError(f"Export input {index} changed tensor identity/alias specialization")
        if (value.dtype, value.device, value.requires_grad, value.layout, value.is_conj(), value.is_neg()) != (
            expected.dtype, expected.device, expected.requires_grad, expected.layout, expected.conjugate, expected.negative,
        ):
            raise ValueError(f"Export input {index} changed dtype, device, requires_grad, layout, or view flags")
        if value.storage_offset() != expected.storage_offset or _contiguity(value) != expected.contiguous_formats:
            raise ValueError(f"Export input {index} changed storage offset or contiguity specialization")
        strides = expected.static_strides if expected.static_strides is not None else _canonical_strides(value)
        if value.stride() != strides:
            raise ValueError(f"Export input {index} changed its specialized stride layout")


def _guard_only_graph(module: fx.GraphModule) -> fx.GraphModule | None:
    guards = [node for node in module.graph.nodes if node.op == "call_module" and node.target == "_guards_fn"]
    if not guards:
        return None
    needed: set[fx.Node] = set()

    def include(node: fx.Node) -> None:
        if node in needed:
            return
        needed.add(node)
        for dependency in node.all_input_nodes:
            include(dependency)

    for guard in guards:
        include(guard)
    # Keep every USER_INPUT, including unused and specialized constant inputs.
    needed.update(node for node in module.graph.nodes if node.op == "placeholder")
    for node in needed:
        if node.op == "placeholder" or node in guards:
            continue
        if node.op == "get_attr":
            value: Any = module
            for part in str(node.target).split("."):
                value = getattr(value, part)
            if isinstance(value, (torch.Tensor, nn.Module)):
                raise ValueError(f"Export input guard unexpectedly depends on model state: {node.target}")
        elif node.op in ("call_function", "call_method") and "val" in node.meta:
            leaves, _ = pytree.tree_flatten(node.meta["val"])
            if any(not isinstance(value, (int, float, bool, str, type(None), torch.SymInt, torch.SymFloat,
                                          torch.SymBool)) for value in leaves):
                raise ValueError(f"Export input guard unexpectedly depends on a data operation: {node.target}")
        else:
            raise ValueError(f"Unsupported export input guard dependency: {node.op} {node.target}")
    graph = fx.Graph()
    copied: dict[fx.Node, fx.Node] = {}
    for node in module.graph.nodes:
        if node in needed:
            copied[node] = graph.node_copy(node, lambda other: copied[other])
    graph.output(None)
    # A new Graph uses ordinary flat CodeGen, not the source module's pytree CodeGen.
    result = fx.GraphModule(module, graph, class_name="ExportInputConstraints")
    if any(True for _ in result.parameters()) or any(True for _ in result.buffers()):
        raise ValueError("Export input guard must not own model parameters or buffers")
    return result


class ExportInputGuard(nn.Module):
    """Check canonical inputs using export's own structure and generated constraints.

    The module owns only the generated guard subgraph, never the exported model.
    Mark it impure because its successful result is intentionally unused.
    """

    _is_impure = True

    def __init__(self, exported: ExportedProgram, session: CaptureSession) -> None:
        super().__init__()
        if exported.call_spec.in_spec is None or exported.example_inputs is None:
            raise ValueError("Export input guards require a recorded input tree and example inputs")
        self._input_spec = session.tree_spec
        self._mapping_keys = _mapping_key_signature(session.example_values)
        leaves, _ = pytree.tree_flatten_with_path(exported.example_inputs)
        self._input_paths = tuple(path for path, _ in leaves)
        self._metadata = _input_metadata(exported, list(session.export_values))
        self._ranges = tuple((str(symbol), str(bounds)) for symbol, bounds in exported.range_constraints.items())
        self.constraints = _guard_only_graph(exported.module())
        if self.constraints is None:
            # The placeholder/range checker does not evaluate extra predicates
            # recorded by capture. It is a complete fallback only without those.
            extra_guards = getattr(exported, "_guards_code", None)
            if extra_guards is None or extra_guards:
                raise ValueError("Generated export guards are missing; refusing to drop capture predicates")
        self._fallback = None if self.constraints is not None else _ExportConstraintAdapter(exported)

    def forward(self, canonical_values: tuple[Any, ...]) -> None:
        # Fixed dataflow specializes observable mapping iteration order as well
        # as keys and values; reordering leaves would change authored semantics.
        if (pytree.tree_structure(canonical_values) != self._input_spec
                or _mapping_key_signature(canonical_values) != self._mapping_keys):
            raise ValueError("Runtime input structure or mapping key order changed since capture")
        flat_values = self._input_spec.flatten_up_to(canonical_values)
        # Its prefix semantics permit a subtree at a recorded leaf. Export inputs
        # require the complete structure, so those values must remain pytree leaves.
        if any(not pytree.tree_structure(value).is_leaf() for value in flat_values):
            raise ValueError("Runtime input structure contains a subtree at an exported input leaf")
        _check_metadata(flat_values, self._metadata)
        try:
            if self.constraints is not None:
                self.constraints(*flat_values)
            else:
                if self._fallback is None:
                    raise RuntimeError("Export input constraints are unavailable")
                self._fallback(list(zip(self._input_paths, flat_values, strict=True)))
        except (AssertionError, RuntimeError) as error:
            shapes = [(index, tuple(value.shape)) for index, value in enumerate(flat_values)
                      if isinstance(value, torch.Tensor)]
            raise ValueError(f"Runtime inputs exceed the captured shape/value contract: {error}; "
                             f"tensor_shapes={shapes}, captured_ranges={self._ranges}. "
                             "Rebuild/capture with matching dynamic_shapes or a sufficient max_tokens bound.") from error


__all__ = ["ExportInputGuard", "graph_value_at_path"]
