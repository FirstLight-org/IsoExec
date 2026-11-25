"""Bind a referenced stateful op without inspecting model metadata."""

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

from isoexec.core.bindings import BoundOp
from isoexec.state.contract import StateRef
from isoexec.state.validation import ContractError, validate_id

from .contract import ModelContract


@dataclass(frozen=True)
class StatefulOpModulePathMapping:
    """Map one authored ``StateRef`` path to its reconstructed native module path.

    This record is needed only when contract construction and execution use
    different module trees. A host must not derive the native path from names,
    layer indexes, geometry, or traversal order.
    """

    authored_module_path: str
    native_module_path: str
    op_id: str


def validate_stateful_op_module_path_mappings(
    contract: ModelContract,
    mappings: tuple[StatefulOpModulePathMapping, ...],
    *,
    native_model: nn.Module,
) -> None:
    """Require one exact, op-compatible native path mapping per ``StateRef``.

    Validate path mappings while the intact native meta tree still exists.
    Errors must be raised before PP slicing, sharding, and realization since:
    - PP slicing removes off-stage modules
    - sharding mutates the native model
    - runtime realization allocates state
    """
    if not isinstance(contract, ModelContract):
        raise ContractError("module path mapping validation requires a ModelContract")
    if type(mappings) is not tuple:
        raise ContractError("stateful op module path mappings must be a tuple")
    if not isinstance(native_model, nn.Module):
        raise ContractError("native module path mapping validation requires an nn.Module")

    ref_by_authored_path = {ref.module_path: ref for ref in contract.state_refs}
    mapping_by_authored_path = {}
    native_paths = set()

    for mapping in mappings:
        if not isinstance(mapping, StatefulOpModulePathMapping):
            raise ContractError("expected a StatefulOpModulePathMapping")
        if not isinstance(mapping.authored_module_path, str):
            raise ContractError("StatefulOpModulePathMapping.authored_module_path must be a string")
        if not isinstance(mapping.native_module_path, str):
            raise ContractError("StatefulOpModulePathMapping.native_module_path must be a string")
        validate_id(mapping.op_id, "StatefulOpModulePathMapping.op_id")

        authored_path = mapping.authored_module_path
        if authored_path in mapping_by_authored_path:
            raise ContractError(f"duplicate authored_module_path mapping: {authored_path!r}")
        ref = ref_by_authored_path.get(authored_path)
        if ref is None:
            raise ContractError(f"unresolved authored_module_path: {authored_path!r}")
        if mapping.op_id != ref.op_id:
            raise ContractError(
                f"authored_module_path {authored_path!r}: mapping op_id {mapping.op_id!r} "
                f"does not match StateRef.op_id {ref.op_id!r}"
            )
        if mapping.native_module_path in native_paths:
            raise ContractError(f"duplicate native_module_path mapping: {mapping.native_module_path!r}")
        mapping_by_authored_path[authored_path] = mapping
        native_paths.add(mapping.native_module_path)

        context = (
            f"authored_module_path {mapping.authored_module_path!r}, "
            f"native_module_path {mapping.native_module_path!r}, op_id {mapping.op_id!r}"
        )
        _resolve_stateful_op(
            native_model,
            module_path=mapping.native_module_path,
            op_id=mapping.op_id,
            context=context,
        )

    missing = ref_by_authored_path.keys() - mapping_by_authored_path.keys()
    if missing:
        raise ContractError(f"missing module path mappings for StateRefs: {sorted(missing)!r}")


def bind_stateful_op(
    model: nn.Module,
    ref: StateRef,
    fn: Callable[..., object] | BoundOp,
    *,
    mapping: StatefulOpModulePathMapping | None = None,
) -> None:
    """Bind one callable or provenance-carrying BoundOp through its resolved module path."""
    if not isinstance(model, nn.Module):
        raise ContractError("bind_stateful_op requires an nn.Module")
    if not isinstance(ref, StateRef):
        raise ContractError("bind_stateful_op requires a StateRef")

    is_authored = mapping is None
    if is_authored:
        module_path = ref.module_path
        context = f"module_path {ref.module_path!r}, op_id {ref.op_id!r}"
    else:
        if not isinstance(mapping, StatefulOpModulePathMapping):
            raise ContractError("bind_stateful_op mapping must be a StatefulOpModulePathMapping or None")
        if mapping.authored_module_path != ref.module_path:
            raise ContractError(
                f"mapping authored_module_path {mapping.authored_module_path!r} "
                f"does not match StateRef.module_path {ref.module_path!r}"
            )
        if mapping.op_id != ref.op_id:
            raise ContractError(f"mapping op_id {mapping.op_id!r} does not match StateRef.op_id {ref.op_id!r}")
        module_path = mapping.native_module_path
        context = (
            f"authored_module_path {mapping.authored_module_path!r}, "
            f"native_module_path {mapping.native_module_path!r}, op_id {mapping.op_id!r}"
        )
    _bind_stateful_op(model, module_path=module_path, op_id=ref.op_id, fn=fn, context=context)


def _bind_stateful_op(
    model: nn.Module,
    *,
    module_path: str,
    op_id: str,
    fn: Callable[..., object] | BoundOp,
    context: str,
) -> None:
    if isinstance(fn, BoundOp):
        if fn.op != op_id:
            raise ContractError(f"{context}: BoundOp declares op {fn.op!r}")
    elif not callable(fn):
        raise ContractError(f"{context}: stateful op binding must be callable or a BoundOp")

    _module, bind = _resolve_stateful_op(model, module_path=module_path, op_id=op_id, context=context)
    try:
        bind(op_id, fn)
    except ContractError as exc:
        raise ContractError(f"{context}: {exc}") from exc


def _resolve_stateful_op(
    model: nn.Module,
    *,
    module_path: str,
    op_id: str,
    context: str,
):
    try:
        module = model.get_submodule(module_path)
    except AttributeError as exc:
        raise ContractError(f"{context}: module_path no longer resolves") from exc

    declared_op_id = getattr(module, "stateful_op_id", None)
    if declared_op_id != op_id:
        raise ContractError(f"{context}: module declares stateful_op_id {declared_op_id!r}")

    bind = getattr(module, "bind_stateful_op", None)
    if not callable(bind):
        raise ContractError(f"{context}: module must provide callable bind_stateful_op(op_id, binding)")
    return module, bind
