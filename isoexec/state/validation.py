"""Validate portable state fields and model-wide identity and reference invariants."""

from collections.abc import Container
from typing import TYPE_CHECKING

from isoexec.schema import ShardAxis
from isoexec.state.contract import StateDecl, StateRef, StateSpec

if TYPE_CHECKING:
    from isoexec.compiler.contract.validation import validate_model_contract


class ContractError(ValueError):
    """Portable model declarations cannot represent the requested semantics."""


def validate_id(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")


def validate_state_shard(shard: ShardAxis, *, ndim: int) -> None:
    if not isinstance(shard, ShardAxis):
        raise ContractError("StateSpec.shard must be a ShardAxis")
    if shard.reduce != "none":
        raise ContractError("state sharding requires reduce='none'")
    if type(shard.stride) is not int or shard.stride != 1:
        raise ContractError("state sharding requires stride=1")
    if shard.units is not None or shard.segments is not None:
        raise ContractError("state sharding forbids units and segments")
    if shard.kind == "replicated":
        if shard.dim is not None:
            raise ContractError("replicated state forbids dim")
        return
    if shard.kind != "col":
        raise ContractError(f"unsupported state shard kind: {shard.kind!r}")
    if type(shard.dim) is not int or not 0 <= shard.dim < ndim:
        raise ContractError(f"col state requires an explicit dim in [0, {ndim})")


def validate_state_spec(spec: StateSpec) -> None:
    if not isinstance(spec, StateSpec):
        raise ContractError("expected a StateSpec")
    validate_id(spec.name, "StateSpec.name")
    validate_id(spec.dtype, "StateSpec.dtype")
    if type(spec.shape) is not tuple or not spec.shape or any(type(d) is not int or d <= 0 for d in spec.shape):
        raise ContractError("StateSpec.shape must be a tuple of positive concrete dimensions")
    validate_state_shard(spec.shard, ndim=len(spec.shape))

    if spec.scope not in ("request", "segment"):
        raise ContractError("unknown StateSpec.scope")
    if spec.scope == "segment" and (spec.granularity != "per_token" or spec.retention != "sliding_window"):
        raise ContractError("segment scope is a per_token sliding window")
    if spec.granularity == "per_request":
        for field in ("tokens_per_entry", "retention", "window_tokens"):
            if getattr(spec, field) is not None:
                raise ContractError(f"per_request forbids {field}")
        return
    if spec.granularity != "per_token":
        raise ContractError("unknown StateSpec.granularity")
    if type(spec.tokens_per_entry) is not int or spec.tokens_per_entry <= 0:
        raise ContractError("per_token requires tokens_per_entry >= 1")
    if spec.retention == "full_history":
        if spec.window_tokens is not None:
            raise ContractError("full_history forbids window_tokens")
        return
    if spec.retention != "sliding_window":
        raise ContractError("per_token requires retention='full_history' or 'sliding_window'")
    if type(spec.window_tokens) is not int or spec.window_tokens <= 0:
        raise ContractError("sliding_window requires window_tokens > 0")
    if spec.window_tokens % spec.tokens_per_entry:
        raise ContractError("window_tokens must be a multiple of tokens_per_entry: partial retention of one logical entry is not representable")


def validate_state_declarations(
    state_decls: tuple[StateDecl, ...],
    state_refs: tuple[StateRef, ...],
    *,
    op_ids: Container[str],
) -> None:
    """Validate state fields and references against IDs declared by the op registry.

    Sema checks live module paths and compares each module's inferred specs;
    this function validates the materialized declarations and references.
    """
    if type(state_decls) is not tuple or type(state_refs) is not tuple:
        raise ContractError("state_decls and state_refs must be tuples")

    # Validate state declarations.
    decls = {}
    for decl in state_decls:
        if not isinstance(decl, StateDecl):
            raise ContractError("expected a StateDecl")
        validate_id(decl.state_id, "StateDecl.state_id")
        if decl.state_id in decls:
            raise ContractError(f"duplicate StateDecl.state_id: {decl.state_id!r}")
        if type(decl.specs) is not tuple or not decl.specs:
            raise ContractError(f"state_id {decl.state_id!r} requires a non-empty specs tuple")
        names = set()
        for spec in decl.specs:
            validate_state_spec(spec)
            if spec.name in names:
                raise ContractError(f"state_id {decl.state_id!r}: duplicate StateSpec.name {spec.name!r}")
            names.add(spec.name)
        decls[decl.state_id] = decl

    # Validate state references.
    module_paths = set()
    referenced_state_ids = set()
    for ref in state_refs:
        if not isinstance(ref, StateRef):
            raise ContractError("expected a StateRef")
        for field in ("op_id", "state_id"):
            validate_id(getattr(ref, field), f"StateRef.{field}")
        if not isinstance(ref.module_path, str):
            raise ContractError("StateRef.module_path must be a string (empty for the root)")
        if ref.module_path in module_paths:
            raise ContractError(f"duplicate StateRef.module_path: {ref.module_path!r}")
        module_paths.add(ref.module_path)
        if ref.op_id not in op_ids:
            raise ContractError(f"unknown StateRef.op_id: {ref.op_id!r}")
        if ref.state_id not in decls:
            raise ContractError(f"unresolved StateRef.state_id: {ref.state_id!r}")
        if ref.state_id in referenced_state_ids:
            raise ContractError(f"state_id {ref.state_id!r} has multiple StateRefs; state sharing is unsupported")
        referenced_state_ids.add(ref.state_id)

    unused = decls.keys() - referenced_state_ids
    if unused:
        raise ContractError(f"StateDecl has no StateRef: {sorted(unused)}")


def __getattr__(name: str):
    if name == "validate_model_contract":
        from isoexec.compiler.contract.validation import validate_model_contract

        return validate_model_contract
    raise AttributeError(name)


__all__ = [
    "ContractError",
    "validate_id",
    "validate_model_contract",
    "validate_state_declarations",
    "validate_state_shard",
    "validate_state_spec",
]
