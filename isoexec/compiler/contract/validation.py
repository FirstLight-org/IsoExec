"""Validate materialized model declarations without importing a runtime."""

from collections.abc import Container, Mapping
from types import MappingProxyType

from isoexec.state.validation import ContractError, validate_state_declarations

from .contract import ModelContract
from .inference import OpFactsRule


def validate_model_contract(
    contract: ModelContract,
    *,
    op_ids: Container[str],
    op_facts_registry: Mapping[str, OpFactsRule] | None = None,
) -> None:
    """Validate the portable fields already materialized in ``contract``."""
    if not isinstance(contract, ModelContract):
        raise ContractError("expected a ModelContract")
    validate_state_declarations(contract.state_decls, contract.state_refs, op_ids=op_ids)
    if type(contract.op_facts) is not MappingProxyType:
        raise ContractError("ModelContract.op_facts must be a read-only mapping")

    refs_by_path = {ref.module_path: ref for ref in contract.state_refs}
    if op_facts_registry is None:
        if contract.op_facts:
            raise ContractError("ModelContract.op_facts requires an op facts registry")
        return
    if not isinstance(op_facts_registry, Mapping):
        raise ContractError("op_facts_registry must be a Mapping or None")

    expected_paths = {ref.module_path for ref in contract.state_refs if ref.op_id in op_facts_registry}
    if set(contract.op_facts) != expected_paths:
        raise ContractError(
            "ModelContract.op_facts must contain exactly one value for every "
            "StateRef whose op_id is in op_facts_registry"
        )

    for module_path, facts in contract.op_facts.items():
        ref = refs_by_path[module_path]
        context = f"module_path {module_path!r}, op_id {ref.op_id!r}"
        rule = op_facts_registry[ref.op_id]
        if not isinstance(rule, OpFactsRule):
            raise ContractError(f"{context}: op facts rule must be an OpFactsRule")
        if not callable(rule.validate):
            raise ContractError(f"{context}: op facts validation must be callable")
        try:
            rule.validate(facts)
        except ContractError as exc:
            raise ContractError(f"{context}: {exc}") from exc


__all__ = ["validate_model_contract"]
