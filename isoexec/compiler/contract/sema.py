"""Assemble a model contract without executing or tracing its forward."""

from collections.abc import Callable, Container, Mapping
from types import MappingProxyType

from torch import nn

from isoexec.nn.inspect import executed_modules
from isoexec.state.contract import StateRef, StateSpec
from isoexec.state.sema import _infer_state_decls_and_refs
from isoexec.state.validation import ContractError

from .contract import ModelContract
from .inference import OpFactsRule
from .validation import validate_model_contract


def build_model_contract(
    model: nn.Module,
    *,
    op_ids: Container[str],
    inference_registry: Mapping[str, Callable[..., tuple[StateSpec, ...]]],
    op_facts_registry: Mapping[str, OpFactsRule] | None = None,
) -> ModelContract:
    """Infer, validate, and materialize fields from the model's executed modules.

    ``op_ids`` comes from op registration; ``inference_registry`` maps stateful
    op IDs to state inference functions. ``op_facts_registry=None`` preserves
    state-only collection; otherwise, its keys select the stateful op IDs whose
    authoring metadata also produces facts.
    """
    if not isinstance(model, nn.Module):
        raise ContractError("build_model_contract requires an nn.Module")

    state_decls, state_refs = _infer_state_decls_and_refs(
        model,
        modules=executed_modules(model, remove_duplicate=False),
        op_ids=op_ids,
        inference_registry=inference_registry,
    )
    op_facts = _infer_stateful_op_facts(model, state_refs, op_facts_registry)
    contract = ModelContract(state_decls=state_decls, state_refs=state_refs, op_facts=op_facts)
    validate_model_contract(contract, op_ids=op_ids, op_facts_registry=op_facts_registry)
    return contract


def _infer_stateful_op_facts(
    model: nn.Module,
    state_refs: tuple[StateRef, ...],
    registry: Mapping[str, OpFactsRule] | None,
) -> Mapping[str, object]:
    if registry is None:
        return MappingProxyType({})
    if not isinstance(registry, Mapping):
        raise ContractError("op_facts_registry must be a Mapping or None")

    inferred = {}
    for ref in state_refs:
        if ref.op_id not in registry:
            continue
        rule = registry[ref.op_id]
        context = f"module_path {ref.module_path!r}, op_id {ref.op_id!r}"
        if not isinstance(rule, OpFactsRule):
            raise ContractError(f"{context}: op facts rule must be an OpFactsRule")
        if not callable(rule.infer):
            raise ContractError(f"{context}: op facts inference must be callable")
        if not callable(rule.validate):
            raise ContractError(f"{context}: op facts validation must be callable")
        module = model.get_submodule(ref.module_path)
        try:
            inferred[ref.module_path] = rule.infer(module.spec)
        except ContractError as exc:
            raise ContractError(f"{context}: {exc}") from exc
    return MappingProxyType(dict(sorted(inferred.items())))


__all__ = ["build_model_contract"]
