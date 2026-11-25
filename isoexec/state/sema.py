"""Infer state declarations and references for model contract assembly.

Stateful modules expose ``stateful_op_id`` and ``spec``. The module path is the
state identity. A parent may contain multiple stateful children alongside
stateless work; each child has its own declaration and reference. An ordinary
``op_id`` attribute does not declare state.

Each stateful module must execute exactly once per root forward. Static analysis
rejects multiple registered paths for one instance, but cannot detect repeated
or skipped calls inside ``forward()``; those require runtime enforcement.
"""

from collections.abc import Callable, Container, Iterable, Mapping
from typing import TYPE_CHECKING

from torch import nn

from isoexec.state.contract import StateDecl, StateRef, StateSpec
from isoexec.state.validation import ContractError, validate_id, validate_state_spec

if TYPE_CHECKING:
    from isoexec.compiler.contract.sema import build_model_contract

_ROOT_STATE_ID = "<root>"


def _infer_state_decls_and_refs(
    model: nn.Module,
    *,
    modules: Iterable[tuple[str, nn.Module]],
    op_ids: Container[str],
    inference_registry: Mapping[str, Callable[..., tuple[StateSpec, ...]]],
) -> tuple[tuple[StateDecl, ...], tuple[StateRef, ...]]:
    """Check live module metadata and infer the model's state requirements.

    ``op_ids`` comes from op registration; ``inference_registry`` maps stateful
    op IDs to inference functions. Each stateful module's ``spec`` is passed
    unchanged to its op's function once per build. ``modules`` is the model
    layer's execution-aware walk. Model contract assembly validates the returned
    declarations and references.
    """
    decls: dict[str, StateDecl] = {}
    refs: dict[str, StateRef] = {}
    module_path_by_object: dict[int, str] = {}

    for module_path, module in modules:
        if hasattr(module, "state_id"):
            raise ContractError(
                f"module {module_path!r}: explicit state_id is unsupported; state_id is derived from module_path"
            )
        if not hasattr(module, "stateful_op_id"):
            continue

        op_id = module.stateful_op_id
        validate_id(op_id, f"module {module_path!r} stateful_op_id")
        if op_id not in op_ids:
            raise ContractError(f"module {module_path!r}: unknown op_id {op_id!r}")
        if op_id not in inference_registry:
            raise ContractError(f"module {module_path!r}: missing state inference for op_id {op_id!r}")
        infer_state_specs = inference_registry[op_id]
        if not callable(infer_state_specs):
            raise ContractError(f"module {module_path!r}: inference for op_id {op_id!r} must be callable")

        previous_path = module_path_by_object.get(id(module))
        if previous_path is not None:
            raise ContractError(
                f"same stateful module appears at {previous_path!r} and {module_path!r}; "
                "multiple module paths are unsupported"
            )
        module_path_by_object[id(module)] = module_path

        if module_path in refs:
            raise ContractError(f"duplicate stateful module_path: {module_path!r}")
        try:
            resolved = model.get_submodule(module_path)
        except AttributeError as exc:
            raise ContractError(f"module_path {module_path!r} cannot be resolved from the model") from exc
        if resolved is not module:
            raise ContractError(f"module_path {module_path!r} resolves to a different module")
        if not hasattr(module, "spec"):
            raise ContractError(f"module_path {module_path!r}: missing module.spec")

        state_id = module_path or _ROOT_STATE_ID

        specs = infer_state_specs(module.spec)
        if type(specs) is not tuple or not specs:
            raise ContractError(f"module_path {module_path!r}: state derivation must return a non-empty tuple")
        for spec in specs:
            validate_state_spec(spec)

        if state_id in decls:
            raise ContractError(f"derived state_id {state_id!r} is not unique; state sharing is unsupported")
        decls[state_id] = StateDecl(state_id, specs)

        refs[module_path] = StateRef(module_path, op_id, state_id)

    return (
        tuple(decls[key] for key in sorted(decls)),
        tuple(refs[key] for key in sorted(refs)),
    )


def __getattr__(name: str):
    if name == "build_model_contract":
        from isoexec.compiler.contract.sema import build_model_contract

        return build_model_contract
    raise AttributeError(name)


__all__ = ["build_model_contract"]
