"""Local capture records. FX executes the program; these records describe it.

This is not a wire format. Workers receive JSON source/configuration fields and
capture locally; modules, callables, and GraphModules are never sent here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from torch import fx, nn
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS

from isoexec.compiler.contract.contract import ModelContract
from isoexec.nn.inspect import Inventory


class PipelineError(ValueError):
    """An actionable capture/mapping refusal, with fields for worker error reports."""

    def __init__(
        self, code: str, *, phase: str, model: str, location: str, reason: str,
        action: str, required_information: str = "none", context: Mapping[str, object] | None = None,
    ) -> None:
        self.code, self.phase, self.model = code, phase, model
        self.location, self.reason, self.action = location, reason, action
        self.required_information = required_information
        self.context = dict(context or {})
        super().__init__(
            f"[{code}] {reason}\nphase: {phase}\nmodel: {model}\nlocation: {location}\n"
            f"required_information: {required_information}\naction: {action}\ncontext: {self.context}"
        )

    def to_dict(self) -> dict[str, object]:
        """Expose diagnostic fields to adapters without serializing an exception object."""
        return {name: getattr(self, name) for name in (
            "code", "phase", "model", "location", "reason", "required_information", "action", "context",
        )}


@dataclass(frozen=True)
class ModuleCall:
    index: int
    path: str
    module_type: type[nn.Module]
    parent: int | None
    nodes: tuple[str, ...]


@dataclass(frozen=True)
class LogicalLayer:
    index: int
    path: str
    call: int
    nodes: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionGraph:
    program: fx.GraphModule
    calls: tuple[ModuleCall, ...]
    layers: tuple[LogicalLayer, ...]
    layer_source: str
    head: str
    inventory: Inventory
    contract: ModelContract
    state_nodes: dict[str, str]
    training_modes: tuple[tuple[str, bool], ...]
    execution_modes: tuple[bool, bool, tuple[tuple[str, bool, str], ...]]


def get_callable_nodes(nodes: Iterable[fx.Node]) -> tuple[fx.Node, ...]:
    """Return callable nodes by their original order."""
    return tuple(node for node in nodes if node.op in CALLABLE_NODE_OPS)
