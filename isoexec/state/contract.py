"""Portable state declarations and module references."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from isoexec.schema import ShardAxis

if TYPE_CHECKING:
    from isoexec.compiler.contract.contract import ModelContract


@dataclass(frozen=True)
class StateSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    shard: ShardAxis
    granularity: Literal["per_token", "per_request"]
    tokens_per_entry: int | None = None
    retention: Literal["full_history", "sliding_window"] | None = None
    window_tokens: int | None = None


@dataclass(frozen=True)
class StateDecl:
    state_id: str
    specs: tuple[StateSpec, ...]


@dataclass(frozen=True)
class StateRef:
    module_path: str
    op_id: str
    state_id: str


def __getattr__(name: str):
    """Retain the former state-contract import without owning the model type."""
    if name == "ModelContract":
        from isoexec.compiler.contract.contract import ModelContract

        return ModelContract
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ModelContract", "StateDecl", "StateRef", "StateSpec"]
