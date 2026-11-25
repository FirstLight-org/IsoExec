"""Placement, inventory and binding inspection for component trees. Tensor shape, dtype and ties
come from registered parameters; execution walks skip unexecuted subtrees while weight walks
retain them.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import torch
from torch import nn

from isoexec.core.bindings import BoundOp
from isoexec.nn.component import Component, PlacementError
from isoexec.nn.unexecuted import is_unexecuted_subtree
from isoexec.schema import TORCH_DTYPES, LogicalTensor, Placement, ShardAxis, TieDecl

#: ``torch.dtype -> held dtype name``, inverted from the vocabulary's one table.
HELD_DTYPES = {getattr(torch, name): held for held, name in TORCH_DTYPES.items()}

# Duck-typed hook for narrowing held tensors without resolving operations.
NARROW_HELD = "narrow_held"


class InventoryError(ValueError):
    """A tree the inspection walk refuses."""


@dataclass(frozen=True)
class Inventory(Mapping[str, LogicalTensor]):
    """Ordered ``{FQN: LogicalTensor}`` mapping with ties, aliases and synced buffers."""

    tensors: Mapping[str, LogicalTensor]
    buffers: Mapping[str, LogicalTensor] = field(default_factory=dict)
    ties: tuple[TieDecl, ...] = ()
    aliases: tuple[str, ...] = ()

    def __getitem__(self, name: str) -> LogicalTensor:
        return self.tensors[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.tensors)

    def __len__(self) -> int:
        return len(self.tensors)


def _split_walk(
    model: nn.Module,
    *,
    prefix: str = "",
    remove_duplicate: bool = True,
) -> Iterator[tuple[str, nn.Module, bool]]:
    """Yield the live module walk with execution status, omitting descendants of unexecuted roots.
    Keep it lazy so placement can prune a component's own subtree before descent.
    """
    skip: list[str] = []
    for path, module in model.named_modules(prefix=prefix, remove_duplicate=remove_duplicate):
        if any(path == root or path.startswith(f"{root}.") for root in skip):
            continue
        if is_unexecuted_subtree(module):
            skip.append(path)
            yield path, module, False
            continue
        yield path, module, True


def executed_modules(
    model: nn.Module,
    *,
    prefix: str = "",
    remove_duplicate: bool = True,
) -> Iterator[tuple[str, nn.Module]]:
    """Yield modules outside unexecuted subtrees, lazily. Weight inventory and held-tensor
    narrowing use the complete tree. Set ``remove_duplicate=False`` when every registered path
    must remain visible for alias validation.
    """
    for path, module, executed in _split_walk(model, prefix=prefix, remove_duplicate=remove_duplicate):
        if executed:
            yield path, module


def _narrow_held(root: nn.Module, placement: Placement, *, prefix: str) -> None:
    """Narrow held tensors under an unexecuted root without resolving operations or claiming sites.
    Snapshot modules before narrowing because it can delete children.
    """
    for path, module in list(root.named_modules(prefix=prefix)):
        narrow = getattr(module, NARROW_HELD, None)
        if callable(narrow):
            narrow(placement, site=path)


def _held_dtype(tensor: torch.Tensor, name: str) -> str:
    try:
        return HELD_DTYPES[tensor.dtype]
    except KeyError:
        raise InventoryError(f"{name}: held dtype {tensor.dtype} is not one of {sorted(TORCH_DTYPES)}") from None


def declared_axes(model: nn.Module, *, prefix: str = "") -> dict[str, ShardAxis]:
    """Return shard axes for the entire tree. Reject missing declarations, unnamed parameters and
    axes with no matching parameter.
    """
    axes: dict[str, ShardAxis] = {}
    for path, module in model.named_modules(prefix=prefix):
        direct = {name: p for name, p in module._parameters.items() if p is not None}
        # Duck-typed on the METHOD, not on the base class: the walk asks a module what its tensors
        # declare, and a model author's own module that answers is as good as a Component.
        declares = getattr(module, "tensor_axes", None)
        declared = declares() if callable(declares) else None
        if declared is None:
            if direct:
                raise InventoryError(
                    f"{path or '<root>'}: {type(module).__name__} holds parameters "
                    f"{sorted(direct)} and declares no tensor_axes(); every parameter of the tree needs "
                    f"a ShardAxis its owner states, because the sharder plans an undeclared tensor as "
                    f"whole-on-every-rank rather than refusing it"
                )
            continue
        missing = sorted(set(direct) - set(declared))
        extra = sorted(set(declared) - set(direct))
        if missing or extra:
            raise InventoryError(
                f"{path or '<root>'}: tensor_axes() and the registered parameters are not the same set "
                f"(registered with no axis: {missing}; declared with no parameter: {extra})"
            )
        for name, axis in declared.items():
            axes[f"{path}.{name}" if path else name] = axis
    return axes


def inventory(model: nn.Module, *, prefix: str = "") -> Inventory:
    """Return logical tensors in parameter-registration order. ``prefix`` names the subtree's
    position in the full model.
    """
    axes = declared_axes(model, prefix=prefix)
    tensors: dict[str, LogicalTensor] = {}
    owner_of: dict[int, str] = {}
    alias_of: dict[str, list[str]] = {}
    for name, param in model.named_parameters(prefix=prefix, remove_duplicate=False):
        if id(param) in owner_of:  # two names for one Parameter object: that is what a tie is
            alias_of.setdefault(owner_of[id(param)], []).append(name)
            continue
        owner_of[id(param)] = name
        tensors[name] = LogicalTensor(name, tuple(param.shape), _held_dtype(param, name), axes[name])

    buffers: dict[str, LogicalTensor] = {}
    for path, module in model.named_modules(prefix=prefix):
        declares = getattr(module, "synced_buffers", None)
        for role, axis in (declares() if callable(declares) else {}).items():
            buf = module._buffers.get(role)
            if buf is None:
                raise InventoryError(f"{path or '<root>'}: synced_buffers() names {role!r}, which is not registered")
            fqn = f"{path}.{role}" if path else role
            buffers[fqn] = LogicalTensor(fqn, tuple(buf.shape), _held_dtype(buf, fqn), axis)

    ties = tuple(TieDecl(logical=o, aliases=tuple(a), owner="first") for o, a in alias_of.items())
    aliases = tuple(name for names in alias_of.values() for name in names)
    return Inventory(tensors=tensors, buffers=buffers, ties=ties, aliases=aliases)


def resolved_ops(model: nn.Module) -> dict[str, dict[str, BoundOp]]:
    """Return ``{site: {slot: BoundOp}}`` for executed components. Reject unplaced components and
    skip explicitly unexecuted subtrees.
    """
    out: dict[str, dict[str, BoundOp]] = {}
    for path, module in executed_modules(model):
        if not isinstance(module, Component):
            continue
        if module.site is None:
            raise InventoryError(
                f"{path or '<root>'}: {type(module).__name__} was never placed, so it has no site and no "
                f"resolved ops; apply_placement must reach every component of the tree"
            )
        out[module.site] = module.slots.resolved()
    return out


def place(model: nn.Module, placement: Placement, *, prefix: str = "", op_table=None) -> tuple[str, ...]:
    """Resolve each executed component and return its site in traversal order. A component root
    requires a nonempty ``prefix``.

    Walk lazily through all containers, allowing components to prune their own descendants
    during placement. Unexecuted subtrees are narrowed for weight ownership without resolving
    slots. Engine attachment and readiness checks happen after this walk.
    """
    if isinstance(model, Component) and not prefix:
        raise PlacementError(
            f"place(): {type(model).__name__} was handed in as the root with no prefix, so its site would "
            f"be empty. Pass prefix= the FQN this subtree occupies in the model (the path "
            f"named_modules() would give it), so that every refusal it raises names where it is."
        )
    if op_table is None:
        from isoexec.ops.table import OpTable

        op_table = OpTable()
    placed: list[str] = []
    for path, module, executed in _split_walk(model, prefix=prefix):
        if not executed:
            _narrow_held(module, placement, prefix=path)
            continue
        if isinstance(module, Component):
            module.apply_placement(placement, site=path, op_table=op_table)
            placed.append(path)
    return tuple(placed)


def verify_placed(model: nn.Module, *, required_only: bool = False) -> dict[str, dict[str, str]]:
    """Return unfilled slots and their setup verbs by site. ``required_only`` excludes
    configuration-absent or fused-away slots using explicit flags. Reject unplaced executed
    components; skip unexecuted subtrees.
    """
    out: dict[str, dict[str, str]] = {}
    for path, module in executed_modules(model):
        if not isinstance(module, Component):
            continue
        if module.site is None:
            raise InventoryError(f"{path or '<root>'}: {type(module).__name__} was never placed")
        out[module.site] = {
            slot: u.fill_verb for slot, u in module.slots.unfilled().items() if u.required or not required_only
        }
    return out


__all__ = [
    "HELD_DTYPES",
    "NARROW_HELD",
    "Inventory",
    "InventoryError",
    "declared_axes",
    "executed_modules",
    "inventory",
    "place",
    "resolved_ops",
    "verify_placed",
]
