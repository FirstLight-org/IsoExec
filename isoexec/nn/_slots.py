"""Closed operation slots retaining immutable ``BoundOp`` records. Forward reads the same bindings
that inspection reports; unfilled slots name their required setup step.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from isoexec.core.bindings import BoundOp

#: The provider recorded for a slot filled by assigning a bare callable.
TEST_PROVIDER = "test"


class SlotError(AttributeError):
    """A write this slot set refuses: a name outside the declared set, or a value for another op."""


class UnfilledSlotError(RuntimeError):
    """A slot was called before anything filled it."""


@dataclass(frozen=True)
class Unbound:
    """Callable sentinel for an unfilled slot. ``fill_verb`` names its setup step or explains why
    this configuration requires no binding.
    """

    site: str
    slot: str
    fill_verb: str
    required: bool = True

    def __call__(self, *args: Any, **kwargs: Any):
        raise UnfilledSlotError(f"{self.site}.{self.slot}: nothing has filled this slot; {self.fill_verb}")


#: Names ``OpSlots`` keeps for itself; everything else is a slot.
_RESERVED = frozenset({"site", "_ops", "_values", "_verbs", "_frozen"})


class OpSlots:
    """Attribute-addressed slots for one component. Reading returns a callable; assigning a
    ``BoundOp`` records provenance, while assigning a bare callable records a test substitution.
    Updating ``site`` renames unfilled sentinels.
    """

    def __init__(self, slot_ops: Mapping[str, str], *, site: str, fill_verb: str = "apply_placement fills it") -> None:
        object.__setattr__(self, "_frozen", False)
        object.__setattr__(self, "_ops", dict(slot_ops))
        object.__setattr__(self, "_verbs", {slot: fill_verb for slot in slot_ops})
        object.__setattr__(self, "site", site)
        object.__setattr__(self, "_values", {s: Unbound(site, s, fill_verb) for s in slot_ops})

    @property
    def frozen(self) -> bool:
        """Whether execution validation has finalized this component's bindings."""
        return self._frozen

    def freeze(self) -> None:
        """Prevent a validated binding from changing without rebuilding the model."""
        object.__setattr__(self, "_frozen", True)


    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name.startswith("_"):  # reached only before __init__ ran; never recurse into _values
            raise AttributeError(name)
        try:
            value = self._values[name]
        except KeyError:
            raise SlotError(f"{self.site}: {name!r} is not one of the declared slots {tuple(self._ops)}") from None
        # An unfilled slot yields the refusing sentinel itself, so the caller always has something
        # callable in hand and the refusal reads as an installation mistake rather than a TypeError.
        return value.fn if isinstance(value, BoundOp) else value

    def __contains__(self, name: str) -> bool:
        return name in self._ops

    def __iter__(self) -> Iterator[str]:
        return iter(self._ops)

    def op_of(self, slot: str) -> str:
        """The op id the component declares for ``slot``."""
        try:
            return self._ops[slot]
        except KeyError:
            raise SlotError(f"{self.site}: {slot!r} is not one of the declared slots {tuple(self._ops)}") from None

    def resolved(self) -> dict[str, BoundOp]:
        """``{slot: BoundOp}`` for the filled slots, in declaration order."""
        return {s: v for s, v in self._values.items() if isinstance(v, BoundOp)}

    def unfilled(self) -> dict[str, Unbound]:
        """``{slot: Unbound}`` for the slots nothing has filled, in declaration order."""
        return {s: v for s, v in self._values.items() if isinstance(v, Unbound)}


    def __setattr__(self, name: str, value: Any) -> None:
        if self._frozen:
            raise SlotError(f"{self.site}: operation bindings are finalized; rebuild before changing {name!r}")
        if name == "site":
            self._retarget(value)
            return
        if name in _RESERVED:
            object.__setattr__(self, name, value)
            return
        if name not in self._ops:
            raise SlotError(f"{self.site}: {name!r} is not one of the declared slots {tuple(self._ops)}")
        self._values[name] = self._record(name, value)

    def declare_unbound(self, slot: str, fill_verb: str, *, required: bool = True) -> None:
        """Leave ``slot`` unfilled and say what fills it -- an engine's ``attach_core``, or nothing."""
        if self._frozen:
            raise SlotError(f"{self.site}: operation bindings are finalized; rebuild before changing {slot!r}")
        self.op_of(slot)  # refuses a slot outside the declared set
        self._verbs[slot] = fill_verb
        self._values[slot] = Unbound(self.site, slot, fill_verb, required)

    def _record(self, slot: str, value: Any) -> BoundOp | Unbound:
        if isinstance(value, Unbound):
            return value
        if isinstance(value, BoundOp):
            self._refuse_other_op(slot, value)
            return value
        if not callable(value):
            raise TypeError(f"{self.site}.{slot}: a slot holds a callable or a BoundOp, got {type(value).__name__}")
        # A bare callable is a substitution. It is recorded as one so that it shows up in the
        # identity report as `test`, where a peer comparison sees it, rather than passing for
        # whatever the resolver would have put here.
        name = getattr(value, "__qualname__", None) or getattr(value, "__name__", type(value).__name__)
        module = getattr(value, "__module__", "?")
        return BoundOp(self._ops[slot], f"{module}.{name}", None, TEST_PROVIDER, {}, value)

    def _refuse_other_op(self, slot: str, value: BoundOp) -> None:
        """Require the binding to compute the slot's declared op and to be recorded under an op
        this component declares. This allows local fusion while rejecting a core from another
        component kind.
        """
        declared = self._ops[slot]
        if declared not in value.ops:
            raise SlotError(
                f"{self.site}.{slot} takes {declared!r}, and {value.op}:{value.impl} computes "
                f"{list(value.ops)}: a resolution that does not compute this slot's op leaves it with "
                f"no owner in the composition and nothing to compare against the peer"
            )
        if value.op not in self._ops.values():
            raise SlotError(
                f"{self.site}.{slot} takes {declared!r}, and {value.op}:{value.impl} answers under "
                f"{value.op!r}, which is no slot of {tuple(self._ops)}: the owner record this site states "
                f"would name an op this kind does not run, and be compared against the wrong peer entry"
            )

    def _retarget(self, site: str) -> None:
        object.__setattr__(self, "site", site)
        for slot, value in self._values.items():
            if isinstance(value, Unbound):
                self._values[slot] = Unbound(site, slot, self._verbs[slot], value.required)

    def __repr__(self) -> str:
        parts = [f"{s}={v.impl if isinstance(v, BoundOp) else 'UNBOUND'}" for s, v in self._values.items()]
        return f"OpSlots({self.site}: {', '.join(parts)})"


__all__ = [
    "TEST_PROVIDER",
    "BoundOp",
    "OpSlots",
    "SlotError",
    "Unbound",
    "UnfilledSlotError",
]
