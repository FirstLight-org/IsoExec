"""Component declarations and one-time placement. Constructors register global meta parameters;
placement resolves operation slots and activation geometry, while the sharder owns parameter
storage.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isoexec.ops.table import OpTable

from torch import nn

from isoexec.nn._seams import Seams
from isoexec.nn._slots import OpSlots
from isoexec.schema import Placement, ShardAxis, StateSpec, TensorPlanMetadata
from isoexec.schema.token_layout import TokenLayoutContract


class PlacementError(ValueError):
    """A placement a component refuses, or a component asked to run before it was placed."""


#: Placements applied in this process (``apply_placement``): a host that resolved a tree's components once
#: compares it to see that no component was placed since.
_PLACEMENTS = [0]


def placement_generation() -> int:
    """The number of component placements applied in this process so far."""
    return _PLACEMENTS[0]


class Component(nn.Module):
    """Base component: operation slots, site identity and placement protocol.

    Numerical configuration is fixed at placement. Changing constructor settings
    such as epsilon or gamma convention requires rebuilding and revalidating the
    model. Slot finalization prevents binding replacement; it does not intercept
    arbitrary Python attribute assignment or mutate PyTorch's parameter lifecycle.
    """

    #: ``{slot: op id}``. ``SLOTS`` -- the closed set a reader addresses -- is its keys.
    SLOT_OPS: Mapping[str, str] = MappingProxyType({})
    SLOTS: tuple[str, ...] = ()

    # CP requires a declared token-wise form or a communication seam.
    # The default refuses CP > 1; only degree 2 is qualified.
    CONTEXT_PARALLEL: str | None = None

    # The planner uses this only for tensors this component directly owns. Subclasses
    # with a different below-units rule state their kind here.
    TENSOR_PLAN_KIND = "other"
    # Native adapters require an explicit token-layout rule under SP.
    NATIVE_LAYOUT: TokenLayoutContract | None = None
    SEQUENCE_GRADIENT_REDUCTION = False

    # Optional arithmetic-specific reason appended to the CP refusal.
    NO_CONTEXT_PARALLEL: str = ""

    # Runtimes bind a placement-bound kind by ``apply_placement`` with the side's full Placement,
    # on the engine and on the trainer alike: the component resolves its own operations and
    # communication, including those of its child components. Built-in kinds keep their
    # runtime-specific binding.
    PLACEMENT_BOUND: bool = False

    # A kind whose trainer forward reads the engine requests behind each packed sequence
    # (``isoexec.schema.rollout``); trainer integrations carry them only into stages that hold one.
    READS_ROLLOUT_REQUESTS: bool = False

    def __init__(self, *, slot_ops: Mapping[str, str] | None = None) -> None:
        super().__init__()
        ops = dict(self.SLOT_OPS if slot_ops is None else slot_ops)
        self.slots = OpSlots(ops, site=f"<unplaced {type(self).__name__}>")
        self.site: str | None = None
        self.placement: Placement | None = None


    def tensor_axes(self) -> Mapping[str, ShardAxis]:
        """Return shard axes for directly registered parameters. Composites default to an empty
        mapping; inspection rejects missing or extra declarations.
        """
        return {}

    def tensor_plan_metadata(self) -> Mapping[str, TensorPlanMetadata]:
        """State planning facts for direct parameters and synchronized buffers.

        A composed model needs no declaration: its parameter-owning components
        provide these facts. Specialized holders may override per-tensor facts.
        """
        roles = (*self.tensor_axes(), *self.synced_buffers())
        return {role: TensorPlanMetadata(kind=self.TENSOR_PLAN_KIND) for role in roles}

    def validate_tensor_plan(self) -> None:
        """Validate owner-specific global structure before PP cutting or EP pruning."""

    def prepare_output_contract(self) -> None:
        """Resolve optional output policy before capture; the default leaves configuration intact."""

    def abstract_output(self, *args: Any, **kwargs: Any) -> Any:
        """Infer output metadata on an isolated FakeTensor copy, leaving this component unplaced."""
        from isoexec.nn.abstract import abstract_output

        return abstract_output(self, args, kwargs)

    def state_specs(self) -> tuple[StateSpec, ...]:
        """The engine-side caches this kind declares; ``()`` for a stateless kind."""
        return ()

    def synced_buffers(self) -> Mapping[str, ShardAxis]:
        """Declare buffers that travel with weights. Undeclared buffers are local scratch and are
        omitted from inventory.
        """
        return {}

    def facts(self) -> Mapping[str, Any]:
        """Return every effective configuration value that affects numerical behavior. Subclasses
        must implement this explicitly because no independent tensor declaration can validate
        its completeness.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.facts(): state the effective configuration this component "
            f"contributes to model identity -- its kind, and every constructor value that changes "
            f"the numbers it computes. Nothing cross-checks this record, so an omission here is a "
            f"config difference two peers cannot see."
        )


    def apply_placement(self, placement: Placement, *, site: str, op_table: OpTable | None = None) -> None:
        """Resolve this component once for ``placement`` at ``site``. Record placement, derive
        activation geometry and fill stateless operations and communication seams; engine state
        slots wait for adapter attachment.
        """
        if self.placement is not None:
            raise PlacementError(
                f"{self.site}: apply_placement already ran for this component. The transform is applied "
                f"ONCE, to the meta model; a second placement would resolve over already-derived "
                f"activation geometry and silently halve it again."
            )
        if not site:
            raise PlacementError(f"{type(self).__name__}.apply_placement needs a non-empty site (a named_modules path)")
        if op_table is None:
            from isoexec.ops.table import OpTable

            op_table = OpTable()
        self._op_table = op_table
        self.site = site
        self.placement = placement
        _PLACEMENTS[0] += 1
        self.slots.site = site
        self._refuse_unstated_context_parallel(placement)
        self._place(placement)

    def bind_op(self, op: str, *, site: str | None = None, inputs=None, convention=None, facts=None):
        """Resolve a component operation using this build's implementation table.

        Component subclasses call this during placement. Runtime adapters may use
        it to finish resource-dependent bindings before execution validation.
        """
        own_site = self.require_placed("bind_op")
        return self._op_table.bind(
            op, site=own_site if site is None else site, inputs=inputs, convention=convention, facts=facts
        )

    def _refuse_unstated_context_parallel(self, placement: Placement) -> None:
        """Reject CP greater than one unless the component declares a supported form."""
        cp = placement.cp.size
        if cp > 1 and self.CONTEXT_PARALLEL is None:
            note = f" ({self.NO_CONTEXT_PARALLEL})" if self.NO_CONTEXT_PARALLEL else ""
            raise PlacementError(
                f"{self.site}: context parallelism (cp={cp}) is implemented for the component kinds that "
                f"state a form, and {type(self).__name__} states no context-parallel "
                f"form{note}. A CP cut gives this rank "
                f"a SUBSET of the tokens; nothing here gathers them back, and the composition a rank would "
                f"record is byte-identical to a cp=1 peer's -- so the handshake would pass while the bits "
                f"differed. Place this model with cp=1, or state the form on the kind (Component.CONTEXT_PARALLEL)."
            )

    def bind_engine_state(self, *, vllm_config: Any, layer_prefix: str) -> None:
        """Attach engine-owned state after allocation; a placement-bound kind with engine caches
        registers its native cache layers here under ``layer_prefix``. The default holds none."""
        del vllm_config, layer_prefix

    def _place(self, placement: Placement) -> None:
        """Derive geometry and fill the slots. Subclasses implement; ``apply_placement`` calls."""
        raise NotImplementedError(f"{type(self).__name__}._place()")

    @property
    def placed(self) -> bool:
        return self.placement is not None

    @property
    def seams(self) -> Seams:
        """Return communication builders for the recorded placement and site. Use
        ``self.seams.at(part)`` for a child operation.
        """
        if self.placement is None:
            raise PlacementError(
                f"{type(self).__name__}.seams: apply_placement has not run, so there is no Placement to "
                f"build placement movement from"
            )
        return Seams(self.placement, self.require_placed("seams"), self._op_table)

    def require_placed(self, what: str) -> str:
        """The site, or a refusal naming what was attempted before placement."""
        if self.site is None:
            raise PlacementError(f"{type(self).__name__}.{what}: apply_placement has not run on this component")
        return self.site


__all__ = ["Component", "PlacementError", "placement_generation"]
