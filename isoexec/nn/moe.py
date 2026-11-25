"""Top-k mixture with routed experts and an optional shared expert. Combine folds terms in
ascending global expert order using fp32 and one final narrowing; EP pruning preserves those
global keys.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, Literal, cast

import torch
from torch import nn

from isoexec.core.bindings import BoundOp
from isoexec.schema import (
    Mlp, Placement, SequenceContext, ShardAxis, ShardError, TensorPlanMetadata,
    col, expert, replicated, row, torch_dtype,
)
from isoexec.nn import _seams
from isoexec.schema.token_layout import TokenLayoutContract, TokenLayoutEffect
from isoexec.nn.component import Component, PlacementError

#: Family policy, not an argument: every expert and shared-expert projection is held bf16.
HELD_DTYPE = "bf16"

# Hold the router and selection bias in fp32 so bf16 rounding cannot alter top-k.
# This mirrors ops/moe/spec.py without importing kernels at construction;
# test_moe_inventory_cpu.py checks the declarations agree.
ROUTER_DTYPE = "fp32"

#: The routings this kind admits. A second routing arrives with its own function and its own gate,
#: not with a field (``ops/moe/routing.py``).
ROUTINGS = ("softmax_renorm", "softmax", "sigmoid_noaux_bias_select")

#: The routing whose selection reads a per-expert correction bias the checkpoint ships.
BIAS_ROUTING = "sigmoid_noaux_bias_select"

# Keyword-only seams exposed by expert bindings, matching runtimes/parallel.py.
SEAM_ROW_PARALLEL = "row_parallel"  # moe.shared_expert: the fc2 K-reduction callable
SEAM_TP_COPY = "tp_copy"  # moe.shared_expert: the fc1 input wire (identity fwd, TP sum bwd)
SEAM_N_LEAVES = "n_leaves"  # moe.expert_mlp: the leaves this rank folds (G // ETP)
SEAM_OWNER_GROUP = "owner_group"  # moe.combine: the expert-tensor group whose two fc2 trees it owns

# Record EP transport while retaining each stage's numerical identity.
EP_DISPATCH = "owner_a2a"  # moe.expert_mlp: tokens to their expert's owner and the rows back
EP_FOLD = "owner_fold"  # moe.combine: the owner already folded the k terms; this adds the shared one


def expert_range(num_experts: int, ep: int, ep_rank: int, *, site: str) -> tuple[int, int]:
    """Return the equal-sized contiguous global expert range ``[lo, hi)`` owned by an EP rank."""
    if ep < 1 or not (0 <= ep_rank < ep):
        raise PlacementError(f"{site}: ep_rank {ep_rank} out of range for ep={ep}")
    if num_experts % ep != 0:
        raise PlacementError(f"{site}: {num_experts} experts do not divide over ep={ep}")
    per = num_experts // ep
    return ep_rank * per, (ep_rank + 1) * per


class _Router(nn.Module):
    """``router.weight`` fp32, and ``router.expert_bias`` as a synced buffer when the routing reads one."""

    def __init__(self, num_experts: int, hidden: int, *, bias: bool, device: torch.device | str) -> None:
        super().__init__()
        dtype = torch_dtype(ROUTER_DTYPE)
        self.weight = nn.Parameter(torch.empty(num_experts, hidden, dtype=dtype, device=device))
        if bias:
            # Selection-only bias receives no gradient but travels as a synced buffer.
            self.register_buffer("expert_bias", torch.zeros(num_experts, dtype=dtype, device=device))
        else:
            self.expert_bias = None

    def tensor_axes(self) -> Mapping[str, ShardAxis]:
        """Replicate the router so every rank scores all tokens against every expert."""
        return {"weight": replicated()}

    def synced_buffers(self) -> Mapping[str, ShardAxis]:
        return {} if self.expert_bias is None else {"expert_bias": replicated()}


class _Expert(nn.Module):
    """One routed expert's SwiGLU pair. Whole on its owner; split by role only at ETP > 1."""

    def __init__(self, inter: int, hidden: int, device: torch.device | str,
                 *, expert_id: int, num_experts: int) -> None:
        super().__init__()
        self.expert_id = expert_id
        self.num_experts = num_experts
        dtype = torch_dtype(HELD_DTYPE)
        self.gate = nn.Parameter(torch.empty(inter, hidden, dtype=dtype, device=device))
        self.up = nn.Parameter(torch.empty(inter, hidden, dtype=dtype, device=device))
        self.down = nn.Parameter(torch.empty(hidden, inter, dtype=dtype, device=device))

    def tensor_axes(self) -> Mapping[str, ShardAxis]:
        """Partition first by expert ID. At ETP greater than one, the sharder further splits
        gate/up output rows and the down contraction dimension according to parameter role.
        """
        return {"gate": expert(), "up": expert(), "down": expert()}

    def tensor_plan_metadata(self) -> Mapping[str, TensorPlanMetadata]:
        """The global owner and ETP split belong to this parameter holder."""
        return {
            "gate": TensorPlanMetadata(expert_id=self.expert_id, num_experts=self.num_experts, expert_axis=col()),
            "up": TensorPlanMetadata(expert_id=self.expert_id, num_experts=self.num_experts, expert_axis=col()),
            "down": TensorPlanMetadata(expert_id=self.expert_id, num_experts=self.num_experts, expert_axis=row()),
        }


class _Shared(nn.Module):
    """The always-on shared expert: an ordinary column/row-parallel SwiGLU, plus its own sigmoid gate."""

    def __init__(self, inter: int, hidden: int, *, gated: bool, device: torch.device | str) -> None:
        super().__init__()
        dtype = torch_dtype(HELD_DTYPE)
        self.gate = nn.Parameter(torch.empty(inter, hidden, dtype=dtype, device=device))
        self.up = nn.Parameter(torch.empty(inter, hidden, dtype=dtype, device=device))
        self.down = nn.Parameter(torch.empty(hidden, inter, dtype=dtype, device=device))
        # One row of weights over the residual: sigmoid(x @ expert_gate.T) scales the shared output.
        self.expert_gate = nn.Parameter(torch.empty(1, hidden, dtype=dtype, device=device)) if gated else None

    def tensor_axes(self) -> Mapping[str, ShardAxis]:
        axes: dict[str, ShardAxis] = {"gate": col(), "up": col(), "down": row()}
        if self.expert_gate is not None:
            # Replicated, and not sum_over_tp: it contracts the full hidden width on every rank, so
            # each rank's gradient for it is already complete rather than a partial over a slice.
            axes["expert_gate"] = replicated()
        return axes


class MoE(Component):
    """A top-k mixture of experts over a packed (thd) batch.

    Parameters are registered at global shape and global expert count on ``device``, in the
    checkpoint vocabulary's order: ``router.weight``, then ``experts.{e}.{gate,up,down}`` for every
    ``e`` in the model, then ``shared.{gate,up,down,expert_gate}`` when the row has a shared expert.
    ``apply_placement`` is what narrows the expert set to this rank's.

    Args:
        hidden: the residual width the router scores and every expert reads.
        num_experts: the global routed-expert count. ``apply_placement`` keeps this rank's slice.
        top_k: experts selected per token. The combine folds exactly this many terms.
        intermediate: the per-expert SwiGLU width.
        routing: ``"softmax_renorm"`` or ``"sigmoid_noaux_bias_select"``. The second reads a
            per-expert correction bias the checkpoint ships and applies ``routed_scale``.
        shared_intermediate: the always-on shared expert's width, or ``0`` for a row with none.
        shared_expert_gate: whether the shared expert carries its own sigmoid gate over the residual.
            Read only when ``shared_intermediate`` is non-zero.
        routed_scale: the checkpoint's scale on the renormalized routed probabilities.
        device: ``"meta"`` by default. Construction allocates no bytes and resolves nothing.
    """

    # These slots record the shared fc2 tree and fc1 gradient wire passed into expert bindings.
    SLOT_OPS = MappingProxyType(
        {
            "route_score": "moe.route_score",
            "route_select": "moe.route_mechanics",
            "expert_mlp": "moe.expert_mlp",
            "shared_expert": "moe.shared_expert",
            "combine": "moe.combine",
            "fc1_tp_copy": "collectives.tp_copy",
            "shared_row_parallel": "collectives.row_parallel",
            "sp_gather": "collectives.sp_gather",
            "sp_scatter": "collectives.sp_scatter",
        }
    )
    SLOTS = tuple(SLOT_OPS)
    NATIVE_LAYOUT = TokenLayoutContract(TokenLayoutEffect.DECODER)

    # Routing, every expert's rows and the fixed-order combine are functions of one token; expert
    # dispatch (also across CP ranks when the EP group spans them) moves whole rows as bytes.
    CONTEXT_PARALLEL = (
        "per token: routing, each expert's rows and the ascending-expert combine read one token; EP dispatch "
        "moves rows only, whichever CP ranks the EP group spans"
    )

    def __init__(
        self,
        *,
        hidden: int,
        num_experts: int,
        top_k: int,
        intermediate: int,
        routing: str = "softmax_renorm",
        weight_after_fc2: bool = False,
        shared_intermediate: int = 0,
        shared_expert_gate: bool = True,
        routed_scale: float = 1.0,
        device: torch.device | str = "meta",
    ) -> None:
        if min(hidden, num_experts, top_k, intermediate) < 1:
            raise ValueError(
                f"MoE: hidden/num_experts/top_k/intermediate must be >= 1, "
                f"got {(hidden, num_experts, top_k, intermediate)}"
            )
        if top_k > num_experts:
            raise ValueError(f"MoE: top_k={top_k} selects more experts than the {num_experts} the row has")
        if routing not in ROUTINGS:
            raise ValueError(f"MoE: routing must be one of {ROUTINGS}, got {routing!r}")
        if shared_intermediate < 0:
            raise ValueError(f"MoE: shared_intermediate must be >= 0, got {shared_intermediate}")
        super().__init__()
        self.hidden, self.num_experts, self.top_k = hidden, num_experts, top_k
        self.intermediate, self.routing = intermediate, routing
        self.shared_intermediate = shared_intermediate
        self.shared_expert_gate = bool(shared_expert_gate) and bool(shared_intermediate)
        self.weight_after_fc2 = weight_after_fc2
        self.routed_scale = float(routed_scale)
        # None preserves the legacy binding result until graph preparation selects a boundary.
        self.output_dtype_policy: Literal["input"] | None = None

        self.router = _Router(num_experts, hidden, bias=routing == BIAS_ROUTING, device=device)
        self.experts = nn.ModuleList(
            _Expert(intermediate, hidden, device, expert_id=index, num_experts=num_experts)
            for index in range(num_experts)
        )
        self.shared = (
            _Shared(shared_intermediate, hidden, gated=self.shared_expert_gate, device=device)
            if shared_intermediate
            else None
        )

        #: This rank's contiguous global expert range, or ``None`` when it holds every expert. The
        #: packed-expert hook reads this name, so it is part of the duck-typed surface.
        self.ep_range: tuple[int, int] | None = None
        #: The free-alias hook for the stacked expert operands. ``None`` until a build path calls
        #: ``ops/moe/packed_experts.attach``; ``None`` from the hook means "stack".
        self.expert_views: Callable[[], tuple[torch.Tensor, torch.Tensor] | None] | None = None

        # Build the routing schema argument from this component's own configuration.
        self._spec = Mlp(
            intermediate_size=intermediate,
            num_experts=num_experts,
            top_k=top_k,
            shared_intermediate=shared_intermediate,
            routing=routing,
            routed_scale=self.routed_scale,
            shared_expert_gate=self.shared_expert_gate,
        )


    # ``tensor_axes`` is the base class's ``{}``: the mixture holds no parameters directly, and the
    # router, the experts and the shared expert each declare their own.

    def validate_tensor_plan(self) -> None:
        """Require all global experts before the placement walk prunes remote owners."""
        if set(self.experts._modules) != {str(index) for index in range(self.num_experts)}:
            raise ShardError(f"MoE: collect global expert facts before pruning ({self.num_experts} expected)")
        seen: set[int] = set()
        for key, module in self.experts._modules.items():
            if not isinstance(module, _Expert):
                raise ShardError(f"MoE: expert {key} is absent or has an unknown parameter holder")
            if id(module) in seen or module.expert_id != int(key) or module.num_experts != self.num_experts:
                raise ShardError(f"MoE: expert {key} has conflicting global ownership")
            seen.add(id(module))

    def facts(self) -> Mapping[str, Any]:
        facts = {
            "kind": "moe",
            "hidden": self.hidden,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "intermediate": self.intermediate,
            "routing": self.routing,
            "routed_scale": self.routed_scale,
            "shared_intermediate": self.shared_intermediate,
            "shared_expert_gate": self.shared_expert_gate,
            "activation": self._spec.activation,
            "router_dtype": ROUTER_DTYPE,
            "held_dtype": HELD_DTYPE,
        }
        if self.output_dtype_policy is not None:
            facts["boundary_out_dtype"] = self.output_dtype_policy
        if self.weight_after_fc2:
            facts["weight_after_fc2"] = True
        return facts

    def prepare_output_contract(self) -> None:
        """Return graph-facing results in the input dtype; legacy builds skip this."""
        if self.placed:
            raise PlacementError("MoE output precision must be selected before placement")
        # TODO: Confirm the output boundary and native fusion rules with Alex.
        # Graph builds round the complete combine to the original input dtype. A kernel
        # that already rounded to BF16 cannot recover wider precision by upcasting.
        self.output_dtype_policy = "input"

    def _output(self, value: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        return value if self.output_dtype_policy is None else value.to(input_dtype)

    def _place(self, placement: Placement) -> None:
        site = self.require_placed("_place")
        ep, ep_rank = placement.ep.size, placement.ep.rank
        etp = placement.etp.size
        # Prune held experts before resolving bindings against the local set.
        self.narrow_held(placement, site=site)

        # Routing selects the score/mechanics functions and is recorded in both bindings.
        self.slots.route_score = self.bind_op(
            "moe.route_score",
            site=f"{site}.score",
            inputs={"routing": self.routing},
            facts={"routing": self.routing},
        )
        # Side selects forward-only engine recipes; numerical identity is compared by twin class.
        self.slots.route_select = self.bind_op(
            "moe.route_mechanics",
            site=f"{site}.mechanics",
            inputs={"routing": self.routing, "top_k": self.top_k, "routed_scale": self.routed_scale},
            facts={"routing": self.routing, "side": placement.side},
        )
        # Record the declared act, leaves and leaf_dtype inputs from the placement plan.
        # A fused expert kernel may replace act with its own epilogue identity.
        self.slots.expert_mlp = self.bind_op(
            "moe.expert_mlp",
            site=f"{site}.experts",
            inputs={
                "act": self._spec.activation,
                "leaves": int(placement.leaves),
                "leaf_dtype": placement.leaf_dtype,
            },
            facts={"weight_after_fc2": True} if self.weight_after_fc2 else {},
        )
        self.slots.combine = self.bind_op(
            "moe.combine",
            site=f"{site}.combine",
            inputs={"top_k": self.top_k},
            facts={"side": placement.side, **({"weight_after_fc2": True} if self.weight_after_fc2 else {})},
        )
        if self.shared is not None:
            self.slots.shared_expert = self.bind_op(
                "moe.shared_expert",
                site=f"{site}.shared",
                inputs={"gated": self.shared_expert_gate},
                facts={"side": placement.side, **({"weight_after_fc2": True} if self.weight_after_fc2 else {})},
            )
        else:
            self.slots.declare_unbound("shared_expert", "nothing does: this row has no shared expert", required=False)

        # Placement movement, from the Placement this component receives once.
        seams = self.seams
        self.slots.sp_gather = seams.sp_gather()
        self.slots.sp_scatter = seams.sp_scatter()
        # Shared and ETP-routed fc1 projections use one gradient-sum wire over the gathered stream.
        wire = seams.at("fc1").tp_grad_sum()
        self.slots.fc1_tp_copy = wire
        tp_copy = wire.fn if wire.impl != _seams.NO_MOVEMENT else None
        # Decide root ownership once for expert partials, shared partials and combine.
        owner = self._combine_owns_reductions(ep=ep, etp=etp)
        self._install_shared_seams(placement, tp_copy=tp_copy, owner=owner)
        self._install_expert_parallel(placement, ep=ep, ep_rank=ep_rank, etp=etp, tp_copy=tp_copy, owner=owner)

    def _combine_owns_reductions(self, *, ep: int, etp: int) -> bool:
        """Return whether the combine exposes ``owner_group`` at EP=1 and ETP greater than one.
        Such bindings execute the expert and shared fc2 trees at the token owner.
        """
        if ep != 1 or etp <= 1:
            return False
        combine = self.slots.resolved()["combine"]
        owner = _seams.accepts_kw(combine.fn, SEAM_OWNER_GROUP)
        placement = self.placement
        if placement is None:
            raise PlacementError("MoE combine ownership requires placement")
        if self._op_table.mode == "production" and placement.side == "engine" and not owner:
            raise PlacementError(
                f"{self.site}: production engine ETP requires moe.combine ({combine.impl!r}) "
                f"to expose the {SEAM_OWNER_GROUP!r} seam for its deferred fc2 reductions"
            )
        return owner

    def narrow_held(self, placement: Placement, *, site: str) -> int:
        """Keep this rank's contiguous expert range, preserving original global keys and ascending
        order. Return the number removed.

        This operation is idempotent and resolves no slots or sites, so it also applies to held
        but unexecuted MTP subtrees. ``site`` supplies diagnostics before placement.
        """
        ep, ep_rank = placement.ep.size, placement.ep.rank
        self.ep_range = None if ep == 1 else expert_range(self.num_experts, ep, ep_rank, site=site)
        if self.ep_range is None:
            return 0
        lo, hi = self.ep_range
        removed = 0
        for key in list(self.experts._modules):
            if not lo <= int(key) < hi:
                del self.experts._modules[key]
                removed += 1
        return removed

    def _install_shared_seams(self, placement: Placement, *, tp_copy, owner: bool) -> None:
        """Install the shared expert's fc1 gradient wire and fixed-tree fc2 reduction at TP greater
        than one. The replicated gate needs no gradient sum.

        When ``owner`` defers the tree root to the combine, require a shared binding that can
        return the partial; otherwise the tree would execute twice.
        """
        if self.shared is None:
            self.slots.declare_unbound(
                "shared_row_parallel", "nothing does: this row has no shared expert", required=False
            )
            return
        if placement.tp.size == 1:
            # Owner mode needs ETP > 1 with ETP equal to TP, so it cannot occur at TP=1.
            self.slots.declare_unbound(
                "shared_row_parallel",
                "nothing does: at tp=1 the shared expert's own kernel owns its whole fc2 contraction",
                required=False,
            )
            return
        resolved = self.slots.resolved()["shared_expert"]
        if owner and not getattr(resolved.fn, "accepts_partial_row_parallel", False):
            raise PlacementError(
                f"{self.site}: moe.combine exposes the {SEAM_OWNER_GROUP!r} seam but moe.shared_expert "
                f"({resolved.impl!r}) cannot hand it a SharedExpertPartial; a placement that gives the combine "
                f"both fc2 trees must resolve the owner-aware shared expert with it, or the shared expert's "
                f"fc2 tree would run twice"
            )
        # The shared expert runs inside sp_gather; its fc2 returns the full stream.
        # sp_scatter slices the joined result once, after combine.
        rp = self.seams.at("shared").row_parallel(inside_sp_gather=True, deferred_root=owner)
        self.slots.shared_row_parallel = rp
        wires: dict[str, Any] = {SEAM_ROW_PARALLEL: rp.fn}
        if tp_copy is not None:
            wires[SEAM_TP_COPY] = tp_copy
        self.slots.shared_expert = _seams.bind_seam(resolved, site=f"{self.site}.shared", **wires)

    def _install_expert_parallel(
        self, placement: Placement, *, ep: int, ep_rank: int, etp: int, tp_copy, owner: bool
    ) -> None:
        """Wrap expert execution and combine with EP or ETP movement, preserving their resolved
        implementation IDs. EP and ETP greater than one are mutually exclusive.

        Owner mode passes partials and the reduction group to combine together. The kernel
        checks this deferred layout bitwise against the in-place trees and falls back across the
        group on disagreement. Expert terms retain ascending global order across layouts.
        """
        if etp > 1:
            from isoexec.ops.moe.expert_parallel import expert_reduce

            base = self.slots.resolved()["expert_mlp"]
            # ETP must divide the recorded global leaf count so each rank owns whole leaves.
            fc2_leaves = int(base.inputs["leaves"])
            if etp > fc2_leaves or fc2_leaves % etp:
                raise PlacementError(
                    f"{self.site}: expert fc2 leaves {fc2_leaves} not divisible by etp={etp} (every rank "
                    f"owns whole leaves of the fixed tree)"
                )
            n_leaves = fc2_leaves // etp
            inner = _seams.bind_seam(base, site=f"{self.site}.experts", **{SEAM_N_LEAVES: n_leaves})
            # A single rounded leaf can travel losslessly as bf16 when the kernel supports it.
            # Multi-leaf roots stay fp32; the owner widens compact leaves before the rank tree.
            compact_root = owner and n_leaves == 1 and getattr(base.fn, "supports_compact_root", False)
            fn = expert_reduce(
                inner.fn,
                etp=etp,
                group=placement.etp.group,
                deferred=owner,
                tp_copy=tp_copy,
                compact_root=compact_root,
            )
            self.slots.expert_mlp = BoundOp(
                base.op,
                base.impl,
                base.version,
                base.provider,
                {**inner.inputs, "etp": etp, "fc2_leaves_local": n_leaves},
                fn,
                owns=base.owns,
            )
            if owner:
                # Bind the group for both deferred trees. The layout preserves numerical identity
                # and is checked by the owner kernel's per-geometry comparison.
                combine = self.slots.resolved()["combine"]
                bound = _seams.bind_seam(
                    combine, site=f"{self.site}.combine", **{SEAM_OWNER_GROUP: placement.etp.group}
                )
                self.slots.combine = BoundOp(
                    bound.op,
                    bound.impl,
                    bound.version,
                    bound.provider,
                    {**bound.inputs, "etp": etp},
                    bound.fn,
                    owns=bound.owns,
                )
        if ep > 1:
            from isoexec.ops.moe.expert_parallel import ep_combine, engine_expert_dispatch, expert_dispatch

            base, combine = self.slots.resolved()["expert_mlp"], self.slots.resolved()["combine"]
            # Engines replay captured decode graphs: their dispatch keeps static shapes and no host sync.
            dispatch = engine_expert_dispatch if placement.side == "engine" else expert_dispatch
            fn = dispatch(
                base.fn,
                ep=ep,
                ep_rank=ep_rank,
                ep_group=placement.ep.group,
                combine=combine.fn,
                placement=placement,
            )
            self.slots.expert_mlp = BoundOp(
                base.op,
                base.impl,
                base.version,
                base.provider,
                {**base.inputs, "ep": ep, "ep_movement": EP_DISPATCH, "dispatch_shard": placement.tp.size},
                fn,
                owns=base.owns,
            )
            # The returned rows are already folded, so the combine's remaining job is the shared add.
            self.slots.combine = BoundOp(
                combine.op,
                combine.impl,
                combine.version,
                combine.provider,
                {**combine.inputs, "ep": ep, "ep_movement": EP_FOLD},
                ep_combine(combine.fn, ep=ep),
                owns=combine.owns,
            )


    def held_experts(self) -> list[_Expert]:
        """This rank's experts in ascending global expert order -- the combine's fold order."""
        return [cast(_Expert, expert) for expert in self.experts]

    def expert_weight_views(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(gate_up[E, 2I, H], down[E, H, I])`` over the held experts. One call, one alias check."""
        packed = self.expert_views() if self.expert_views is not None else None
        if packed is not None:
            return packed
        held = self.held_experts()
        gate_up = torch.stack([torch.cat([e.gate, e.up], dim=0) for e in held], dim=0)
        down = torch.stack([e.down for e in held], dim=0)
        if torch.is_grad_enabled():
            # Provenance for a grouped weight-gradient kernel that accumulates each expert's block straight
            # into its parameters' gradient storage (duck-typed; read by ops/mm/wgrad.expert_sink).
            gate_up._isoexec_expert_params = tuple((e.gate, e.up) for e in held)
            down._isoexec_expert_params = tuple((e.down,) for e in held)
        return gate_up, down

    def gate_up_view(self) -> torch.Tensor:
        return self.expert_weight_views()[0]

    def down_view(self) -> torch.Tensor:
        return self.expert_weight_views()[1]


    def forward(
        self, x: torch.Tensor, seq: SequenceContext, aux: dict[str, torch.Tensor] | None = None,
        *, return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del seq  # a mixture reads no sequence geometry: every stage is per token
        input_dtype = x.dtype
        # Gather the replicated router/shared inputs; scatter outputs with matching gradient ownership.
        x = self.slots.sp_gather(x)
        scores = self.slots.route_score(x, self.router.weight)
        want = aux is not None or return_aux
        if self.routing == BIAS_ROUTING:
            routing = self.slots.route_select(scores, self._spec, want, expert_bias=self.router.expert_bias)
        else:
            routing = self.slots.route_select(scores, self._spec, want)
        if aux is not None and routing.scores_full is not None:
            name = f"{self.require_placed('forward')}.router.scores_full"
            if name in aux:
                raise KeyError(f"aux entry {name!r} added twice in one forward")
            aux[name] = routing.scores_full
        gate_up, down = self.expert_weight_views()
        routed = (self.slots.expert_mlp(x, routing, gate_up, down, weight_after_fc2=True)
                  if self.weight_after_fc2 else self.slots.expert_mlp(x, routing, gate_up, down))
        shared = None
        if self.shared is not None:
            s = self.shared
            shared = self.slots.shared_expert(x, s.gate, s.up, s.down, s.expert_gate)
        if routed.layout == "combined_local":
            if shared is not None and not isinstance(shared, torch.Tensor):
                raise TypeError(f"{self.site}: the combined_local layout needs a full shared Tensor or None")
            local_shared = None if shared is None else self.slots.sp_scatter(shared)
            combined = self.slots.combine(routed, local_shared)
            # Return caller-owned storage: the residual join writes in place, so an EP result
            # aliasing the dispatcher's autograd input must be copied.
            owned = combined.clone() if shared is None else combined
            output = self._output(owned, input_dtype)
        else:
            output = self._output(self.slots.sp_scatter(self.slots.combine(routed, shared)), input_dtype)
        if return_aux:
            if routing.scores_full is None:
                raise RuntimeError("MoE routing did not produce the requested full scores")
            # Explicit outputs follow the caller's token stream, including under SP.
            return output, self.slots.sp_scatter(routing.scores_full)
        return output

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden}, num_experts={self.num_experts}, top_k={self.top_k}, "
            f"intermediate={self.intermediate}, routing={self.routing!r}, "
            f"shared_intermediate={self.shared_intermediate}, shared_expert_gate={self.shared_expert_gate}"
        )


__all__ = ["BIAS_ROUTING", "HELD_DTYPE", "ROUTER_DTYPE", "ROUTINGS", "MoE", "expert_range"]
