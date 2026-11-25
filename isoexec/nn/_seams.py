"""Placement-dependent communication and fixed-tree reduction seams. Each seam records its
operation binding, including an explicit identity when no movement is needed.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from isoexec.core.bindings import COLLECTIVES_PROVIDER as PROVIDER
from isoexec.core.bindings import NO_MOVEMENT, BoundOp, no_movement
from isoexec.core.bindings import no_movement_binding as _identity
from isoexec.schema import Placement


class SeamError(ValueError):
    """A ``Placement`` a seam refuses."""


def accepts_kw(fn, name: str) -> bool:
    """Return whether the resolved callable accepts the named keyword."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    p = params.get(name)
    if p is not None and p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
        return True
    return any(q.kind is inspect.Parameter.VAR_KEYWORD for q in params.values())


def bind_seam(resolved: BoundOp, *, site: str, **kw: Any) -> BoundOp:
    """Bind a communication keyword, refusing callables that do not expose it. Preserve operation
    ownership and record the seam name in the binding inputs.
    """
    from isoexec.nn.component import PlacementError

    for k in kw:
        if not accepts_kw(resolved.fn, k):
            raise PlacementError(
                f"{site}: {resolved.op} binding {resolved.impl!r} has no keyword-only seam {k!r}; the "
                f"component cannot hand it the collective it needs at this placement (a binding that "
                f"reduces over a sharded K must expose the seam)"
            )
    fn = resolved.fn

    def bound(*a, **k):
        return fn(*a, **kw, **k)

    bound.__name__ = bound.__qualname__ = getattr(fn, "__name__", "bound") + "|sharded"
    return BoundOp(
        resolved.op, resolved.impl, resolved.version, resolved.provider, {**resolved.inputs, "seams": tuple(kw)}, bound, owns=resolved.owns
    )


def _declared_version(op: str, impl_id: str) -> int:
    """Read the version declared for this collective implementation."""
    from isoexec.ops.collectives.spec import IMPLS

    for spec in IMPLS:
        if spec.op == op and spec.impl_id == impl_id:
            return spec.version
    raise SeamError(f"{op}:{impl_id} is not declared in ops/collectives/spec.py")


@dataclass(frozen=True)
class Seams:
    """Build communication seams from a placement and a site. ``site`` is the FQN used in errors
    and operation records.
    """

    placement: Placement
    site: str
    op_table: Any = None

    def at(self, part: str) -> Seams:
        """Return the same placement under the child site ``f"{site}.{part}"``."""
        if not part:
            raise SeamError(f"{self.site}: Seams.at needs a non-empty part name")
        return Seams(self.placement, f"{self.site}.{part}", self.op_table)


    def tp_copy(self) -> BoundOp:
        """Return the column-parallel input seam: identity forward and fixed-tree TP sum backward.
        Under sequence parallelism, gather tokens in forward and reduce-scatter their partial
        gradients in backward.
        """
        placement = self.placement
        tp, side, sp = placement.tp.size, placement.side, bool(placement.sp)
        if tp == 1 or not (side == "trainer" or sp):
            return _identity("collectives.tp_copy", tp=tp, side=side, sp=sp)
        from isoexec.ops.collectives.tp_bytes import make_seq_all_gather, make_tp_copy

        impl = "seq_all_gather_fwd_tree_rs_bwd" if sp else "identity_fwd_tree_sum_bwd"
        fn = make_seq_all_gather(placement) if sp else make_tp_copy(placement)
        inputs = {"tp": tp, "side": side, "sp": sp, "leaves": int(placement.leaves), "leaf_dtype": placement.leaf_dtype}
        version = _declared_version("collectives.tp_copy", impl)
        return BoundOp("collectives.tp_copy", impl, version, PROVIDER, inputs, fn)

    def tp_grad_sum(self) -> BoundOp:
        """Return identity forward and a fixed-tree TP sum backward. Use inside an already-gathered
        region, where another token gather would duplicate movement.
        """
        placement = self.placement
        tp, side = placement.tp.size, placement.side
        if tp == 1 or side != "trainer":
            return _identity("collectives.tp_copy", tp=tp, side=side)
        from isoexec.ops.collectives.tp_bytes import make_tp_copy

        impl = "identity_fwd_tree_sum_bwd"
        inputs = {"tp": tp, "side": side, "leaves": int(placement.leaves), "leaf_dtype": placement.leaf_dtype}
        version = _declared_version("collectives.tp_copy", impl)
        return BoundOp("collectives.tp_copy", impl, version, PROVIDER, inputs, make_tp_copy(placement))


    def kv_gather(self, *, kv_heads: int) -> BoundOp:
        """Reassemble whole KV heads from trainer ranks holding head fractions. Engines replicate
        below the head granule and need no gather.
        """
        placement = self.placement
        tp, side = placement.tp.size, placement.side
        if kv_heads < 1:
            raise SeamError(f"{self.site}: kv_heads must be >= 1, got {kv_heads}")
        if tp == 1 or side != "trainer" or kv_heads >= tp:
            return _identity("collectives.all_gather", tp=tp, side=side, kv_heads=kv_heads)
        from isoexec.ops.collectives.tp_bytes import make_kv_gather

        impl = "nccl_bytes_tree_rs_bwd"
        fn = make_kv_gather(placement, kv_heads)  # refuses a tp that is not a multiple of the head count
        inputs = {"tp": tp, "side": side, "kv_heads": kv_heads}
        version = _declared_version("collectives.all_gather", impl)
        return BoundOp("collectives.all_gather", impl, version, PROVIDER, inputs, fn)


    def sp_gather(self) -> BoundOp:
        """Gather token chunks for replicated-weight consumers. Backward slices the complete
        gradient; summing replicated copies would multiply it by TP.
        """
        op, impl = "collectives.sp_gather", "seq_all_gather_replicated_fwd_slice_bwd"
        placement = self.placement
        tp, sp = placement.tp.size, bool(placement.sp)
        if not sp:
            return _identity(op, tp=tp, sp=sp)
        from isoexec.ops.collectives.tp_bytes import make_seq_all_gather_replicated

        inputs = {"tp": tp, "sp": sp, "rank": placement.tp.rank}
        fn = make_seq_all_gather_replicated(placement)
        return BoundOp(op, impl, _declared_version(op, impl), PROVIDER, inputs, fn)

    def sp_scatter(self) -> BoundOp:
        """Return a copied token slice in forward and all-gather in backward. The copy permits the
        residual join to write in place without aliasing an autograd input.
        """
        op, impl = "collectives.sp_scatter", "seq_scatter_fwd_all_gather_bwd"
        placement = self.placement
        tp, sp = placement.tp.size, bool(placement.sp)
        if not sp:
            return _identity(op, tp=tp, sp=sp)
        from isoexec.ops.collectives.tp_bytes import make_seq_scatter

        inputs = {"tp": tp, "sp": sp, "rank": placement.tp.rank}
        return BoundOp(op, impl, _declared_version(op, impl), PROVIDER, inputs, make_seq_scatter(placement))


    def cp_attention(self, core: BoundOp | None) -> tuple[BoundOp, BoundOp | None]:
        """Return the gather binding and its wrapped attention core together. The gqa_sm100 trainer
        core at head dim 256 takes its own CP form (``gqa_sm100/context_parallel.py``): K/V gathered
        forward, every local row's dQ/dK/dV from the CP=1 tile that owns it, bitwise the CP=1 rows.
        Any other core is wrapped generically: gather and unzigzag the global packed stream, run the
        core, keep local token rows; backward sums the two contributors before reduce-scattering.

        Preserve the core identity and record movement separately. Only degrees 1 and 2 are
        supported because the backward sum is order-independent for two contributors. A deferred
        engine core is allowed only at CP=1.
        """
        op, impl = "collectives.cp_gather", "all_gather_unzigzag_fwd_sum_rs_bwd"
        placement = self.placement
        cp = placement.cp.size
        if cp == 1:
            return _identity(op, cp=cp), core
        if core is None:
            raise SeamError(
                f"{self.site}.cp_attention: cp={cp} and no attention core was resolved at this site. The "
                f"context-parallel seam IS the core, gathered; recording the gather next to a slot nothing "
                f"has filled would state a movement no callable performs"
            )
        from isoexec.ops.attention.gqa_sm100.spec import TRAINER_IMPL as GQA_SM100_TRAINER
        from isoexec.ops.collectives.cp_gather import make_cp_attention_core, make_cp_gather

        if core.impl == GQA_SM100_TRAINER and core.inputs.get("head_dim") == 256:
            # The kernel's own CP form: every local row from the CP=1 tile that owns it, no sum, half the work.
            from isoexec.ops.attention.gqa_sm100.context_parallel import make_trainer_core

            impl = "kv_all_gather_fwd_owner_tiles_bwd"
            wrapped = make_trainer_core(placement)
        else:
            wrapped = make_cp_attention_core(core.fn, placement)
        gather = BoundOp(op, impl, _declared_version(op, impl), PROVIDER, {"cp": cp}, make_cp_gather(placement))
        # The composed binding explicitly owns the gather. records_from_tree checks this claim
        # against the separate gather binding instead of emitting another input-less record.
        owns = {**core.owns, core.op: core.impl, op: impl}
        return gather, BoundOp(core.op, core.impl, core.version, core.provider, core.inputs, wrapped, owns=owns)


    def cp_heads(self, *, key_heads: int, value_heads: int, key_head_dim: int, value_head_dim: int) -> BoundOp:
        """Return the head-wise context-parallel region of a recurrent mixer. At CP>1, an all-to-all
        hands each rank every token of ``1/cp`` of the heads, the mixer runs per head, and the reverse
        all-to-all returns this rank's tokens; the adjoints are the reverse movements, with no sum.
        """
        op, impl = "collectives.cp_heads", "head_all_to_all_both_directions"
        placement = self.placement
        cp = placement.cp.size
        if cp == 1:
            return _identity(op, cp=cp)
        from isoexec.ops.collectives.cp_heads import make_gdn_region

        fn = make_gdn_region(placement, key_heads=key_heads, value_heads=value_heads,
                             key_head_dim=key_head_dim, value_head_dim=value_head_dim)
        return BoundOp(op, impl, _declared_version(op, impl), PROVIDER, {"cp": cp}, fn)

    def row_parallel(self, *, inside_sp_gather: bool = False, deferred_root: bool = False) -> BoundOp:
        """Build the fixed leaf-tree K reduction, including at TP=1. Each leaf rounds to
        ``leaf_dtype``, the balanced tree folds in fp32, and the root rounds once.

        Sequence parallelism returns a reduce-scattered token chunk unless ``inside_sp_gather``
        requests the full stream. The explicit reference table uses its distinct
        ``ref.leaf_tree`` identity and has no reduce-scatter implementation.

        ``deferred_root`` returns this rank's partial and the tree owed by the owner combine. It
        preserves the numerical identity: the owner path checks live results bitwise against the
        in-place trees and falls back across the group on disagreement.
        """
        placement, site = self.placement, self.site
        tp, g = placement.tp.size, int(placement.leaves)
        if deferred_root and (not inside_sp_gather or tp == 1):
            raise SeamError(
                f"{site}.row_parallel: deferred_root is the owner-combine layout of a contraction that runs "
                f"inside an already-gathered region at tp>1 (got inside_sp_gather={inside_sp_gather}, tp={tp}); "
                f"there is nothing to defer at tp=1 and no combine owns a reduce-scattered root"
            )
        # Named here, like kv_gather's head count, so the refusal carries the site; the rest of the leaf
        # plan's domain is the factory's own (``tp_tree._tree_domain`` / ``ref_row_parallel``).
        if tp > g or g % tp != 0:
            raise SeamError(
                f"{site}.row_parallel: tp={tp} must divide leaves={g} (every rank owns whole leaves of the fixed tree)"
            )
        op = "collectives.row_parallel"
        inputs = {"tp": tp, "leaves": g, "leaf_dtype": placement.leaf_dtype}
        # The same fixed tree, scattered: this rank keeps its own token slice of the root.
        scattered = bool(placement.sp) and tp > 1 and not inside_sp_gather
        from isoexec.ops.table import REFERENCE_PROVIDER, OpTable, reference_binding

        table = self.op_table if self.op_table is not None else OpTable()
        if table.mode == "abstract":
            return table.bind(op, site=f"{site}.row_parallel", inputs=inputs)
        if table.mode == "reference":
            if deferred_root:
                raise SeamError(
                    f"{site}.row_parallel: the owner-combine layout hands on an unreduced partial with the pik "
                    f"tree attached to it, and this host serves {op!r} from the reference table, which states the "
                    f"whole tree on one rank and nothing that defers it. Recording the pik impl id from here "
                    f"would put a reference in the kernel's twin class"
                )
            if scattered:
                raise SeamError(
                    f"{site}.row_parallel: a sequence-parallel reduce-scatter of the fixed tree has no "
                    f"plain-torch statement, and this host serves {op!r} from the reference table. Recording "
                    f"the pik impl id from here would put a reference in the kernel's twin class and digest "
                    f"EQUAL to it, which is the one comparison a reference must always lose"
                )
            from isoexec.ops.collectives.reference import ref_row_parallel

            ref_impl = reference_binding(op, site=f"{site}.row_parallel").impl
            fn = ref_row_parallel(leaves=g, leaf_dtype=placement.leaf_dtype, tp=tp)
            return BoundOp(op, ref_impl, None, REFERENCE_PROVIDER, inputs, fn)
        table.require_production(op, site=f"{site}.row_parallel")
        from isoexec.ops.collectives.tp_tree import make_row_parallel, make_row_parallel_partial, make_row_parallel_rs

        impl = "pik_tree_rs" if scattered else "pik_tree"
        if deferred_root:
            fn = make_row_parallel_partial(placement)
        else:
            fn = make_row_parallel_rs(placement) if scattered else make_row_parallel(placement)
        return BoundOp(op, impl, _declared_version(op, impl), PROVIDER, inputs, fn)


__all__ = [
    "NO_MOVEMENT",
    "PROVIDER",
    "SeamError",
    "Seams",
    "accepts_kw",
    "bind_seam",
    "no_movement",
]
