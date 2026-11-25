"""Bind trainer operations without placing or materializing registered tensors.

``Component.apply_placement`` combines operation binding, local activation
geometry, and collective construction. Independent binding lets native
constructors select operations without applying those placement changes.
Existing callers retain their current API during migration; the duplicated
binding rules must be removed after their replacements are complete.
"""

# TODO(jiangjiawei1103): Complete migration to independent binding:
# - Move common binding rules into helpers shared by Megatron and vLLM, keeping
#   native core, cache, and process-group bindings in their respective adapters.
# - Replace TP1 constants and corresponding refusal cases when parallel support lands.
# - Retire duplicate binding in Component._place and obsolete placement helpers
#   after migrating their callers; preserve local geometry logic still needed.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import distributed as dist
from torch import fx

from isoexec.core.bindings import BoundOp
from isoexec.compiler.realization.tp import tensor_axis
from isoexec.nn import GDN, MLP, Attention, Embedding, LMHead, MoE, RMSNorm
from isoexec.nn._seams import Seams, _identity
from isoexec.nn.component import Component
from isoexec.nn.gdn import CORE_FUSED_OPS
from isoexec.nn.inspect import executed_modules
from isoexec.ops.collectives.tp_bytes import bind_kv_gather, bind_tp_copy, bind_vocab_parallel_embed
from isoexec.ops.collectives.tp_tree import bind_row_parallel
from isoexec.ops.table import OpResolutionError, OpTable
from isoexec.runtimes.parallel import validate_component_placement
from isoexec.schema import Axis, Placement

if TYPE_CHECKING:
    from megatron.core.process_groups_config import ProcessGroupCollection


def placement_from_groups(pg_collection: ProcessGroupCollection, *, sp: bool = False,
                          has_moe: bool = True) -> Placement:
    """Return the trainer Placement that MoE and context-parallel bindings use.

    Leaves and leaf dtype match ``bind_row_parallel``. EP greater
    than one holds whole experts (ETP=1); otherwise ETP equals TP and uses the
    TP group.
    """
    from isoexec.ops.collectives.pik_tp_invariant import _leaf_dtype, _num_leaves

    def axis(group: dist.ProcessGroup) -> Axis:
        return Axis(group.size(), group.rank(), group)

    tp, ep = axis(pg_collection.tp), axis(pg_collection.ep)
    etp = tp if ep.size == 1 else Axis(1, 0, None)
    if has_moe and pg_collection.expt_tp.size() != etp.size:
        raise ValueError(f"Native MoE requires expert_tensor_parallel_size={etp.size} at TP={tp.size}, EP={ep.size}")
    return Placement(tp=tp, ep=ep, dp=axis(pg_collection.dp), pp=axis(pg_collection.pp), etp=etp,
                     cp=axis(pg_collection.cp), side="trainer", leaves=_num_leaves(),
                     leaf_dtype="bf16" if _leaf_dtype() == torch.bfloat16 else "fp32", sp=sp)


def bind_trainer_ops(
    graph_module: fx.GraphModule, *, tp_group: dist.ProcessGroup, sp: bool = False,
    placement: Placement | None = None, op_table: OpTable | None = None,
    cp_group: dist.ProcessGroup | None = None, cp_layout: str = "contiguous", cp_transport_factory=None,
) -> None:
    """Bind trainer slots to the supplied TP group without changing tensor storage.

    The caller supplies an independently owned graph module and a native TP
    group. Row-parallel callables retain this group; their factory owns reduction
    settings. Under ``sp``, components receive this rank's sequence chunk and
    return it; ``placement.sp`` must agree. MoE components bind through
    ``apply_placement`` with ``placement``, which also removes experts other EP
    ranks own; call this before allocation. Binding failure may leave the target
    partially modified; the caller must discard it rather than publish it for
    execution.

    Context parallelism comes from ``placement.cp`` (or an explicit ``cp_group``). ``cp_layout``
    selects the mechanism: ``"contiguous"`` (any degree; every rank computes its own rows with the CP=1
    kernels, and key/value prefixes, recurrent states and conv tails move point-to-point,
    ``runtimes/context_parallel.py``; ``cp_transport_factory`` overrides the transport, e.g. for
    single-process simulation) or ``"zigzag"`` (CP=2: the gathered attention core / gqa_sm100 owner
    tiles and GDN's head-wise all-to-all, ``Seams.cp_attention``/``cp_heads``).
    """
    if placement is not None and placement.sp != sp:
        raise ValueError(f"Trainer binding has sp={sp} but its MoE placement has sp={placement.sp}")
    table = OpTable() if op_table is None else op_table
    for site, module in executed_modules(graph_module):
        if isinstance(module, MoE):
            if placement is None:
                raise OpResolutionError(f"{site}: MoE trainer binding requires a placement")
            module.apply_placement(placement, site=site, op_table=table)
        elif isinstance(module, Component):
            _bind_component(module, site, table, tp_group=tp_group, sp=sp, placement=placement, cp_group=cp_group,
                            cp_layout=cp_layout, cp_transport_factory=cp_transport_factory)
    if placement is not None:
        # Construction check: the leaf plan and, at EP>1, a combine in the korder_fp32 twin class.
        validate_component_placement(graph_module, placement)


def _bind_component(module: Component, site: str, table: OpTable, *, tp_group: dist.ProcessGroup,
                    sp: bool, placement: Placement | None = None, cp_group: dist.ProcessGroup | None = None,
                    cp_layout: str = "contiguous", cp_transport_factory=None) -> None:
    """Bind one component's trainer operations from its declared configuration. Under context
    parallelism the ``contiguous`` layout binds attention's KV-prefix core and GDN's state hand-off
    (``runtimes/context_parallel.py``, any degree); the ``zigzag`` layout binds attention's gathered core
    and GDN's head-wise region (``Seams.cp_attention``/``cp_heads``, CP=2). A component that states no
    CP form is refused."""
    module.site = module.slots.site = site
    if cp_group is None and placement is not None and placement.cp.size > 1:
        cp_group = placement.cp.group
    cp = 1 if cp_group is None else cp_group.size()
    if cp_layout not in ("contiguous", "zigzag"):
        raise OpResolutionError(f"{site}: cp_layout must be 'contiguous' or 'zigzag', got {cp_layout!r}")
    contiguous = cp > 1 and cp_layout == "contiguous"
    if cp > 1 and not contiguous and (placement is None or placement.cp.size != cp):
        raise OpResolutionError(f"{site}: the zigzag CP layout binds from a Placement with cp={cp}")
    if cp > 1 and module.CONTEXT_PARALLEL is None:
        note = f" ({module.NO_CONTEXT_PARALLEL})" if module.NO_CONTEXT_PARALLEL else ""
        raise OpResolutionError(f"{site}: {type(module).__name__} states no context-parallel form{note}")
    tp = tp_group.size()
    cp = 1 if cp_group is None else cp_group.size()
    # Replicated weights used inside the gathered region sum partial gradients without a token gather.
    grad_sum = bind_tp_copy(tp_group=tp_group, site=f"{site}.tp_copy")
    if "tp_copy" in module.slots:
        module.slots.tp_copy = bind_tp_copy(tp_group=tp_group, site=f"{site}.tp_copy", sp=True) if sp else grad_sum

    def bind(slot: str, **kwargs) -> BoundOp:
        bound = table.bind(module.slots.op_of(slot), site=f"{site}.{slot}", **kwargs)
        setattr(module.slots, slot, bound)
        return bound

    if isinstance(module, RMSNorm):
        bind("norm", inputs={"eps": module.eps}, convention=module.convention, facts={"width": module.hidden})
    elif isinstance(module, Embedding):
        module.slots.lookup = bind_vocab_parallel_embed(bind("lookup", inputs={"tp": tp}), tp_group=tp_group)
    elif isinstance(module, MLP):
        bind("mm")
        bind("act", facts={"activation": module.activation})
    elif isinstance(module, LMHead):
        module.tp_group = tp_group
        bind("head", inputs={"tp": tp, "boundary_out_dtype": "fp32"})
        bind("lm_head_slice")
        log_softmax = bind("log_softmax")
        if tp > 1:
            from isoexec.nn._seams import bind_seam

            module.slots.log_softmax = bind_seam(log_softmax, site=f"{site}.log_softmax", group=tp_group)
    elif isinstance(module, Attention):
        bind("mm")
        _bind_qk_rope(module, bind, grad_sum.fn)
        core = bind("core", inputs={"scale": module.scale, "head_dim": module.head_dim})
        if contiguous:
            from isoexec.runtimes.context_parallel import bind_cp_attention

            module.slots.cp_gather, module.slots.core = bind_cp_attention(
                core, group=cp_group, site=f"{site}.core", transport_factory=cp_transport_factory)
        elif cp > 1:
            module.slots.cp_gather, module.slots.core = Seams(placement, site).cp_attention(core)
        else:
            module.slots.cp_gather = _identity("collectives.cp_gather", cp=1)
        if module.out_gate is None:
            module.slots.declare_unbound("gate", "This attention has no output gate", required=False)
        else:
            bind("gate", inputs={"gate": module.out_gate}, facts={"gate": module.out_gate})
    elif isinstance(module, GDN):
        bind("mm")
        out_norm = bind("out_norm", inputs={"eps": module.eps}, convention=module.out_norm_convention,
                        facts={"width": module.value_head_dim, "activation": module.output_activation})
        if tp > 1:
            copy = grad_sum.fn

            def norm(output, gate, weight, eps):
                return out_norm.fn(output, gate, copy(weight), eps)

            module.slots.out_norm = replace(out_norm, fn=norm)
        core = bind("core")
        missing = [op for op in CORE_FUSED_OPS if op not in core.owns]
        if missing:
            raise OpResolutionError(f"{site}.core: {core.impl!r} does not own required operations {missing}")
        if "conv.short" in core.owns:
            module.slots.declare_unbound("conv", f"The {core.impl} core owns conv.short", required=False)
        else:
            bind("conv")
        if cp > 1:
            if contiguous:
                from isoexec.runtimes.context_parallel import bind_cp_gdn

                filled = module.slots.resolved()
                module.slots.core, wrapped_conv = bind_cp_gdn(filled["core"], filled.get("conv"), group=cp_group,
                                                              site=site, transport_factory=cp_transport_factory)
                module.slots.conv = wrapped_conv
                # Heads stay put: the mixer runs on this rank's tokens from the handed-off state.
                module.slots.cp_heads = _identity("collectives.cp_heads", cp=cp, layout="contiguous")
            else:
                # The region re-cuts this TP rank's heads: the in-projection columns it receives are local.
                module.slots.cp_heads = Seams(placement, site).cp_heads(
                    key_heads=module.key_heads_local, value_heads=module.value_heads_local,
                    key_head_dim=module.key_head_dim, value_head_dim=module.value_head_dim)
        else:
            module.slots.cp_heads = _identity("collectives.cp_heads", cp=1)
    else:
        raise OpResolutionError(f"{site}: trainer binding does not support {type(module).__name__}")

    if "row_parallel" in module.slots:
        module.slots.row_parallel = bind_row_parallel(tp_group=tp_group, site=f"{site}.row_parallel", sp=sp)
    if isinstance(module, Attention):
        module.slots.kv_gather = bind_kv_gather(tp_group=tp_group, kv_heads=module.kv_heads, site=f"{site}.kv_gather")

    missing = [slot for slot, value in module.slots.unfilled().items() if value.required]
    if missing:
        raise OpResolutionError(f"{site}: required trainer operation slots remain unbound: {missing}")


def _bind_qk_rope(module: Attention, bind: Callable[..., BoundOp],
                  copy: Callable[[torch.Tensor], torch.Tensor]) -> None:
    """Bind separate or fused q/k normalization and rotation from ownership metadata."""
    norm = None
    if module.qk_norm is None:
        module.slots.declare_unbound("qk_norm", "This attention has no q/k norm", required=False)
    else:
        norm = bind("qk_norm", inputs={"eps": module.eps}, convention=module.qk_norm)

    rotary = module._rotary
    if norm is not None and "rope.apply" in norm.owns:
        fused, spec = norm.fn, module.spec

        def stage(q, k, wq, wk, eps, seq):
            return fused(q, k, copy(wq), copy(wk), eps, seq=seq, rotary=rotary, spec=spec)

        module.slots.declare_unbound("qk_norm", f"The {norm.impl} qk_rope binding owns norms.qk", required=False)
        module.slots.qk_rope = BoundOp(norm.op, norm.impl, norm.version, norm.provider, norm.inputs,
                                      stage, owns=norm.owns)
        return

    rope = bind("qk_rope",
                inputs={"style": rotary.style, "rotary_dim": rotary.rotary_dim, "theta": rotary.theta},
                facts={"style": rotary.style, "rotary_dim": rotary.rotary_dim, "theta": rotary.theta,
                       "mrope_section": module.mrope_section, "mrope_interleaved": module.mrope_interleaved,
                       "qk_norm": module.qk_norm})
    norm_fn, rope_fn = None if norm is None else norm.fn, rope.fn

    def stage(q, k, wq, wk, eps, seq):
        if norm_fn is not None:
            q = norm_fn(q, copy(wq), eps)
            k = norm_fn(k, copy(wk), eps)
        return rope_fn(q, k, None, style=rotary.style, rotary_dim=rotary.rotary_dim, seq=seq, rotary=rotary)

    owns = {"rope.apply": rope.impl}
    if norm is not None:
        owns["norms.qk"] = norm.impl
    module.slots.qk_rope = BoundOp(rope.op, rope.impl, rope.version, rope.provider, rope.inputs, stage, owns=owns)


def set_tensor_parallel_attributes(graph_module: fx.GraphModule) -> None:
    """Set native shard attributes before DDP and optimizer construction."""
    from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes

    for logical_name, parameter in graph_module.named_parameters():
        axis = tensor_axis(graph_module, logical_name)
        parallel = axis.kind != "replicated"
        set_tensor_model_parallel_attributes(parameter, parallel, axis.dim if parallel else -1, axis.stride)


def set_sequence_parallel_attributes(graph_module: fx.GraphModule) -> None:
    """Mark RMSNorm weights so Megatron sums their gradients over TP under SP.

    Under SP every RMSNorm reads this rank's sequence chunk of the residual
    stream, so its weight gradient is partial. Megatron's
    ``finalize_model_grads`` sums ``sequence_parallel`` Parameters over the TP
    group when ``config.sequence_parallel`` is set. Set this before DDP and
    optimizer construction.
    """
    for _, module in executed_modules(graph_module):
        if isinstance(module, RMSNorm):
            for parameter in module.parameters():
                parameter.sequence_parallel = True


def set_expert_parallel_attributes(graph_module: fx.GraphModule) -> None:
    """Mark held experts so Megatron DDP reduces them over the expert DP group.

    Megatron treats ``allreduce=False`` Parameters as expert-parallel; set it
    only at EP>1, before DDP and optimizer construction.
    """
    for module in graph_module.modules():
        if isinstance(module, MoE):
            for parameter in module.experts.parameters():
                parameter.allreduce = False
