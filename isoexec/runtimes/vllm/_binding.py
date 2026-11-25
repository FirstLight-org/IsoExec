"""Bind component operations for native vLLM calls."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import distributed as dist
from torch import fx

from isoexec.core.bindings import BoundOp
from isoexec.nn import GDN, MLP, Attention, Embedding, LMHead, MoE, RMSNorm
from isoexec.nn._seams import _identity
from isoexec.nn.component import Component
from isoexec.nn.inspect import executed_modules
from isoexec.ops.collectives.tp_bytes import bind_vocab_parallel_embed
from isoexec.ops.collectives.tp_tree import bind_row_parallel
from isoexec.ops.table import OpResolutionError, OpTable
from isoexec.runtimes.engine_ops import SLICE_IMPL, engine_resolution, gdn_core_resolution, lm_head_slice_sampled_rows
from isoexec.runtimes.parallel import validate_component_placement
from isoexec.schema import Axis, Placement

from .bindings import PagedAttentionBinding

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def cudagraph_mode(vllm_config: VllmConfig) -> str:
    """Return the CUDA graph mode name vLLM resolved; ``enforce_eager`` resolves to NONE."""
    mode = getattr(vllm_config.compilation_config, "cudagraph_mode", None)
    return "NONE" if mode is None else getattr(mode, "name", str(mode))


#: CUDA graph modes the native model supports. FULL graphs cover uniform decode only (GDN prefill
#: reads host boundaries); FULL_AND_PIECEWISE replays mixed batches through vLLM's breakable
#: PIECEWISE graphs with attention and GDN cores as eager breaks (``graph_breaks``).
SUPPORTED_CUDAGRAPH_MODES = ("NONE", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE")
#: Modes whose mixed batches replay breakable PIECEWISE graphs.
PIECEWISE_CUDAGRAPH_MODES = ("FULL_AND_PIECEWISE",)


def placement_from_groups(*, tp_group: dist.ProcessGroup, pp_group: dist.ProcessGroup,
                          ep_group: dist.ProcessGroup | None = None) -> Placement:
    """Return the engine Placement that MoE components bind with.

    Without EP, ETP equals TP and uses the TP group. With vLLM's expert-parallel group, each EP
    rank holds whole experts (ETP=1) in its contiguous global range and the TP group shards the
    dispatched tokens. Leaves and leaf dtype match ``bind_row_parallel``.
    """
    from isoexec.ops.collectives.pik_tp_invariant import _leaf_dtype, _num_leaves

    tp, one = Axis(tp_group.size(), tp_group.rank(), tp_group), Axis(1, 0, None)
    ep = one if ep_group is None or ep_group.size() == 1 else Axis(ep_group.size(), ep_group.rank(), ep_group)
    return Placement(tp=tp, ep=ep, dp=one, pp=Axis(pp_group.size(), pp_group.rank(), pp_group),
                     etp=tp if ep.size == 1 else one, cp=one, side="engine", leaves=_num_leaves(),
                     leaf_dtype="bf16" if _leaf_dtype() == torch.bfloat16 else "fp32")


def bind_body_ops(body: fx.GraphModule, *, tp_group: dist.ProcessGroup, placement: Placement | None = None) -> None:
    """Bind body operations using the supplied native TP group.

    The graph and registered tensors remain unchanged. Row-parallel callables
    retain the native TP group. MoE components bind through
    ``apply_placement`` with ``placement``. Attention/GDN cores and GDN
    convolution remain required and unbound until native cache binding
    determines their ownership. Binding failure may leave the body partially
    bound; the caller must discard it.
    """
    table = OpTable()
    for site, module in executed_modules(body, prefix="body"):
        if isinstance(module, MoE):
            if placement is None:
                raise OpResolutionError(f"{site}: MoE engine binding requires a placement")
            module.apply_placement(placement, site=site, op_table=table)
        elif isinstance(module, (Embedding, RMSNorm, MLP, Attention, GDN)):
            _bind_component(module, site, table, tp_group=tp_group)
    if placement is not None:
        # Construction check of the MoE leaf plan.
        validate_component_placement(body, placement)


def bind_attention_cores(body: fx.GraphModule, *, vllm_config: VllmConfig, prefix: str = "") -> None:
    """Bind Attention cores to registered native vLLM layers.

    The caller supplies an allocated body in vLLM's configuration and allocation
    context. Native layers are registered as ``core`` children; vLLM owns their KV
    storage. Failure may leave the body and native registrations partially modified.
    """
    for site, module in executed_modules(body, prefix="body"):
        if isinstance(module, Attention):
            _bind_attention_core(module, site, vllm_config=vllm_config, prefix=prefix)


def _bind_attention_core(module: Attention, site: str, *, vllm_config: VllmConfig, prefix: str) -> None:
    """Bind Attention core to one registered native vLLM layer."""
    from vllm.model_executor.layers.attention import Attention as NativeAttention

    from isoexec.ops.attention.varlen_backend import PyTorchVarlenAttentionBackend

    layer_name = f"{prefix}.{site}.core" if prefix else f"{site}.core"
    layer = NativeAttention(num_heads=module.heads_local, head_size=module.head_dim,
                            scale=module.scale, num_kv_heads=module.kv_heads_local,
                            cache_config=vllm_config.cache_config, quant_config=None,
                            prefix=layer_name, attn_backend=PyTorchVarlenAttentionBackend)
    from .context_parallel import Settings

    # PCP/DCP route the core through the gathered-KV path in context_parallel.py.
    object.__setattr__(layer, "_isoexec_cp_settings", Settings.from_config(vllm_config))
    module.add_module("core", layer)

    from .graph_breaks import attention_core_into

    def core(q, k, v, cu_seqlens, max_seqlen, scale):
        del cu_seqlens, max_seqlen, scale  # Native layer and ForwardContext supply these fields.
        output = q.new_empty((q.shape[0], q.shape[1], v.shape[-1]))
        # The KV write and the paged kernel read this step's metadata: an eager break under a
        # breakable PIECEWISE capture, inline otherwise.
        attention_core_into(layer, q, k, v, output)
        return output

    module.slots.core = engine_resolution("attention.core.gqa", PagedAttentionBinding.impl_id(),
                                         core, provider="runtimes.vllm")


def bind_gdn_cores(body: fx.GraphModule, *, vllm_config: VllmConfig, prefix: str = "") -> None:
    """Bind GDN cores and convolution ownership to registered native state layers.

    The caller supplies allocated GDN weights in the native allocation context.
    Pool tensors retain the selected kernel's shapes and dtypes. Failure may leave
    the body and native registrations partially modified.
    """
    table = OpTable()
    for site, module in executed_modules(body, prefix="body"):
        if isinstance(module, GDN):
            _bind_gdn_core(module, site, table, vllm_config=vllm_config, prefix=prefix)


def _bind_gdn_core(module: GDN, site: str, table: OpTable, *, vllm_config: VllmConfig, prefix: str) -> None:
    """Bind one GDN layer using the existing CPR request lifecycle."""
    from ._gdn import GDNStateLayer
    from .context_parallel import Settings

    if Settings.from_config(vllm_config) is not None:
        raise ValueError(f"{site}: prefill/decode context parallelism is implemented for softmax attention only")
    from .gdn_engine_patch import (
        STATE_RESUMING_ENGINE_ARGS,
        assert_engine_args_compatible,
        enable_state_features,
        install_cpr_lazy_driver,
        lift_gdn_batch_invariance_veto,
    )

    engine_args = {
        "enable_prefix_caching": vllm_config.cache_config.enable_prefix_caching,
        "enable_chunked_prefill": vllm_config.scheduler_config.enable_chunked_prefill,
        "speculative_config": vllm_config.speculative_config,
        "mamba_cache_mode": getattr(vllm_config.cache_config, "mamba_cache_mode", None),
    }
    # This worker may not have inherited the flags register_native exported (an engine started in another Ray
    # actor), so it turns on the ones its own configuration needs before the gate reads them.
    enable_state_features([name for name in STATE_RESUMING_ENGINE_ARGS if engine_args[name]])
    assert_engine_args_compatible(engine_args, capturing=cudagraph_mode(vllm_config) != "NONE")
    lift_gdn_batch_invariance_veto()
    install_cpr_lazy_driver()
    layer_name = f"{prefix}.{site}.core" if prefix else f"{site}.core"
    layer = GDNStateLayer(module, vllm_config=vllm_config, prefix=layer_name)
    module.add_module("core", layer)
    module.attach_core(gdn_core_resolution(site, partial(layer, conv_weight=module.conv_w),
                                          table=table, provider="runtimes.vllm"))


def _bind_component(module: Component, site: str, table: OpTable, *, tp_group: dist.ProcessGroup) -> None:
    """Bind one component's operations, leaving native cache bindings pending."""
    module.site = module.slots.site = site
    tp = tp_group.size()

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
    elif isinstance(module, Attention):
        bind("mm")
        _bind_qk_rope(module, bind)
        if module.out_gate is None:
            module.slots.declare_unbound("gate", "This attention has no output gate", required=False)
        else:
            bind("gate", inputs={"gate": module.out_gate}, facts={"gate": module.out_gate})
        module.slots.kv_gather = _identity("collectives.all_gather", tp=tp, side="engine", kv_heads=module.kv_heads)
        module.slots.cp_gather = _identity("collectives.cp_gather", cp=1)
        module.slots.declare_unbound("core", "Native vLLM cache binding supplies this core")
    elif isinstance(module, GDN):
        bind("mm")
        bind("out_norm", inputs={"eps": module.eps}, convention=module.out_norm_convention,
             facts={"width": module.value_head_dim, "activation": module.output_activation})
        module.slots.declare_unbound("core", "Native vLLM cache binding supplies this core")
        module.slots.declare_unbound("conv", "Native vLLM core binding determines convolution ownership")
        # The engine runs every head of its TP shard on its own tokens (no prefill/decode CP for GDN).
        module.slots.cp_heads = _identity("collectives.cp_heads", cp=1)

    if isinstance(module, (MLP, Attention, GDN)):
        module.slots.tp_copy = _identity("collectives.tp_copy", tp=tp, side="engine", sp=False)
        module.slots.row_parallel = bind_row_parallel(tp_group=tp_group, site=f"{site}.row_parallel")


def _bind_qk_rope(module: Attention, bind: Callable[..., BoundOp]) -> None:
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
            return fused(q, k, wq, wk, eps, seq=seq, rotary=rotary, spec=spec)

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
            q = norm_fn(q, wq, eps)
            k = norm_fn(k, wk, eps)
        return rope_fn(q, k, None, style=rotary.style, rotary_dim=rotary.rotary_dim, seq=seq, rotary=rotary)

    owns = {"rope.apply": rope.impl}
    if norm is not None:
        owns["norms.qk"] = norm.impl
    module.slots.qk_rope = BoundOp(rope.op, rope.impl, rope.version, rope.provider, rope.inputs, stage, owns=owns)


def bind_head_ops(head: LMHead, *, tp_group: dist.ProcessGroup) -> None:
    """Bind head operations for rows already selected by the native runner.

    Registered parameters remain unchanged. Log-softmax is outside this call path.
    """
    table = OpTable()
    tp = tp_group.size()
    head.tp_group = tp_group
    head.site = head.slots.site = "head"
    head.slots.head = table.bind("logprobs.head", site="head.head",
                                 inputs={"tp": tp, "boundary_out_dtype": "fp32"})
    head.slots.lm_head_slice = engine_resolution("logprobs.lm_head_slice", SLICE_IMPL,
                                               lm_head_slice_sampled_rows, provider="runtimes.vllm")
    head.slots.tp_copy = _identity("collectives.tp_copy", tp=tp, side="engine", sp=False)
