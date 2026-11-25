"""The blocks of the CED stacks as placement-bound opaque components.

``EncoderBlock`` (causal encoder layer: SWA, CSA2 FULL or REUSE, Engram on two layers), ``DecoderEntry`` (the CED
projection: the decoder FULL layer's attention boundary and its token-wise global-KV, index-key and window-KV
projection, on EVERY token) and ``DecoderBlock`` (decoder layer on the decoder segments' rows: the FULL layer's
query side, REUSE, REINDEX). A block runs, in order: the optional Engram gated residual, the attention
boundary (mHC coefficients, collapse and RMSNorm of the four streams), the sparse attention (window + compressed
KV, indexer, top-k, learned-sink dual-rail attention, grouped ``wo_a``, the fixed 8-leaf ``wo_b`` tree), the FFN
boundary (which also applies the attention expand) and the MoE (sqrt-softplus router, clamped-SwiGLU experts,
``config.MOE_RECIPE``). The FFN expand is left DEFERRED in the returned carry: the next boundary fuses it with its
collapse and RMSNorm, the head stage (or the CED handoff) settles the last one.

Carry, as explicit tensors: ``x [B, T, 4, H]`` bf16 streams, ``pre [B, T, 4]`` fp32 shifted collapse weights,
and the deferred expand ``(out [B, T, H], post [B, T, 4], comb [B, T, 4, 4])`` (``None`` before the first block).
Inside the encoder the cross-layer attention state of a CSA2 group is explicit: its FULL layer returns ``kv``,
``index_codes`` / ``index_sf`` and ``topk``; its REUSE layers take them (``EncoderBlock.consumes``). In the
decoder every block reads ``decoder.entry``'s global KV and the decoder chain's top-k / candidate pool.

GPU execution: the encoder blocks run the confirmed composition of the attention package (the same kernels and
order as the validated ``deepseek_v41_bf16_actor_v1`` rows). The decoder components' GPU composition (the
entry's projection split from the FULL layer's attention call, segment-row attention with the window truncated
at the segment start) needs op entry points the attention lane has not delivered (``model/REQUESTS.md``);
until then they refuse a CUDA forward by name and the CPU reference program (``reference.py``) states the
arithmetic.

Seams resolved once at placement (``providers.PROVIDERS[recipe]``): ``linear`` (one provider per projection
kind), ``moe_forward`` (per side and EP degree), the TP reducer (prepared at placement; its reductions run in
its execution scope, held by an owning runtime or entered here), the EP transport and, on the engine, the
MegaMoE workspace (``prepare_expert_workspace``). Derived weights (BF16 and FP32 copies, concatenations,
``wo_b`` leaves) live in the kernels' per-holder caches and are re-derived in place by ``refresh_derived_weights``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from isoexec.nn.component import Component, PlacementError
from isoexec.schema import Placement, ShardAxis, replicated

from .admission import admit
from .config import MOE_RECIPE, RECIPE, SINK_RECIPE, CedGeometry
from .holders import AttentionWeights, EngramWeights, EntryProjection, MoEWeights, Norm
from .providers import PROVIDERS, ep_comm, tp_reducer

#: Cross-row attention state, in the order a block returns what it produces.
STATE_KEYS = ("kv", "index_codes", "index_sf", "topk", "candidates")

#: Rows up to which Engram's ``wkv`` output columns are split over the TP ranks at inference (``_engram_kv``; the
#: same bits as the replicated projection). Measured on a TP2 pair: the half-width GEMM saves ~23 us at any row
#: count, but the all-gather and the column interleave cost more than that above ~16 rows.
ENGRAM_TP_SPLIT_ROWS = 16


class _WeightRows:
    """A linear-provider operand holding a contiguous row block of a weight (the provider reads ``.weight``)."""

    def __init__(self, weight):
        self.weight = weight


def _zero_engine_caches(engine_state) -> int:
    """Zero a row's engine cache pages (window KV, compressed KV, index keys, compressor state) that are mapped;
    returns the tensors zeroed. Run at a weight-update commit and right after every KV-cache wake (the wake maps
    fresh pages of unspecified content), so both leave the same bytes. Pages of a pool that is asleep are
    unmapped (writing them faults) and are zeroed by the wake that maps them."""
    if engine_state is None:
        return 0
    from isoexec.runtimes.vllm.weights_sleep import cumem_asleep
    from .engine import cache_tensors
    n = 0
    with torch.no_grad():
        for tensor in cache_tensors(engine_state):
            if not cumem_asleep(tensor):
                tensor.zero_()
                n += 1
    return n


def _rms_norm(weight, x, eps):
    from isoexec.ops.norms.norms_sm100.ops import rms_norm
    shape = x.shape
    return rms_norm(x.reshape(-1, shape[-1]).contiguous(), weight, eps, False).view(shape)


class _Carry:
    """Streams between two mHC boundaries with an optional deferred ``expand(out, x, post, comb)`` and the
    side-stream event that produced pre/post/comb; ``settle`` joins and expands (the eager order's values)."""

    __slots__ = ("deferred", "event", "pre", "x")

    def __init__(self, x, pre, event=None, deferred=None):
        self.x, self.pre, self.event, self.deferred = x, pre, event, deferred

    def join(self):
        if self.event is not None:
            torch.cuda.current_stream(self.x.device).wait_event(self.event)
            self.event = None

    def settle(self):
        self.join()
        if self.deferred is not None:
            from isoexec.ops.residual.deepseek_mhc_sm100.ops import expand
            out, post, comb = self.deferred
            self.x, self.deferred = expand(out, self.x, post, comb), None
        return self.x, self.pre


def _step_rows(step, rows):
    """The rows of an engine step's attention: its real tokens (``engine.real_tokens``). A padded PIECEWISE step's
    padding rows carry no attention metadata; eager and FULL steps have none."""
    return rows if rows.shape[0] == step.tokens else rows[:step.tokens]


def _padded_rows(out, rows: int):
    """``out [real, ...]`` back on the step's ``rows`` with zero padding rows (the mature binding's ``output[n:]``
    zero); unchanged when there is no padding."""
    if out.shape[0] == rows:
        return out
    return torch.nn.functional.pad(out, (0, 0) * (out.dim() - 1) + (0, rows - out.shape[0]))


def boundary(cfg, fn, scale, base, norm_weight, carry, *, sublayer):
    """``carry`` -> ``(streams, (pre, post, comb, event), h)``. Inference at H = 5120 on CUDA: one launch for
    [expand +] collapse + RMSNorm, the coefficients on a side stream at decode sizes; otherwise the deferred
    expand, the fused split-with-collapse and the RMSNorm (autograd-capable)."""
    from isoexec.ops.residual.deepseek_mhc_sm100.ops import mhc_params, mhc_params_async, mhc_split_shifted
    kw = dict(rms_eps=cfg.rms_norm_eps, sinkhorn_eps=cfg.hc_eps, iterations=cfg.hc_sinkhorn_iters)
    if not torch.is_grad_enabled() and carry.x.shape[-1] == 5120 and carry.x.is_cuda:
        from isoexec.ops.residual.deepseek_mhc_sm100.ops import boundary_rows
        carry.join()
        out, post, comb = carry.deferred if carry.deferred is not None else (None, None, None)
        got = boundary_rows(carry.x, carry.pre, norm_weight, cfg.rms_norm_eps, out=out, post=post, comb=comb,
                            encode=False)
        if got is not None:
            x, h = got
            pre, post, comb, ev = mhc_params_async(x, fn, scale, base, sublayer=sublayer, **kw)
            return x, (pre, post, comb, ev), h
    x, shifted_pre = carry.settle()
    if not torch.is_grad_enabled() and x.shape[-1] == 5120:
        from isoexec.ops.attention.deepseek_v41_cache_sm100.paged import collapse_rms_norm
        new_pre, post, comb = mhc_params(x, fn, scale, base, **kw)
        h = collapse_rms_norm(x, shifted_pre, norm_weight, cfg.rms_norm_eps, encode=False)
        if h is not None:
            return x, (new_pre, post, comb, None), h
    new_pre, post, comb, col = mhc_split_shifted(x, fn, scale, base, shifted_pre, **kw)
    return x, (new_pre, post, comb, None), _rms_norm(norm_weight, col, cfg.rms_norm_eps)


@dataclass
class _TP:
    size: int = 1
    rank: int = 0
    group: Any = None

    def reducer(self):
        return tp_reducer(self.group)

    def scope(self):
        """The reducer's execution scope around this row's reductions (none at TP = 1, or when the calling
        thread already holds it: an owning runtime's capture / replay scope)."""
        from contextlib import nullcontext
        return nullcontext() if self.size == 1 else self.reducer().scope()


@dataclass
class _EP:
    """This block's expert ownership and exchange: contiguous expert ranges, the shared expert at home."""

    size: int = 1
    rank: int = 0
    group: Any = None
    partitioned: bool = False
    expert_map: Any = None
    local: tuple[int, ...] = ()
    static_home_rows: int | None = None
    mega: Any = field(default=None, repr=False)       # the engine's BF16 MegaMoE workspace (EP > 1)
    _comm: Any = field(default=None, repr=False)

    def comm(self):
        if self._comm is None:
            self._comm = ep_comm(self.group)
        return self._comm


def _mhc_parameters(module, cfg, parts, device) -> None:
    mix, n = (2 + cfg.hc_mult) * cfg.hc_mult, cfg.hc_mult
    for part in parts:
        for name, shape in ((f"hc_{part}_fn", (mix, n * cfg.hidden_size)), (f"hc_{part}_base", (mix,)),
                            (f"hc_{part}_scale", (3,))):
            module.register_parameter(name, nn.Parameter(torch.empty(shape, dtype=torch.float32, device=device),
                                                         requires_grad=False))


class _StageMember:
    """Pipeline stage starts declared by the model (``DeepseekV41ForCausalLM.declare_pipeline_stages``)."""

    stage_starts: tuple[int, ...] | None = None
    ced_prefill: str | None = None

    def program_identity(self) -> dict:
        """The forward program's identity (``config.program_identity``), streamed with every weight update."""
        from .config import program_identity
        return program_identity(self.cfg, self.ced_prefill)

    def program_recorded(self) -> dict:
        from .config import program_recorded
        return program_recorded()

    def _admit(self, placement, site):
        starts = self.stage_starts
        if starts is None and placement.pp.size > 1 and placement.side == "engine":
            # vLLM's own layer partition (the engine has no other place to declare it): the ranges the model's
            # ``pipeline_assignments`` hands the vLLM factory, so the admitted starts are the stages vLLM builds
            # (defect 13: a re-derived split disagreed with it)
            from .admission import vllm_layer_partition
            starts = tuple(lo for lo, _ in vllm_layer_partition(self.cfg, placement.pp.size))
        admit(self.cfg, placement, site, program=self.ced_prefill, stage_starts=starts)


def _tp_sum_backward(h, placement):
    """``h`` for a TP-sharded consumer on the trainer: identity forward, rank-ordered TP sum of its gradient backward
    (the column-parallel input seam, ``Seams.tp_copy``). The consumers behind it (attention heads split over TP, and
    the projections shared by those heads) leave a per-rank partial input gradient; summing it here gives every rank
    the same complete gradient, so everything below is recomputed with the same bits. Identity at TP 1, outside
    autograd and on the engine."""
    if placement is None or placement.side != "trainer" or placement.tp.size == 1 or not torch.is_grad_enabled():
        return h
    from isoexec.ops.collectives.parallel_autograd import copy_to_parallel
    return copy_to_parallel(h, placement.tp.group)


def _agree_break_mode(placement) -> str:
    """The piecewise break structure (``piecewise.agree_mode``), agreed over this stage's TP group at bind time
    (collective; the first bound row of every rank, outside capture)."""
    from .piecewise import agree_mode
    tp = placement.tp
    return agree_mode(tp.group if tp.size > 1 else None)


def _join_before_break(ev):
    """A side stream forked inside a recorded segment must be joined before the segment ends at a break."""
    if ev is not None:
        torch.cuda.current_stream().wait_event(ev)
    return None


def _group_agrees(group, flag: bool) -> bool:
    """``flag`` on EVERY rank of ``group`` (collective; a rank-local predicate in front of a collective setup)."""
    import torch.distributed as dist
    if group is None or not dist.is_initialized() or dist.get_world_size(group) == 1:
        return bool(flag)
    flags = [None] * dist.get_world_size(group)
    dist.all_gather_object(flags, bool(flag), group=group)
    return all(flags)


class V41Layer(_StageMember, Component):
    """One backbone layer ``index`` (shared machinery of the encoder and decoder blocks); parameters on the
    holders ``attn``, ``ffn``, ``engram``, the norms, and the mHC tensors ``hc_{attn,ffn}_{fn,base,scale}``
    (the decoder FULL layer's attention boundary and projection belong to ``DecoderEntry``)."""

    PLACEMENT_BOUND = True
    NO_CONTEXT_PARALLEL = "CP schedule of the windows, compressed KV and indexer not designed yet (briefs/CP.md)"

    def __init__(self, cfg: CedGeometry, index: int, *, recipe: str = RECIPE, device="meta") -> None:
        if recipe not in PROVIDERS:
            raise ValueError(f"DeepSeek V4.1 block recipe must be one of {sorted(PROVIDERS)}, got {recipe!r}")
        super().__init__()
        self.cfg, self.index, self.recipe = cfg, index, recipe
        self.kind = cfg.kind(index)
        entry = self.kind.is_entry
        _mhc_parameters(self, cfg, ("ffn",) if entry else ("attn", "ffn"), device)
        if not entry:
            self.attn_norm = Norm(cfg.hidden_size, device=device)
        self.attn = AttentionWeights(cfg, index, device, projection=not entry)
        self.ffn_norm = Norm(cfg.hidden_size, device=device)
        self.ffn = MoEWeights(cfg, device)
        self.engram = EngramWeights(cfg, index, device) if self.kind.engram is not None else None
        self.tp, self.ep = _TP(), _EP()
        self.linear = self.moe_forward = None
        self.engine_state = None
        self.expert_path = None           # the expert execution the last forward took (set by the MoE seam)
        self._graph_reads = None          # (name, address, shape, dtype) of the parameters graphs read in place
        self._graph_slots = ()            # (name, owner's parameter dict, leaf, parameter, address) of the same
        self._packed = None               # engine: (gate_up, down, anchors) of ``packed_experts`` (see there)

    # ------------------------------------------------------------------------------ structure
    @property
    def ratio(self) -> int:
        return self.cfg.compress_ratios[self.index]

    def consumes(self) -> tuple[str, ...]:
        """State inputs this row reads from earlier rows."""
        cfg, i = self.cfg, self.index
        if not self.ratio:
            return ()
        keys = [] if i in cfg.kv_source_layer_ids else ["kv"]
        if i in cfg.index_source_layer_ids:
            if i not in cfg.kv_source_layer_ids:
                keys += ["index_codes", "index_sf"]
            if i > cfg.candidate_source_layer_id:
                keys.append("candidates")
        else:
            keys.append("topk")
        return tuple(keys)

    def produces(self) -> tuple[str, ...]:
        cfg, i = self.cfg, self.index
        keys = []
        if self.ratio and i in cfg.kv_source_layer_ids:
            keys.append("kv")
            if i in cfg.index_source_layer_ids:
                keys += ["index_codes", "index_sf"]
        if self.ratio and i in cfg.index_source_layer_ids:
            keys.append("topk")
            if i == cfg.candidate_source_layer_id:
                keys.append("candidates")
        return tuple(k for k in STATE_KEYS if k in keys)

    def tensor_axes(self) -> Mapping[str, ShardAxis]:
        return {name: replicated() for name in self._parameters}

    def state_specs(self):
        from .program import ced_states
        return _specs(ced_states(self.cfg), self.site_prefix())

    def site_prefix(self) -> str:
        b = self.cfg.boundary
        return f"encoder.blocks.{self.index}" if self.index < b else f"decoder.blocks.{self.index - b}"

    def facts(self) -> Mapping[str, Any]:
        cfg, k = self.cfg, self.kind
        return {"kind": f"deepseek_v41_{k.stack}_block", "recipe": self.recipe,
                "program": PROVIDERS[self.recipe].program, "moe": MOE_RECIPE, "stack": k.stack,
                "mode": k.mode.value, "kv_source": k.kv_source, "index_source": k.index_source,
                "candidate": k.candidate, "window_clamp": k.window_clamp, "ced_prefill": self.ced_prefill,
                "sink_recipe": SINK_RECIPE, "layer": self.index, "checkpoint_layer": cfg.layer_sources[self.index],
                "compress_ratio": self.ratio, "engram": self.engram is not None,
                "hidden": cfg.hidden_size, "heads": cfg.num_attention_heads, "head_dim": cfg.head_dim,
                "o_groups": cfg.o_groups, "experts": cfg.n_routed_experts, "top_k": cfg.num_experts_per_tok,
                "intermediate": cfg.moe_intermediate_size, "routed_scale": cfg.routed_scaling_factor,
                "swiglu_limit": cfg.swiglu_limit, "rms_eps": cfg.rms_norm_eps, "hc_eps": cfg.hc_eps,
                "hc_iterations": cfg.hc_sinkhorn_iters, "consumes": self.consumes(), "produces": self.produces()}

    # ------------------------------------------------------------------------------ placement
    def narrow_held(self, placement: Placement, *, site: str) -> int:
        """Keep this rank's contiguous routed-expert range under EP (global ``experts.{e}`` names); the shared
        expert is computed at home on every rank. Idempotent. Returns the number removed."""
        ep, E = placement.ep.size, self.cfg.n_routed_experts
        if ep == 1:
            return 0
        if E % ep:
            from .admission import EP_EXPERT_SPLIT, Refusal
            raise Refusal(EP_EXPERT_SPLIT, site, f"{E} routed experts do not split over EP = {ep}")
        lo, hi = placement.ep.rank * E // ep, (placement.ep.rank + 1) * E // ep
        removed = 0
        for key in list(self.ffn.experts._modules):
            if not lo <= int(key) < hi:
                del self.ffn.experts._modules[key]
                removed += 1
        return removed

    def _place(self, placement: Placement) -> None:
        from isoexec.ops.moe.ep_fixed_order import ExpertMap
        site = self.require_placed("_place")
        cfg, tp, ep = self.cfg, placement.tp.size, placement.ep.size
        self._admit(placement, site)
        self.narrow_held(placement, site=site)
        if placement.side == "trainer":
            from .training import mark_trainable
            mark_trainable(self)
        providers = PROVIDERS[self.recipe]
        self.linear = {kind: providers.linear() for kind in ("attention", "indexer", "engram")}
        self.moe_forward = providers.moe(placement.side, ep)
        self.tp = _TP(tp, placement.tp.rank, placement.tp.group)
        if tp > 1 and torch.cuda.is_available():
            # collective (every TP rank places its rows in the same order, outside capture): the tree
            # transport's workspace, shared by every row of the group
            self.tp.reducer().prepare(torch.device("cuda", torch.cuda.current_device()))
        emap = ExpertMap.contiguous(cfg.n_routed_experts, ep) if ep > 1 else None
        self.ep = _EP(ep, placement.ep.rank, placement.ep.group, partitioned=ep > 1 and tp == 1, expert_map=emap,
                      local=tuple(e for e in emap.local_experts(placement.ep.rank) if e < cfg.n_routed_experts)
                      if emap is not None else tuple(range(cfg.n_routed_experts)))
        if placement.side == "trainer" and self.ep.partitioned:
            # Token-disjoint EP training: ep_moe_train already sums the router's and the shared expert's
            # gradients over the EP group (ordered, every rank the same bits); Megatron must not reduce them over
            # it again, so they are reduced as expert tensors (over the expert data-parallel group).
            self.ffn.mark_ep_summed()
        if self.ffn.held_experts() != self.ep.local:
            raise PlacementError(f"{site}: held experts {self.ffn.held_experts()[:4]}... differ from the EP map")

    def bind_engine_state(self, *, vllm_config: Any, layer_prefix: str) -> None:
        """Register this row's native vLLM V4.1 cache layers (``engine.bind_row_caches``) and, for captured
        decode under EP, prepare the static-capacity routes of the largest capture size (equal on every peer)."""
        from .engine import bind_row_caches, graphs_enabled
        if self.placement is None or self.placement.side != "engine":
            raise PlacementError(f"{layer_prefix}: engine state binds an engine-placed row")
        if self.kind.is_entry:            # layer b's caches are decoder.entry's
            self.engine_state, graphs = None, graphs_enabled(vllm_config)
        else:
            self.engine_state = bind_row_caches(self.cfg, self.index, vllm_config,
                                                canonical_prefix(layer_prefix, self.index))
            graphs = self.engine_state.graphs
        _agree_break_mode(self.placement)
        if self.ep.size > 1 and graphs:
            sizes = list(getattr(vllm_config.compilation_config, "cudagraph_capture_sizes", None) or ())
            largest = max(sizes) if sizes else getattr(vllm_config.compilation_config, "max_cudagraph_capture_size", 0)
            if type(largest) is not int or largest < 1:
                raise ValueError("captured EP decode needs positive cudagraph capture sizes")
            rows = -(-largest // self.ep.size)
            import torch.distributed as dist
            mine = torch.tensor([rows], dtype=torch.int64, device=self.ffn.gate.weight.device)
            peers = [torch.empty_like(mine) for _ in range(self.ep.size)]
            dist.all_gather(peers, mine, group=self.ep.group)
            if any(int(p) != rows for p in peers):
                raise RuntimeError("EP peers disagree on the static route capacity")
            self.prepare_static_ep(rows)

    def prepare_expert_workspace(self) -> bool:
        """The engine's canonical BF16 MegaMoE workspace for this row (EP > 1; EP = 1 creates its own on first use).
        Collective: every EP rank calls it after the weights load, outside capture, for the same rows in the same
        order; the first row of a group compiles every tile geometry. Returns whether the row runs MegaMoE."""
        if self.ep.size == 1 or self.placement is None or self.placement.side != "engine":
            return False
        gate_up, down = self.packed_experts()
        # Every predicate in front of the collective setup is agreed by the whole EP group: a rank that skipped the
        # setup alone would leave its peers waiting in the rendezvous (no setup counter can catch that).
        from .providers import EP_EXCHANGE_ENV, fallback_forced, record_transport
        if not _group_agrees(self.ep.group, gate_up.is_cuda and not fallback_forced(EP_EXCHANGE_ENV)):
            record_transport("ep_exchange", self.ep.group, f"fixed_order:{EP_EXCHANGE_ENV}|host_masters")
            return False            # host-resident masters, or ISOEXEC_DS_EP_EXCHANGE=fixed_order (on any rank):
                                    # every rank keeps the fixed-order exchange (NCCL), same bits
        from isoexec.ops.collectives.node_local import NodeLocalityError, symmetric_admission
        from isoexec.ops.collectives.symm_setup import SymmetricSetupSkew
        from isoexec.ops.moe.deepseek_v41_sm100 import mega

        from .providers import note_transport_choice, prepare_expert_workspace
        # MegaMoE EP maps every peer's device memory: one host, pairwise distinct GPUs. The group-agreed peer
        # admission (one gather of (host, device UUID, ordinal), the same verdict on every rank) decides it once;
        # an EP group across hosts or with two ranks on one GPU keeps the fixed-order exchange (NCCL, same bits)
        verdict = symmetric_admission(self.ep.group)
        if not verdict.symmetric:
            if record_transport("ep_exchange", self.ep.group, f"fixed_order:{verdict.reason}"):
                note_transport_choice("EP exchange", self.ep.group,
                                      f"fixed-order NCCL exchange ({verdict.reason}: {verdict.detail})")
            return False
        try:
            ws = prepare_expert_workspace(self.ep.group, gate_up.device, self.cfg)
        except (NodeLocalityError, SymmetricSetupSkew) as exc:
            # raised on every rank alike (the workspace's own peer refusal, or symm_setup's count agreement)
            if record_transport("ep_exchange", self.ep.group, f"fixed_order:{type(exc).__name__}"):
                note_transport_choice("EP exchange", self.ep.group, f"fixed-order NCCL exchange ({exc})")
            return False
        mega.check_expert_map(self.ep.expert_map, ws.world, ws.rank)
        if not ws.warmed:
            with torch.no_grad():
                mega.warm(ws, gate_up.detach(), down.detach(), self.cfg.swiglu_limit)
        self.ep.mega = ws
        if record_transport("ep_exchange", self.ep.group, "mega"):
            note_transport_choice("EP exchange", self.ep.group, f"MegaMoE symmetric memory ({verdict.reason})")
        return True

    def prepare_engine_execution(self) -> None:
        """After the checkpoint load (``VllmModel`` load hook, collective, outside capture): pack the experts into
        their expert-major storage, prepare the MegaMoE workspace (EP > 1), and record the address of every
        parameter the kernels read in place (DeepGEMM projections, MegaMoE masters, packed experts): captured
        graphs keep reading those addresses, so a weight update must write into them
        (``refresh_derived_weights`` refuses one that moved a parameter)."""
        if self.placement is None or self.placement.side != "engine":
            raise PlacementError(f"{type(self).__name__}: engine execution prepares an engine-placed row")
        self._packed = None
        self.packed_experts()
        self.pack_shared_input()
        self.prepare_expert_workspace()
        self._graph_reads = self._read_addresses()
        slots = []
        for owner_name, owner in self.named_modules():
            for leaf, p in owner._parameters.items():
                if p is not None:
                    slots.append((f"{owner_name}.{leaf}" if owner_name else leaf, owner._parameters, leaf, p,
                                  p.data_ptr()))
        self._graph_slots = tuple(slots)
        self._cache_packed_experts()

    def pack_shared_input(self) -> int:
        """Pack this row's projections that read the same input into one storage each, through the attention
        ops (values unchanged; weight updates write through the parameters into it; the consumer runs one GEMM
        when the members are adjacent rows): ``wq_a | wkv`` (| the derived indexer weights on index-source layers,
        ``ops.pack_shared_input_rows``, attention request M-D2) and, on index-source layers, ``wq_b | indexer
        wq_b`` (``ops.pack_qr_rows``, M-D3). Engine CUDA weights only, after the rank's TP shard is in place: the
        trainer's parameters live in its data-parallel buffers (unpacked: separate projections, the same bits).
        Returns the number of packed groups."""
        from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops as v41_ops
        packed = 0
        # the decoder's FULL layer (index b) has no window projection here: decoder.entry owns its wkv (projection=False)
        if (hasattr(self.attn, "wkv") and self.attn.wq_a.weight.is_cuda
                and v41_ops.pack_shared_input_rows(self.attn, self.cfg, self.index) is not None):
            packed += 1
        if self.attn.wq_b.weight.is_cuda and v41_ops.pack_qr_rows(self.attn, self.cfg, self.index) is not None:
            packed += 1
        return packed

    def _read_addresses(self):
        return tuple((name, p.data_ptr(), tuple(p.shape), p.dtype) for name, p in self.named_parameters())

    def moved_parameters(self) -> list[str]:
        """Parameters whose storage moved since ``prepare_engine_execution`` (graphs would read the old one)."""
        if self._graph_reads is None:
            return []
        now = self._read_addresses()
        if len(now) != len(self._graph_reads):
            return ["<parameter set changed>"]
        return [old[0] for old, new in zip(self._graph_reads, now) if old != new]

    def moved_parameter_sample(self, start: int, count: int) -> list[str]:
        """``moved_parameters`` over ``count`` of the recorded parameters from ``start`` (cyclic), with no module
        walk: a parameter replaced in its owner or rebound to another storage since the engine prepared."""
        slots = self._graph_slots
        out = []
        for i in range(start, start + min(count, len(slots))):
            name, params, leaf, p, address = slots[i % len(slots)]
            if params.get(leaf) is not p or p.data_ptr() != address:
                out.append(name)
        return out

    def graph_parameter_count(self) -> int:
        return len(self._graph_slots)

    def engine_execution_scopes(self) -> list:
        """The runtime-owned execution scope of this row's TP tree reducer (engine, TP > 1, prepared transport):
        a runtime that holds it around capture, replay and eager forwards gets the tree in captured graphs."""
        if self.tp.size == 1 or self.placement is None or self.placement.side != "engine":
            return []
        reducer = self.tp.reducer()
        return [reducer] if reducer.prepared else []

    def engine_expert_exchange(self):
        """The bound expert exchange (EP > 1) the engine replays in captured graphs, or None."""
        return self.moe_forward if self.ep.size > 1 else None

    def prepare_static_ep(self, max_home_rows: int) -> None:
        """Fixed-capacity EP routes for captured graph buckets (prepare once, outside capture, equal on every
        peer); ``use_static_ep`` switches them per forward."""
        if self.ep.size > 1:
            if type(max_home_rows) is not int or max_home_rows < 1:
                raise ValueError("BF16 static EP requires positive max_home_rows")
            self.ep.expert_map.tables(self.ep.comm().rank, self.ffn.gate.weight.device)
            self.ep.static_home_rows = max_home_rows

    def select_ep_routes(self, step) -> bool:
        """Per engine forward, no collective: graph decode buckets run the static-capacity routes prepared
        once at bind time (per-bucket capacity ``min(prepared, ceil(T / ep))``), eager prefill / mixed /
        profiling steps the host-counted routes. Never prepares."""
        if self.ep.size == 1 or self.ep.static_home_rows is None:
            return False
        static = bool(step is not None and step.graph)
        if self.index == 0:          # one record per forward (Python runs at capture, not at replay)
            from .engine import record_routes
            record_routes(static, None if step is None else step.tokens,
                          bool(step is not None and step.capturing))
        return self.use_static_ep(static)

    def use_static_ep(self, enabled: bool) -> bool:
        if self.ep.size == 1:
            return False
        if self.ep.static_home_rows is None:
            raise RuntimeError("prepare_static_ep(max_home_rows=...) before switching static EP routes")
        self.ep.comm().static_routes = bool(enabled)
        return True

    # ------------------------------------------------------------------------------ derived weights
    def packed_experts(self):
        """``gate_up [E_local + 1, 2I, H]`` (w1 rows then w3 rows) and ``down [E_local + 1, H, I]``, the shared
        expert last: aliases of the per-expert masters, which are re-pointed once at one expert-major storage
        (values unchanged; later in-place updates land in it). Differentiable to the per-expert masters.

        A prepared engine row (``prepare_engine_execution``) returns the two aliases it recorded there, without a
        per-forward walk of its experts (no gradient on the engine). They are views of the masters' storage, so an
        in-place weight update is read through them, exactly as the captured graphs read the same addresses; a
        parameter that moved is what neither may see: ``refresh_derived_weights`` refuses a move and re-records
        the aliases at every update commit, and each call checks the anchors (first and last master of each
        operand: replaced in its owner or rebound) and re-derives on a mismatch. The trainer has no cache (its
        operands are stacked copies of parameters the optimizer changes every step)."""
        cached = self._packed
        if cached is not None and not torch.is_grad_enabled():
            gate_up, down, anchors = cached
            if all(params.get(leaf) is p and p.data_ptr() == address for params, leaf, p, address in anchors):
                return gate_up, down
            self._packed = None
        from isoexec.ops.moe.clamped_swiglu_sm100.packing import Stack, packed_slices, stacked
        experts = [self.ffn.experts[str(e)] for e in self.ep.local] + [self.ffn.shared_experts]
        gu = [p for e in experts for p in (e.w1.weight, e.w3.weight)]
        dn = [e.w2.weight for e in experts]
        G, (I, H) = len(experts), gu[0].shape
        if self.placement is not None and self.placement.side == "trainer":
            # The trainer's parameters live in the data-parallel buffers, one buffer slice each, so they are
            # not re-pointed: the packed operands are stacked copies (same values) for the forward, the gradient
            # returns as per-expert slices. The copies carry their parameters, so the MoE backward saves the
            # parameters and re-stacks (no layer keeps its copy from forward to backward).
            return stacked(gu, (G, 2 * I, H), 2), stacked(dn, (G, H, I), 1)
        with torch.no_grad():
            packed_slices(gu, (G, 2 * I, H))
            packed_slices(dn, (G, H, I))
        out = Stack.apply((G, 2 * I, H), *gu), Stack.apply((G, H, I), *dn)
        if cached is not None:              # a prepared engine row whose anchors moved: record the new aliases
            self._cache_packed_experts()
        return out

    def _cache_packed_experts(self) -> None:
        """Record the engine's packed expert aliases (``packed_experts``) and their anchor masters."""
        self._packed = None
        if self.placement is None or self.placement.side != "engine":
            return
        from isoexec.ops.moe.clamped_swiglu_sm100.packing import alias_slices
        experts = [self.ffn.experts[str(e)] for e in self.ep.local] + [self.ffn.shared_experts]
        gu = [p for e in experts for p in (e.w1.weight, e.w3.weight)]
        dn = [e.w2.weight for e in experts]
        G, (I, H) = len(experts), gu[0].shape
        gate_up, down = alias_slices(gu, (G, 2 * I, H)), alias_slices(dn, (G, H, I))
        if gate_up is None or down is None:
            return                          # not packed (yet): every call walks the experts
        owners = ([(e.w1, "weight") for e in experts[:1]] + [(experts[-1].w3, "weight")]
                  + [(experts[0].w2, "weight"), (experts[-1].w2, "weight")])
        anchors = tuple((m._parameters, leaf, m._parameters[leaf], m._parameters[leaf].data_ptr())
                        for m, leaf in owners)
        self._packed = (gate_up, down, anchors)

    def refresh_derived_weights(self) -> int:
        """Re-derive every cached copy of this block's weights into its existing storage (captured graphs keep
        their addresses). Call after a weight update, before replay; refuses a layout change. An engine row
        also clears its cache pages (no request is in flight across an update)."""
        from isoexec.ops.attention.deepseek_v41_cache_sm100.ops import refresh_derived_weights
        self._packed = None
        moved = self.moved_parameters()
        if moved:
            raise RuntimeError(f"layer {self.index}: a weight update moved {moved[:4]}; the kernels and captured "
                               "graphs read these parameters in place (update in place, or rebuild the engine)")
        n = refresh_derived_weights(self)
        if self._graph_reads is not None:
            self._cache_packed_experts()
        if self.engine_state is not None:
            from .engine import commit_weights
            # A colocated update runs with the KV pool still asleep (SkyRL wakes "weights", syncs, then wakes
            # "kv_cache"): only mapped pages are cleared here; the KV wake zeroes the pages it maps
            # (``zero_engine_caches``, called by the worker's wake hook).
            self.zero_engine_caches()
            if self.index == 0:
                commit_weights()
        return n

    def zero_engine_caches(self) -> int:
        return _zero_engine_caches(self.engine_state)

    def stale_derived_weights(self) -> list[str]:
        """Derived copies whose sources changed since derivation (a write that bypassed the refresh)."""
        return self.derived_weight_guard().stale()

    def derived_weight_guard(self):
        """The per-step check of this row's derived copies (``DerivedWeightGuard``: entries collected once, a
        version compare per call); guards of several rows combine into one (``DerivedWeightGuard.combine``)."""
        guard = self.__dict__.get("_derived_guard")
        if guard is None:
            from isoexec.ops.attention.deepseek_v41_cache_sm100.ops import DerivedWeightGuard
            guard = self.__dict__["_derived_guard"] = DerivedWeightGuard([self], [f"layer {self.index}"])
        return guard

    # ------------------------------------------------------------------------------ forward
    def _engram(self, x, hashes):
        from isoexec.ops.embed.engram_sm100.ops import engram_gate_residual, fp8_rowwise_lookup
        cfg, eng = self.cfg, self.engram
        ids = hashes[..., cfg.hash_column(self.index), :]
        # A host-resident table is read through its device mapping (``engram_host_tables``): same bytes.
        weight, scale = getattr(eng.embed, "device_view", None) or (eng.embed.weight, eng.embed.scale)
        rows = fp8_rowwise_lookup(weight, scale, ids.to(x.device), validate=False)
        kv = self._engram_kv(eng, rows.flatten(-2))
        S, H = cfg.hc_mult, cfg.hidden_size
        key, value = kv.split((S * H, H), dim=-1)
        return engram_gate_residual(x, key.reshape(x.shape), value.contiguous(), eng.q_weight, eng.k_weight,
                                    cfg.rms_norm_eps)

    def _engram_kv(self, eng, rows):
        """Engram's ``wkv`` projection ``rows [..., width] -> [..., (hc_mult + 1) * hidden]``.

        At TP > 1 (replicated tokens, inference, at most ``ENGRAM_TP_SPLIT_ROWS`` rows) rank r computes the output
        columns ``[r n, (r + 1) n)`` from its row block of the SAME replicated weight through the same provider,
        and one all-gather plus the rank-order column interleave rebuilds ``[..., N]`` on every rank (native
        vLLM's column split). The weight's published ``mm_sm100`` segment count is 1 (policy v4: a function of K only,
        so the same at both widths), so every output column is the provider's one-chain program whichever rank
        computes it: the replicated bits.
        """
        lin, tp, w = self.linear["engram"], self.tp.size, eng.wkv.weight
        if (tp == 1 or rows.numel() > ENGRAM_TP_SPLIT_ROWS * rows.shape[-1] or torch.is_grad_enabled()
                or not rows.is_cuda or w.dtype != torch.bfloat16 or w.shape[0] % (16 * tp)):
            return lin(eng.wkv, rows)
        from isoexec.ops.mm.mm_sm100 import segments as seg
        n, k = w.shape[0] // tp, w.shape[1]
        if (not seg.supported(k, n) or not seg.supported(k, w.shape[0])
                or seg.segment_count(k, torch.bfloat16) != 1):
            return lin(eng.wkv, rows)
        import torch.distributed as dist
        r = self.tp.rank
        x2 = rows.reshape(-1, k)
        M = x2.shape[0]
        local = lin(_WeightRows(w[r * n:(r + 1) * n]), x2).contiguous()
        flat = torch.empty((tp * M, n), dtype=local.dtype, device=local.device)
        if M:
            dist.all_gather_into_tensor(flat, local, group=self.tp.group)
        return flat.view(tp, M, n).permute(1, 0, 2).reshape(*rows.shape[:-1], tp * n)

    def _attention(self, h, state, step=None):
        from isoexec.ops.attention.deepseek_v41_cache_sm100.ops import attention_full_gen, drive
        from isoexec.ops.collectives.parallel_autograd import copy_to_parallel, reduce_from_parallel
        lin, tp = self.linear["attention"], self.tp
        training = torch.is_grad_enabled()
        if step is not None:          # engine step over vLLM's pages: the packed tokens as rows
            from isoexec.ops.attention.deepseek_v41_cache_sm100.paged import attention_paged_gen
            rows = h.reshape(-1, h.shape[-1])
            gen = attention_paged_gen(self.attn, _step_rows(step, rows), self.cfg, self.index, step.pools,
                                      step.batch, linear_fn=lin, dequant_fn=lin.dequant_fn, tp=tp.size)
            if tp.size == 1:
                step.note("reducer", "none")
                return _padded_rows(drive(gen), rows.shape[0]).view(h.shape)
            reducer = tp.reducer()
            with tp.scope():
                step.note("reducer", reducer.transport())
                return _padded_rows(reducer.run(gen), rows.shape[0]).view(h.shape)
        if training and tp.size > 1:
            h = copy_to_parallel(h, tp.group)
        gen = attention_full_gen(self.attn, h, self.cfg, self.index, state, linear_fn=lin, dequant_fn=lin.dequant_fn,
                                 recipe=SINK_RECIPE, tp=tp.size)
        if tp.size == 1:
            return drive(gen)
        reducer = tp.reducer()
        with tp.scope():
            if not training:
                return reducer.run(gen)
            try:
                msg = next(gen)
                while True:
                    msg = gen.send([reducer.gather_sum(s) for s in msg[1]])
            except StopIteration as stop:
                return reduce_from_parallel(stop.value, reducer)

    def _attention_break(self, h, seq):
        """The attention sublayer as ONE eager break of a recorded PIECEWISE segment (``piecewise.core``): the step's
        paged metadata, page writes, compressor, indexer, top-k, dual-rail attention, output projection and TP
        reduction run eagerly on every replay with the live forward context; ``h`` and the output are the segment
        tensors around it. ``seq`` is the capture-time context: only its static positions buffer is read."""
        from .piecewise import core
        out = torch.empty_like(h)
        core(self._attention_core, (h, seq), (out,))
        return out

    def _attention_core(self, h, seq):
        from .engine import engine_step
        step = engine_step(self.cfg, self.index, self.engine_state, seq)
        return self._attention(h, {}, step)

    def _capture_routes(self) -> None:
        """A recorded PIECEWISE segment runs the experts with the static-capacity EP routes (host-counted routes sync
        the host): the same choice on every EP rank (it depends on the capture state only)."""
        if self.ep.size > 1 and self.ep.static_home_rows is not None:
            self.use_static_ep(True)

    def _state_in(self, given: Mapping[str, Any], B: int) -> dict:
        state = {}
        if given.get("kv") is not None:
            state["compress_kv_native"] = given["kv"]
        if given.get("index_codes") is not None:
            state["index_k_native"] = (given["index_codes"], given["index_sf"])
        if given.get("topk") is not None:
            state["topk_native"] = given["topk"]
        if given.get("candidates") is not None:
            state["candidates_native"] = {b: given["candidates"][b] for b in range(B)}
        missing = [k for k in self.consumes() if given.get(k) is None]
        if missing:
            raise ValueError(f"layer {self.index} needs {missing} from its source rows")
        return state

    def _state_out(self, state: Mapping[str, Any], x) -> tuple:
        B, T = x.shape[:2]
        out = []
        for key in self.produces():
            if key == "kv":
                out.append(state["compress_kv_native"])
            elif key in ("index_codes", "index_sf"):
                pair = state.get("index_k_native")
                if pair is None:      # no compressed row yet (T < ratio): nothing reads the keys
                    pair = self._empty_index_keys(x, B)
                out.append(pair[0] if key == "index_codes" else pair[1])
            elif key == "topk":
                out.append(state["topk_native"])
            else:
                cands = state.get("candidates_native")
                out.append(torch.stack([cands[b] for b in range(B)]) if cands else
                           torch.empty((B, T, self.cfg.candidate_topk_blocks), dtype=torch.int32, device=x.device))
        return tuple(out)

    @staticmethod
    def _empty_index_keys(x, B):
        return (torch.empty((B, 0, 128), dtype=torch.uint8, device=x.device),
                torch.empty((B, 0), dtype=torch.int32, device=x.device))

    def _composed_forward(self, x, pre, out=None, post=None, comb=None, hashes=None, kv=None, index_codes=None,
                          index_sf=None, topk=None, candidates=None, seq=None):
        """``(x, pre, out, post, comb, *produced)``; see the module docstring for shapes.

        Engine rows read and write vLLM's pages through the forward's paged step (``engine.engine_step``);
        their cross-row state travels in the pages and the step, and the produced outputs are empty. Without
        paged metadata (vLLM's profiling pass) an engine row runs the full-sequence program."""
        if self.placement is None:
            raise PlacementError(f"{type(self).__name__}: apply_placement has not run")
        step = None
        core = False
        if self.placement.side == "engine":
            if self.engine_state is None:
                raise RuntimeError("bind_engine_state has not run for this engine row")
            from .piecewise import core_capturing
            core = core_capturing()
            if core:            # recording a PIECEWISE segment: the step's metadata is read inside the attention break
                self._capture_routes()
            else:
                from .engine import engine_step
                step = engine_step(self.cfg, self.index, self.engine_state, seq)
                self.select_ep_routes(step)
        cfg = self.cfg
        if x.is_cuda:
            from .providers import install_projection_gemm
            install_projection_gemm()
        carry = _Carry(x, pre, None, None if out is None else (out, post, comb))
        if self.engram is not None:
            if hashes is None:
                raise ValueError(f"Engram layer {self.index} needs the n-gram hashes")
            xs, shifted = carry.settle()
            carry = _Carry(self._engram(xs, hashes), shifted)
        x, (pre_a, post, comb, ev), h = boundary(cfg, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
                                                 self.attn_norm.weight, carry, sublayer="attn")
        # A packed trainer forward attends per sequence: the right-padded view of its rows (``training``).
        layout = None
        if self.placement.side == "trainer":
            from .training import sequence_rows
            layout = sequence_rows(seq, h)
        h_att = h if layout is None else layout.pad(h)
        given = dict(kv=kv, index_codes=index_codes, index_sf=index_sf, topk=topk, candidates=candidates)
        if layout is not None:
            given = layout.unpack_state(given, self.ratio)
        state = {} if step is not None or core else self._state_in(given, h_att.shape[0])
        if core:
            ev = _join_before_break(ev)
            a = self._attention_break(h_att, seq)
        else:
            a = self._attention(h_att, state, step)
        if layout is not None:
            a = layout.unpad(a)
        x, (pre_f, post, comb, ev), h = boundary(cfg, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
                                                 self.ffn_norm.weight, _Carry(x, pre_a, ev, (a, post, comb)),
                                                 sublayer="ffn")
        rows = h.reshape(-1, h.shape[-1])
        y = self.moe_forward(self, rows).view(h.shape)
        _Carry(x, pre_f, ev).join()       # the coefficients' side stream overlapped the MoE
        if core:
            return (x, pre_f, y, post, comb, *self._placeholders(x, None))
        if step is not None:
            step.note("experts", self.expert_path)
            step.note("routes", "local" if self.ep.size == 1 else
                      "static" if getattr(self.ep.comm(), "static_routes", False) else "counted")
            if self.index == cfg.num_hidden_layers - 1:
                from .engine import record_program
                record_program(step)
            # the produced state lives in the pages and the step: unwritten tensors of the captured shapes
            return (x, pre_f, y, post, comb, *self._placeholders(x, step))
        produced = self._state_out(state, h_att)
        if layout is not None:     # the captured packed geometry between rows
            produced = layout.pack_state(self.produces(), produced, self.ratio)
        return (x, pre_f, y, post, comb, *produced)

    def _placeholders(self, x, step) -> tuple:
        """The produced-state outputs of an engine forward: never written, never read (consumers read the pages),
        but the captured boundary requires each one's exact shape and strides, offset 0 and no alias of an input.
        They persist per row and padded token count: graph bucket sizes keep theirs, eager steps share one
        slot replaced when the token count changes; none is allocated inside a capture (a graph's private pool
        must not own a tensor that outlives it), and a row never shares one with another row (whose outputs are
        its inputs)."""
        if not self.produces():
            return ()
        key = (tuple(x.shape[:2]), x.device)
        cache = self.__dict__.setdefault("_placeholder_cache", {})
        hit = cache.get(key)
        if hit is None:
            made = tuple(torch.empty(shape, dtype=dtype, device=x.device)
                         for shape, dtype in self._produced_shapes(*key[0]))
            if x.is_cuda and torch.cuda.is_current_stream_capturing():
                return made
            if not step.graph:
                for k in [k for k, (kept, _) in cache.items() if not kept]:
                    del cache[k]
            hit = cache[key] = (bool(step.graph), made)
        return hit[1]

    def _produced_shapes(self, B: int, T: int) -> list:
        """``(shape, dtype)`` of each produced state output, in ``produces()`` order."""
        cfg = self.cfg
        C = T // self.ratio if self.ratio else 0
        Cp = (C + 127) // 128 * 128
        shapes = {"kv": ((B * C, cfg.head_dim), torch.bfloat16),
                  "index_codes": ((B, Cp, cfg.index_head_dim), torch.uint8),
                  "index_sf": ((B, Cp), torch.int32), "topk": ((B * T, cfg.index_topk), torch.int32),
                  "candidates": ((B, T, cfg.candidate_topk_blocks), torch.int32)}
        return [shapes[key] for key in self.produces()]

    def abstract_output(self, x, pre, out=None, post=None, comb=None, hashes=None, kv=None, index_codes=None,
                        index_sf=None, topk=None, candidates=None, seq=None):
        """Shape and dtype rules only; no kernel runs."""
        cfg = self.cfg
        B, T = x.shape[0], x.shape[1]
        S, H = cfg.hc_mult, cfg.hidden_size
        dev = x.device
        result = [torch.empty((B, T, S, H), dtype=torch.bfloat16, device=dev),
                  torch.empty((B, T, S), dtype=torch.float32, device=dev),
                  torch.empty((B, T, H), dtype=torch.bfloat16, device=dev),
                  torch.empty((B, T, S), dtype=torch.float32, device=dev),
                  torch.empty((B, T, S, S), dtype=torch.float32, device=dev)]
        result += [torch.empty(shape, dtype=dtype, device=dev) for shape, dtype in self._produced_shapes(B, T)]
        return tuple(result)


def _specs(states, owner):
    from isoexec.schema import StateSpec
    return tuple(StateSpec(st.name, (st.record_bytes,), "u8", "per_request" if st.tokens_per_entry == 0 else
                           "per_token", st.reuse) for st in states if st.owner == owner)


class EncoderBlock(V41Layer):
    """Causal encoder layer ``index`` (every token): SWA (layers 0-1), CSA2 FULL (compressor ratio 2, own
    indexer) or REUSE (the group's FULL layer's compressed KV and top-k); Engram on its Engram layers."""

    def __init__(self, cfg: CedGeometry, index: int, **kw) -> None:
        if cfg.kind(index).stack != "encoder":
            raise ValueError(f"layer {index} is a decoder layer")
        super().__init__(cfg, index, **kw)

    def forward(self, x, pre, out=None, post=None, comb=None, hashes=None, kv=None, index_codes=None,
                index_sf=None, topk=None, candidates=None, seq=None):
        from .piecewise import core_capturing, run
        args = (x, pre, out, post, comb, hashes, kv, index_codes, index_sf, topk, candidates, seq)
        if core_capturing():         # attention-core breaks inside the row (piecewise ``core``)
            return self._composed_forward(*args)
        return run(self, self._composed_forward, args, {})


def ced_ops():
    """The attention package's CED entry points (model/REQUESTS.md R1-R3), or a named refusal until they land:
    ``entry_projection(holder, h, cfg, layer, positions, *, linear_fn, step=None)`` and
    ``decoder_attention_gen(holder, h, cfg, layer, rows, decoder_kv, shared, *, linear_fn, dequant_fn, tp=1,
    step=None)`` (signatures in ``DecoderBlock.forward`` / ``DecoderEntry.forward``)."""
    try:
        from isoexec.ops.attention.deepseek_v41_cache_sm100 import ced
    except ImportError:
        raise NotImplementedError(DECODER_GPU_PENDING) from None
    return ced


DECODER_GPU_PENDING = ("the decoder's GPU composition (entry projection split from the FULL layer's attention, "
                       "segment-row attention with the window truncated at the segment start, the replay tail) "
                       "needs attention-lane entry points (model/REQUESTS.md R1-R3); CPU: reference.py")


def canonical_prefix(layer_prefix: str, index: int) -> str:
    """The cache-layer prefix of backbone layer ``index`` (``...layers.{index}``) from a component's module path
    (``encoder.blocks.{i}``, ``decoder.entry``, ``decoder.blocks.{j}``): native vLLM's names, so the cache layers
    of every stack resolve their KV source by layer id."""
    import re
    out, n = re.subn(r"(^|\.)(?:encoder\.blocks\.\d+|decoder\.blocks\.\d+|decoder\.entry|layers\.\d+)$",
                     rf"\g<1>layers.{index}", layer_prefix)
    if n != 1:
        raise ValueError(f"cannot locate the layer of {layer_prefix!r}")
    return out


class _CedForward:
    """One decoder forward's host context, created by ``decoder.entry`` and read by the decoder blocks of the same
    forward: the decoder rows, the step (engine) and the decoder chain (``shared``: top-k, candidate pool and the
    attention package's per-forward views). Nothing in it crosses forwards."""

    current = None

    def __init__(self, rows, step, all_rows):
        self.rows, self.step, self.all_rows, self.shared = rows, step, all_rows, {}


def _decoder_rows(cfg, x, seq, step, mode, side):
    """``(DecoderRows, all_rows)`` of this forward: engine = the published step plan (all rows in a captured
    decode step, where the floor cannot bind); an engine forward WITHOUT paged metadata (vLLM's memory profiling
    with ``skip_attn``: no ``EngineStep``) runs the full-sequence composition on every row, segment start 0 (the
    dummy plan's program); trainer = the rollout's segments (``training_ced``)."""
    from .program import DecoderRows, DecoderSegments
    T, dev = x.shape[1], x.device
    pos = seq.positions.reshape(-1)[:T].to(torch.int64)
    if step is not None:
        if step.graph:
            idx = torch.arange(T, device=dev)
            seq_of = step.batch.tok_seq[:T].long()
            return DecoderRows(idx, pos, seq_of, torch.zeros_like(pos), None, seq.cu_seqlens), True
        from .engine_ced import current_plan
        plan = current_plan()
        if plan is None or not 0 < plan.tokens <= T:       # T may carry padding rows after the scheduled tokens
            raise RuntimeError(f"CED engine step of {T} tokens without its decoder-row plan (published: "
                               f"{None if plan is None else plan.tokens} tokens; model state prepare_inputs / "
                               f"prepare_dummy_inputs)")
        # all rows only without padding rows: a padded PIECEWISE step gathers its decoder rows (the padding rows keep
        # the identity carry), so no decoder kernel sees a row outside vLLM's metadata
        return (DecoderRows.from_plan(plan, pos, step.batch.tok_seq[:T], seq.cu_seqlens),
                plan.all_rows and plan.tokens == T)
    from .training_ced import segments_for_forward
    cu = (seq.cu_seqlens_cpu if seq.cu_seqlens_cpu is not None else seq.cu_seqlens.cpu()).tolist()
    if side == "engine":
        segments = DecoderSegments.all_rows(cu)
    else:
        segments = segments_for_forward(mode, cu, cfg.replay_window)
    if tuple(segments.cu_seqlens) != tuple(cu):
        raise ValueError("decoder segments describe another packing than the forward's")
    if len(set(segments.rows.tolist())) != len(segments):
        raise ValueError("CED_OVERLAPPING_SEGMENTS: segments that share positions (multi-turn) need the per-segment "
                         "output form; this binding scatters decoder rows back onto the tokens")
    rows = DecoderRows.from_segments(segments, cfg.replay_window, dev)
    return rows, len(segments) == T


def _gather(ctx, tensors):
    if ctx.all_rows:
        return tensors
    from isoexec.ops.residual.mhc_sm100.ops import gather_rows
    return gather_rows(ctx.rows.index, tensors)


class DecoderBlock(V41Layer):
    """Decoder layer ``index`` on the decoder segments' rows: the FULL layer's query side (index queries over the
    whole prefix of the decoder global KV, the candidate pool), REUSE, REINDEX (masked to the pool). Every
    decoder layer after the FULL one truncates its window at the segment start (``LayerKind.window_clamp``).

    Captured form: per-token tensors in and out (``[1, T, ...]``); the block computes the decoder rows of the
    forward (``_CedForward``) and writes them back, the other tokens passing through with an identity deferred
    expand (``out = 0``, ``post = 0``, ``comb = I``). Inputs: the carry ``(x, pre, out, post, comb)`` (for the FULL
    layer: ``decoder.entry``'s ``(streams, pre_a, h, post_a, comb_a)``) and the decoder global KV
    ``(main_kv, index_codes, index_sf, window_kv)`` (trainer: dequantized rows; engine: in the pages)."""

    def __init__(self, cfg: CedGeometry, index: int, **kw) -> None:
        if cfg.kind(index).stack != "decoder":
            raise ValueError(f"layer {index} is an encoder layer")
        super().__init__(cfg, index, **kw)

    def consumes(self) -> tuple[str, ...]:
        return ()          # stack-level wiring: decoder.entry's global KV and the decoder chain

    def produces(self) -> tuple[str, ...]:
        return ()

    def forward(self, x, pre, out, post, comb, main_kv, index_codes, index_sf, window_kv, seq=None):
        from .piecewise import DECODER_ALL_ROWS_MAX, core_capturing, run
        if core_capturing() and x.shape[-3] <= DECODER_ALL_ROWS_MAX:
            return self._forward_all_rows(x, pre, out, post, comb, main_kv, index_codes, index_sf, window_kv, seq)
        return run(self, self._forward_impl, (x, pre, out, post, comb, main_kv, index_codes, index_sf, window_kv),
                   {"seq": seq})

    def _forward_all_rows(self, x, pre, out, post, comb, main_kv, index_codes, index_sf, window_kv, seq):
        """The block while a PIECEWISE segment of at most ``piecewise.DECODER_ALL_ROWS_MAX`` tokens records
        (``piecewise`` ``core``): the boundaries and the experts run on EVERY token inside the segment (their kernels
        are row-independent, so a decoder row gets the bits of the row-space form); the decoder-row selection, the
        decoder attention on exactly those rows and the scatter of its output onto the tokens (zero elsewhere) are
        one eager break (``_decoder_core``) that reads the step's plan live. Tokens that are not decoder rows carry
        values nothing reads (their logits are a discarded partial-prefill sample; no decoder KV is written for
        them)."""
        if self.placement is None:
            raise PlacementError(f"{type(self).__name__}: apply_placement has not run")
        cfg = self.cfg
        self._capture_routes()
        if x.is_cuda:
            from .providers import install_projection_gemm
            install_projection_gemm()
        S, H = cfg.hc_mult, cfg.hidden_size
        xr, pr, outr, postr, combr = (x.reshape(-1, S, H), pre.reshape(-1, S), out.reshape(-1, H),
                                      post.reshape(-1, S), comb.reshape(-1, S, S))
        if self.kind.is_entry:
            pre_a, h, post, comb, ev = pr, outr, postr, combr, None
        else:
            xr, (pre_a, post, comb, ev), h = boundary(cfg, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
                                                      self.attn_norm.weight, _Carry(xr, pr, None, (outr, postr, combr)),
                                                      sublayer="attn")
        ev = _join_before_break(ev)
        from .piecewise import core
        a = torch.empty_like(h)
        core(self._decoder_core, (h, main_kv, index_codes, index_sf, window_kv, seq), (a,))
        xr, (pre_f, post_f, comb_f, ev), h2 = boundary(cfg, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
                                                       self.ffn_norm.weight, _Carry(xr, pre_a, ev, (a, post, comb)),
                                                       sublayer="ffn")
        y = self.moe_forward(self, h2.reshape(-1, H)).view(h2.shape)
        _Carry(xr, pre_f, ev).join()
        lead = x.shape[:-2]
        return (xr.view(*lead, S, H), pre_f.view(*lead, S), y.view(*lead, H), post_f.view(*lead, S),
                comb_f.view(*lead, S, S))

    def _decoder_core(self, h, main_kv, index_codes, index_sf, window_kv, seq):
        """Break body of ``_forward_all_rows``: ``h [T, H]`` -> the decoder attention at this step's decoder rows
        (``_CedForward`` of this forward, published by ``decoder.entry``'s break), zero on every other token."""
        from .program import DecoderGlobalKV
        ops = ced_ops()
        ctx = _CedForward.current
        if ctx is None:
            raise RuntimeError("decoder block without the decoder.entry of its forward")
        step = ctx.step
        if step is not None and not self.kind.is_entry:
            from .engine import engine_step
            step = engine_step(self.cfg, self.index, self.engine_state, seq)        # binds this layer's window
        decoder_kv = DecoderGlobalKV(main_kv=main_kv, index_k=(index_codes, index_sf), window_kv=window_kv, rows={})
        if ctx.all_rows:
            return self._decoder_attention(ops, h, ctx.rows, decoder_kv, ctx.shared, step)
        idx = ctx.rows.index
        a = self._decoder_attention(ops, h.index_select(0, idx), ctx.rows, decoder_kv, ctx.shared, step)
        full = torch.zeros(h.shape, dtype=h.dtype, device=h.device)
        if a.shape[0]:
            full.index_copy_(0, idx, a)
        return full

    def _forward_impl(self, x, pre, out, post, comb, main_kv, index_codes, index_sf, window_kv, seq=None):
        from .program import DecoderGlobalKV
        ops = ced_ops()
        if self.placement is None:
            raise PlacementError(f"{type(self).__name__}: apply_placement has not run")
        ctx = _CedForward.current
        if ctx is None:
            raise RuntimeError("decoder block without the decoder.entry of its forward")
        cfg, step = self.cfg, ctx.step
        if step is not None and not self.kind.is_entry:
            from .engine import engine_step
            step = engine_step(cfg, self.index, self.engine_state, seq)        # binds this layer's window
        if step is not None:
            self.select_ep_routes(step)
        if x.is_cuda:
            from .providers import install_projection_gemm
            install_projection_gemm()
        S, H = cfg.hc_mult, cfg.hidden_size
        xt, pt = x.reshape(-1, S, H), pre.reshape(-1, S)
        # the segment gather (G): one launch for the five per-token tensors (mhc_sm100.gather_rows)
        xr, pr, outr, postr, combr = _gather(ctx, [xt, pt, out.reshape(-1, H), post.reshape(-1, S),
                                                   comb.reshape(-1, S, S)])
        if self.kind.is_entry:
            pre_a, h, post, comb, ev = pr, outr, postr, combr, None
        else:
            xr, (pre_a, post, comb, ev), h = boundary(cfg, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
                                                      self.attn_norm.weight, _Carry(xr, pr, None, (outr, postr, combr)),
                                                      sublayer="attn")
        decoder_kv = DecoderGlobalKV(main_kv=main_kv, index_k=(index_codes, index_sf), window_kv=window_kv, rows={})
        a = self._decoder_attention(ops, h, ctx.rows, decoder_kv, ctx.shared, step)
        xr, (pre_f, post_f, comb_f, ev), h2 = boundary(cfg, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
                                                       self.ffn_norm.weight, _Carry(xr, pre_a, ev, (a, post, comb)),
                                                       sublayer="ffn")
        y = self.moe_forward(self, h2.reshape(-1, H)).view(h2.shape)
        _Carry(xr, pre_f, ev).join()
        if step is not None:
            step.note("experts", self.expert_path)
            if self.index == cfg.num_hidden_layers - 1:
                from .engine import record_program
                record_program(step)
        T = xt.shape[0]
        if ctx.all_rows:
            x_out, pre_out, y_out, post_out, comb_out = xr, pre_f, y, post_f, comb_f
        else:
            idx = ctx.rows.index
            x_out = xt.index_copy(0, idx, xr)
            pre_out = pt.index_copy(0, idx, pre_f)
            y_out = y.new_zeros((T, H)).index_copy(0, idx, y)
            post_out = post_f.new_zeros((T, S)).index_copy(0, idx, post_f)
            eye = torch.eye(S, dtype=comb_f.dtype, device=comb_f.device).expand(T, S, S)
            comb_out = eye.index_copy(0, idx, comb_f)
        lead = x.shape[:-2]
        return (x_out.view(*lead, S, H), pre_out.view(*lead, S), y_out.view(*lead, H), post_out.view(*lead, S),
                comb_out.view(*lead, S, S))

    def _decoder_attention(self, ops, h, rows, decoder_kv, shared, step):
        from isoexec.ops.attention.deepseek_v41_cache_sm100.ops import drive
        from isoexec.ops.collectives.parallel_autograd import copy_to_parallel, reduce_from_parallel
        lin, tp = self.linear["attention"], self.tp
        training = torch.is_grad_enabled()
        if training and tp.size > 1:
            h = copy_to_parallel(h, tp.group)
        gen = ops.decoder_attention_gen(self.attn, h, self.cfg, self.index, rows, decoder_kv, shared, linear_fn=lin,
                                        dequant_fn=lin.dequant_fn, tp=tp.size, step=step)
        if tp.size == 1:
            return drive(gen)
        reducer = tp.reducer()
        with tp.scope():
            if not training:
                return reducer.run(gen)
            try:
                msg = next(gen)
                while True:
                    msg = gen.send([reducer.gather_sum(s) for s in msg[1]])
            except StopIteration as stop:
                return reduce_from_parallel(stop.value, reducer)

    def abstract_output(self, x, pre, out, post, comb, main_kv, index_codes, index_sf, window_kv, seq=None):
        S, H = self.cfg.hc_mult, self.cfg.hidden_size
        lead, dev = x.shape[:-2], x.device
        return (torch.empty((*lead, S, H), dtype=torch.bfloat16, device=dev),
                torch.empty((*lead, S), dtype=torch.float32, device=dev),
                torch.empty((*lead, H), dtype=torch.bfloat16, device=dev),
                torch.empty((*lead, S), dtype=torch.float32, device=dev),
                torch.empty((*lead, S, S), dtype=torch.float32, device=dev))


class DecoderEntry(_StageMember, Component):
    """The CED projection (report eq. (1)): the decoder FULL layer's attention boundary (its mHC coefficients
    from the handoff streams, the collapse with the last encoder FFN's ``pre``, ``attn_norm``) and its token-wise
    projections: window KV, the ratio-1 compressor (decoder global KV) and the index keys, on EVERY token.
    Checkpoint names: ``layers.{b}.{hc_attn_*, attn_norm, attn.wkv, attn.kv_norm, attn.compressor.*,
    attn.indexer.wk, attn.indexer.k_norm}``.

    Input: the last encoder block's carry ``(x, pre, out, post, comb)`` (its FFN expand deferred); the settled
    streams and ``pre`` are the handoff (``EncoderOutput``). Output: ``(streams, pre_a, h, post_a, comb_a,
    main_kv [T, 512], index_codes [Tp, 128], index_sf [Tp], window_kv [T, 512])`` (``Tp``: T rounded up to 128;
    the engine's records are in the pages and these four are unwritten placeholders). It also opens the decoder
    forward (``_CedForward``): the decoder rows of the step (engine) or of the rollout's segments (trainer)."""

    PLACEMENT_BOUND = True
    NO_CONTEXT_PARALLEL = "token-wise; admitted with the CP design of the decoder blocks (briefs/CP.md)"
    READS_ROLLOUT_REQUESTS = True        # trainer: the decoder segments come from the rollout's engine requests

    def __init__(self, cfg: CedGeometry, *, recipe: str = RECIPE, device="meta") -> None:
        super().__init__()
        self.cfg, self.index, self.recipe = cfg, cfg.boundary, recipe
        _mhc_parameters(self, cfg, ("attn",), device)
        self.attn_norm = Norm(cfg.hidden_size, device=device)
        self.attn = EntryProjection(cfg, cfg.boundary, device)
        self.engine_state = None

    def tensor_axes(self) -> Mapping[str, ShardAxis]:
        return {name: replicated() for name in self._parameters}

    def state_specs(self):
        from .program import ced_states
        return _specs(ced_states(self.cfg), "decoder.entry")

    def facts(self) -> Mapping[str, Any]:
        cfg = self.cfg
        return {"kind": "deepseek_v41_decoder_entry", "recipe": self.recipe, "layer": cfg.boundary,
                "checkpoint_layer": cfg.layer_sources[cfg.boundary], "ced_prefill": self.ced_prefill,
                "hidden": cfg.hidden_size, "heads": cfg.num_attention_heads, "head_dim": cfg.head_dim,
                "index_head_dim": cfg.index_head_dim, "experts": cfg.n_routed_experts,
                "intermediate": cfg.moe_intermediate_size, "rms_eps": cfg.rms_norm_eps, "hc_eps": cfg.hc_eps,
                "hc_iterations": cfg.hc_sinkhorn_iters}

    def _place(self, placement: Placement) -> None:
        site = self.require_placed("_place")
        self._admit(placement, site)
        self._lin = PROVIDERS[self.recipe].linear()
        if placement.side == "trainer":
            from .training import mark_trainable
            mark_trainable(self)

    def bind_engine_state(self, *, vllm_config: Any, layer_prefix: str) -> None:
        """Register the boundary layer's cache layers (window, decoder global KV, index keys) under its layer id."""
        from .engine import bind_row_caches
        if self.placement is None or self.placement.side != "engine":
            raise PlacementError(f"{layer_prefix}: engine state binds an engine-placed entry")
        self.engine_state = bind_row_caches(self.cfg, self.index, vllm_config,
                                            canonical_prefix(layer_prefix, self.index))
        _agree_break_mode(self.placement)

    def refresh_derived_weights(self) -> int:
        from isoexec.ops.attention.deepseek_v41_cache_sm100.ops import refresh_derived_weights
        n = refresh_derived_weights(self)
        self.zero_engine_caches()          # as every row does at an update commit
        return n

    def zero_engine_caches(self) -> int:
        return _zero_engine_caches(getattr(self, "engine_state", None))

    def forward(self, x, pre, out=None, post=None, comb=None, seq=None):
        from .piecewise import core_capturing, run
        if core_capturing():
            return self._forward_core(x, pre, out, post, comb, seq)
        return run(self, self._forward_impl, (x, pre, out, post, comb), {"seq": seq})

    def _forward_core(self, x, pre, out, post, comb, seq):
        """While a PIECEWISE segment records (``piecewise`` ``core``): the attention boundary inside the segment, then
        ONE eager break (``_entry_core``) for the token-wise projection into the pages and the step's decoder rows."""
        if self.placement is None:
            raise PlacementError(f"{type(self).__name__}: apply_placement has not run")
        cfg = self.cfg
        if x.is_cuda:
            from .providers import install_projection_gemm
            install_projection_gemm()
        S, H = cfg.hc_mult, cfg.hidden_size
        deferred = None if out is None else (out.reshape(-1, H), post.reshape(-1, S), comb.reshape(-1, S, S))
        xs, (pre_a, post_a, comb_a, ev), h = boundary(cfg, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
                                                      self.attn_norm.weight,
                                                      _Carry(x.reshape(-1, S, H), pre.reshape(-1, S), None, deferred),
                                                      sublayer="attn")
        _Carry(xs, pre_a, ev).join()
        from .piecewise import core
        core(self._entry_core, (x, h, seq), ())
        main_kv, (codes, sf), window_kv = self._placeholders(xs.shape[0], xs.device)
        lead = x.shape[:-2]
        return (xs.view(*lead, S, H), pre_a.view(*lead, S), h.view(*lead, H), post_a.view(*lead, S),
                comb_a.view(*lead, S, S), main_kv, codes, sf, window_kv)

    def _entry_core(self, x, h, seq):
        """Break body of ``_forward_core``: the step (live metadata), the projection of its real rows into the pages,
        and the decoder rows of the step (``_CedForward``, read by the decoder blocks' breaks of this forward)."""
        from .engine import engine_step
        ops = ced_ops()
        cfg = self.cfg
        step = engine_step(cfg, self.index, self.engine_state, seq)       # binds layer b's window
        positions = seq.positions.reshape(-1)[:h.shape[0]]
        proj = ops.entry_projection(self.attn, _step_rows(step, h), cfg, self.index, positions, linear_fn=self._lin,
                                    step=step)
        if proj is not None:
            raise RuntimeError("the engine's decoder.entry projection writes the pages and returns nothing")
        rows, all_rows = _decoder_rows(cfg, x, seq, step, self.ced_prefill, self.placement.side)
        _CedForward.current = _CedForward(rows, step, all_rows)

    def _forward_impl(self, x, pre, out=None, post=None, comb=None, seq=None):
        ops = ced_ops()
        if self.placement is None:
            raise PlacementError(f"{type(self).__name__}: apply_placement has not run")
        cfg = self.cfg
        if x.shape[0] != 1:
            raise ValueError("decoder.entry takes the packed tokens of one forward ([1, T, ...])")
        step = None
        if self.placement.side == "engine":
            if self.engine_state is None:
                raise RuntimeError("bind_engine_state has not run for decoder.entry")
            from .engine import engine_step
            step = engine_step(cfg, self.index, self.engine_state, seq)       # binds layer b's window
        if x.is_cuda:
            from .providers import install_projection_gemm
            install_projection_gemm()
        S, H = cfg.hc_mult, cfg.hidden_size
        deferred = None if out is None else (out.reshape(-1, H), post.reshape(-1, S), comb.reshape(-1, S, S))
        xs, (pre_a, post_a, comb_a, ev), h = boundary(cfg, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
                                                      self.attn_norm.weight,
                                                      _Carry(x.reshape(-1, S, H), pre.reshape(-1, S), None, deferred),
                                                      sublayer="attn")
        _Carry(xs, pre_a, ev).join()             # the coefficients exist before the segment gather (R10)
        T = xs.shape[0]
        positions = seq.positions.reshape(-1)[:T]
        rows_in = _step_rows(step, h) if step is not None else _tp_sum_backward(h, self.placement)
        proj = ops.entry_projection(self.attn, rows_in, cfg, self.index, positions, linear_fn=self._lin, step=step)
        rows, all_rows = _decoder_rows(cfg, x, seq, step, self.ced_prefill, self.placement.side)
        _CedForward.current = _CedForward(rows, step, all_rows)
        if proj is None:
            main_kv, (codes, sf), window_kv = self._placeholders(T, xs.device)
        else:
            main_kv, (codes, sf), window_kv = proj
        lead = x.shape[:-2]
        return (xs.view(*lead, S, H), pre_a.view(*lead, S), h.view(*lead, H), post_a.view(*lead, S),
                comb_a.view(*lead, S, S), main_kv, codes, sf, window_kv)

    def _placeholders(self, T, device):
        """Engine: the global-KV outputs live in the pages; the captured boundary needs tensors of the abstract
        shapes, allocated outside any capture and kept per token count."""
        cache = self.__dict__.setdefault("_placeholder_cache", {})
        hit = cache.get((T, device))
        if hit is None:
            shapes = self._kv_shapes(T)
            made = tuple(torch.empty(shape, dtype=dtype, device=device) for shape, dtype in shapes)
            if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                hit = made
            else:
                hit = cache[(T, device)] = made
        return hit[0], (hit[1], hit[2]), hit[3]

    def _kv_shapes(self, T):
        Tp = -(-T // 128) * 128
        return [((T, self.cfg.head_dim), torch.bfloat16), ((Tp, self.cfg.index_head_dim), torch.uint8),
                ((Tp,), torch.int32), ((T, self.cfg.head_dim), torch.bfloat16)]

    def abstract_output(self, x, pre, out=None, post=None, comb=None, seq=None):
        S, H = self.cfg.hc_mult, self.cfg.hidden_size
        lead, dev = x.shape[:-2], x.device
        T = x.shape[-3]
        kv = [torch.empty(shape, dtype=dtype, device=dev) for shape, dtype in self._kv_shapes(T)]
        return (torch.empty((*lead, S, H), dtype=torch.bfloat16, device=dev),
                torch.empty((*lead, S), dtype=torch.float32, device=dev),
                torch.empty((*lead, H), dtype=torch.bfloat16, device=dev),
                torch.empty((*lead, S), dtype=torch.float32, device=dev),
                torch.empty((*lead, S, S), dtype=torch.float32, device=dev), *kv)


__all__ = [
    "DECODER_GPU_PENDING",
    "STATE_KEYS",
    "DecoderBlock",
    "DecoderEntry",
    "EncoderBlock",
    "V41Layer",
    "boundary",
]
