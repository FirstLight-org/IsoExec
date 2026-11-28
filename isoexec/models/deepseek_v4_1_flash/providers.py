"""Operation providers of the ``sm100_bf16_actor`` recipe, selected once at placement.

A block resolves its seams from ``PROVIDERS[recipe]``:

* ``linear``: one provider per projection kind (``attention``, ``indexer``, ``engram``), the
  ``linear_fn(holder, x)`` / ``dequant_fn(holder)`` pair the V4.1 attention composition takes. The actor
  recipe runs a BF16 projection on vendored DeepGEMM's BF16 GEMM when its weight's published ``mm_sm100``
  segment count is 1 (``mm/deepgemm_sm100/bf16.py``: the same one-chain program; under segment policy v4 every
  weight) and the grouped ``wo_a`` on DeepGEMM's grouped einsum (``grouped_deepgemm``, the ``moe_sm100.fc1``
  program), narrow weights (N <= ``NARROW_N``) above ``NARROW_ROWS`` rows and every other projection on the
  batch-invariant ``mm`` provider chain with ``mm_sm100`` on top (``install_projection_gemm``); ``wo_b`` always
  takes the composition's fixed 8-leaf tree. The composition runs its explicit trees (``wo_b``'s leaves) as
  DeepGEMM segment GEMMs plus the same fold (``mm/deepgemm_sm100/segtree.py``) where that is faster. Same bits as
  the incumbent plans. Projections that read the same input (``SHARED_INPUT``: ``wq_a | wkv``, native's
  ``fused_wqa_wkv``) run as one GEMM over their packed weight (``BF16Linear.packed``) once the engine has packed
  them (``pack_rows``); the same bits per output column.
* ``moe``: one expert-execution provider per side, ``moe_forward(block, rows) -> rows``. The trainer runs
  the grouped BF16 experts (custom backward) and, under EP, the fixed-order exchange with backward. The
  engine runs canonical BF16 MegaMoE (``moe/deepseek_v41_sm100/mega.py``; at EP > 1 once the block's MegaMoE
  workspace is prepared, ``prepare_expert_workspace``), else the trainer's forward with the static-capacity
  routes of graph buckets under EP. Same bits on every path. The engine's router (``mega.route_select``:
  ``router_sm100.route``) evaluates the logits program as its segment GEMMs plus the tree-folding selection,
  the trainer's (``route_fused``) in the fused kernel: the same segments, chains and tree, so the same routing
  table at every row count.
* ``tp_reducer``: the FP32 subtree roots of ``wo_b`` on the symmetric-memory tree reducer
  (``collectives/detreduce_sm100``) whenever autograd does not track them, the rank-ordered all-gather
  otherwise; both are ``leaf_tree(stack(all_gather(root)))`` bit for bit.

Another recipe plugs in by adding its entry; components read nothing recipe-specific beyond this table.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .config import FORWARD_PROGRAM, RECIPE

_GEMM_LIB = None


def install_projection_gemm():
    """Route BF16 ``aten::mm``/``addmm`` (every BF16 linear) to the batch-invariant provider chain with
    ``mm_sm100`` on top, and ``aten::bmm`` to ``mm_sm100/general``'s batched path. Idempotent, process wide;
    call before the first GEMM. Backward matmuls of the actor recipe run on the same chain."""
    global _GEMM_LIB
    if _GEMM_LIB is not None:
        return _GEMM_LIB
    from isoexec.ops.mm import mm_batch_invariant as bi
    from isoexec.ops.mm.mm_sm100.general import ops as mm_general
    from isoexec.ops.mm.mm_sm100.integration import install_mm_sm100

    os.environ["ISOEXEC_MM_SM100"] = "1"
    if not install_mm_sm100():
        raise RuntimeError("mm_sm100 provider did not install")

    from isoexec.ops.mm.mm_sm100 import ops as mm_ops

    def mm(a, b):
        if a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16:
            # mm_sm100's recipe shapes on it; every other bf16 layout (the projections' dgrad and wgrad,
            # K < 256, narrow N) on mm_sm100/general; what neither serves on the incumbent kernel
            if mm_ops.supported_shape(a, b):
                return bi.mm_batch_invariant(a, b)
            if mm_general.supported(a, b):
                return mm_general.mm(a, b)
            return bi._isoexec_stock_matmul_persistent(a, b)
        # other dtypes keep the stock kernel (the vendored Triton kernel would run FP32 as TF32)
        return torch.ops.aten.mm.out(a, b, out=a.new_empty((a.shape[0], b.shape[1])))

    def addmm(bias, a, b, *, beta=1, alpha=1):
        if a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16 and beta == 1 and alpha == 1:
            return bi._isoexec_stock_matmul_persistent(a, b, bias=bias)
        return torch.ops.aten.addmm.out(bias, a, b, beta=beta, alpha=alpha,
                                        out=a.new_empty((a.shape[0], b.shape[1])))

    def bmm(a, b):
        if mm_general.supported(a, b) and a.dim() == 3:
            return mm_general.bmm(a, b)
        return torch.ops.aten.bmm.out(a, b, out=a.new_empty((a.shape[0], a.shape[1], b.shape[2])))

    def bmm_dtype(a, b, out_dtype):
        if out_dtype in (torch.bfloat16, torch.float32) and mm_general.supported(a, b) and a.dim() == 3:
            return mm_general.bmm(a, b, out_dtype=out_dtype)
        return torch.ops.aten.bmm.dtype_out(a, b, out_dtype,
                                            out=a.new_empty((a.shape[0], a.shape[1], b.shape[2]), dtype=out_dtype))

    lib = torch.library.Library("aten", "IMPL")
    lib.impl("aten::mm", mm, "CUDA")
    lib.impl("aten::addmm", addmm, "CUDA")
    lib.impl("aten::bmm", bmm, "CUDA")
    lib.impl("aten::bmm.dtype", bmm_dtype, "CUDA")
    _GEMM_LIB = lib
    return lib


#: Narrow weights (the indexer's ``wq_b``, the TP8 ``wq_b`` shard: N = 4096) above this many rows stay on mm_sm100
#: (measured faster there with cold weights: 16.8 vs 18.2 us at M = 2048). The same one-chain program, so the choice
#: is per call and bit-neutral.
NARROW_N, NARROW_ROWS = 4096, 256

#: Weights ``(N, K)`` that run on mm_sm100 at every row count: the packed ``wq_a | wkv``, measured faster there than
#: on DeepGEMM at every M (B200, graphs, cold weights: 8.5 vs 10.6 us at M = 1, 10.2 vs 11.2 at M = 128, 115.4 vs
#: 117.3 at M = 8192; dense lane round 1). The same one-chain program (the DeepGEMM gate), so the choice is a
#: fixed placement by weight shape and bit-neutral. (1856, 5120) is wq_a | wkv | the indexer's padded weights_proj
#: (index-source layers): 8.8 vs DeepGEMM 12.5 us at M = 1 and 113.2 us at M = 8192 (dense lane r4-1).
MM_SM100_WEIGHTS = frozenset({(1792, 5120), (1856, 5120)})


class BF16Linear:
    """``F.linear(x, holder.weight)``: DeepGEMM's BF16 GEMM for the weights it serves, the installed provider
    chain otherwise (``mm/deepgemm_sm100/bf16.linear``); ``dequant_fn`` returns the BF16 master.

    It deliberately has no ``supports_leaf_tree``: ``wo_b`` then runs the composition's fixed 8-leaf tree.
    ``grouped_deepgemm``: the grouped ``wo_a`` forward on DeepGEMM's grouped einsum (the same program).
    """

    implementation = "isoexec_bf16_segment_tree"
    grouped_deepgemm = True

    def __call__(self, holder, x):
        w = holder.weight
        if x.is_cuda:
            install_projection_gemm()
            if (tuple(w.shape) not in MM_SM100_WEIGHTS
                    and not (w.shape[0] <= NARROW_N and x.numel() // x.shape[-1] > NARROW_ROWS)):
                from isoexec.ops.mm.deepgemm_sm100 import bf16
                return bf16.linear(x, w)
        # ``aten::mm`` itself, as ``F.linear`` without a bias decomposes (fold the leading dims, ``mm`` with
        # the transposed weight): a runtime that overrides the composite ops (vLLM's batch-invariant mode
        # registers ``aten::linear`` / ``aten::matmul``) cannot move this projection off the provider chain.
        lead = x.shape[:-1]
        return torch.mm(x.reshape(-1, x.shape[-1]), w.t()).view(*lead, w.shape[0])

    def dequant_fn(self, holder):
        return holder.weight

    def packs(self, holders) -> bool:
        """True when ``packed(holders, x)`` runs ONE projection: the members' weights are adjacent row blocks of
        one storage (``pack_rows``)."""
        return adjacent_rows([h.weight for h in holders]) is not None

    def packed(self, holders, x):
        """``tuple(self(h, x) for h in holders)`` for projections that read the same ``x``, as ONE projection
        over the packed weight ``[sum N_i, K]`` when the members are adjacent row blocks of one storage, else
        one projection per member. Output column block i is member i's output bit for bit: every output column
        is its own one-chain K reduction whose program depends on K only (segment policy v4), never on which
        columns share the launch. The members' outputs are then column views of one ``[..., sum N_i]`` tensor
        (row stride ``sum N_i``)."""
        ws = [h.weight for h in holders]
        w = adjacent_rows(ws)
        if w is None or (torch.is_grad_enabled() and (x.requires_grad or any(p.requires_grad for p in ws))):
            return tuple(self(h, x) for h in holders)
        return self(_Rows(w), x).split([p.shape[0] for p in ws], dim=-1)


class _Rows:
    """A linear-provider operand holding a weight (the provider reads ``.weight``)."""

    def __init__(self, weight):
        self.weight = weight


# ------------------------------------------------------------------------------ shared-input projections
#: Projections of one attention row that read the same input (the layer's attention input ``x``), packed so that
#: one GEMM serves them, in output-column order: native's ``fused_wqa_wkv``.
SHARED_INPUT = (("wq_a", "wkv"),)


def adjacent_rows(weights):
    """The ``[sum N_i, K]`` tensor whose consecutive row blocks are ``weights`` when they already tile one storage
    contiguously in this order (same dtype, device and K), else None."""
    w0 = weights[0]
    if len(weights) < 2 or w0.dim() != 2:
        return None
    K, base, expect = w0.shape[1], w0.untyped_storage().data_ptr(), w0.data_ptr()
    for w in weights:
        if (w.dim() != 2 or w.shape[1] != K or not w.is_contiguous() or w.dtype != w0.dtype
                or w.device != w0.device or w.untyped_storage().data_ptr() != base or w.data_ptr() != expect):
            return None
        expect += w.numel() * w.element_size()
    out = torch.empty(0, dtype=w0.dtype, device=w0.device)
    out.set_(w0.untyped_storage(), w0.storage_offset(), (sum(w.shape[0] for w in weights), K), (K, 1))
    return out


def pack_rows(params):
    """Re-point each parameter at its row block of ONE contiguous ``[sum N_i, K]`` storage (one copy of the bytes,
    values unchanged; later in-place writes such as weight updates land in it) and return that tensor; an alias,
    no copy, when they already are adjacent. None (nothing changed) for parameters that cannot share a storage
    (ragged K, mixed dtype or device, meta)."""
    out = adjacent_rows(params)
    if out is not None:
        return out
    p0 = params[0]
    if (len(params) < 2 or any(p.dim() != 2 or p.shape[1] != p0.shape[1] or p.dtype != p0.dtype
                               or p.device != p0.device or p.is_meta for p in params)):
        return None
    buf = torch.empty((sum(p.shape[0] for p in params), p0.shape[1]), dtype=p0.dtype, device=p0.device)
    row = 0
    with torch.no_grad():
        for p in params:
            view = buf[row:row + p.shape[0]]
            view.copy_(p.detach())
            p.data = view
            row += p.shape[0]
    return adjacent_rows(params)


# ------------------------------------------------------------------------------------------- MoE
def _kw(cfg):
    return dict(top_k=cfg.num_experts_per_tok, routed_scale=cfg.routed_scaling_factor, limit=cfg.swiglu_limit)


def _local_moe(block, rows):
    """EP = 1: grouped BF16 experts; the custom backward whenever autograd records."""
    gate, (gate_up, down) = block.ffn.gate, block.packed_experts()
    if torch.is_grad_enabled() and (rows.requires_grad or gate_up.requires_grad):
        from isoexec.ops.moe.deepseek_v41_sm100.training import moe_train
        return moe_train(rows, gate.weight, gate.bias, gate_up, down, **_kw(block.cfg))
    from isoexec.ops.moe.deepseek_v41_sm100.ops import moe_forward
    return moe_forward(rows, gate.weight, gate.bias, gate_up, down, **_kw(block.cfg))


def home_rows(tokens: int, world: int) -> list[tuple[int, int]]:
    """Canonical contiguous partition of replicated token rows into EP homes (ragged and empty homes)."""
    return [(tokens * r // world, tokens * (r + 1) // world) for r in range(world)]


def _ep_moe(block, rows):
    """EP > 1: each replicated row has one home rank; the fixed-order exchange runs the routed experts and
    completed BF16 rows travel back without arithmetic. Static routes (graph buckets, no autograd) use the
    prepared capacity, else host-counted routes."""
    from isoexec.ops.collectives.parallel_autograd import gather_home_rows, scatter_home_rows
    from isoexec.ops.moe.deepseek_v41_sm100.ep import ep_moe_forward, ep_moe_train
    ep = block.ep
    comm = ep.comm()
    gate, (gate_up, down) = block.ffn.gate, block.packed_experts()
    spans = home_rows(rows.shape[0], ep.size)
    home = rows if ep.partitioned else scatter_home_rows(rows, spans, ep.group)
    kw = dict(_kw(block.cfg), emap=ep.expert_map, comm=comm)
    if not getattr(comm, "static_routes", False) and rows.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("a captured EP step must run the static-capacity routes (host-counted routes sync)")
    if getattr(comm, "static_routes", False):
        if torch.is_grad_enabled():
            raise NotImplementedError("fixed-capacity BF16 EP transport requires no_grad inference")
        capacity = ep.static_home_rows
        if capacity is None:
            raise RuntimeError("static EP routes need prepare_static_ep(max_home_rows=...)")
        if not ep.partitioned:
            # every EP peer derives the same bucket capacity from the agreed padded token count
            # (ceil(T / ep) bounds every canonical home), bounded by the prepared one
            capacity = min(capacity, -(-rows.shape[0] // ep.size))
        count = home.shape[0]
        if count > capacity:
            raise ValueError("BF16 EP home rows exceed prepared graph capacity")
        home = torch.cat((home, home.new_zeros((capacity - count, home.shape[1]))), 0)
        result = ep_moe_forward(home, gate.weight, gate.bias, gate_up, down, **kw)[:count]
    else:
        result = ep_moe_train(home, gate.weight, gate.bias, gate_up, down, **kw)
    return result if ep.partitioned else gather_home_rows(result, spans, ep.group)


@dataclass(frozen=True)
class Providers:
    program: str
    linear: Callable[[], Any]
    moe: Callable[[str, int], Callable]


def _tracked(*tensors):
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def padding_rows(tokens: int):
    """vLLM's padding flags of this forward's ``tokens`` rows (``forward_context.is_padding``: the padding rows of a
    padded graph step, set from the scheduler output identically on every rank, captured with every row flagged), or
    None outside a forward context / when vLLM does not flag padding (``VLLM_MOE_SKIP_PADDING=0``)."""
    from vllm.forward_context import get_forward_context, is_forward_context_available
    if not is_forward_context_available():
        return None
    flags = getattr(get_forward_context(), "is_padding", None)
    if flags is None or flags.shape[0] < tokens:
        return None
    return flags[:tokens]


def _engine_local_moe(block, rows):
    """EP = 1: canonical BF16 MegaMoE (one fused kernel; its workspace is created and warmed on the first eager
    call); host tensors and grad-enabled passes run the grouped experts."""
    gate, (gate_up, down) = block.ffn.gate, block.packed_experts()
    if not rows.is_cuda or _tracked(rows, gate.weight, gate_up, down):
        block.expert_path = "grouped"
        return _local_moe(block, rows)
    from isoexec.ops.moe.deepseek_v41_sm100.ops import moe_forward_mega
    block.expert_path = "mega"
    return moe_forward_mega(rows, gate.weight, gate.bias, gate_up, down, skip=padding_rows(rows.shape[0]),
                            **_kw(block.cfg))


def _mega_width(block, home):
    """Rows per rank of a partitioned MegaMoE call, equal on every EP rank: the prepared capacity in captured
    steps (static routes), else the widest home agreed by one all-reduce (an eager step's host sync)."""
    ep = block.ep
    if getattr(ep.comm(), "static_routes", False):
        if home.shape[0] > ep.static_home_rows:
            raise ValueError("BF16 EP home rows exceed prepared graph capacity")
        return ep.static_home_rows
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("a captured partitioned MegaMoE step needs prepare_static_ep and static routes")
    import torch.distributed as dist
    agreed = torch.tensor([home.shape[0]], dtype=torch.int64, device=home.device)
    dist.all_reduce(agreed, op=dist.ReduceOp.MAX, group=ep.group)
    return int(agreed.item())


def _engine_ep_moe(block, rows):
    """EP > 1 on canonical BF16 MegaMoE once the block's workspace is prepared: replicated rows feed this rank's
    canonical home slice and get every slice back through the fused NVLink gather (no host sync; the width is
    ceil(M / ep) on every rank); partitioned rows feed themselves. Otherwise (no workspace, grad-enabled) the
    fixed-order exchange."""
    ep = block.ep
    gate, (gate_up, down) = block.ffn.gate, block.packed_experts()
    if ep.mega is None or _tracked(rows, gate.weight, gate_up, down):
        block.expert_path = "fixed_order"
        return _ep_moe(block, rows)
    from isoexec.ops.moe.deepseek_v41_sm100.ep import mega_ep_forward
    block.expert_path = "mega"
    kw = dict(_kw(block.cfg), emap=ep.expert_map, ws=ep.mega)
    if ep.partitioned:
        return mega_ep_forward(rows, gate.weight, gate.bias, gate_up, down, width=_mega_width(block, rows), **kw)
    if rows.shape[0] == 0:
        # replicated rows: every EP rank holds the same (empty) set, so all skip the collective together (a step
        # with no rows for this block, e.g. a prefill chunk before a long prompt's decoder segment)
        return rows.new_empty((0, rows.shape[-1]))
    spans = home_rows(rows.shape[0], ep.size)
    lo, hi = spans[ep.rank]
    # padding rows of a padded graph step get no routed expert (native vLLM's VLLM_MOE_SKIP_PADDING): every EP rank
    # reads the same scheduler-derived flags, so the in-kernel route counts stay group-consistent
    skip = padding_rows(rows.shape[0])
    return mega_ep_forward(rows[lo:hi], gate.weight, gate.bias, gate_up, down, width=max(b - a for a, b in spans),
                           out_rows=rows.shape[0], skip=None if skip is None else skip[lo:hi], **kw)


# Captured engine steps are device-sized: MegaMoE agrees its width without a host sync, and the fixed-order
# exchange runs the static-capacity routes (the row selects them for every captured forward and the
# host-counted routes refuse a capture).
_engine_ep_moe.graph_capturable = True


def _actor_moe(side: str, ep: int) -> Callable:
    # The trainer keeps the grouped experts (its backward reads their stage tensors); the engine runs MegaMoE.
    if side == "engine":
        return _engine_local_moe if ep == 1 else _engine_ep_moe
    return _local_moe if ep == 1 else _ep_moe


def prepare_expert_workspace(group, device, cfg):
    """The BF16 MegaMoE workspace of an EP group (pooled per group, device and geometry). Collective at EP > 1:
    every EP rank calls it outside capture, in the same order."""
    from isoexec.ops.moe.deepseek_v41_sm100 import mega
    return mega.Workspace.prepare(group, device, cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.hidden_size,
                                  cfg.moe_intermediate_size)


PROVIDERS = {RECIPE: Providers(program=FORWARD_PROGRAM, linear=BF16Linear, moe=_actor_moe)}


def vocab_shard_is_bit_neutral(cfg, tp: int) -> bool:
    """The FP32 head (``LMHead(logits_dtype="fp32")``, ``logprobs.head`` ``fp32_root``) computes this rank's
    contiguous vocab shard, ``linear(h, w[rows], root_fp32=True)``, and the shards are gathered in rank order.
    A logit's bits are fixed by the segment count S of the reduction (mm_sm100: every plan at one S gives the same
    bits, rows are independent). Under segment policy v4 S is a function of K only, so a shard is the replicated
    head whenever mm_sm100 serves its width."""
    from isoexec.ops.mm.mm_sm100 import segments
    V, K = cfg.vocab_size, cfg.hidden_size
    if tp == 1:
        return True
    return V % tp == 0 and segments.supported(K, V // tp)


# ------------------------------------------------------------------------- shared communicators
_COMMS: dict = {}
_REDUCERS: dict = {}


def ep_comm(group):
    """One fixed-order EP transport per group, shared by every block (one static-route switch)."""
    key = id(group)
    if key not in _COMMS:
        from isoexec.ops.moe.ep_fixed_order import DistComm
        _COMMS[key] = (group, DistComm(group))
    return _COMMS[key][1]


#: Fallback switches (cluster bring-up): each names the known-correct NCCL realization of a symmetric-memory fast
#: path, read once when the component is built and agreed by its group (every rank takes the same path; same bits).
TP_REDUCER_ENV = "ISOEXEC_DS_TP_REDUCER"        # "tree" (default: detreduce_sm100 TreeReducer) | "all_gather"
EP_EXCHANGE_ENV = "ISOEXEC_DS_EP_EXCHANGE"      # "mega" (default: MegaMoE EP over NVLink) | "fixed_order" (NCCL)


#: What each group's symmetric-memory fast path resolved to in this process (receipts for gates and logs):
#: ``{"tp_reducer": {ranks: "tree" | "all_gather:<why>"}, "ep_exchange": {ranks: "mega" | "fixed_order:<why>"}}``;
#: ``<why>`` is the group-agreed reason: the switch's env name, ``host_masters``, a peer-admission reason of
#: ``collectives/node_local.symmetric_admission`` (``multi_host``, ``shared_device``, ``overlapping_ordinals``,
#: ``no_cuda_device``) or ``SymmetricSetupSkew``. Every rank of a group records the same receipt.
TRANSPORT_RECEIPTS: dict = {"tp_reducer": {}, "ep_exchange": {}}


def record_transport(kind: str, group, choice: str) -> bool:
    """Record ``choice`` for the group; True when it is new (first row of the group, or a changed choice)."""
    from isoexec.ops.collectives.symm_setup import group_key
    try:
        key = str(group_key(group)) if group is not None else "()"
    except Exception:          # stand-in groups in CPU tests
        key = repr(group)
    new = TRANSPORT_RECEIPTS[kind].get(key) != choice
    TRANSPORT_RECEIPTS[kind][key] = choice
    return new


def fallback_forced(env: str) -> bool:
    """Whether the switch ``env`` asks for the NCCL fallback (rank-local; the caller agrees it over its group)."""
    import os
    choices = {TP_REDUCER_ENV: ("tree", "all_gather"), EP_EXCHANGE_ENV: ("mega", "fixed_order")}[env]
    value = os.environ.get(env, choices[0]).strip().lower()
    if value not in choices:
        raise ValueError(f"{env}={value!r}: expected one of {choices}")
    return value == choices[1]


class TreeScopeNotHeld(RuntimeError):
    """A tree-eligible TP reduction outside the tree's execution scope (the transport would become a per-thread,
    hence possibly per-rank, decision)."""


def note_transport_choice(what: str, group, text: str) -> None:
    """One log line per group and fast path whose NCCL fallback was selected, with the named reason."""
    from isoexec.ops.collectives.symm_setup import group_key
    try:
        key = group_key(group) if group is not None else "()"
    except Exception:          # stand-in groups in CPU tests
        key = repr(group)
    print(f"[ISOEXEC-V41] {what} for group {key}: {text}", flush=True)


class ActorTPReducer:
    """TP reducer of the actor recipe: every FP32 subtree root that autograd does not track goes through the
    symmetric-memory ``TreeReducer`` (``collectives/detreduce_sm100``: bitwise ``leaf_tree(stack(all_gather))``)
    -- rollout, logprob scoring and the trainer's forward, whose reduction runs inside
    ``parallel_autograd._Reduce.forward`` (identity backward). Payloads autograd tracks, non-FP32 payloads and
    host tensors keep the rank-ordered all-gather + the same tree (``DistributedReducer``). Same bits either way.

    The tree transport runs inside its execution scope: an owning runtime holds it (graph capture and replay),
    otherwise the caller enters ``scope()`` around its reductions. A capture by a runtime that holds no scope
    records the all-gather form (replayable without one; same bits)."""

    def __init__(self, tree, gather):
        self.tree, self.gather = tree, gather
        self.world = gather.world
        self.prepared = False
        #: False when the group agreed the all-gather form at ``prepare`` (switch, peer admission: hosts / shared
        #: GPU / ordinals, or a setup skew): every reduction takes it (the same bits; ``collectives/node_local``).
        self.tree_local = True
        #: ``ISOEXEC_DS_TP_REDUCER=all_gather`` on any rank: the whole group keeps the all-gather form.
        self.force_all_gather = fallback_forced(TP_REDUCER_ENV)

    def prepare(self, device):
        """Collective (every TP rank, outside capture, same order): the group agrees ONCE which transport it takes.
        The tree's workspace when the switch allows it on every rank and the peer admission admits the group;
        otherwise the all-gather form, by a named reason recorded in the receipt (``all_gather:<reason>``):
        ``ISOEXEC_DS_TP_REDUCER`` (the switch on some rank), ``multi_host`` / ``shared_device`` /
        ``overlapping_ordinals`` / ``no_cuda_device`` (``node_local.symmetric_admission``: every rank decides from
        the same gathered (host, device UUID, ordinal) rows, never from a local exception), or
        ``SymmetricSetupSkew`` (raised on every rank alike by ``symm_setup``'s count agreement)."""
        from isoexec.ops.collectives.node_local import NodeLocalityError, symmetric_admission
        from isoexec.ops.collectives.symm_setup import SymmetricSetupSkew

        from .block import _group_agrees
        group = getattr(self.tree, "group", None)
        if self.prepared or not self.tree_local:
            return self                     # decided by the group at its first row (every rank alike)
        if not _group_agrees(group, not self.force_all_gather):
            return self._all_gather_form(group, TP_REDUCER_ENV, "fallback switch set on some rank")
        verdict = symmetric_admission(group)
        if not verdict.symmetric:
            return self._all_gather_form(group, verdict.reason, verdict.detail)
        try:
            self.tree.prepare(device)
        except (NodeLocalityError, SymmetricSetupSkew) as exc:
            # defence in depth: both are raised on every rank alike (the admission above already excludes the
            # first; the second is symm_setup's count agreement), so every rank takes the all-gather form together
            return self._all_gather_form(group, type(exc).__name__, str(exc))
        self.prepared = True
        if record_transport("tp_reducer", group, "tree"):
            note_transport_choice("TP reducer", group, f"symmetric-memory tree ({verdict.reason}: {verdict.detail})")
        return self

    def _all_gather_form(self, group, reason: str, detail: str):
        self.tree_local = False
        if record_transport("tp_reducer", group, f"all_gather:{reason}"):
            note_transport_choice("TP reducer", group, f"NCCL all-gather form ({reason}: {detail})")
        return self

    def _tree_held(self) -> bool:
        return self.tree_local and self.tree.holds_execution_scope()

    def transport(self) -> str:
        """The transport this thread's untracked FP32 CUDA roots take now: ``"tree"`` inside the execution
        scope, else ``"all_gather"`` (a capture by a runtime that holds no scope records this form)."""
        return "tree" if self._tree_held() else "all_gather"

    def execution_scope(self, *, capture=False):
        from contextlib import nullcontext
        return self.tree.execution_scope(capture=capture) if self.tree_local else nullcontext()

    def scope(self):
        """This reducer's execution scope, unless the calling thread already holds it (an owning runtime's) or a
        capture without one is running (its reductions then take the all-gather form)."""
        from contextlib import nullcontext
        capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        if not self.tree_local or self.tree.holds_execution_scope() or capturing:
            return nullcontext()
        return self.tree.execution_scope()

    def gather_sum(self, x):
        """One TP reduction. The transport is fixed by facts every rank of the group shares, never by which thread
        happens to run the call:

        * the payload class (autograd-tracked, dtype, device): identical on every rank (same program, same row);
        * ``tree_local``: agreed by the group at ``prepare``;
        * the execution scope: every model call site reduces inside ``scope()``, which enters the tree's scope on
          the calling thread unless that thread already holds it (vLLM's V2 runner: ``execute_model`` around eager
          forwards and replays, ``capture_model`` around warmups and captures; ``profile_run`` / ``_dummy_run``
          and the trainer enter it per row on their own thread). So a tree-eligible payload on a tree group is
          ALWAYS reduced inside the scope, except during a CUDA graph capture that no scope covers (a runtime that
          installs no execution scope: every rank captures the same graphs scope-free, so every rank records the
          all-gather form; structural, the same on every rank).

        Anything else, e.g. a reduction issued from a thread that does not own the scope while another thread
        does (vLLM's dual-batch-overlap micro-batch threads, an async helper thread), or a direct call outside any
        scope, would let ranks pick different transports (DEFECT-12's failure class: a collective chosen by a
        rank-local condition): it is refused by name (``TreeScopeNotHeld``) instead of silently taking NCCL."""
        if ((torch.is_grad_enabled() and x.requires_grad) or x.dtype != torch.float32 or not x.is_cuda
                or not self.tree_local):
            return self.gather.gather_sum(x)
        if self.tree.holds_execution_scope():
            return self.tree.gather_sum(x)
        owner = getattr(self.tree, "execution_scope_owner", lambda: None)()
        capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        if capturing and owner is None:
            return self.gather.gather_sum(x)        # scope-free capture: the all-gather form on every rank
        raise TreeScopeNotHeld(
            "deepseek_v4_1_flash TP reducer: a tree-eligible reduction (untracked FP32 CUDA root, group agreed on "
            f"the symmetric-memory tree) was issued outside the tree's execution scope (scope owner: "
            f"{'thread ' + str(owner) if owner is not None else 'none'}, this thread {threading.get_ident()}, "
            f"capturing={capturing}). Taking NCCL here would let ranks choose different transports; reduce inside "
            "ActorTPReducer.scope() on the thread that owns it (or set ISOEXEC_DS_TP_REDUCER=all_gather).")

    def run(self, gen):
        try:
            msg = next(gen)
            while True:
                msg = gen.send([self.gather_sum(s) for s in msg[1]])
        except StopIteration as stop:
            return self.gather_sum(stop.value)


def tp_reducer(group):
    """The TP reducer of ``wo_b``'s FP32 subtree roots (``ActorTPReducer``), one per group."""
    key = id(group)
    if key not in _REDUCERS:
        from isoexec.ops.attention.deepseek_v41_cache_sm100.tp import DistributedReducer
        from isoexec.ops.collectives.detreduce_sm100.ops import TreeReducer
        _REDUCERS[key] = (group, ActorTPReducer(TreeReducer(group), DistributedReducer(group)))
    return _REDUCERS[key][1]


# ------------------------------------------------------------------------------- Engram hashing
def _is_prime(number: int) -> bool:
    import math
    return number >= 2 and all(number % d for d in range(2, math.isqrt(number) + 1))


def engram_hash_constants(cfg):
    """``(primes [L, N-1, heads], offsets [L, (N-1) heads], multipliers [L, N])`` int64 of the released hash
    layout (CPU); bit for bit the checkpoint's n-gram hash program."""
    import numpy as np
    layers, seen = [], set()
    for _ in cfg.released_engram_layers:
        groups = []
        for _ in range(cfg.engram_max_ngram_size - 1):
            current, primes = cfg.engram_vocab_size - 1, []
            for _ in range(cfg.engram_n_heads):
                current += 1
                while current in seen or not _is_prime(current):
                    current += 1
                seen.add(current)
                primes.append(current)
            groups.append(primes)
        layers.append(groups)
    primes = torch.tensor(layers, dtype=torch.int64)
    flat = primes.flatten(1)
    if list(flat.sum(-1).tolist()) != list(cfg.released_engram_rows):
        raise ValueError("Engram table rows do not match the derived prime layout")
    offsets = flat.cumsum(-1) - flat
    bound = max(1, (np.iinfo(np.int64).max // cfg.engram_compressed_vocab_size) // 2)
    multipliers = torch.stack([torch.from_numpy(np.random.default_rng(10007 * layer).integers(
        0, bound, size=cfg.engram_max_ngram_size, dtype=np.int64) * 2 + 1) for layer in cfg.released_engram_layers])
    return primes, offsets, multipliers


__all__ = ["ActorTPReducer", "TreeScopeNotHeld", "BF16Linear", "PROVIDERS", "Providers", "SHARED_INPUT", "adjacent_rows",
           "engram_hash_constants", "ep_comm", "home_rows", "install_projection_gemm", "pack_rows",
           "prepare_expert_workspace", "tp_reducer", "vocab_shard_is_bit_neutral"]
