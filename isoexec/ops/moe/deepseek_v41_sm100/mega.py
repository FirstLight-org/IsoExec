"""``deepseek_v41_bf16_actor_v1`` forward on vendored DeepGEMM BF16 MegaMoE (``isoexec_canonical=True``).

The SAME arithmetic program as the grouped-kernel forward (``ops.stages``), in one persistent kernel per call:

* route      unchanged (``ops.route``: router_sm100 ``sqrtsoftplus_bias_select``); ids / weights go to MegaMoE
* FC1        BF16 x BF16 -> FP32, one ascending K chain of 16-wide tcgen05 steps from a non-accumulating first
             step (k_block 0..H/64), no split-K, rounded once RNE to BF16 (= moe_sm100 ``fc1``'s stored projection)
* act        h = RNE_bf16((silu(min(g, L)) * clamp(u, -L, L)) * w), FP32, IEEE expf / divide (= ``swiglu_rows``)
* FC2        one K chain over I, root rounded RNE to BF16 (= moe_sm100 ``fc2`` with one leaf)
* combine    ascending GLOBAL expert id, then the shared expert, FP32 ``__fadd_rn`` from +0 (every contribution added),
             one BF16 RNE (= moe_sm100 ``combine`` / ``ep._fold_f32``)
The shared expert runs fused (MegaMoE's SharedLinear phases, weight 1.0: x * 1.0 is exact). Per output element the
K order does not depend on block_m, UMMA N, the 2-CTA tile, the row's pool position (integer atomics allocate rows,
they never order sums), the EP degree or the expert placement, so the outputs are bitwise the grouped recipe's at
every M and EP degree (GPU-gated: ``tests/test_actor_mega_gpu.py``, ``tests/_actor_mega_ep_child.py``).

Forward only: grad-enabled passes run the grouped kernels (``training.moe_train``, the same forward bits); a
grad-enabled call that would need autograd through MegaMoE is refused (``check_no_grad_pass``).

Weights: the masters are used IN PLACE (zero copy). The canonical BF16 specialization reads L1 in the master layout
``[E, 2I, H]`` (gate rows, then up rows) through a 5D TMA map that delivers MegaMoE's granularity-8 gate/up
interleave in shared memory, and L2 ``[E, H, I]`` as stored; routed experts are ``gate_up[:-1]`` / ``down[:-1]`` and
the shared expert ``gate_up[-1]`` / ``down[-1]`` (views). Trainer updates are therefore seen by the next call with no
re-transform.

EP. MegaMoE dispatches over a REAL EP process group (symmetric memory); every rank must pass the same per-rank row
count (tile geometry is a JIT constant chosen from it), so callers pass ``width`` (the widest home slice) and the rows
are padded with top-k ids -1 (never dispatched; their shared-expert rows are discarded). Calls above the per-rank
capacity run as equal chunks. Expert placement must be the contiguous map (rank r owns [r E/W, (r+1) E/W)) with the
shared expert at home.

Launches per call: the router (``route_select``: ``router_sm100.route``, whose ids / weights are ``ops.route``'s:
the logits program's segment GEMMs and the tree-folding selection; the
grouped dispatch plan ``route`` also builds is not used by MegaMoE and is not built), ONE prep
kernel (``moe/deepgemm_sm100/csrc/mega_ep_io.cu`` ``prep_bf16``: the rows, int64 top-k ids, weights and padding
rows written straight into MegaMoE's input buffer), MegaMoE, and at EP > 1 with replicated rows ONE gather kernel
that reassembles every rank's home slice from the ranks' symmetric output regions over NVLink (the protocol of
``moe/deepgemm_sm100/mega_ep.py``: per-rank call counter advanced by prep, release / acquire flags, a flag ahead of
the tag traps). Data movement only.

Progress. Persistent num_SMs grid with grid syncs and NVLink barriers: it must run alone on the device (no concurrent
kernel on another stream, no MPS), like the checkpoint recipe's MegaMoE (``moe/deepgemm_sm100/mega_ep``).
"""
from __future__ import annotations

import os

import torch

from isoexec.ops.moe.deepgemm_sm100.mega_ep import chunks

IMPLEMENTATION = "vendored_deepgemm_sm100_canonical_bf16_mega_v1"
MMA_TYPE = "bf16xbf16"
#: rows per rank one call serves without chunking (larger calls run as equal chunks)
MAX_ROWS_PER_RANK = 8192
#: The prep -> MegaMoE -> gather chain launches without programmatic dependent launch.
PDL = False


def check_no_grad_pass(*tensors):
    """Refuse a grad-enabled call that would need autograd through MegaMoE (forward only; grad-enabled passes run
    the grouped kernels)."""
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors):
        raise RuntimeError("BF16 MegaMoE is a no-grad forward; grad-enabled passes run the grouped kernels "
                           "(training.moe_train / ep.ep_moe_train)")


def max_rows_per_rank() -> int:
    """Rows per rank one workspace call serves (``ISOEXEC_DS_ACTOR_MEGA_ROWS_PER_RANK``); capacity only."""
    return int(os.environ.get("ISOEXEC_DS_ACTOR_MEGA_ROWS_PER_RANK", MAX_ROWS_PER_RANK))


def _dg():
    from isoexec.ops.mm.deepgemm_sm100 import ops as mm
    return mm._dg()


class _LocalGroup:
    """Single-rank SymmBuffer protocol (EP = 1): no process-global distributed group."""

    def size(self):
        return 1

    def rank(self):
        return 0

    def barrier(self):
        return None


def check_geometry(E, H, I):
    if H % 256 or I % 128:
        raise ValueError("BF16 MegaMoE needs H % 256 == 0 and I % 128 == 0 (2-CTA 128-column tiles of both layers)")


class Workspace:
    """One BF16 MegaMoE symmetric buffer per (EP group's ranks, device, geometry, rows per rank), shared by every layer
    (calls are serialized on the stream, like every IsoExec collective).

    Setup. At EP > 1 construction rendezvouses two symmetric buffers (MegaMoE's ``SymmBuffer``, then ``OutRegion``)
    through ``moe.symm_rendezvous`` and is COLLECTIVE: every EP rank prepares the same workspace, outside capture, in
    the same order. ``prepare`` never creates one lazily inside a forward at EP > 1: the engine prepares it eagerly
    after the weights load (``prepare_expert_workspace``) and a forward without one takes the fixed-order exchange.
    At EP = 1 (``ops.moe_forward_mega``) it is created on first use, which is local (no rendezvous, plain memory).
    A rank-skewed setup sequence raises ``RendezvousSkewError`` on every rank (keep the fixed-order exchange).

    Pool key. The group's RANK TUPLE (``symm_rendezvous.group_ranks``), never its size or ``id``: a buffer
    rendezvoused with one set of ranks is unusable by another group of the same size, and ``id`` of a destroyed group
    can be reused by a new group over other ranks. Two groups over the same ranks share one workspace (the same
    peers; its call counter then orders their calls, which every rank issues in the same order on one stream)."""

    _POOLS: dict = {}

    def __init__(self, group, device, num_experts, rows_per_rank, top_k, hidden, intermediate):
        dg = _dg()
        check_geometry(num_experts, hidden, intermediate)
        self.group, self.device = group, torch.device(device)
        if isinstance(group, _LocalGroup):
            self.world, self.rank = 1, 0
        else:
            import torch.distributed as dist
            self.world, self.rank = dist.get_world_size(group), dist.get_rank(group)
        if num_experts % self.world:
            raise ValueError(f"{num_experts} experts do not split over EP = {self.world}")
        if self.world > 1:
            from isoexec.ops.collectives.node_local import require_symmetric_peers

            # NVLink symmetric buffers between distinct GPUs of one host only: an EP group across hosts or with two
            # ranks on one GPU is refused by name and runs the fixed-order NCCL exchange
            require_symmetric_peers(group, "BF16 MegaMoE expert parallelism (moe/deepseek_v41_sm100)")
        self.E, self.top_k, self.hidden, self.intermediate = num_experts, top_k, hidden, intermediate
        with torch.cuda.device(self.device):
            self.buffer = dg.SymmBuffer(group, num_experts, rows_per_rank, top_k, hidden, intermediate,
                                        num_shared_experts=1, mma_type=MMA_TYPE)
        self.cap = int(self.buffer.num_max_tokens_per_rank)
        self.compiled: set[int] = set()       # block_m geometries whose JIT kernel exists in this process
        self.warmed = False
        self._block: dict[int, int] = {}
        for n in range(1, self.cap + 1):      # every per-rank row count a call can use: no host state in capture
            self.block_m(n)
        if self.world > 1:
            # second symmetric buffer: gather flags, call counter (advanced by prep), output rows
            from isoexec.ops.moe.deepgemm_sm100.mega_ep import OutRegion
            self.out = OutRegion(group, self.world, self.device, self.cap, hidden)
            self.counter = self.out.counter
        else:
            self.out = None
            self.counter = torch.zeros(1, dtype=torch.int64, device=self.device)

    @staticmethod
    def pool_key(group, device, num_experts, top_k, hidden, intermediate, rows_per_rank=None):
        """``(ranks, device, E, k, H, I, rows per rank)``: ``ranks`` is ``"local"`` at EP = 1, else the group's rank
        tuple."""
        local = group is None or isinstance(group, _LocalGroup)
        if local:
            ranks = "local"
        else:
            from isoexec.ops.moe.symm_rendezvous import group_ranks
            ranks = group_ranks(group)
        rows = max_rows_per_rank() if rows_per_rank is None else int(rows_per_rank)
        return (ranks, str(torch.device(device)), num_experts, top_k, hidden, intermediate, rows)

    @classmethod
    def prepare(cls, group, device, num_experts, top_k, hidden, intermediate, rows_per_rank=None):
        """The pooled workspace (created outside capture; collective at EP > 1: see the class docstring)."""
        device = torch.device(device)
        local = group is None or isinstance(group, _LocalGroup)
        key = cls.pool_key(group, device, num_experts, top_k, hidden, intermediate, rows_per_rank)
        ws = cls._POOLS.get(key)
        if ws is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("BF16 MegaMoE workspaces must be prepared outside CUDA graph capture")
            ws = cls(_LocalGroup() if local else group, device, num_experts, key[-1], top_k, hidden, intermediate)
            cls._POOLS[key] = ws
        return ws

    def block_m(self, n: int) -> int:
        """MegaMoE's block_m for an n-row call (a function of the agreed geometry only; cached at construction)."""
        hit = self._block.get(n)
        if hit is None:
            hit = self._block[n] = int(_dg().get_block_m_for_mega_moe(self.world, self.E, self.cap, n, self.top_k,
                                                                      MMA_TYPE))
        return hit

    def geometries(self) -> list[int]:
        """One representative per-rank row count for each distinct tile geometry (JIT specialization)."""
        seen, out = set(), []
        for n in range(1, self.cap + 1):
            b = self.block_m(n)
            if b not in seen:
                seen.add(b)
                out.append(n)
        return out


def masters(gate_up, down, shared=None):
    """Routed + shared operand views: the packed local masters ``[n + 1, 2I, H]`` / ``[n + 1, H, I]`` (shared last),
    or routed-only ``[n, ...]`` masters with ``shared=(gate_up [2I, H], down [H, I])`` (the trainer's split optimizer
    identities). No copy; contiguous inputs give contiguous views."""
    if not (gate_up.is_contiguous() and down.is_contiguous()):
        raise ValueError("BF16 MegaMoE reads the masters in place: they must be contiguous")
    if shared is not None:
        return gate_up, down, shared[0], shared[1]
    return gate_up[:-1], down[:-1], gate_up[-1], down[-1]


def _io():
    from isoexec.ops.moe.deepgemm_sm100._extension import extension
    return extension()


def route_select(x, router_w, bias, *, top_k, routed_scale):
    """``(ids int32 [T, k + 1], weights fp32 [T, k + 1])``: ``ops.route``'s router kernel alone (the shared expert as
    slot k, id E, weight 1.0); MegaMoE consumes the first k columns. No grouped plan (MegaMoE does not use it)."""
    from isoexec.ops.moe.router_sm100 import ops as router
    from .spec import ROUTING
    with torch.no_grad():
        return router.route(x, router_w, routing=ROUTING, top_k=top_k, expert_bias=bias,
                            routed_scale=routed_scale, num_shared=1)


def skip_rows(ids, weights, skip, top_k):
    """Rows flagged in ``skip`` (bool ``[n]``: vLLM's ``forward_context.is_padding``, the padding rows of a padded
    graph step) get no routed expert: their first ``top_k`` ids become -1 (never dispatched) and weights 0, as native
    vLLM's router does under ``VLLM_MOE_SKIP_PADDING``. In place, graph-safe (device ops on persistent buffers).
    Rows are independent in MegaMoE, so the other rows' bits do not change; a skipped row keeps its shared expert."""
    if skip is None or ids is None:
        return
    flag = skip[:ids.size(0)].unsqueeze(1)
    ids[:, :top_k].masked_fill_(flag, -1)
    weights[:, :top_k].masked_fill_(flag, 0.0)


def _call(ws: Workspace, rows_n, x, ids, weights, ops_w, limit, y=None):
    """One MegaMoE launch on exactly ``rows_n`` rows per rank (real rows first, then padding) into ``y [rows_n, H]``
    (default: a new tensor). ``ids`` int32 / ``weights`` fp32 hold >= k columns (the router's [n, k + 1] output or
    a slice of it)."""
    l1, l2, sl1, sl2 = ops_w
    dg, buf = _dg(), ws.buffer
    real = x.size(0)
    block = ws.block_m(rows_n)
    if block not in ws.compiled and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"BF16 MegaMoE: warm the block_m={block} geometry ({rows_n} rows) before CUDA graph capture")
    if y is None:
        y = torch.empty((rows_n, ws.hidden), dtype=torch.bfloat16, device=ws.device)
    if real and ids.dtype != torch.int32:
        raise TypeError("BF16 MegaMoE prep takes the router's int32 ids")
    if not real:
        x = ids = weights = buf.x[:0]
    _io().prep_bf16(x, ids, weights, ws.top_k, rows_n, buf.x, buf.topk_idx, buf.topk_weights, ws.counter, PDL)
    prev = dg.get_pdl()
    dg.set_pdl(PDL)
    try:
        dg.bf16_mega_moe(y, l1, l2, buf, shared_l1_weights=sl1, shared_l2_weights=sl2,
                         activation_clamp=float(limit), isoexec_canonical=True)
    finally:
        dg.set_pdl(prev)
    ws.compiled.add(block)
    return y


def _check_ops(ws: Workspace, ops_w):
    l1, l2, sl1, sl2 = ops_w
    nl = ws.E // ws.world
    if (l1.dtype, l2.dtype, sl1.dtype, sl2.dtype) != (torch.bfloat16,) * 4:
        raise ValueError("the BF16 actor MegaMoE runs BF16 masters")
    if tuple(l1.shape) != (nl, 2 * ws.intermediate, ws.hidden) or tuple(l2.shape) != (nl, ws.hidden, ws.intermediate):
        raise ValueError(f"local masters must be gate_up [{nl}, 2I, H] / down [{nl}, H, I] (+ the shared expert)")
    if tuple(sl1.shape) != (2 * ws.intermediate, ws.hidden) or tuple(sl2.shape) != (ws.hidden, ws.intermediate):
        raise ValueError("shared expert masters must be [2I, H] / [H, I]")
    for t in ops_w:
        if not t.is_contiguous() or t.device != ws.device or t.data_ptr() % 16:
            raise ValueError("MegaMoE masters must be contiguous, 16-byte aligned, on the workspace device")


def warm(ws: Workspace, gate_up, down, limit, shared=None):
    """Compile every tile geometry outside capture on all-padding rows (top-k ids -1: nothing dispatched). At EP > 1
    this is COLLECTIVE: every EP rank must call it with the same workspace, in the same order."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm BF16 MegaMoE outside CUDA graph capture")
    ops_w = masters(gate_up, down, shared)
    _check_ops(ws, ops_w)
    with torch.no_grad():
        for n in ws.geometries():
            if ws.block_m(n) not in ws.compiled:
                _call(ws, n, ws.buffer.x[:0], None, None, ops_w, limit)
    torch.cuda.synchronize(ws.device)
    ws.warmed = True


def experts(ws: Workspace, x, ids, weights, gate_up, down, *, limit, width=None, shared=None, out_rows=None):
    """MegaMoE over routed rows: ``x [n, H]`` bf16 with ``ids [n, >= k]`` (int32 global routed expert ids in the first
    k columns) and ``weights [n, >= k]`` fp32; ``gate_up`` / ``down`` are this rank's local masters (``masters``).
    Every EP rank passes the same ``width >= n`` (default n, valid at EP = 1). Returns ``[n, H]`` bf16, or with
    ``out_rows=M`` (EP > 1, replicated rows: ``x`` is this rank's canonical home slice of M rows) the
    reassembled replicated ``[M, H]``."""
    n = x.size(0)
    width = n if width is None else int(width)
    if width < n or (ws.world > 1 and width < 1):
        raise ValueError("MegaMoE width must cover this rank's rows")
    if x.dtype != torch.bfloat16 or x.size(-1) != ws.hidden or (n and ids.size(1) < ws.top_k):
        raise ValueError("MegaMoE call disagrees with its prepared geometry")
    ops_w = masters(gate_up, down, shared)
    _check_ops(ws, ops_w)
    gather = out_rows is not None
    if gather and ws.world == 1:
        raise ValueError("the replicated gather reassembles home slices at EP > 1")
    if width == 0:
        return x.new_empty((0, ws.hidden))
    x = x.contiguous()
    out = torch.empty((out_rows, ws.hidden), dtype=torch.bfloat16, device=ws.device) if gather else None
    outs, start = [], 0
    for per in chunks(width, ws.cap):
        take = max(0, min(n, start + per) - start)
        sl = slice(start, start + take)
        y = _call(ws, per, x[sl], ids[sl] if take else None, weights[sl] if take else None, ops_w, limit,
                  y=ws.out.y[:per] if gather else None)
        if gather:
            _io().gather(ws.out.ptrs, ws.rank, ws.out.flags_off, ws.out.y_off, ws.counter, start, per, out, PDL)
        else:
            outs.append(y[:take])
        start += per
    if gather:
        return out
    return outs[0] if len(outs) == 1 else torch.cat(outs, 0)


def check_expert_map(emap, world, rank):
    """MegaMoE's expert-to-rank rule: rank r owns [r E/W, (r+1) E/W), the shared expert at home."""
    E = emap.E
    if E % world or list(emap.owner[:E]) != [e // (E // world) for e in range(E)] or not emap.shared_home:
        raise ValueError("BF16 MegaMoE EP requires the contiguous expert map with the shared expert at home")
    if emap.local_experts(rank) != [*range(rank * (E // world), (rank + 1) * (E // world)), E]:
        raise ValueError("local masters must be this rank's contiguous experts, shared last")


__all__ = ["IMPLEMENTATION", "MAX_ROWS_PER_RANK", "Workspace", "check_expert_map", "check_geometry",
           "check_no_grad_pass", "chunks", "experts", "masters", "max_rows_per_rank", "route_select", "warm"]
