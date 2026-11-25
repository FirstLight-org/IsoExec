"""Engine TP > 1 without EP: the experts sharded over the TP group so that every rank evaluates ``deepseek_v41_bf16_actor_v1``
bit for bit (the TP = 1, EP and trainer program at every TP degree).

Placement of rank ``r`` of ``tp`` (load time, ``shard_masters``): FC1 gate and up rows of the intermediate indices
``[r I/tp, (r+1) I/tp)``; FC2 ``down`` OUTPUT rows ``[r H/tp, (r+1) H/tp)`` with the whole K = I. Per call, with the token
rows replicated in the TP group (every rank routes them and builds the same plan, no communication):

1. FC1 on the rank's I slice and the recipe's weighted clamped SwiGLU rows -> ``h_r [R, I/tp]`` (R = 7 T expert rows);
2. ``h`` exchange: ``h [R, I]`` with the columns in global I order on every rank;
3. FC2 on the rank's output slice, one K chain over the global I index (the recipe's FC2, BF16 root);
4. the recipe's combine on the slice (columns are independent) -> ``y_r [T, H/tp]``;
5. output exchange: ``[T, H]`` replicated (no arithmetic on the wire; it replaces native's all-reduce).

Every output element is computed by exactly the operations of the TP = 1 program, so the bits are the same at every
TP degree and whatever realizes steps 2 and 5. Realizations: ``NcclExchange`` (any group; the named fallback and the
off-node path) and ``SimExchange`` (single-process simulation of all ranks, for the battery); the node-local NVLink store
path writes the same bytes to the same positions. This module runs the stages on the base grouped kernels.
"""
from __future__ import annotations

import torch

from . import rows as cs
from .spec import FC2_LEAVES, ROUTING

DEGREES = (1, 2, 4, 8)


def check_tp(intermediate: int, hidden: int, tp: int) -> None:
    """Named refusal of a TP degree the sharded program does not admit."""
    if tp not in DEGREES:
        raise ValueError(f"deepseek_v41 TP-sharded experts admit TP in {DEGREES}, got {tp}")
    if intermediate % tp or (intermediate // tp) % 8 or hidden % tp or (hidden // tp) % 8:
        raise ValueError(f"deepseek_v41 TP-sharded experts need I / TP and H / TP multiples of 8 "
                         f"(I = {intermediate}, H = {hidden}, TP = {tp})")


def shard_masters(gate_up: torch.Tensor, down: torch.Tensor, tp: int, rank: int):
    """``(gate_up_r [G, 2 I/tp, H], down_r [G, H/tp, I])`` of rank ``rank`` from the packed masters ``gate_up [G, 2I, H]``
    (gate rows, then up rows) and ``down [G, H, I]`` (copies; load-time)."""
    G, I2, H = gate_up.shape
    I = I2 // 2
    check_tp(I, H, tp)
    if not 0 <= rank < tp or tuple(down.shape) != (G, H, I):
        raise ValueError("shard_masters: bad rank or masters")
    Is, Hs = I // tp, H // tp
    gu = torch.cat((gate_up[:, rank * Is:(rank + 1) * Is], gate_up[:, I + rank * Is:I + (rank + 1) * Is]), 1)
    return gu.contiguous(), down[:, rank * Hs:(rank + 1) * Hs].contiguous()


class NcclExchange:
    """Steps 2 and 5 as two all-gathers over a ``torch.distributed`` group (NCCL) plus one reorder each."""

    def __init__(self, group=None):
        import torch.distributed as dist
        self.group = group
        self.tp = dist.get_world_size(group)
        self.rank = dist.get_rank(group)

    def _gather_cols(self, part: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist
        rows, cols = part.shape
        buf = torch.empty((self.tp, rows, cols), dtype=part.dtype, device=part.device)
        dist.all_gather_into_tensor(buf, part.contiguous(), group=self.group)
        return buf.permute(1, 0, 2).reshape(rows, self.tp * cols)

    def gather_h(self, h_r):
        return self._gather_cols(h_r)

    def gather_out(self, y_r):
        return self._gather_cols(y_r)


def _route(x, router_w, bias, top_k, routed_scale, num_experts):
    return cs.route(x, router_w, bias, top_k=top_k, routed_scale=routed_scale, num_experts=num_experts,
                    routing=ROUTING)


def _front(x, rt, gu_r, limit):
    from isoexec.ops.moe.moe_sm100 import ops as base
    x_rows, gp = base.dispatch(x, rt.probs_ext, rt.plan)
    proj = base.fc1(x_rows, gu_r, rt.plan.grouped)
    return cs.swiglu_rows(proj, gp, limit)


def _back(h, rt, dn_r):
    from isoexec.ops.moe.moe_sm100 import ops as base
    roots = base.fc2(h, dn_r, rt.plan.grouped, leaves=FC2_LEAVES)
    return base.combine(roots, rt.plan, dtype=torch.bfloat16)


def tp_moe_forward(x, router_w, bias, gu_r, dn_r, *, top_k, routed_scale, limit, exchange):
    """One rank's call: ``x [T, H]`` replicated bf16 -> ``[T, H]`` replicated bf16 (the recipe's bits)."""
    if x.size(0) == 0:
        return torch.empty_like(x)
    with torch.no_grad():
        rt = _route(x.contiguous(), router_w, bias, top_k, routed_scale, gu_r.size(0) - 1)
        h = exchange.gather_h(_front(x, rt, gu_r, limit))
        return exchange.gather_out(_back(h, rt, dn_r))


def tp_moe_forward_sim(x, router_w, bias, gate_up, down, *, top_k, routed_scale, limit, tp):
    """Single-process simulation of all ``tp`` ranks (the battery's TP relation without a process group): every rank's
    shard computed in turn, the exchanges as concatenations. ``[T, H]`` bf16."""
    if x.size(0) == 0:
        return torch.empty_like(x)
    with torch.no_grad():
        shards = [shard_masters(gate_up, down, tp, r) for r in range(tp)]
        rt = _route(x.contiguous(), router_w, bias, top_k, routed_scale, gate_up.size(0) - 1)
        h = torch.cat([_front(x, rt, gu_r, limit) for gu_r, _ in shards], 1)
        return torch.cat([_back(h, rt, dn_r) for _, dn_r in shards], 1)


__all__ = ["DEGREES", "NcclExchange", "check_tp", "shard_masters", "tp_moe_forward", "tp_moe_forward_sim"]
