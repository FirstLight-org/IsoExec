"""Non-executable parameter holders of the V4.1 components.

Holders have no ``forward``: an opaque capture component may not contain executable children. They keep
the checkpoint's module layout (``attn.wq_a.weight``, ``ffn.experts.{e}.w1.weight``...), which is also the
attribute layout the SM100 V4.1 kernels read. Every FP8/FP4 projection is held as its decoded BF16 master
(``storage = "bf16"``); the Engram tables keep their FP8 rows and UE8M0 scales as checkpoint buffers.

Each holder declares its tensors' shard axes. Under TP, attention heads split ``wq_b`` rows and
``attn_sink``, output groups split ``wo_a`` rows and ``wo_b`` columns; the other attention tensors are
replicated and receive per-rank partial gradients (``sum_over_tp``). Everything else is replicated;
routed experts are held whole and narrowed by expert id under EP.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

from isoexec.schema import ShardAxis, TensorPlanMetadata, col, replicated, row


def _p(shape, dtype, device):
    return nn.Parameter(torch.empty(shape, dtype=dtype, device=device), requires_grad=False)


class Holder(nn.Module):
    """Parameters with declared shard axes and no execution of their own."""

    AXES: Mapping[str, ShardAxis] = {}

    def tensor_axes(self) -> Mapping[str, ShardAxis]:
        return {name: self.AXES.get(name, replicated()) for name, p in self._parameters.items() if p is not None}

    def tensor_plan_metadata(self) -> Mapping[str, TensorPlanMetadata]:
        return {name: TensorPlanMetadata() for name in self.tensor_axes()}


class Proj(Holder):
    """One projection ``weight [N, K]`` BF16 (decoded master of an FP8/FP4 checkpoint tensor, or BF16 as stored)."""

    storage = "bf16"

    def __init__(self, n: int, k: int, axis: ShardAxis | None = None, *, device="meta",
                 expert_id: int | None = None, num_experts: int | None = None) -> None:
        super().__init__()
        self.out_features, self.in_features = n, k
        self.weight = _p((n, k), torch.bfloat16, device)
        self._axis = axis
        self.expert_id, self.num_experts = expert_id, num_experts

    def tensor_axes(self) -> Mapping[str, ShardAxis]:
        return {"weight": self._axis or replicated()}

    def tensor_plan_metadata(self) -> Mapping[str, TensorPlanMetadata]:
        if self.expert_id is None:
            return {"weight": TensorPlanMetadata()}
        return {"weight": TensorPlanMetadata(kind="expert", expert_id=self.expert_id, num_experts=self.num_experts)}


class Norm(Holder):
    def __init__(self, n: int, axis: ShardAxis | None = None, *, device="meta") -> None:
        super().__init__()
        self.weight = _p((n,), torch.bfloat16, device)
        self._axis = axis

    def tensor_axes(self) -> Mapping[str, ShardAxis]:
        return {"weight": self._axis or replicated()}


_PARTIAL = replicated(reduce="sum_over_tp")   # replicated attention tensors behind head-local derivatives


class Compressor(nn.Module):
    def __init__(self, cfg, ratio: int, device) -> None:
        super().__init__()
        self.wkv = Proj(cfg.head_dim, cfg.hidden_size, _PARTIAL, device=device)
        if ratio > 1:
            self.wgate = Proj(cfg.head_dim, cfg.hidden_size, _PARTIAL, device=device)
        self.norm = Norm(cfg.head_dim, _PARTIAL, device=device)


class Indexer(nn.Module):
    """The frozen selector: replicated on every TP rank (each rank scores every index head). ``queries``: the
    query side (``wq_b``, ``weights_proj``); ``keys``: the index-key side (``wk``, ``k_norm``) of a KV source."""

    def __init__(self, cfg, owns_keys: bool, device, *, queries: bool = True) -> None:
        super().__init__()
        if queries:
            self.wq_b = Proj(cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank, _PARTIAL, device=device)
            self.weights_proj = Proj(cfg.index_n_heads, cfg.hidden_size, _PARTIAL, device=device)
        if owns_keys:
            self.wk = Proj(cfg.index_head_dim, cfg.head_dim, _PARTIAL, device=device)
            self.k_norm = Norm(cfg.index_head_dim, _PARTIAL, device=device)


class AttentionWeights(Holder):
    """The attribute layout ``deepseek_v41_cache_sm100`` reads (checkpoint names).

    ``projection=False``: the decoder's FULL layer, whose token-wise projection (window ``wkv`` / ``kv_norm``,
    the compressor, the index-key side of the indexer) is owned by ``decoder.entry`` (``EntryProjection``);
    this holder keeps the query side, the sink and the output projection."""

    AXES = {"attn_sink": col()}

    def __init__(self, cfg, index: int, device, *, projection: bool = True) -> None:
        super().__init__()
        H, D = cfg.num_attention_heads, cfg.head_dim
        self.attn_sink = _p((H,), torch.float32, device)
        self.wq_a = Proj(cfg.q_lora_rank, cfg.hidden_size, _PARTIAL, device=device)
        self.q_norm = Norm(cfg.q_lora_rank, _PARTIAL, device=device)
        self.wq_b = Proj(H * D, cfg.q_lora_rank, col(), device=device)
        if projection:
            self.wkv = Proj(D, cfg.hidden_size, _PARTIAL, device=device)
            self.kv_norm = Norm(D, _PARTIAL, device=device)
        self.wo_a = Proj(cfg.o_groups * cfg.o_lora_rank, H * D // cfg.o_groups, col(), device=device)
        self.wo_b = Proj(cfg.hidden_size, cfg.o_groups * cfg.o_lora_rank, row(), device=device)
        if projection and index in cfg.kv_source_layer_ids:
            self.compressor = Compressor(cfg, cfg.compress_ratios[index], device)
        if index in cfg.index_source_layer_ids:
            self.indexer = Indexer(cfg, projection and index in cfg.kv_source_layer_ids, device)


class EntryProjection(Holder):
    """``decoder.entry``'s part of the decoder FULL layer's attention (checkpoint ``layers.{b}.attn.*``): the
    window-KV projection, the ratio-1 compressor and the index-key side of the indexer, all token-wise in the
    CED handoff."""

    def __init__(self, cfg, index: int, device) -> None:
        super().__init__()
        D = cfg.head_dim
        self.wkv = Proj(D, cfg.hidden_size, _PARTIAL, device=device)
        self.kv_norm = Norm(D, _PARTIAL, device=device)
        self.compressor = Compressor(cfg, cfg.compress_ratios[index], device)
        self.indexer = Indexer(cfg, True, device, queries=False)


class Gate(Holder):
    """Router: BF16 scores weight, FP32 selection bias (the vision bias ``bias_vl`` is excluded with the vision
    tower: not built, its checkpoint key ignored)."""

    def __init__(self, cfg, device) -> None:
        super().__init__()
        E = cfg.n_routed_experts
        self.weight = _p((E, cfg.hidden_size), torch.bfloat16, device)
        self.bias = _p((E,), torch.float32, device)


class Expert(nn.Module):
    """``w1`` (gate) and ``w3`` (up) ``[I, H]``, ``w2`` (down) ``[H, I]`` decoded BF16 masters."""

    def __init__(self, hidden: int, inter: int, expert_id: int | None, num_experts: int, device) -> None:
        super().__init__()
        kw = dict(device=device, expert_id=expert_id, num_experts=num_experts)
        self.w1, self.w3, self.w2 = Proj(inter, hidden, **kw), Proj(inter, hidden, **kw), Proj(hidden, inter, **kw)


class MoEWeights(nn.Module):
    """Router, routed experts under their global ids (``experts.{e}``) and the shared expert."""

    def __init__(self, cfg, device) -> None:
        super().__init__()
        E, H, I = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
        self.num_experts = E           # global routed experts (EP holds a contiguous range of them)
        self.gate = Gate(cfg, device)
        self.experts = nn.ModuleDict({str(e): Expert(H, I, e, E, device) for e in range(E)})
        self.shared_experts = Expert(H, I, None, E, device)

    def held_experts(self) -> tuple[int, ...]:
        return tuple(sorted(int(e) for e in self.experts))

    def mark_ep_summed(self) -> None:
        """Token-disjoint EP training: the router's and the shared expert's gradients are already summed over the EP
        group (``ep_moe_train``, every rank the same bits), so Megatron reduces them as expert tensors
        (``allreduce=False``: over the expert data-parallel group). They stay whole replicas on every EP rank:
        ``ep_replicated`` tells the weight stream to send them as they are, not as EP-held experts."""
        for p in (*self.gate.parameters(), *self.shared_experts.parameters()):
            p.allreduce = False
            p.ep_replicated = True


class EngramTable(nn.Module):
    """The frozen FP8 Engram rows ``weight [rows, dim]`` and UE8M0 ``scale [rows, dim / 32]``: checkpoint
    buffers loaded byte for byte (not streamed by weight updates: they are checkpoint provenance). An engine
    keeps them in host memory, one copy per node shared by its ranks, and reads rows through a device mapping
    (``engine_state.map_host_tables``)."""

    HOST_RESIDENT_BUFFERS = ("weight", "scale")

    def __init__(self, rows: int, dim: int, device) -> None:
        super().__init__()
        self.rows, self.dim = rows, dim
        self.register_buffer("weight", torch.empty((rows, dim), dtype=torch.float8_e4m3fn, device=device))
        self.register_buffer("scale", torch.empty((rows, math.ceil(dim / 32)), dtype=torch.float8_e8m0fnu,
                                                  device=device))


class EngramWeights(Holder):
    def __init__(self, cfg, index: int, device) -> None:
        super().__init__()
        S, H = cfg.hc_mult, cfg.hidden_size
        rows = cfg.engram_num_embeddings[cfg.engram_layer_ids.index(index)]
        self.q_weight = _p((S, H), torch.bfloat16, device)
        self.k_weight = _p((S, H), torch.bfloat16, device)
        self.wkv = Proj(H * (S + 1), cfg.engram_width(), device=device)
        self.embed = EngramTable(rows, cfg.engram_head_dim, device)


__all__ = [
    "AttentionWeights",
    "EngramTable",
    "EngramWeights",
    "EntryProjection",
    "Expert",
    "Gate",
    "Holder",
    "MoEWeights",
    "Norm",
    "Proj",
]
