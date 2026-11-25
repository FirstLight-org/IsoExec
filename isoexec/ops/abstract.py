"""Logical OP metadata for FakeTensor/meta execution, without native kernel selection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensor

from isoexec.core.bindings import BoundOp
from isoexec.ops.moe.carriers import ExpertOutputs, RoutingResult
from isoexec.ops.table import OpResolutionError, OpTable, reference_binding


# Reuse fake-compatible arithmetic so its casts define the metadata.
_PURE_OPS = frozenset({
    "mm", "mlp.act", "logprobs.head", "logprobs.lm_head_slice",
    "logprobs.log_softmax", "norms.rms.pre_attn", "norms.rms.pre_mlp",
    "norms.rms.final", "norms.qk", "norms.mla", "norms.gated_out",
    "attention.gate", "residual.add.attn", "residual.add.mlp",
    "moe.route_score", "moe.route_mechanics", "moe.shared_expert", "moe.combine",
})


def _require_metadata(value: Any) -> None:
    """Refuse real tensors even when callers obtain a binding directly."""
    if isinstance(value, torch.Tensor):
        if not isinstance(value, FakeTensor) and value.device.type != "meta":
            raise TypeError("abstract operation requires FakeTensor or meta tensor inputs")
    elif isinstance(value, Mapping):
        for child in value.values():
            _require_metadata(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _require_metadata(child)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _require_metadata(vars(value)[field.name] if hasattr(value, "__dict__") else getattr(value, field.name))


def _guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    def abstract(*args: Any, **kwargs: Any) -> Any:
        _require_metadata(args)
        _require_metadata(kwargs)
        return fn(*args, **kwargs)

    return abstract


def _packed(tensor: torch.Tensor, rank: int, label: str) -> None:
    if tensor.ndim != rank:
        raise ValueError(f"{label} requires rank {rank}, got shape {tuple(tensor.shape)}")


def _embedding(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # PyTorch's embedding meta kernel does not reject floating-point indices,
    # although its real operator does. Keep the missing admission check here.
    if ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("embed.lookup requires int32 or int64 token indices")
    _packed(weight, 2, "embedding weight")
    from isoexec.ops.embed.reference import ref_embed_lookup

    return ref_embed_lookup(ids, weight)


def _rope(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor | None = None,
          *, style: str, rotary_dim: int, seq: Any = None, rotary: Any = None,
          input_pairs: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer fresh RoPE outputs; angle-cache preparation remains runtime-owned."""
    del seq, rotary
    for label, value in (("rope query", q), ("rope key", k)):
        _packed(value, 3, label)
        torch._check(value.shape[-1] >= rotary_dim)
    if style not in ("neox", "interleaved") or rotary_dim < 0 or rotary_dim % 2:
        raise ValueError("rope.apply needs an even rotary width and a supported lane style")
    if input_pairs is not None and input_pairs != style:
        if (input_pairs, style) != ("interleaved", "neox"):
            raise ValueError("rope.apply cannot convert the requested lane styles")
    torch._check(q.shape[0] == k.shape[0])
    if freqs is not None:
        torch._check(freqs.shape[-1] * 2 == rotary_dim)
    return q.new_empty(q.shape), k.new_empty(k.shape)


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               cu_seqlens: torch.Tensor, max_seqlen: int, scale: float) -> torch.Tensor:
    """Metadata of attention/reference.py::ref_attention_core, without reading cu."""
    del max_seqlen, scale
    for label, value in (("attention query", q), ("attention key", k), ("attention value", v)):
        _packed(value, 3, label)
    _packed(cu_seqlens, 1, "packed sequence offsets")
    torch._check(q.shape[0] == k.shape[0])
    torch._check(k.shape[:2] == v.shape[:2])
    torch._check(q.shape[-1] == k.shape[-1])
    torch._check(k.shape[1] > 0)
    torch._check(q.shape[1] % k.shape[1] == 0)
    return q.new_empty((q.shape[0], q.shape[1], v.shape[-1]))


def _mla(q_nope: torch.Tensor, q_pe: torch.Tensor, c: torch.Tensor,
         k_pe: torch.Tensor, uk: torch.Tensor, uv: torch.Tensor,
         seq: Any, scale: float) -> torch.Tensor:
    """Reuse the unabsorbed reference's tensor algebra before its packed core."""
    if c.dtype != uk.dtype or c.dtype != uv.dtype:
        raise TypeError("attention.core.mla requires matching latent and up-projection dtypes")
    heads = q_nope.shape[1]
    latent = c.unsqueeze(0).expand(heads, -1, -1)
    k_nope = torch.bmm(latent, uk.transpose(1, 2)).transpose(0, 1)
    values = torch.bmm(latent, uv.transpose(1, 2)).transpose(0, 1)
    q = torch.cat((q_nope, q_pe), dim=-1)
    k = torch.cat((k_nope, k_pe.unsqueeze(1).expand(-1, heads, -1)), dim=-1)
    return _attention(q, k, values, seq.cu_seqlens, seq.max_q, scale)


def _conv(x: torch.Tensor, w: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Short convolution is fresh [T,C], rounded to x.dtype by its operation."""
    _packed(x, 2, "short convolution activation")
    _packed(w, 3, "short convolution weight")
    _packed(cu_seqlens, 1, "packed sequence offsets")
    torch._check(w.shape[0] == x.shape[-1])
    torch._check(w.shape[1] == 1)
    torch._check(w.shape[2] > 0)
    return torch.empty_like(x)


def _gdn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
         alpha: torch.Tensor, beta: torch.Tensor, A_log: torch.Tensor,
         dt_bias: torch.Tensor, cu_seqlens: torch.Tensor,
         *, cu_seqlens_cpu: torch.Tensor | None = None) -> torch.Tensor:
    """The recurrent core's explicit final cast is to v.dtype, not its FP32 state."""
    del cu_seqlens_cpu
    for label, value in (("gdn query", q), ("gdn key", k), ("gdn value", v)):
        _packed(value, 3, label)
    _packed(cu_seqlens, 1, "packed sequence offsets")
    torch._check(q.shape == k.shape)
    torch._check(q.shape[0] == v.shape[0])
    torch._check(q.shape[1] > 0)
    torch._check(v.shape[1] % q.shape[1] == 0)
    torch._check(alpha.shape == v.shape[:2])
    torch._check(beta.shape == alpha.shape)
    torch._check(A_log.shape == v.shape[1:2])
    torch._check(dt_bias.shape == A_log.shape)
    return v.new_empty(v.shape)


def _experts(x: torch.Tensor, routing: RoutingResult, gate_up: torch.Tensor,
             down: torch.Tensor, *, weight_after_fc2: bool = False) -> ExpertOutputs:
    """Per-slot FP32 rows are the carrier contract, prior to combine/output cast."""
    _packed(x, 2, "expert activation")
    _packed(gate_up, 3, "expert gate/up weight")
    _packed(down, 3, "expert down weight")
    _packed(routing.indices, 2, "routing indices")
    torch._check(routing.indices.shape == routing.probs.shape)
    torch._check(routing.indices.shape[0] == x.shape[0])
    torch._check(gate_up.shape[0] == down.shape[0])
    torch._check(gate_up.shape[1] == 2 * down.shape[-1])
    torch._check(gate_up.shape[-1] == x.shape[-1])
    torch._check(down.shape[1] == x.shape[-1])
    if routing.indices.dtype != torch.int32 or routing.probs.dtype != torch.float32:
        raise TypeError("expert routing requires int32 indices and FP32 probabilities")
    rows = x.new_empty((*routing.indices.shape, x.shape[-1]), dtype=torch.float32)
    return ExpertOutputs(rows=rows, layout="per_slot", routing=routing)


_OP_RULES: Mapping[str, Callable[..., Any]] = {
    "embed.lookup": _embedding,
    "rope.apply": _rope,
    "attention.core.gqa": _attention,
    "attention.core.mla": _mla,
    "conv.short": _conv,
    "gdn.core": _gdn,
    "moe.expert_mlp": _experts,
}


class AbstractOpTable(OpTable):
    """Metadata-only bindings with identities distinct from reference and production."""

    def __init__(self) -> None:
        object.__setattr__(self, "mode", "abstract")
        object.__setattr__(self, "arch", "meta")

    def bind(self, op: str, *, site: str, inputs: Mapping[str, Any] | None = None,
             convention: str | None = None, facts: Mapping[str, Any] | None = None) -> BoundOp:
        if not site:
            raise ValueError("AbstractOpTable.bind requires a component site")
        recorded = dict(inputs or {})
        selectors = dict(facts or {})
        for key in recorded.keys() & selectors.keys():
            if recorded[key] != selectors[key]:
                raise OpResolutionError(f"{site}: conflicting abstract selection fact {key!r}")
        if convention is not None:
            recorded["convention"] = convention
        owns: dict[str, str] = {}
        if op == "collectives.row_parallel":
            from isoexec.ops.collectives.reference import ref_row_parallel

            fn = ref_row_parallel(leaves=int(recorded.get("leaves", 1)),
                                  leaf_dtype=recorded.get("leaf_dtype", "bf16"),
                                  tp=int(recorded.get("tp", 1)))
        elif op in _OP_RULES:
            fn = _OP_RULES[op]
            if op == "gdn.core":
                owns = {name: f"abstract.{op}" for name in ("gdn.l2norm", "gdn.gating")}
        elif op in _PURE_OPS:
            # Only attention.gate uses non-convention selectors in the reused
            # reference families. Geometry is checked by the tensor operations.
            ref_facts = {"gate": selectors["gate"]} if op == "attention.gate" and "gate" in selectors else None
            binding = reference_binding(op, site=site, convention=convention, facts=ref_facts)
            fn = binding.fn
            recorded.update(binding.inputs)
        else:
            raise OpResolutionError(f"{site}: no abstract rule for operation {op!r}; add its metadata contract")
        return BoundOp(op, f"abstract.{op}", None, "ops.abstract", recorded, _guard(fn), owns=owns)

    def require_production(self, op: str, *, site: str) -> None:
        raise OpResolutionError(f"{site}: metadata execution cannot construct a native kernel for {op!r}")


__all__ = ["AbstractOpTable"]
