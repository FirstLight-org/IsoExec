"""Derive logical tensor inventories from model configuration independently of live components.
Preserve held dtypes and normalization conventions required by checkpoint conversion.
"""

from __future__ import annotations

from typing import Protocol

from isoexec.schema.model import GDN, Attn, Mlp, ModelConfig
from isoexec.schema.params import LogicalTensor, TieDecl, col, expert, replicated, row


class TiedHeadConfig(Protocol):
    """The config fact shared by schema configs and authored model configs."""

    @property
    def tied_head(self) -> bool:
        """Whether the output head aliases the input embedding."""
        ...


# Per-head norms use SUM_OVER_TP because each rank contributes gradients for its own heads.
REPL = replicated()
SUM_OVER_TP = replicated(reduce="sum_over_tp")
ROW1 = row()
EXPERT = expert()


def attention_vocabulary(ns: str, spec: Attn, hidden: int) -> tuple[LogicalTensor, ...]:
    q_dim, kv_dim = spec.num_q_heads * spec.head_dim, spec.num_kv_heads * spec.head_dim
    out = [
        LogicalTensor(f"{ns}.mixer.q", (q_dim, hidden), "bf16", col()),
        LogicalTensor(f"{ns}.mixer.k", (kv_dim, hidden), "bf16", col(units=spec.num_kv_heads)),
        LogicalTensor(f"{ns}.mixer.v", (kv_dim, hidden), "bf16", col(units=spec.num_kv_heads)),
    ]
    if spec.out_gate is not None:
        out.append(LogicalTensor(f"{ns}.mixer.gate", (q_dim, hidden), "bf16", col()))
    if spec.qkv_bias:
        out += [
            LogicalTensor(f"{ns}.mixer.q_bias", (q_dim,), "bf16", col()),
            LogicalTensor(f"{ns}.mixer.k_bias", (kv_dim,), "bf16", col(units=spec.num_kv_heads)),
            LogicalTensor(f"{ns}.mixer.v_bias", (kv_dim,), "bf16", col(units=spec.num_kv_heads)),
        ]
    if spec.qk_norm is not None:
        out += [
            LogicalTensor(f"{ns}.mixer.q_norm", (spec.head_dim,), "bf16", SUM_OVER_TP),
            LogicalTensor(f"{ns}.mixer.k_norm", (spec.head_dim,), "bf16", SUM_OVER_TP),
        ]
    out.append(LogicalTensor(f"{ns}.mixer.o", (hidden, q_dim), "bf16", ROW1))
    return tuple(out)


def gdn_vocabulary(ns: str, spec: GDN, hidden: int) -> tuple[LogicalTensor, ...]:
    hk, hv = spec.num_key_heads, spec.num_value_heads
    key_dim, value_dim = spec.key_dim, spec.value_dim
    return (
        LogicalTensor(f"{ns}.mixer.q", (key_dim, hidden), "bf16", col(units=hk)),
        LogicalTensor(f"{ns}.mixer.k", (key_dim, hidden), "bf16", col(units=hk)),
        LogicalTensor(f"{ns}.mixer.v", (value_dim, hidden), "bf16", col(units=hv)),
        LogicalTensor(f"{ns}.mixer.z", (value_dim, hidden), "bf16", col(units=hv)),
        LogicalTensor(f"{ns}.mixer.beta", (hv, hidden), "bf16", col(units=hv)),
        LogicalTensor(f"{ns}.mixer.alpha", (hv, hidden), "bf16", col(units=hv)),
        # channel axis = q | k | v concat, and the three segments do NOT have the same unit count:
        # q and k count key heads, v counts value heads (weights.py sharp edge 7).
        LogicalTensor(
            f"{ns}.mixer.conv_w",
            (2 * key_dim + value_dim, 1, spec.conv_kernel),
            "bf16",
            col(segments=((key_dim, hk), (key_dim, hk), (value_dim, hv))),
        ),
        LogicalTensor(f"{ns}.mixer.A_log", (hv,), "bf16", col(units=hv)),
        LogicalTensor(f"{ns}.mixer.dt_bias", (hv,), "bf16", col(units=hv)),
        LogicalTensor(f"{ns}.mixer.out_norm", (spec.value_head_dim,), "bf16", SUM_OVER_TP),
        LogicalTensor(f"{ns}.mixer.o", (hidden, value_dim), "bf16", ROW1),
    )


def moe_vocabulary(ns: str, spec: Mlp, hidden: int) -> tuple[LogicalTensor, ...]:
    inter = spec.intermediate_size
    out = [LogicalTensor(f"{ns}.mlp.router.weight", (spec.num_experts, hidden), "fp32", REPL)]
    if spec.routing == "sigmoid_noaux_bias_select":
        out.append(LogicalTensor(f"{ns}.mlp.router.expert_bias", (spec.num_experts,), "fp32", REPL))
    for e in range(spec.num_experts):
        out += [
            LogicalTensor(f"{ns}.mlp.experts.{e}.gate", (inter, hidden), "bf16", EXPERT),
            LogicalTensor(f"{ns}.mlp.experts.{e}.up", (inter, hidden), "bf16", EXPERT),
            LogicalTensor(f"{ns}.mlp.experts.{e}.down", (hidden, inter), "bf16", EXPERT),
        ]
    if spec.shared_intermediate:
        si = spec.shared_intermediate
        out += [
            LogicalTensor(f"{ns}.mlp.shared.gate", (si, hidden), "bf16", col()),
            LogicalTensor(f"{ns}.mlp.shared.up", (si, hidden), "bf16", col()),
            LogicalTensor(f"{ns}.mlp.shared.down", (hidden, si), "bf16", ROW1),
            # the shared expert's own sigmoid gate: one row of weights over the residual
        ]
        if spec.shared_expert_gate:
            out.append(LogicalTensor(f"{ns}.mlp.shared.expert_gate", (1, hidden), "bf16", REPL))
    return tuple(out)


def dense_vocabulary(ns: str, spec: Mlp, hidden: int) -> tuple[LogicalTensor, ...]:
    return (
        LogicalTensor(f"{ns}.mlp.gate", (spec.intermediate_size, hidden), "bf16", col()),
        LogicalTensor(f"{ns}.mlp.up", (spec.intermediate_size, hidden), "bf16", col()),
        LogicalTensor(f"{ns}.mlp.down", (hidden, spec.intermediate_size), "bf16", ROW1),
    )


def stack_vocabulary(cfg: ModelConfig) -> tuple[LogicalTensor, ...]:
    h, v = cfg.hidden, cfg.padded_vocab
    out = [
        LogicalTensor("lm.embed.weight", (v, h), "bf16", col()),
        LogicalTensor("lm.final_norm.weight", (h,), "bf16", REPL),
    ]
    if not cfg.tied_head:
        out.append(LogicalTensor("lm.head.weight", (v, h), "bf16", col()))
    return tuple(out)


def ties_for(cfg: TiedHeadConfig) -> tuple[TieDecl, ...]:
    """The tied-tensor declarations of a config. ``tied_head`` is the only tie any shipped model
    declares, and it is a bool on the config rather than a name table because there is exactly one
    pair it could name: the embedding and the output head."""
    if not cfg.tied_head:
        return ()
    return (TieDecl(logical="lm.embed.weight", aliases=("lm.head.weight",), owner="first"),)


__all__ = [
    "REPL",
    "SUM_OVER_TP",
    "ROW1",
    "EXPERT",
    "attention_vocabulary",
    "gdn_vocabulary",
    "moe_vocabulary",
    "dense_vocabulary",
    "stack_vocabulary",
    "ties_for",
]
