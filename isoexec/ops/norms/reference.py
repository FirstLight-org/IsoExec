"""Plain-torch norm references selected by stored gamma convention.
RMS reads gamma directly and zero-centered RMS reads gamma-1 then applies 1+w; unsupported conventions refuse."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isoexec.core.bindings import BoundOp

if TYPE_CHECKING:  # the plan is DUCK-TYPED at runtime -- ops/reference.py imports this module, never the reverse
    from isoexec.ops.reference import ReferencePlan


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.to(torch.float32)
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


def ref_rms_zero_centered(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """``rms(x) * (1 + w)``: HF Qwen3.5 RMSNorm form (weights stored as gamma - 1)."""
    return (_rms(x, eps) * (1.0 + w.to(torch.float32))).to(x.dtype)


def ref_rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return (_rms(x, eps) * w.to(torch.float32)).to(x.dtype)


def ref_qk_norm_zero_centered(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return ref_rms_zero_centered(x, w, eps)


def ref_qk_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head ``rms(x) * w`` with gamma held as is (HF Qwen3 q/k norm), one output rounding."""
    return ref_rms(x, w, eps)


def ref_gated_out_zero_centered(x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float,
                              *, activation: str = "silu") -> torch.Tensor:
    """``rms(x) * (1 + w) * silu(z)`` per value head -> [T, hv*dv] in x's dtype."""
    y = _rms(x, eps) * (1.0 + w.to(torch.float32))
    y = y * _gate(z, activation)
    return y.to(x.dtype).reshape(x.shape[0], -1)


def ref_gated_out(x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float,
                  *, activation: str = "silu") -> torch.Tensor:
    y = _rms(x, eps) * w.to(torch.float32)
    y = y * _gate(z, activation)
    return y.to(x.dtype).reshape(x.shape[0], -1)


def _gate(z: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "silu":
        return F.silu(z.to(torch.float32))
    if activation == "sigmoid":
        return torch.sigmoid(z.to(torch.float32))
    raise ValueError(f"norms.gated_out: unsupported activation {activation!r}")


# One plain-gamma RMS callable and identity shared across its operation sites.
def _rms_binding(op: str) -> BoundOp:
    return BoundOp(op, "ref.rms", None, "ops.reference", {"convention": "rms"}, ref_rms)


# op -> gamma-storage convention -> reference binding.
# OpTable selects per component; ReferencePlan supplies catalog defaults.
NORM_CONVENTIONS: Mapping[str, Mapping[str, BoundOp]] = {
    "norms.rms.pre_attn": {
        "rms_zero_centered": BoundOp(
            "norms.rms.pre_attn",
            "ref.rms_zero_centered",
            None,
            "ops.reference",
            fn=ref_rms_zero_centered,
            inputs={"convention": "rms_zero_centered"},
        ),
        "rms": _rms_binding("norms.rms.pre_attn"),
    },
    "norms.rms.pre_mlp": {
        "rms_zero_centered": BoundOp(
            "norms.rms.pre_mlp",
            "ref.rms_zero_centered",
            None,
            "ops.reference",
            fn=ref_rms_zero_centered,
            inputs={"convention": "rms_zero_centered"},
        ),
        "rms": _rms_binding("norms.rms.pre_mlp"),
    },
    "norms.rms.final": {
        "rms_zero_centered": BoundOp(
            "norms.rms.final",
            "ref.rms_zero_centered",
            None,
            "ops.reference",
            fn=ref_rms_zero_centered,
            inputs={"convention": "rms_zero_centered"},
        ),
        "rms": _rms_binding("norms.rms.final"),
    },
    # Zero-centred (Qwen3.5) and plain-gamma (Qwen3) q/k norms are separate implementations.
    "norms.qk": {
        "rms_zero_centered": BoundOp(
            "norms.qk",
            "ref.qk_zero_centered",
            None,
            "ops.reference",
            fn=ref_qk_norm_zero_centered,
            inputs={"convention": "rms_zero_centered"},
        ),
        "rms": BoundOp(
            "norms.qk",
            "ref.qk_rms",
            None,
            "ops.reference",
            fn=ref_qk_norm,
            inputs={"convention": "rms"},
        ),
    },
    "norms.mla": {"rms": _rms_binding("norms.mla")},
    "norms.gated_out": {
        "gated_rms_zero_centered": BoundOp(
            "norms.gated_out",
            "ref.gated_rms_zero_centered",
            None,
            "ops.reference",
            fn=ref_gated_out_zero_centered,
            inputs={"convention": "gated_rms_zero_centered"},
        ),
        "gated_rms": BoundOp(
            "norms.gated_out",
            "ref.gated_rms",
            None,
            "ops.reference",
            fn=ref_gated_out,
            inputs={"convention": "gated_rms"},
        ),
    },
}


# Explicit defaults keep declaration reordering from changing a reference forward.
DEFAULT_NORM_CONVENTIONS: Mapping[str, str] = {
    "norms.rms.pre_attn": "rms_zero_centered",
    "norms.rms.pre_mlp": "rms_zero_centered",
    "norms.rms.final": "rms_zero_centered",
    "norms.qk": "rms_zero_centered",
    "norms.mla": "rms",
    "norms.gated_out": "gated_rms_zero_centered",
}


def bindings(plan: "ReferencePlan") -> dict[str, BoundOp]:
    """Bind each norm op using plan.conventions or its declared default.
    Unknown ops and unsupported gamma conventions refuse before interpreting checkpoint weights."""
    chosen = dict(DEFAULT_NORM_CONVENTIONS)
    for op, convention in (plan.conventions or {}).items():
        variants = NORM_CONVENTIONS.get(op)
        if variants is None:
            raise ValueError(
                f"reference_table: {op!r} is not a norm op id this module states a convention for "
                f"(stated: {tuple(sorted(NORM_CONVENTIONS))})"
            )
        if convention not in variants:
            raise ValueError(
                f"reference_table: no reference binding for op id {op!r} in norm convention "
                f"{convention!r} (on offer: {tuple(variants)})"
            )
        chosen[op] = convention
    result = {op: NORM_CONVENTIONS[op][convention] for op, convention in chosen.items()}
    activation = (plan.selectors or {}).get("norms.gated_out", {}).get("activation", "silu")
    if activation not in ("silu", "sigmoid"):
        raise ValueError(f"norms.gated_out: unsupported activation {activation!r}")
    if activation != "silu":
        bound = result["norms.gated_out"]
        result["norms.gated_out"] = replace(
            bound, impl=f"{bound.impl}_{activation}", fn=partial(bound.fn, activation=activation),
            inputs={**bound.inputs, "activation": activation},
        )
    return result


__all__ = [
    "DEFAULT_NORM_CONVENTIONS",
    "NORM_CONVENTIONS",
    "bindings",
    "ref_gated_out",
    "ref_gated_out_zero_centered",
    "ref_qk_norm",
    "ref_qk_norm_zero_centered",
    "ref_rms",
    "ref_rms_zero_centered",
]
