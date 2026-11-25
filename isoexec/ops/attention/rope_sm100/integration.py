"""SM100 rotary and fused q/k-norm callable factories, mirroring rope_sm90/integration.py's seam."""

from __future__ import annotations

import os
from typing import Any

import torch

MAX_POSITIONS_FLAG = "ISOEXEC_ROPE_SM100_MAX_POSITIONS"
ROPE_IMPL = "sm100_table_v1"
QK_IMPL = "sm100_qk_norm_rope_v1"


def max_positions() -> int:
    return int(os.environ.get(MAX_POSITIONS_FLAG, "262144"))


# One read-only table per (device, constants, length) for the whole process. Every attention layer
# builds its own ``rope.apply`` closure; sharing the table keeps 64 MiB (262144 x 64 fp32) per
# process instead of one copy per layer, which stays resident through offload and sleep.
_SHARED_TABLES: dict[tuple, Any] = {}


class _TableSet:
    """Tables keyed by constants and device; filled on first use (construction-scoped, not per forward)."""

    def __init__(self) -> None:
        self._tables: dict[tuple, Any] = {}

    def resident_tensors(self):
        return {repr(key): value.table for key, value in self._tables.items()}

    def get(self, device, theta: float, rotary_dim: int, sections=None, interleaved: bool = False):
        key = (device, float(theta), int(rotary_dim), tuple(sections) if sections else None, bool(interleaved))
        tab = self._tables.get(key)
        if tab is None:
            shared_key = (*key, max_positions())
            tab = _SHARED_TABLES.get(shared_key)
            if tab is None:
                from isoexec.ops.attention.rope_sm100.ops import build_table

                tab = build_table(float(theta), int(rotary_dim), max_positions(), device, mrope_section=sections,
                                  mrope_interleaved=bool(interleaved))
                _SHARED_TABLES[shared_key] = tab
            self._tables[key] = tab
        return tab


def _rope_constants(cfg):
    """(theta, rotary_dim, mrope_section, mrope_interleaved) shared by the model's rope rows."""
    from isoexec.schema import MLA, Attn

    keys = set()
    for mixer, _ in cfg.layers:
        if isinstance(mixer, Attn):
            keys.add((float(mixer.rope_theta), int(mixer.rotary_dim), mixer.mrope_section, bool(mixer.mrope_interleaved)))
        elif isinstance(mixer, MLA):
            keys.add((float(mixer.rope_theta), int(mixer.rotary_dim), None, False))
    if len(keys) > 1:
        raise ValueError(f"rope_sm100: rows disagree on rope constants {sorted(keys, key=repr)}")
    return keys.pop() if keys else None


def rope_apply(cfg):
    """``rope.apply`` callable for ``cfg`` (signature of ``core/bindings.py`` plus ``input_pairs``)."""
    return rope_apply_for(_rope_constants(cfg))


def rope_apply_for(constants):
    """Build the RoPE callable from theta, rotary_dim, mrope_section and mrope_interleaved."""
    from isoexec.ops.attention.rope_sm100.ops import rope

    tables = _TableSet()

    def apply(q, k, freqs=None, *, style: str, rotary_dim: int, seq=None, rotary=None, input_pairs: str | None = None):
        if style != "neox":
            raise ValueError(f"rope_sm100 rope.apply binds rotate-half rows only, got style={style!r}")
        if constants is None or rotary_dim != constants[1]:
            raise ValueError(f"rope_sm100: rotary_dim {rotary_dim} does not match the model constants {constants}")
        theta, rot, sections, inter = constants
        tab = tables.get(q.device, theta, rot, sections, inter)
        pairs = input_pairs or "neox"
        if pairs == "neox" and not torch.is_grad_enabled() and not q.requires_grad and not k.requires_grad:
            # Engine / scoring forward: q and k are this layer's own projection views and nothing else
            # reads them un-rotated, so rotate in place (same bits; no allocation, no pass-through copy).
            try:
                return rope(q, k, seq.positions, tab, style="neox", inplace=True)
            except ValueError:  # operands on odd element offsets: out of place
                pass
        return rope(q, k, seq.positions, tab, style=pairs, out_style="neox")

    apply.resident_tensors = tables.resident_tensors
    return apply


def _row_packed(t: torch.Tensor) -> bool:
    """``[T, H, D]`` whose rows are adjacent (``.contiguous()`` would not copy it)."""
    return t.dim() == 3 and t.stride(2) == 1 and t.stride(1) == t.shape[2] and (
        t.shape[0] <= 1 or t.stride(0) == t.shape[1] * t.shape[2])


def qk_norm_rope(cfg=None, *, zero_centered: bool = True):
    """Build fn(q, k, wq, wk, eps, *, seq, rotary, spec) -> (q, k).
    The unused cfg preserves the signature; constants arrive on rotary and spec. ``spec.py`` declares
    the zero-centred convention only, so the norms resolver never asks for ``zero_centered=False``."""
    del cfg
    if not zero_centered:
        raise ValueError("rope_sm100 declares its fused q/k norm for rms_zero_centered only (spec.py); "
                         "the plain-gamma path has not been verified on SM100")
    from isoexec.ops.attention.rope_sm100.ops import qk_norm_rope as op

    tables = _TableSet()

    def fused(q, k, wq, wk, eps, *, seq, rotary, spec):
        if rotary.style != "neox":
            raise ValueError(f"rope_sm100 q/k norm + rope binds rotate-half rows only, got {rotary.style!r}")
        tab = tables.get(q.device, rotary.theta, rotary.rotary_dim, spec.mrope_section, spec.mrope_interleaved)
        if not torch.is_grad_enabled() and not q.requires_grad and not k.requires_grad and _row_packed(q):
            # Engine / scoring forward: q and k are this layer's own projection views and nothing else
            # reads them un-normalized, so write the result over them (same bits; no allocation, and the
            # launch touches each byte once instead of twice). Mirrors rope_apply's in-place fast path.
            # A q that is a column window of a wider GEMM row (the packed q|k|v|gate projection) is
            # instead written out of place: the attention core reads q packed, and an in-place q would
            # cost it a separate [T, Hq, D] copy kernel. Same kernel, same bits either way.
            try:
                return op(q, k, wq, wk, seq.positions, tab, eps, zero_centered=zero_centered,
                          inplace=True)
            except ValueError:  # operands on odd element offsets: out of place
                pass
        return op(q, k, wq, wk, seq.positions, tab, eps, zero_centered=zero_centered)

    fused.resident_tensors = tables.resident_tensors
    fused.fused_qk_norm_rope = True
    return fused


__all__ = [
    "MAX_POSITIONS_FLAG",
    "QK_IMPL",
    "ROPE_IMPL",
    "max_positions",
    "qk_norm_rope",
    "rope_apply",
    "rope_apply_for",
]
