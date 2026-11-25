"""SM90 rotary and fused q/k-norm callable factories.

The attention and norm family resolvers select these implementations. Fused
ownership is recorded on the BoundOp returned by the norm resolver.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from isoexec.ops.attention.rope_sm90.spec import QK_IMPL, ROPE_IMPL


MAX_POSITIONS_FLAG = "ISOEXEC_ROPE_SM90_MAX_POSITIONS"


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
                from isoexec.ops.attention.rope_sm90.ops import build_table

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
        raise ValueError(f"rope_sm90: rows disagree on rope constants {sorted(keys, key=repr)}")
    return keys.pop() if keys else None


def rope_apply(cfg):
    """``rope.apply`` callable for ``cfg`` (signature of ``core/bindings.py`` plus ``input_pairs``)."""
    return rope_apply_for(_rope_constants(cfg))


def rope_apply_for(constants):
    """Build the RoPE callable from theta, rotary_dim, mrope_section and mrope_interleaved.
    The config binding and component resolver share this closure. The canonical
    call accepts ``rotary`` for reference angle preparation; this kernel uses its
    bound constants and sequence positions directly.
    """
    from isoexec.ops.attention.rope_sm90.ops import rope

    tables = _TableSet()

    def apply(q, k, freqs=None, *, style: str, rotary_dim: int, seq: Any = None, rotary=None, input_pairs: str | None = None):
        if style != "neox":
            raise ValueError(f"rope_sm90 rope.apply binds rotate-half rows only, got style={style!r}")
        if constants is None or rotary_dim != constants[1]:
            raise ValueError(f"rope_sm90: rotary_dim {rotary_dim} does not match the model constants {constants}")
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

    setattr(apply, "resident_tensors", tables.resident_tensors)
    return apply


def qk_norm_rope(cfg=None, *, zero_centered: bool = True):
    """Build fn(q, k, wq, wk, eps, *, seq, rotary, spec) -> (q, k).
    The unused cfg preserves the signature; constants arrive on rotary and spec. ``zero_centered``
    is the row's gamma convention (``rms_zero_centered`` -> ``(x*inv)*(1+w)``, ``rms`` ->
    ``(x*inv)*w``); the norms resolver derives it from the convention the binding records."""
    del cfg
    from isoexec.ops.attention.rope_sm90.ops import qk_norm_rope as op

    tables = _TableSet()
    zero_centered = bool(zero_centered)

    def fused(q, k, wq, wk, eps, *, seq, rotary, spec):
        if rotary.style != "neox":
            raise ValueError(f"rope_sm90 q/k norm + rope binds rotate-half rows only, got {rotary.style!r}")
        tab = tables.get(q.device, rotary.theta, rotary.rotary_dim, spec.mrope_section, spec.mrope_interleaved)
        return op(q, k, wq, wk, seq.positions, tab, eps, zero_centered=zero_centered)

    setattr(fused, "resident_tensors", tables.resident_tensors)
    setattr(fused, "fused_qk_norm_rope", True)
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
