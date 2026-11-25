"""Final collapse of the four streams and the final RMSNorm, ahead of the FP32 ``LMHead``."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from isoexec.nn.component import Component
from isoexec.schema import Placement, ShardAxis, replicated

from .config import Geometry


class DeepSeekV41HeadStage(Component):
    """Settles the last row's deferred expand, collapses the streams with the shifted ``pre`` and applies the
    final RMSNorm: ``(x, pre, out, post, comb)`` -> ``h [B, T, H]`` bf16."""

    def __init__(self, cfg: Geometry, *, device="meta") -> None:
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.empty((cfg.hidden_size,), dtype=torch.bfloat16, device=device),
                                   requires_grad=False)

    def tensor_axes(self) -> Mapping[str, ShardAxis]:
        return {"weight": replicated()}

    def facts(self) -> Mapping[str, Any]:
        return {"kind": "deepseek_v41_head_stage", "hidden": self.cfg.hidden_size, "rms_eps": self.cfg.rms_norm_eps}

    def _place(self, placement: Placement) -> None:
        self.weight.requires_grad_(placement.side == "trainer")

    def forward(self, x, pre, out, post, comb):
        from isoexec.ops.norms.norms_sm100.ops import rms_norm
        from isoexec.ops.residual.deepseek_mhc_sm100.ops import boundary_rows, collapse, expand
        if not torch.is_grad_enabled() and x.is_cuda and x.shape[-1] == 5120:
            # inference: expand + collapse + RMSNorm in ONE launch (the mHC boundary's row kernel: mhc_sm100's expand
            # and collapse programs and norms_sm100 recipe v2, bit for bit); None when its operands do not fit
            got = boundary_rows(x, pre, self.weight, self.cfg.rms_norm_eps, out=out, post=post, comb=comb,
                                encode=False)
            if got is not None:
                return got[1]
        x = expand(out, x, post, comb)
        h = collapse(x, pre)
        shape = h.shape
        return rms_norm(h.reshape(-1, shape[-1]).contiguous(), self.weight, self.cfg.rms_norm_eps, False).view(shape)

    def abstract_output(self, x, pre, out, post, comb):
        return torch.empty(out.shape, dtype=torch.bfloat16, device=out.device)


__all__ = ["DeepSeekV41HeadStage"]
