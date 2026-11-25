"""Decoder block with pre-norm mixer and MLP sublayers, each followed by a residual join. The
placement walk visits children independently; a block may prune its own subtree but must not
restructure its parent.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from isoexec.schema import Placement, SequenceContext
from isoexec.schema.token_layout import TokenLayoutContract, TokenLayoutEffect
from isoexec.nn.component import Component
from isoexec.nn.norm import RMSNorm


class DecoderBlock(Component):
    """Compose ``mixer(h, seq)`` and ``mlp(h, seq, aux)`` with two pre-norms and residual slots.
    Each residual join rounds once to bf16 after the sublayer boundary.

    ``hidden``, ``eps`` and ``convention`` configure both norms; parameters default to meta.
    Registration follows checkpoint order: pre-attention norm, mixer, pre-MLP norm, MLP.
    """

    SLOT_OPS: Mapping[str, str] = MappingProxyType(
        {"residual_attn": "residual.add.attn", "residual_mlp": "residual.add.mlp"}
    )
    SLOTS = tuple(SLOT_OPS)
    NATIVE_LAYOUT = TokenLayoutContract(TokenLayoutEffect.DECODER)

    # Residual joins operate per token; each child independently declares its CP support.
    CONTEXT_PARALLEL = "per token: the residual joins are pointwise; each child states its own form"

    def __init__(
        self,
        *,
        mixer: nn.Module,
        mlp: nn.Module,
        hidden: int,
        eps: float = 1e-6,
        convention: str = "rms_zero_centered",
        device: torch.device | str = "meta",
    ) -> None:
        for role, sublayer in (("mixer", mixer), ("mlp", mlp)):
            if not isinstance(sublayer, nn.Module):
                raise TypeError(
                    f"DecoderBlock: {role} must be an nn.Module, got {type(sublayer).__name__}. A bare "
                    f"callable would register no parameters, so the row's tensors would be missing from "
                    f"the inventory and from every shard plan built off it."
                )
        super().__init__()
        self.hidden, self.eps, self.convention = hidden, float(eps), convention
        self.pre_attn_norm = RMSNorm(hidden=hidden, eps=eps, convention=convention, role="pre_attn", device=device)
        self.mixer = mixer
        self.pre_mlp_norm = RMSNorm(hidden=hidden, eps=eps, convention=convention, role="pre_mlp", device=device)
        self.mlp = mlp


    def facts(self) -> Mapping[str, Any]:
        """The row's own configuration. The sublayers' is on the sublayers, at their own sites."""
        return {"kind": "decoder_block", "hidden": self.hidden, "eps": self.eps, "convention": self.convention}


    def _place(self, placement: Placement) -> None:
        del placement  # a residual join is elementwise: no seam, no side dependence, no degree
        self.slots.residual_attn = self.bind_op("residual.add.attn", site=f"{self.site}.seam_attn.residual")
        self.slots.residual_mlp = self.bind_op("residual.add.mlp", site=f"{self.site}.seam_mlp.residual")


    def forward(
        self, h: torch.Tensor, seq: SequenceContext, aux: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """``aux`` is the trainer's side-output dict, threaded to the MLP half and written in place
        by the rows that produce one. It is an output-collection choice, never a numerics fork."""
        h = self.slots.residual_attn(h, self.mixer(self.pre_attn_norm(h), seq))
        return self.slots.residual_mlp(h, self.mlp(self.pre_mlp_norm(h), seq, aux))

    def extra_repr(self) -> str:
        return f"hidden={self.hidden}, eps={self.eps}, convention={self.convention!r}"


__all__ = ["DecoderBlock"]
