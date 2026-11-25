"""Vocabulary-parallel input embedding. Shards gather candidate rows and select their owner without
arithmetic; backward masks non-owner gradients.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from isoexec.core.bindings import VOCAB_PARALLEL, BoundOp
from isoexec.schema import Placement, ShardAxis, col, torch_dtype
from isoexec.schema.token_layout import TokenLayoutContract, TokenLayoutEffect
from isoexec.nn.component import Component, PlacementError

#: Family policy, not an argument: the embedding table is held bf16 on both sides.
HELD_DTYPE = "bf16"


class Embedding(Component):
    """Embedding table with ``vocab`` padded rows and ``hidden`` columns. The padded vocabulary
    must match the head and divide the TP degree; construction defaults to meta.
    """

    SLOT_OPS = MappingProxyType({"lookup": "embed.lookup"})
    SLOTS = tuple(SLOT_OPS)
    NATIVE_LAYOUT = TokenLayoutContract(TokenLayoutEffect.EMBEDDING)

    # Lookup operates per token, so CP only changes which token IDs this rank receives.
    CONTEXT_PARALLEL = "per token: a lookup of one id, folded over vocab shards and never across tokens"

    def __init__(self, *, vocab: int, hidden: int, device: torch.device | str = "meta") -> None:
        if min(vocab, hidden) < 1:
            raise ValueError(f"Embedding: vocab and hidden must be >= 1, got {(vocab, hidden)}")
        super().__init__()
        self.vocab, self.hidden = vocab, hidden
        self.weight = nn.Parameter(torch.empty(vocab, hidden, dtype=torch_dtype(HELD_DTYPE), device=device))


    def tensor_axes(self) -> Mapping[str, ShardAxis]:
        """Shard vocabulary rows with the same axis as the head, including when both share a
        parameter.
        """
        return {"weight": col()}

    def facts(self) -> Mapping[str, Any]:
        return {"kind": "embed", "vocab": self.vocab, "hidden": self.hidden, "held_dtype": HELD_DTYPE}


    def _place(self, placement: Placement) -> None:
        tp = placement.tp.size
        if self.vocab % tp:
            raise PlacementError(
                f"{self.site}: the padded vocabulary {self.vocab} does not divide tp={tp}; the vocab-parallel "
                f"select maps a token id to its owner by integer division and needs equal shards"
            )
        base = self.bind_op("embed.lookup", site=f"{self.site}.lookup", inputs={"tp": tp})
        if tp == 1:
            self.slots.lookup = base
            return
        from isoexec.ops.collectives.tp_bytes import make_vocab_parallel_embed

        # Gather-and-select preserves lookup bits and implementation identity; record only the movement.
        self.slots.lookup = BoundOp(
            base.op,
            base.impl,
            base.version,
            base.provider,
            {**base.inputs, "vocab_parallel": VOCAB_PARALLEL},
            make_vocab_parallel_embed(base.fn, placement),
            owns=base.owns,
        )


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.slots.lookup(input_ids, self.weight)

    def extra_repr(self) -> str:
        return f"vocab={self.vocab}, hidden={self.hidden}"


__all__ = ["HELD_DTYPE", "VOCAB_PARALLEL", "Embedding"]
