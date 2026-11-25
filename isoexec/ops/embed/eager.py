"""Trainer vocabulary lookup using F.embedding.
Tensor-parallel callers wrap it with the vocab-parallel gather."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def embed_lookup(ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """mcore ``VocabParallelEmbedding`` at TP=1 == ``F.embedding``."""
    return F.embedding(ids, w)


__all__ = ["embed_lookup"]
