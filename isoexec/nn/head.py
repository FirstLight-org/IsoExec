"""Vocabulary-parallel output projection, row selection and log-softmax. Trainers resolve all
stages at placement; engines attach sampler-dependent stages, and tied heads share the embedding
parameter.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from isoexec.nn import _seams
from isoexec.nn.component import Component, PlacementError
from isoexec.schema import Placement, ShardAxis, col, torch_dtype
from isoexec.schema.token_layout import TokenLayoutContract, TokenLayoutEffect

#: Family policy, not an argument: the head is held bf16, tied or not.
HELD_DTYPE = "bf16"

#: The keyword-only seam ``logprobs.log_softmax`` exposes for the vocab-parallel process group,
#: named identically to ``runtimes/parallel.py::SEAM_GROUP`` so a reader can put the two side by
#: side. A binding that lacks it at TP>1 is refused by name rather than run over a column shard.
SEAM_GROUP = "group"


class LMHead(Component):
    """Output head mapping ``[T, hidden]`` to fp32 logits trimmed to the real ``vocab``.
    ``padded_vocab`` is the stored row count and must divide TP; ``tied_to`` shares the
    embedding parameter by identity. Construction defaults to meta.

    The head GEMM computes in bf16 and widens at the output boundary. Padding columns are
    trimmed before callers select tokens or normalize probabilities.

    Attributes:
        tp_group: Borrowed native TP group for masking local vocabulary padding.
    """

    SLOT_OPS = MappingProxyType(
        {
            "head": "logprobs.head",
            "lm_head_slice": "logprobs.lm_head_slice",
            "log_softmax": "logprobs.log_softmax",
            "tp_copy": "collectives.tp_copy",
        }
    )
    SLOTS = tuple(SLOT_OPS)
    NATIVE_LAYOUT = TokenLayoutContract(TokenLayoutEffect.HEAD)

    # All stages score individual tokens; the host reassembles CP results.
    CONTEXT_PARALLEL = "per token: every stage scores one row, and the cross-rank reassembly is the host's"

    def __init__(
        self,
        *,
        vocab: int,
        hidden: int,
        padded_vocab: int | None = None,
        tied_to: nn.Parameter | None = None,
        device: torch.device | str = "meta",
    ) -> None:
        padded = vocab if padded_vocab is None else padded_vocab
        if min(vocab, hidden) < 1:
            raise ValueError(f"LMHead: vocab and hidden must be >= 1, got {(vocab, hidden)}")
        if padded < vocab:
            raise ValueError(f"LMHead: padded_vocab={padded} is smaller than vocab={vocab}")
        super().__init__()
        self.vocab, self.hidden, self.padded_vocab = vocab, hidden, padded
        self.tied = tied_to is not None
        self.tp_group: torch.distributed.ProcessGroup | None = None
        if tied_to is None:
            self.weight = nn.Parameter(torch.empty(padded, hidden, dtype=torch_dtype(HELD_DTYPE), device=device))
        else:
            if tuple(tied_to.shape) != (padded, hidden):
                raise ValueError(
                    f"LMHead: tied_to has shape {tuple(tied_to.shape)}, not the head's {(padded, hidden)}; a tie "
                    f"is ONE tensor under two names, so a reshape would make it two"
                )
            # The same object, deliberately. Two names for one Parameter is what a tie is.
            self.weight = tied_to


    def tensor_axes(self) -> Mapping[str, ShardAxis]:
        """Shard vocabulary rows with the same axis as the embedding so a tied parameter has one
        shard plan.
        """
        return {"weight": col()}

    def facts(self) -> Mapping[str, Any]:
        return {
            "kind": "lm_head",
            "vocab": self.vocab,
            "padded_vocab": self.padded_vocab,
            "hidden": self.hidden,
            "tied": self.tied,
            "held_dtype": HELD_DTYPE,
        }


    def _place(self, placement: Placement) -> None:
        tp = placement.tp.size
        if self.padded_vocab % tp:
            raise PlacementError(
                f"{self.site}: the padded vocabulary {self.padded_vocab} does not divide tp={tp}; the "
                f"vocab-parallel log-softmax sums equal column shards"
            )
        # Vocabulary shards need a fixed-tree sum of partial input gradients.
        # Under SP, the forward seam first gathers the full token stream.
        self.slots.tp_copy = self.seams.tp_copy()
        self.slots.head = self.bind_op(
            "logprobs.head", site=f"{self.site}.head", inputs={"tp": tp, "boundary_out_dtype": "fp32"}
        )
        # The two remaining stages, by side. See "Which side owes them" in the module docstring.
        if placement.side == "trainer":
            self._resolve_trainer_stages(placement)
        else:
            self.slots.declare_unbound(
                "lm_head_slice",
                "the engine adapter assigns it: the rows its sampler selected before compute_logits "
                "(vLLM's logits_indices, SGLang's LogitsProcessor)",
            )
            self.slots.declare_unbound(
                "log_softmax",
                "the engine adapter assigns it: it carries the engine's own vocab-parallel group, which "
                "is a host handle a component never creates",
            )

    def _resolve_trainer_stages(self, placement: Placement) -> None:
        """Bind identity row selection and log-softmax on the trainer. At TP greater than one,
        inject the vocabulary process group and reject a binding that cannot accept it.
        """
        site = self.site
        self.slots.lm_head_slice = self.bind_op("logprobs.lm_head_slice", site=f"{site}.lm_head_slice")
        log_softmax = self.bind_op("logprobs.log_softmax", site=f"{site}.log_softmax")
        if placement.tp.size > 1:
            log_softmax = _seams.bind_seam(log_softmax, site=f"{site}.log_softmax", **{SEAM_GROUP: placement.tp.group})
        self.slots.log_softmax = log_softmax


    def head_output(self, hidden: torch.Tensor, *, replicated_tokens: bool = False) -> torch.Tensor:
        """The raw head GEMM. ``replicated_tokens`` skips the input seam when the caller already
        holds the full token set on every rank -- an SP engine's sampler hands it globally selected
        rows, so gathering them again would duplicate them."""
        return self.slots.head(hidden if replicated_tokens else self.slots.tp_copy(hidden), self.weight)

    def trim_to_vocab(self, logits: torch.Tensor) -> torch.Tensor:
        """Trim global logits; preserve equal TP shards with padding masked to -inf."""
        if self.tp_group is not None:
            tp, rank = self.tp_group.size(), self.tp_group.rank()
        elif self.placement is not None:
            tp, rank = self.placement.tp.size, self.placement.tp.rank
        else:
            tp, rank = 1, 0
        if tp > 1 and logits.shape[-1] != self.padded_vocab:
            # A trainer retains equally sized physical vocabulary shards. Padding
            # must contribute neither probability mass nor parameter gradients.
            width = self.padded_vocab // tp
            if logits.shape[-1] != width:
                raise RuntimeError(f"{self.site}: head output is neither the global vocabulary nor its local shard")
            if (rank + 1) * width <= self.vocab:
                return logits
            columns = torch.arange(width, device=logits.device) + rank * width
            return logits.masked_fill(columns >= self.vocab, float("-inf"))
        return logits if logits.shape[-1] == self.vocab else logits[..., : self.vocab]

    def forward(self, hidden: torch.Tensor, *, replicated_tokens: bool = False) -> torch.Tensor:
        """fp32 logits for the rows ``lm_head_slice`` selects, padding trimmed."""
        h = self.slots.lm_head_slice(hidden)
        return self.trim_to_vocab(self.head_output(h, replicated_tokens=replicated_tokens).to(torch.float32))

    def log_probs(self, logits_fp32: torch.Tensor, targets: torch.Tensor | None, temperature: float = 1.0):
        """``logprobs.log_softmax`` on ``logits / T``, before any truncation the sampler applies."""
        return self.slots.log_softmax(logits_fp32, targets, temperature)

    def log_probs_entropy(self, logits_fp32: torch.Tensor, targets: torch.Tensor, temperature: float = 1.0):
        """``(log_probs, entropy)`` of ``logits / T`` without grad. ``entropy`` comes from the
        log-softmax's own leaf pass where the bound provider emits it, else ``None`` (the caller computes
        it); ``log_probs`` are the bits :meth:`log_probs` returns either way."""
        return self.slots.log_softmax(logits_fp32, targets, temperature, with_entropy=True)

    def extra_repr(self) -> str:
        return f"vocab={self.vocab}, padded_vocab={self.padded_vocab}, hidden={self.hidden}, tied={self.tied}"


__all__ = ["HELD_DTYPE", "SEAM_GROUP", "LMHead"]
