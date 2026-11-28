"""Trainer side of the V4.1 components: packed sequences, trainable tensors and the trainer's host state.

A Megatron trainer packs its sequences into one row (``input_ids [N]`` with the request boundaries in the
``SequenceContext``). Row-wise operations (embedding, mHC boundaries, norms, Engram, MoE, head) run on the
packed rows; the attention of a row reads its own sequence only, so it runs on the right-padded
``[sequences, longest]`` view of the packed rows (``SequenceRows``), the batch layout of the validated
trainer. The produced cross-row state (compressed KV, index keys, top-k, candidates) stays in that
layout between rows of one forward. Padding positions follow every real token of their row, so no real
token reads them (causal attention, complete compressed blocks only); their rows are zeros and their
gradients are dropped.

Trainable tensors of the ``sm100_bf16_actor`` recipe: every BF16/FP32 parameter except the indexer (the
frozen selector) and the router's selection biases; the Engram FP8 tables are checkpoint buffers.

Debug-layout switch (off by default; the recipe trains the routed experts): ``ISOEXEC_DS_FREEZE_ROUTED_EXPERTS=1``
also freezes every routed expert (``ffn.experts.*``; the shared expert stays trainable), as native's debug hook
``SKYRL_DSV41_FREEZE_ROUTED_EXPERTS=1``. It changes the trained set, not the forward: the trained set is reported
beside the program identity (``config.program_recorded()["trained_set"]``, streamed with every weight update), so a
frozen run is not mistaken for the recipe. Every trainer rank must use the same value.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

#: Parameter-name tests of the tensors the actor recipe keeps frozen.
FROZEN_PARTS = (".indexer.",)
FROZEN_SUFFIXES = ("gate.bias", "gate.bias_vl")

#: The debug-layout switch freezing the routed experts too, and the parameter-name test it adds.
FREEZE_ROUTED_EXPERTS_ENV = "ISOEXEC_DS_FREEZE_ROUTED_EXPERTS"
ROUTED_EXPERTS_PART = ".ffn.experts."
#: ``trained_set()`` values: the actor recipe, and the recipe with the routed experts frozen.
TRAINED_SET_ACTOR = "actor"
TRAINED_SET_ROUTED_FROZEN = "actor_routed_experts_frozen"


def routed_experts_frozen() -> bool:
    """Whether ``ISOEXEC_DS_FREEZE_ROUTED_EXPERTS`` freezes the routed experts (``0`` default, ``1``)."""
    value = os.environ.get(FREEZE_ROUTED_EXPERTS_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{FREEZE_ROUTED_EXPERTS_ENV} must be 0 or 1, got {value!r}")
    return value == "1"


def trained_set() -> str:
    """The trainer's trained set: ``actor`` (the recipe) or ``actor_routed_experts_frozen`` (debug layout)."""
    return TRAINED_SET_ROUTED_FROZEN if routed_experts_frozen() else TRAINED_SET_ACTOR


def frozen_parts() -> tuple[str, ...]:
    """Parameter-name parts kept frozen: the recipe's, plus the routed experts under the debug switch."""
    return FROZEN_PARTS + ((ROUTED_EXPERTS_PART,) if routed_experts_frozen() else ())


def trainable(name: str) -> bool:
    """Whether the trainer trains the parameter ``name`` (relative to a decoder row or the model): the actor
    recipe, with the routed experts frozen under ``ISOEXEC_DS_FREEZE_ROUTED_EXPERTS=1``."""
    qualified = "." + name
    return not (any(part in qualified for part in frozen_parts()) or qualified.endswith(FROZEN_SUFFIXES))


def mark_trainable(module: torch.nn.Module) -> int:
    """Set ``requires_grad`` of ``module``'s parameters by ``trainable``; returns the trainable count."""
    count = 0
    for name, parameter in module.named_parameters():
        train = trainable(name)
        parameter.requires_grad_(train)
        count += train
    return count


class _Gather(torch.autograd.Function):
    """``out = [src; 0][index]`` along dim 0 with ``inverse`` the out position of each src row (or the zero
    row): every src row appears at most once in ``index``, so both directions are pure gathers."""

    @staticmethod
    def forward(ctx, src, index, inverse):
        ctx.save_for_backward(inverse)
        flat = src.reshape(src.shape[0], -1)
        out = torch.cat((flat, flat.new_zeros((1, flat.shape[1]))), 0).index_select(0, index)
        return out.view(index.numel(), *src.shape[1:])

    @staticmethod
    def backward(ctx, grad):
        inverse, = ctx.saved_tensors
        flat = grad.reshape(grad.shape[0], -1)
        back = torch.cat((flat, flat.new_zeros((1, flat.shape[1]))), 0).index_select(0, inverse)
        return back.view(inverse.numel(), *grad.shape[1:]), None, None


def _gather(src, index, inverse):
    if src.is_floating_point():
        return _Gather.apply(src, index, inverse)
    flat = src.reshape(src.shape[0], -1)
    out = torch.cat((flat, flat.new_zeros((1, flat.shape[1]))), 0).index_select(0, index)
    return out.view(index.numel(), *src.shape[1:])


@dataclass(frozen=True, eq=False)
class SequenceRows:
    """The right-padded view of packed sequences: ``rows [N]`` (padded flat index of each packed token, in
    packed order) and ``source [B * L]`` (packed index of each padded position, ``N`` at padding).

    Cross-row state crosses the component boundary in the packed geometry the graph captured (``N`` token
    rows; ``N // ratio`` compressed entries and their 128-padded index keys): each sequence's own entries,
    in order, then zeros. Inside a row it is the padded view again; a padding row of ``topk`` /
    ``candidates`` repeats its sequence's last token row (it reads only that sequence's entries)."""

    sequences: int
    longest: int
    tokens: int
    lengths: tuple[int, ...]
    rows: torch.Tensor
    source: torch.Tensor
    last: torch.Tensor
    entries: dict

    @classmethod
    def from_boundaries(cls, boundaries, device) -> "SequenceRows":
        bounds = [int(b) for b in boundaries]
        lengths = tuple(b - a for a, b in zip(bounds[:-1], bounds[1:]))
        if len(bounds) < 2 or bounds[0] != 0 or any(n <= 0 for n in lengths):
            raise ValueError("packed sequence boundaries must start at 0 and give positive lengths")
        B, L, N = len(lengths), max(lengths), bounds[-1]
        rows, source, last = [], [N] * (B * L), []
        for b, (start, n) in enumerate(zip(bounds[:-1], lengths)):
            for t in range(L):
                last.append(start + min(t, n - 1))
                if t < n:
                    rows.append(b * L + t)
                    source[b * L + t] = start + t
        as_index = dict(dtype=torch.int64, device=device)
        return cls(B, L, N, lengths, torch.tensor(rows, **as_index), torch.tensor(source, **as_index),
                   torch.tensor(last, **as_index), {})

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        """``[1, N, ...]`` packed rows -> ``[B, L, ...]``, zeros at padding; the gradient is a gather."""
        out = _Gather.apply(x.reshape(self.tokens, *x.shape[2:]), self.source, self.rows)
        return out.view(self.sequences, self.longest, *x.shape[2:])

    def unpad(self, y: torch.Tensor) -> torch.Tensor:
        """``[B, L, ...]`` -> ``[1, N, ...]`` packed rows; padding rows receive zero gradient."""
        out = _Gather.apply(y.reshape(self.sequences * self.longest, *y.shape[2:]), self.rows, self.source)
        return out.view(1, self.tokens, *y.shape[2:])

    # ------------------------------------------------------------------ cross-row state
    @staticmethod
    def _entry_rows(count: int, index_keys: bool) -> int:
        return -(-count // 128) * 128 if index_keys else count

    def _entries(self, ratio: int, index_keys: bool):
        """``(per-sequence padded rows, packed rows, index/inverse pack, index/inverse unpack)``."""
        key = (ratio, index_keys)
        hit = self.entries.get(key)
        if hit is not None:
            return hit
        per = self._entry_rows(self.longest // ratio, index_keys)
        packed = self._entry_rows(self.tokens // ratio, index_keys)
        slots, sources, offset = [], [], 0
        for b, n in enumerate(self.lengths):
            for c in range(min(n // ratio, per)):
                slots.append(offset)
                sources.append(b * per + c)
                offset += 1
        if offset > packed:
            raise ValueError("packed compressed entries exceed the captured entry rows")
        S = self.sequences * per
        pack_index, pack_inverse = [S] * packed, [packed] * S          # out: packed rows; src: padded rows
        unpack_index, unpack_inverse = [packed] * S, [S] * packed      # out: padded rows; src: packed rows
        for slot, source in zip(slots, sources):
            pack_index[slot], pack_inverse[source] = source, slot
            unpack_index[source], unpack_inverse[slot] = slot, source
        as_index = dict(dtype=torch.int64, device=self.rows.device)
        hit = (per, packed, *(torch.tensor(v, **as_index) for v in (pack_index, pack_inverse, unpack_index,
                                                                     unpack_inverse)))
        self.entries[key] = hit
        return hit

    def pack_state(self, keys, values, ratio: int) -> tuple:
        """A producer's padded-view state -> the captured packed geometry."""
        out = []
        for key, value in zip(keys, values):
            if key in ("topk", "candidates"):
                flat = value.reshape(self.sequences * self.longest, -1)
                packed = flat.index_select(0, self.rows)
                out.append(packed.view(self.tokens, *value.shape[1:]) if key == "topk"
                           else packed.view(1, self.tokens, *value.shape[2:]))
                continue
            index_keys = key != "kv"
            per, packed, index, inverse, _, _ = self._entries(ratio, index_keys)
            rows = value.reshape(-1, *value.shape[2:]) if index_keys else value
            if rows.shape[0] != self.sequences * per:
                raise ValueError(f"{key}: {rows.shape[0]} produced entries, expected {self.sequences} x {per}")
            got = _gather(rows, index, inverse)
            out.append(got.view(1, packed, *value.shape[2:]) if index_keys else got)
        return tuple(out)

    def unpack_state(self, given: dict, ratio: int) -> dict:
        """A consumer's packed state inputs -> the padded view its attention reads."""
        out = dict(given)
        for key, value in given.items():
            if value is None:
                continue
            if key in ("topk", "candidates"):
                flat = value.reshape(self.tokens, -1).index_select(0, self.last)
                out[key] = (flat.view(self.sequences * self.longest, *value.shape[1:]) if key == "topk"
                            else flat.view(self.sequences, self.longest, *value.shape[2:]))
                continue
            index_keys = key != "kv"
            per, packed, _, _, index, inverse = self._entries(ratio, index_keys)
            rows = value.reshape(-1, *value.shape[2:]) if index_keys else value
            if rows.shape[0] != packed:
                raise ValueError(f"{key}: {rows.shape[0]} packed entries, expected {packed}")
            got = _gather(rows, index, inverse)
            out[key] = got.view(self.sequences, per, *value.shape[2:]) if index_keys else got
        return out


def sequence_rows(seq, x: torch.Tensor) -> SequenceRows | None:
    """The padded view of a packed trainer forward, or None when the batch rows already are the sequences
    (no context, or one packed sequence). The host boundaries come from ``seq.cu_seqlens_cpu``."""
    if seq is None:
        return None
    boundaries = getattr(seq, "cu_seqlens_cpu", None)
    if boundaries is None:
        raise ValueError("a packed trainer forward needs the host sequence boundaries (seq.cu_seqlens_cpu)")
    boundaries = boundaries.tolist()
    if x.shape[0] != 1 or boundaries[-1] != x.shape[1]:
        raise ValueError(f"packed sequence boundaries {boundaries[-1]} do not cover the rows {tuple(x.shape[:2])}")
    if len(boundaries) == 2:
        return None
    cache = getattr(seq, "__dict__", {}).get("_v41_sequence_rows")
    if cache is not None and cache.rows.device == x.device:
        return cache
    layout = SequenceRows.from_boundaries(boundaries, x.device)
    try:
        object.__setattr__(seq, "_v41_sequence_rows", layout)    # one layout per forward's context
    except (AttributeError, TypeError):
        pass
    return layout


# ------------------------------------------------------------------------------- trainer host state
def bind_trainer_state(module: torch.nn.Module, checkpoint_path: str) -> dict:
    """After the trainer loads its weights: the compressed token map of the checkpoint's tokenizer for the
    Engram hash, and each host-resident Engram table mapped into the device address space (the engine's
    ``map_host_tables``). Returns what was bound."""
    from .embed import DeepSeekV41Embed, compressed_token_map
    from .engine_state import map_host_tables

    embed = next((m for m in module.modules() if isinstance(m, DeepSeekV41Embed)), None)
    bound = {"token_map": False, "host_tables": 0}
    if embed is not None and embed.hashing:
        from transformers import AutoTokenizer
        embed.bind_token_map(compressed_token_map(AutoTokenizer.from_pretrained(checkpoint_path)))
        bound["token_map"] = True
    device = torch.device("cuda", torch.cuda.current_device())
    bound["host_tables"] = map_host_tables(module, device)
    return bound


def release_trainer_state(module: torch.nn.Module) -> dict:
    """Before the trainer process exits: drop the CUDA registrations of its host Engram tables piece by piece
    (``engine_state.release_host_tables``). Left to the context teardown, the driver drops each ~92 GiB
    registration in one critical section and stalls every other client of the node's driver. A finalizer does
    the same at interpreter exit; this makes the order explicit."""
    from .engine_state import release_host_tables
    return {"host_tables": release_host_tables(module)}


__all__ = ["FREEZE_ROUTED_EXPERTS_ENV", "FROZEN_PARTS", "FROZEN_SUFFIXES", "ROUTED_EXPERTS_PART",
           "TRAINED_SET_ACTOR", "TRAINED_SET_ROUTED_FROZEN", "SequenceRows", "bind_trainer_state", "frozen_parts",
           "mark_trainable", "release_trainer_state", "routed_experts_frozen", "sequence_rows", "trainable",
           "trained_set"]
