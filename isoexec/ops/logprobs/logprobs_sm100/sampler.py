"""SM100 decode sampler: Gumbel-max token, then the sampled logprob, rank and top-1 under the rowinv row.

Logprob numerics are the trainer's. ``leaf_states`` + ``merge_rows`` of the pinned recipe (``ops.RECIPE``)
produce (M, L32) and the sampled-token logprob ``(v_tok - M) - L32``, the kernels ``sampled_logprobs``
runs on the trainer. ``csrc/sampler.cu`` adds the token draw and a rank / top-1 pass that re-evaluates
the full-row write expression.

Tokens: Gumbel-max with Philox4x32-10 noise keyed by (seed, step offset, row, column), so a row's token
depends on neither the other rows nor the launch geometry. Columns whose score cannot reach the row
maximum are pruned exactly (no noise drawn); partial winners combine in a fixed order, no atomics.
"""

from __future__ import annotations

import torch

from isoexec.ops.logprobs.logprobs_sm100 import ops as _lp
from isoexec.ops.logprobs.logprobs_sm100._extension_sampler import extension
from isoexec.ops.logprobs.logprobs_sm100.spec import LEAVES, STRIP

#: Philox offset advance per sampling step (the counter also carries row and column).
OFFSET_STEP = 4


def split_for(vocab: int) -> int | None:
    """The CTA split of one row: a function of the vocabulary only, never of the batch."""
    for split in (32, 16, 8):
        if vocab % (split * STRIP) == 0:
            return split
    return None


def supports(logits: torch.Tensor) -> bool:
    """Whether ``logits`` [rows, vocab] has a layout the kernels read in place."""
    return (
        logits.ndim == 2
        and logits.is_cuda
        and logits.dtype in (torch.float32, torch.bfloat16)
        and (logits.shape[0] <= 1 or logits.stride(1) == 1)
        and logits.data_ptr() % 16 == 0
        and (logits.shape[0] <= 1 or (logits.stride(0) * logits.element_size()) % 16 == 0)
        and logits.shape[1] % (LEAVES * STRIP) == 0
        and split_for(logits.shape[1]) is not None
    )


def next_philox(device: torch.device) -> tuple[int, int]:
    """(seed, offset) for one sampling step, advancing the device's default CUDA generator."""
    gen = torch.cuda.default_generators[device.index if device.index is not None else torch.cuda.current_device()]
    seed, offset = gen.initial_seed(), gen.get_offset()
    gen.set_offset(offset + OFFSET_STEP)
    return seed & 0xFFFFFFFFFFFFFFFF, offset


def _signed64(v: int) -> int:
    return v - (1 << 64) if v >= (1 << 63) else v


@torch.no_grad()
def sample(x: torch.Tensor, temperature: torch.Tensor | None, seed: int, offset: int, *, prune: bool = True,
           row_max: torch.Tensor | None = None) -> torch.Tensor:
    """int64 [rows] Gumbel-max tokens of the sampling logits ``x``; ``temperature`` None = all greedy.

    ``prune=False`` draws every column's noise; the token is the same.
    ``row_max`` [rows, n]: partial maxima of exactly ``x`` the caller already holds (else one scan).
    """
    rows, vocab = x.shape
    split = split_for(vocab)
    pmax = torch.empty(rows, split, device=x.device, dtype=torch.float32)
    pval = torch.empty_like(pmax)
    pidx = torch.empty(rows, split, device=x.device, dtype=torch.int32)
    token = torch.empty(rows, device=x.device, dtype=torch.int64)
    if temperature is not None:
        temperature = temperature.to(torch.float32).contiguous()
    extension().sample(x, temperature, _signed64(seed), _signed64(offset), split, prune, row_max, pmax, pval,
                       pidx, token)
    return token


def _score_temperature(t, dev):
    return None if t is None else t.reshape(1).to(device=dev, dtype=torch.float32)


def leaf_states(x: torch.Tensor, score_temperature: torch.Tensor | None):
    """The trainer's rowinv leaf states (M_g, S_g) [rows, LEAVES] of ``x`` (pinned recipe)."""
    rows = x.shape[0]
    m = torch.empty(rows, LEAVES, device=x.device, dtype=torch.float32)
    s = torch.empty_like(m)
    if rows:
        _lp._ext(None).leaf_states(x, LEAVES, m, s, _score_temperature(score_temperature, x.device))
    return m, s


@torch.no_grad()
def sampled_logprobs(x: torch.Tensor, token: torch.Tensor, k: int, score_temperature: torch.Tensor | None,
                     *, leaves=None):
    """(indices int32 [rows, 1 + k], logprobs fp32 [rows, 1 + k], ranks int64 [rows]) under the rowinv row.

    Column 0 is the sampled token; with ``k == 1`` column 1 is the top-1 (torch.topk order).
    ``leaves``: the (M_g, S_g) ``leaf_states`` already returned for this ``x`` and temperature.
    """
    rows, vocab = x.shape
    dev = x.device
    st = _score_temperature(score_temperature, dev)
    rM = torch.empty(rows, device=dev, dtype=torch.float32)
    rL = torch.empty_like(rM)
    lp_tok = torch.empty_like(rM)
    out_idx = torch.empty(rows, 1 + k, device=dev, dtype=torch.int32)
    out_lp = torch.empty(rows, 1 + k, device=dev, dtype=torch.float32)
    out_rank = torch.empty(rows, device=dev, dtype=torch.int64)
    if rows == 0:
        return out_idx, out_lp, out_rank
    m, s = leaves if leaves is not None else leaf_states(x, st)
    # The trainer's row merge, which also gathers the sampled logprob (v_tok - M) - L32.
    _lp._ext(None).merge_rows(m, s, rM, rL, None, lp_tok, x, token, None, vocab, st)
    split = split_for(vocab)
    pcnt = torch.empty(rows, split, device=dev, dtype=torch.int32)
    ptop = torch.empty(rows, split, device=dev, dtype=torch.float32)
    ptopi = torch.empty(rows, split, device=dev, dtype=torch.int32)
    extension().rank(x, rM, rL, lp_tok, token, st, split, k, pcnt, ptop, ptopi, out_idx, out_lp, out_rank)
    return out_idx, out_lp, out_rank


@torch.no_grad()
def decode_step(x: torch.Tensor, temperature: torch.Tensor | None, seed: int, offset: int, k: int | None,
                score_temperature: torch.Tensor | None, *, xs: torch.Tensor | None = None):
    """One decode step: (token int64 [rows], (indices, logprobs, ranks) or None when ``k`` is None).

    ``x`` holds the raw logits the logprobs score; ``xs`` the sampling logits when logits processors
    rewrote a copy. Without a separate ``xs`` or a scoring temperature the leaf maxima of ``x`` are the
    row maxima the sampler's prune bound needs, so the leaf pass runs first and the sampler skips its
    own max scan.
    """
    leaves = None
    row_max = None
    if k is not None and xs is None and score_temperature is None and temperature is not None:
        leaves = leaf_states(x, None)
        row_max = leaves[0]
    token = sample(x if xs is None else xs, temperature, seed, offset, row_max=row_max)
    if k is None:
        return token, None
    return token, sampled_logprobs(x, token, k, score_temperature, leaves=leaves)


__all__ = ["OFFSET_STEP", "decode_step", "leaf_states", "next_philox", "sample", "sampled_logprobs",
           "split_for", "supports"]
