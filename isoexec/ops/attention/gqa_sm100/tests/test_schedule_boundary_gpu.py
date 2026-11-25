"""Invariance at the DECLARED segment span's own boundaries, raw bits.

The declared span is ``S = ops.segment_span(d)`` (one segment of ``spec.LONGEST_CONTEXT`` keys). The other
GPU modules' lengths stay below it, so this module puts rows on both sides of every boundary the
declaration has: ``S - 1``, ``S``, ``S + 1`` keys, a second segment with a short tail, the longest
declared context, and a row of more than two segments. For each: trainer rows == paged prefill rows ==
chunked prefill rows == prefill + decode steps, alone and inside batches with shorter and longer
neighbours at every position, repeated calls, both head-dim-256 leaves and every decode split.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(live_device_arch() != "sm100", reason="requires an SM100 GPU")

from isoexec.ops.attention.gqa_sm100 import ops, spec
from isoexec.ops.attention.gqa_sm100.tests._harness import bit_equal, cu, draw, first_diff, page_kv

#: (label, hq, hk, d): a TP=8 rank of the 35B / 0.8B layers, the 9B layer at TP=4, a head-dim-128 rank.
GEOMETRIES = [("d256-2q1kv", 2, 1, 256), ("d256-4q1kv", 4, 1, 256), ("d128-2q2kv", 2, 2, 128)]


def _span(d):
    s = ops.segment_span(d)
    assert s == spec.LONGEST_CONTEXT == 10240
    return s


def _lengths(s):
    """Rows on both sides of the declared span: S-1, S (the longest declared), S+1, a short tail, three segments."""
    return [s - 1, s, s + 1, s + 130, 2 * s + 5]


def _same(a, b, what):
    assert bit_equal(a, b), f"{what}: {first_diff(a, b)}"


def _paged(q_rows, kc, vc, table, q_lens, k_lens, d, **kw):
    used = torch.tensor(list(k_lens), device="cuda", dtype=torch.int32)
    return ops.gqa_sm100_fwd(q_rows.contiguous(), kc, vc, cu(q_lens), used, softmax_scale=ops.default_scale(d),
                             seg_tokens=ops.segment_span(d), max_seqlen_q=max(q_lens), max_seqlen_k=max(k_lens),
                             block_table=table, **kw)


@pytest.mark.parametrize(("label", "hq", "hk", "d"), GEOMETRIES, ids=[g[0] for g in GEOMETRIES])
def test_rows_at_the_declared_span_boundaries(label, hq, hk, d):
    s = _span(d)
    lens = _lengths(s)
    scale = ops.default_scale(d)
    starts = [0, *torch.tensor(lens).cumsum(0).tolist()[:-1]]
    q, k, v = (draw(sum(lens), h, d, seed) for h, seed in ((hq, 3), (hk, 4), (hk, 5)))

    # the trainer path (ops.varlen_fwd is what integration.trainer_core runs), twice
    trainer, trainer_lse = ops.varlen_fwd(q, k, v, cu(lens), max(lens), scale)
    again, again_lse = ops.varlen_fwd(q, k, v, cu(lens), max(lens), scale)
    _same(trainer, again, "trainer repeat")
    _same(trainer_lse, again_lse, "trainer repeat lse")
    assert torch.isfinite(trainer.float()).all()

    # each sequence alone on the trainer path == its rows inside the batch
    for i, (st, n) in enumerate(zip(starts, lens)):
        alone, alone_lse = ops.varlen_fwd(q[st:st + n].contiguous(), k[st:st + n].contiguous(),
                                          v[st:st + n].contiguous(), cu([n]), n, scale)
        _same(alone, trainer[st:st + n], f"trainer row alone, len {n}")
        _same(alone_lse, trainer_lse[st:st + n], f"trainer lse alone, len {n}")

    # the engine path: whole prefill in one call, through permuted pages with poisoned unwritten slots
    kc, vc, table = page_kv(k, v, lens, 1152, permute=True, fill=float("nan"))
    used = torch.tensor(lens, device="cuda", dtype=torch.int32)
    prefill, prefill_lse = ops.paged_fwd(q, kc, vc, cu(lens), used, table, max(lens), max(lens), scale)
    _same(prefill, trainer, "paged prefill")
    _same(prefill_lse, trainer_lse, "paged prefill lse")

    # chunked prefill: the last third of every sequence, then the 300 rows that straddle each boundary
    for q_lens in ([n // 3 for n in lens], [min(n, 300 + n - s if n > s else 300) for n in lens]):
        rows = torch.cat([q[st + n - m:st + n] for st, n, m in zip(starts, lens, q_lens)])
        want = torch.cat([trainer[st + n - m:st + n] for st, n, m in zip(starts, lens, q_lens)])
        got, _ = ops.paged_fwd(rows.contiguous(), kc, vc, cu(q_lens), used, table, max(q_lens), max(lens), scale)
        _same(got, want, f"chunked prefill {q_lens}")

    # decode steps at every position around each boundary, every row in one batch of mixed lengths
    for offset in (-2, -1, 0, 1, 2, 129):
        pos = [min(n - 1, max(0, s - 1 + offset)) for n in lens] + [n - 1 for n in lens]
        idx = list(range(len(lens))) * 2
        rows = torch.stack([q[starts[i] + p] for i, p in zip(idx, pos)])
        want = torch.stack([trainer[starts[i] + p] for i, p in zip(idx, pos)])
        want_lse = torch.stack([trainer_lse[starts[i] + p] for i, p in zip(idx, pos)])
        tab = table[idx].contiguous()
        k_lens = [p + 1 for p in pos]
        got, got_lse = ops.paged_fwd(rows.contiguous(), kc, vc, cu([1] * len(pos)),
                                     torch.tensor(k_lens, device="cuda", dtype=torch.int32), tab, 1, max(lens), scale)
        _same(got, want, f"decode offset {offset}")
        _same(got_lse, want_lse, f"decode lse offset {offset}")
        # row position: the same rows in reverse order, and each row alone
        rev = list(reversed(range(len(pos))))
        got_r, _ = ops.paged_fwd(rows[rev].contiguous(), kc, vc, cu([1] * len(pos)),
                                 torch.tensor([k_lens[i] for i in rev], device="cuda", dtype=torch.int32),
                                 tab[rev].contiguous(), 1, max(lens), scale)
        _same(got_r, want[rev], f"decode reversed, offset {offset}")
        for j in range(len(pos)):
            one, _ = ops.paged_fwd(rows[j:j + 1].contiguous(), kc, vc, cu([1]),
                                   torch.tensor(k_lens[j:j + 1], device="cuda", dtype=torch.int32),
                                   tab[j:j + 1].contiguous(), 1, max(lens), scale)
            _same(one, want[j:j + 1], f"decode alone row {j}, offset {offset}")
        if d == 256:  # both leaves and every decode split
            general = _paged(rows, kc, vc, tab, [1] * len(pos), k_lens, d, leaf=ops.LEAF_GENERAL)
            _same(general[0], want, f"general leaf, offset {offset}")
            for split in (1, 2, 3):
                dec = _paged(rows, kc, vc, tab, [1] * len(pos), k_lens, d, leaf=ops.LEAF_DECODE, decode_split=split)
                _same(dec[0], want, f"decode leaf split {split}, offset {offset}")
                _same(dec[1], want_lse, f"decode leaf split {split} lse, offset {offset}")


@pytest.mark.parametrize("d", (128, 256))
def test_a_boundary_decode_row_does_not_depend_on_its_batch(d):
    """A decode row with S + 1 keys alone, and first, middle and last in batches of 4, 128 and 256 short rows."""
    s = _span(d)
    hq, hk = 2, 1
    long = s + 1
    lens = [long] + [(53 * i) % 900 + 1 for i in range(1, 256)]
    q, k, v = (draw(sum(lens), h, d, seed) for h, seed in ((hq, 21), (hk, 22), (hk, 23)))
    kc, vc, table = page_kv(k, v, lens, 128, permute=True, fill=float("nan"))
    starts = [0, *torch.tensor(lens).cumsum(0).tolist()[:-1]]
    rows = torch.stack([q[st + n - 1] for st, n in zip(starts, lens)])
    scale = ops.default_scale(d)
    t, _ = ops.varlen_fwd(q[:long].contiguous(), k[:long].contiguous(), v[:long].contiguous(), cu([long]), long, scale)
    want = t[-1:]
    for batch in (1, 4, 128, 256):
        for at in sorted({0, batch // 2, batch - 1}):
            order = list(range(1, batch))
            order.insert(at, 0)
            got, _ = ops.paged_fwd(rows[order].contiguous(), kc, vc, cu([1] * batch),
                                   torch.tensor([lens[i] for i in order], device="cuda", dtype=torch.int32),
                                   table[order].contiguous(), 1, long, scale)
            _same(got[at:at + 1], want, f"batch {batch}, position {at}")
