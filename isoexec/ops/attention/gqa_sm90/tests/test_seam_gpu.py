"""THE SEAM: trainer rows == engine paged prefill rows == chunked prefill rows == decode rows, raw bits.

The trainer's ``sm90_fa3_segsplit_v1_varlen`` core attends packed THD rows; the engine's
``sm90_fa3_segsplit_v1_paged`` core attends the same tokens through a paged KV cache with right-aligned
queries (vLLM's layout). Under the fixed segment split (spec.SEG_RECIPE) a row's KV reduction is a
function of its logical position only, so the same token must give the same bits however many queries
share the launch, whatever the page size or page order, and whoever else is in the batch. Shapes are
Qwen3.5-35B-A3B's attention layers per rank (head dim 256: engine TP=8 2q/1kv, trainer TP=4 4q/1kv, TP=1
16q/2kv) plus head dims 128 and 64; ``test_segsplit_gpu.py`` adds the Qwen3.8-27B geometries, every
execution path and rows of many segments.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires an SM90 GPU")

GEOMETRIES = [  # (label, hq, hk, d)
    ("35b-engine-tp8", 2, 1, 256),
    ("35b-trainer-tp4", 4, 1, 256),
    ("35b-tp1", 16, 2, 256),
    ("d128-gqa", 8, 2, 128),
    ("d64-mha", 4, 4, 64),
]
PAGES = (16, 64, 256, 1056)
LENS = (1, 127, 128, 129, 700, 2049)


def _bits(t):
    return t.contiguous().view(torch.int16)


def _cu(lens):
    out = torch.zeros(len(lens) + 1, dtype=torch.int32, device="cuda")
    out[1:] = torch.tensor(lens, device="cuda").cumsum(0)
    return out


def _qkv(lens, hq, hk, d, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lens)
    mk = lambda h: torch.randn(T, h, d, device="cuda", generator=g).to(torch.bfloat16)  # noqa: E731
    return mk(hq), mk(hk), mk(hk)


def _paged(k, v, lens, page, seed):
    """K/V of each sequence in its own pages, in a shuffled page order, padding pages poisoned-finite."""
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    counts = [(n + page - 1) // page for n in lens]
    nblocks = sum(counts) + 2
    hk, d = k.shape[1:]
    kc = torch.full((nblocks, page, hk, d), 3.0, device="cuda", dtype=torch.bfloat16)
    vc = torch.full_like(kc, -2.0)
    order = torch.randperm(nblocks, device="cuda", generator=g)
    table = torch.zeros(len(lens), max(counts), dtype=torch.int32, device="cuda")
    at = src = 0
    for i, (n, c) in enumerate(zip(lens, counts)):
        blocks = order[at:at + c]
        table[i, :c] = blocks.to(torch.int32)
        flat_k = kc[blocks].reshape(c * page, hk, d)
        flat_v = vc[blocks].reshape(c * page, hk, d)
        flat_k[:n] = k[src:src + n]
        flat_v[:n] = v[src:src + n]
        kc[blocks] = flat_k.view(c, page, hk, d)
        vc[blocks] = flat_v.view(c, page, hk, d)
        at += c
        src += n
    return kc, vc, table


def _paged_call(q_rows, kc, vc, table, q_lens, k_lens, scale):
    from isoexec.ops.attention.gqa_sm90.integration import paged_core

    out = torch.empty_like(q_rows)
    paged_core(q_rows, kc, vc, cu_seqlens_q=_cu(q_lens), seqused_k=torch.tensor(k_lens, dtype=torch.int32, device="cuda"),
               block_table=table, max_seqlen_q=max(q_lens), max_seqlen_k=max(k_lens), scale=scale, out=out)
    return out


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize(("label", "hq", "hk", "d"), GEOMETRIES, ids=[g[0] for g in GEOMETRIES])
def test_trainer_rows_equal_paged_prefill_chunked_prefill_and_decode(label, hq, hk, d, page):
    from isoexec.ops.attention.gqa_sm90.integration import trainer_core

    lens = LENS
    q, k, v = _qkv(lens, hq, hk, d, seed=d * 7 + hq)
    scale = d ** -0.5
    trainer = trainer_core(q, k, v, _cu(lens), max(lens), scale).detach()
    kc, vc, table = _paged(k, v, lens, page, seed=page)

    # 1. whole-sequence prefill through the paged cache
    prefill = _paged_call(q, kc, vc, table, lens, lens, scale)
    assert torch.equal(_bits(prefill), _bits(trainer)), f"{label} page={page}: paged prefill != trainer"

    starts = [0, *torch.tensor(lens).cumsum(0).tolist()[:-1]]
    # 2. chunked prefill: the last third of each sequence as new rows over its full prefix
    q_lens = [max(1, n // 3) for n in lens]
    rows = torch.cat([q[s + n - m:s + n] for s, n, m in zip(starts, lens, q_lens)])
    chunk = _paged_call(rows, kc, vc, table, q_lens, lens, scale)
    want = torch.cat([trainer[s + n - m:s + n] for s, n, m in zip(starts, lens, q_lens)])
    assert torch.equal(_bits(chunk), _bits(want)), f"{label} page={page}: chunked prefill != trainer"

    # 3. one decode step for every sequence at once, each at its own position
    for pos_of in (lambda n: n - 1, lambda n: (n - 1) // 2):
        pos = [pos_of(n) for n in lens]
        rows = torch.stack([q[s + p] for s, p in zip(starts, pos)])
        dec = _paged_call(rows, kc, vc, table, [1] * len(lens), [p + 1 for p in pos], scale)
        want = torch.stack([trainer[s + p] for s, p in zip(starts, pos)])
        assert torch.equal(_bits(dec), _bits(want)), f"{label} page={page}: decode != trainer"


@pytest.mark.parametrize(("label", "hq", "hk", "d"), GEOMETRIES[:2], ids=[g[0] for g in GEOMETRIES[:2]])
def test_a_decode_row_does_not_depend_on_its_batch(label, hq, hk, d):
    """One sequence's decode row alone, in a batch of 64, and in a batch of 512 (max_num_seqs)."""
    lens = [3000] + [(37 * i) % 4000 + 1 for i in range(1, 512)]
    q, k, v = _qkv(lens, hq, hk, d, seed=5)
    kc, vc, table = _paged(k, v, lens, 16, seed=9)
    scale = d ** -0.5
    starts = [0, *torch.tensor(lens).cumsum(0).tolist()[:-1]]
    rows = torch.stack([q[s + n - 1] for s, n in zip(starts, lens)])
    ref = None
    for b in (1, 64, 512):
        out = _paged_call(rows[:b], kc, vc, table[:b], [1] * b, lens[:b], scale)[:1]
        ref = out if ref is None else ref
        assert torch.equal(_bits(out), _bits(ref)), f"{label}: decode row changed with batch {b}"
    # and it is the trainer's last row of that sequence
    from isoexec.ops.attention.gqa_sm90.integration import trainer_core

    t = trainer_core(q[:3000], k[:3000], v[:3000], _cu([3000]), 3000, scale)
    assert torch.equal(_bits(ref[0]), _bits(t[-1]))


def test_long_context_decode_matches_the_trainer_row():
    """The production maximum (max_model_len 10240) on the engine's TP=8 geometry."""
    from isoexec.ops.attention.gqa_sm90.integration import trainer_core

    n = 10240
    q, k, v = _qkv([n], 2, 1, 256, seed=11)
    scale = 256 ** -0.5
    t = trainer_core(q, k, v, _cu([n]), n, scale)
    kc, vc, table = _paged(k, v, [n], 16, seed=3)
    for p in (8191, 10239):
        dec = _paged_call(q[p:p + 1], kc, vc, table, [1], [p + 1], scale)
        assert torch.equal(_bits(dec), _bits(t[p:p + 1]))
