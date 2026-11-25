"""The segment split (spec.SEG_RECIPE): raw-bit relations between its execution paths, and its accuracy.

A row's KV axis is cut into fixed logical segments of SEG keys anchored at key 0. Each segment is one FA3
online-softmax pass, and a row's segment results are right-folded with ``fa3/seg_merge.h``. Two execution
paths run that program: FOLD (the unsplit kernel walks a work tile's segments and folds them in an
L2-resident per-CTA slot; the saturated path) and SPLIT (one work tile per segment, ``fa3/seg_combine.cu``).
What may change between calls -- the path, whether the split kernel runs at all (``max_seqlen_k`` vs SEG),
how many queries share a launch, page size and order, the chunked-prefill boundary, the packing, the SM
count the scheduler sees, the head split -- must not change a bit:

* fold == split == auto, for every row;
* one segment == the unsplit (ns1) kernel, whether the unsplit kernel runs, the split kernel writes the
  row directly, or the combine casts the row's only partial;
* trainer (packed varlen) rows == paged prefill == chunked prefill == decode rows;
* a decode row is the same alone and in batches of 64 and 256, and under any SM count;
* TP head splits give each rank exactly the TP=1 bits of its heads.

Shapes are Qwen3.8-27B's full-attention layers (head dim 256; per rank 6q/1kv at TP4, 12q/2kv at TP2,
24q/4kv at TP1) plus head dims 128 and 64. Small SEG values (640, 1280) put many segments inside short rows.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires an SM90 GPU")

GEOMETRIES = [  # (label, hq, hk, d)
    ("27b-tp4", 6, 1, 256),
    ("27b-tp2", 12, 2, 256),
    ("d128-gqa", 8, 2, 128),
    ("d64-mha", 4, 4, 64),
]
SEGS = (640, 1280, 2560)
PATHS = ("split", "fold", "hybrid")
LENS = (1, 79, 80, 81, 639, 640, 641, 1536, 2049, 5200)


def _cu(lens):
    out = torch.zeros(len(lens) + 1, dtype=torch.int32, device="cuda")
    out[1:] = torch.tensor(lens, device="cuda").cumsum(0)
    return out


def _starts(lens):
    return [0, *torch.tensor(lens).cumsum(0).tolist()[:-1]]


def _qkv(lens, hq, hk, d, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lens)
    mk = lambda h, s=1.0: (s * torch.randn(T, h, d, device="cuda", generator=g)).to(torch.bfloat16)  # noqa: E731
    # A scaled Q makes the softmax peaky, so segment maxima differ and the merge weights are not all ~1.
    return mk(hq, 2.0), mk(hk), mk(hk)


def _paged(k, v, lens, page, seed):
    """K/V of each sequence in its own pages, in a shuffled page order, unused page rows poisoned-finite."""
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


def _varlen(q, k, v, lens, seg, *, max_k=None, path="auto"):
    from isoexec.ops.attention.gqa_sm90 import ops

    cu = _cu(lens)
    return ops.varlen_fwd(q, k, v, cu, cu, max(lens), max_k or max(lens), q.shape[-1] ** -0.5, seg_tokens=seg,
                          seg_path=path)


def _paged_call(rows, kc, vc, table, q_lens, k_lens, seg, *, max_k=None, path="auto"):
    from isoexec.ops.attention.gqa_sm90 import ops

    return ops.paged_fwd(rows, kc, vc, _cu(q_lens), torch.tensor(k_lens, dtype=torch.int32, device="cuda"), table,
                         max(q_lens), max_k or max(k_lens), rows.shape[-1] ** -0.5, seg_tokens=seg, seg_path=path)


def _reference(q, k, v, lens):
    """Float64 causal GQA per sequence -> (out [T, Hq, D] float64, lse [Hq, T] float64)."""
    outs, lses = [], []
    hq, hk = q.shape[1], k.shape[1]
    for s, n in zip(_starts(lens), lens):
        qs, ks, vs = (t[s:s + n].double().transpose(0, 1) for t in (q, k, v))
        ks, vs = ks.repeat_interleave(hq // hk, 0), vs.repeat_interleave(hq // hk, 0)
        sc = qs @ ks.transpose(1, 2) * q.shape[-1] ** -0.5
        sc = sc.masked_fill(torch.ones(n, n, dtype=torch.bool, device="cuda").triu(1), float("-inf"))
        lses.append(torch.logsumexp(sc, -1))
        outs.append((sc.softmax(-1) @ vs).transpose(0, 1))
    return torch.cat(outs), torch.cat(lses, 1)


def _rows_within_one_segment(lens, seg):
    """Packed row indices whose border (position + 1) is <= seg: their bits are the unsplit kernel's."""
    return torch.tensor([s + i for s, n in zip(_starts(lens), lens) for i in range(min(n, seg))], device="cuda")


# --------------------------------------------------------------------------------------------- one segment


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("seg", SEGS)
@pytest.mark.parametrize(("label", "hq", "hk", "d"), GEOMETRIES, ids=[g[0] for g in GEOMETRIES])
def test_one_segment_rows_are_the_unsplit_kernels_bits_on_every_path(label, hq, hk, d, seg, path):
    """Rows with border <= SEG: segment split (split kernel + combine cast) == ns1 unsplit kernel, raw bits.

    Three paths reach such a row: the unsplit kernel (max_seqlen_k <= SEG), the split kernel writing a
    one-segment sequence directly, and the combine's cast of a long sequence's early rows."""
    q, k, v = _qkv(LENS, hq, hk, d, seed=d + hq)
    ns1, ns1_lse = _varlen(q, k, v, LENS, None)
    split, split_lse = _varlen(q, k, v, LENS, seg, path=path)
    idx = _rows_within_one_segment(LENS, seg)
    assert bitwise_equal(split[idx], ns1[idx]), f"{label} seg={seg}: one-segment rows differ from ns1"
    assert bitwise_equal(split_lse[:, idx], ns1_lse[:, idx]), f"{label} seg={seg}: one-segment lse differs from ns1"
    # Short sequences only: max_seqlen_k <= SEG runs the unsplit kernel; overstating max_seqlen_k forces the
    # split kernel (every sequence one segment, written directly) -- the same bits.
    short = [n for n in LENS if n <= seg]
    qs, ks, vs = (t[:sum(short)] for t in (q, k, v))
    a, _ = _varlen(qs, ks, vs, short, seg)
    b, _ = _varlen(qs, ks, vs, short, seg, max_k=4 * seg + 1, path=path)
    assert bitwise_equal(a, b), f"{label} seg={seg}: {path} with max_seqlen_k > SEG != the unsplit kernel"


@pytest.mark.parametrize("seg", SEGS)
@pytest.mark.parametrize(("label", "hq", "hk", "d"), GEOMETRIES, ids=[g[0] for g in GEOMETRIES])
def test_fold_split_and_auto_paths_are_one_program(label, hq, hk, d, seg):
    """THE path relation: the saturated path (in-CTA fold) == the split path (partials + combine), raw bits,
    for output and lse, packed and paged, prefill and decode."""
    lens = LENS + (12000,)
    q, k, v = _qkv(lens, hq, hk, d, seed=17 * d + hq)
    got = {p: _varlen(q, k, v, lens, seg, path=p) for p in (*PATHS, "auto")}
    for p in ("fold", "hybrid", "auto"):
        assert bitwise_equal(got[p][0], got["split"][0]), f"{label} seg={seg}: {p} out != split out"
        assert bitwise_equal(got[p][1], got["split"][1]), f"{label} seg={seg}: {p} lse != split lse"
    kc, vc, table = _paged(k, v, lens, 128, seed=2)
    rows = torch.stack([q[s + n - 1] for s, n in zip(_starts(lens), lens)])
    dec = {p: _paged_call(rows, kc, vc, table, [1] * len(lens), list(lens), seg, path=p) for p in PATHS}
    for p in ("fold", "hybrid"):
        assert bitwise_equal(dec[p][0], dec["split"][0]) and bitwise_equal(dec[p][1], dec["split"][1]), p
    # a mixed step: every sequence's last row as decode plus a chunk of the longest one, in one call
    mixed_rows = torch.cat([rows, q[sum(lens) - 3000:]])
    q_lens, k_lens = [1] * len(lens) + [3000], list(lens) + [lens[-1]]
    kc2, vc2, table2 = _paged(torch.cat([k, k[sum(lens) - lens[-1]:]]), torch.cat([v, v[sum(lens) - lens[-1]:]]),
                              k_lens, 128, seed=3)
    mixed = {p: _paged_call(mixed_rows, kc2, vc2, table2, q_lens, k_lens, seg, path=p)[0] for p in (*PATHS, "auto")}
    want_mixed = torch.cat([torch.stack([got["split"][0][s + n - 1] for s, n in zip(_starts(lens), lens)]),
                            got["split"][0][sum(lens) - 3000:]])
    for p, o in mixed.items():
        assert bitwise_equal(o, want_mixed), f"{label} seg={seg}: mixed step on {p} != packed rows"
    want = torch.stack([got["split"][0][s + n - 1] for s, n in zip(_starts(lens), lens)])
    assert bitwise_equal(dec["split"][0], want), f"{label} seg={seg}: paged decode != packed rows"


# --------------------------------------------------------------------------------------------- the seam


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("page", (16, 80, 128, 1040))
@pytest.mark.parametrize("seg", SEGS)
@pytest.mark.parametrize(("label", "hq", "hk", "d"), GEOMETRIES, ids=[g[0] for g in GEOMETRIES])
def test_trainer_rows_equal_paged_prefill_chunked_prefill_and_decode(label, hq, hk, d, seg, page, path):
    """The trainer always runs auto; the engine's calls run the parametrized path."""
    lens = LENS
    q, k, v = _qkv(lens, hq, hk, d, seed=d * 7 + hq)
    trainer, _ = _varlen(q, k, v, lens, seg)
    kc, vc, table = _paged(k, v, lens, page, seed=page)

    prefill = _paged_call(q, kc, vc, table, lens, lens, seg, path=path)[0]
    assert bitwise_equal(prefill, trainer), f"{label} seg={seg} page={page}: paged prefill != trainer"

    starts = _starts(lens)
    for frac in (3, 2):  # chunk boundaries that fall on and off segment boundaries
        q_lens = [max(1, n // frac) for n in lens]
        rows = torch.cat([q[s + n - m:s + n] for s, n, m in zip(starts, lens, q_lens)])
        chunk = _paged_call(rows, kc, vc, table, q_lens, lens, seg, path=path)[0]
        want = torch.cat([trainer[s + n - m:s + n] for s, n, m in zip(starts, lens, q_lens)])
        assert bitwise_equal(chunk, want), f"{label} seg={seg} page={page}: chunked prefill 1/{frac} != trainer"

    for pos_of in (lambda n: n - 1, lambda n: (n - 1) // 2, lambda n: min(n - 1, seg)):
        pos = [pos_of(n) for n in lens]
        rows = torch.stack([q[s + p] for s, p in zip(starts, pos)])
        dec = _paged_call(rows, kc, vc, table, [1] * len(lens), [p + 1 for p in pos], seg, path=path)[0]
        want = torch.stack([trainer[s + p] for s, p in zip(starts, pos)])
        assert bitwise_equal(dec, want), f"{label} seg={seg} page={page}: decode != trainer"


@pytest.mark.parametrize("seg", (640, 2560))
@pytest.mark.parametrize(("label", "hq", "hk", "d"), GEOMETRIES[:2], ids=[g[0] for g in GEOMETRIES[:2]])
def test_a_decode_row_does_not_depend_on_its_batch_or_the_bound(label, hq, hk, d, seg):
    """One sequence's decode row alone, in batches of 64 and 256 (mixed lengths), and with max_seqlen_k
    overstated to a 32k max_model_len (what a captured CUDA graph passes)."""
    lens = [3000] + [(37 * i) % 4000 + 1 for i in range(1, 256)]
    q, k, v = _qkv(lens, hq, hk, d, seed=5)
    kc, vc, table = _paged(k, v, lens, 16, seed=9)
    rows = torch.stack([q[s + n - 1] for s, n in zip(_starts(lens), lens)])
    ref = _paged_call(rows[:1], kc, vc, table[:1], [1], lens[:1], seg)[0]
    for b in (1, 64, 256):
        for max_k in (None, 32768):
            for path in ("auto", *PATHS):
                out = _paged_call(rows[:b], kc, vc, table[:b], [1] * b, lens[:b], seg, max_k=max_k, path=path)[0][:1]
                assert bitwise_equal(out, ref), f"{label} seg={seg}: decode row changed with batch {b}, {max_k}, {path}"
    t, _ = _varlen(q[:3000], k[:3000], v[:3000], [3000], seg)
    assert bitwise_equal(ref[0], t[-1])


@pytest.mark.parametrize("path", (1, 2, 3))  # -(SEG + path): 1 split, 2 fold, 3 hybrid
@pytest.mark.parametrize("sm_margin", (0, 37, 100, 131))
def test_the_sm_count_the_scheduler_sees_changes_no_bit(sm_margin, path):
    """FA3's persistent scheduler sizes its grid from the SM count; the segment split must not see it."""
    from isoexec.ops.attention.gqa_sm90 import ops

    lens = [5000, 1, 900, 3000, 33]
    q, k, v = _qkv(lens, 6, 1, 256, seed=2)
    cu = _cu(lens)
    ref, ref_lse = _varlen(q, k, v, lens, 640)
    o, lse, _, _ = ops._ops().fwd(q, k, v, None, None, None, None, cu, cu, None, None, None, max(lens), max(lens), None,
                                  None, None, None, None, None, None, None, None, 256 ** -0.5, True, -1, 0, 0, 0.0,
                                  True, None, -(640 + path), None, sm_margin)
    assert bitwise_equal(o, ref) and bitwise_equal(lse, ref_lse)


@pytest.mark.parametrize("seg", (640, 2560))
def test_tp_head_splits_give_each_rank_its_tp1_bits(seg):
    """Qwen3.8-27B: TP1 24q/4kv; TP2 ranks 12q/2kv; TP4 ranks 6q/1kv."""
    lens = [1536, 700, 5200, 1]
    q, k, v = _qkv(lens, 24, 4, 256, seed=13)
    full, _ = _varlen(q, k, v, lens, seg)
    for tp in (2, 4):
        hq, hk = 24 // tp, 4 // tp
        for r in range(tp):
            part, _ = _varlen(q[:, r * hq:(r + 1) * hq].contiguous(), k[:, r * hk:(r + 1) * hk].contiguous(),
                              v[:, r * hk:(r + 1) * hk].contiguous(), lens, seg)
            assert bitwise_equal(part, full[:, r * hq:(r + 1) * hq]), f"tp{tp} rank {r} seg={seg}"


@pytest.mark.parametrize("path", PATHS)
def test_forward_is_run_to_run_deterministic(path):
    lens = [8192, 3000, 1]
    q, k, v = _qkv(lens, 6, 1, 256, seed=4)
    o0, l0 = _varlen(q, k, v, lens, 640, path=path)
    for _ in range(4):
        torch.empty(1 << 28, device="cuda").fill_(float("nan"))  # dirty the allocator's free blocks (and scratch)
        o, lse = _varlen(q, k, v, lens, 640, path=path)
        assert bitwise_equal(o, o0) and bitwise_equal(lse, l0)


@pytest.mark.parametrize("path", PATHS)
def test_cuda_graph_replay_equals_eager(path):
    """Decode captured with max_seqlen_k = max_model_len replays the eager bits for new lengths."""
    seg, B, max_len = 640, 32, 8192
    lens = [(311 * i) % 7000 + 1 for i in range(B)]
    q, k, v = _qkv(lens, 6, 1, 256, seed=21)
    kc, vc, table = _paged(k, v, lens, 16, seed=1)
    rows = torch.stack([q[s + n - 1] for s, n in zip(_starts(lens), lens)])
    from isoexec.ops.attention.gqa_sm90 import ops

    cu = _cu([1] * B)
    used = torch.tensor([1] * B, dtype=torch.int32, device="cuda")
    out = torch.empty_like(rows)
    call = lambda: ops.paged_fwd(rows, kc, vc, cu, used, table, 1, max_len, 256 ** -0.5, out=out,  # noqa: E731
                                 seg_tokens=seg, seg_path=path)
    call()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    used.copy_(torch.tensor(lens, dtype=torch.int32, device="cuda"))
    g.replay()
    torch.cuda.synchronize()
    want = _paged_call(rows, kc, vc, table, [1] * B, lens, seg)[0]
    assert bitwise_equal(out, want)


# --------------------------------------------------------------------------------------------- accuracy


@pytest.mark.parametrize("seg", SEGS)
@pytest.mark.parametrize(("label", "hq", "hk", "d"), GEOMETRIES, ids=[g[0] for g in GEOMETRIES])
def test_accuracy_against_float64_matches_the_unsplit_kernels(label, hq, hk, d, seg):
    """The fold of segment partials is as accurate as one pass: max error within 1.25x of ns1's + 1 bf16 ulp."""
    q, k, v = _qkv(LENS, hq, hk, d, seed=3 * d + hq)
    ref, ref_lse = _reference(q, k, v, LENS)
    ns1, ns1_lse = _varlen(q, k, v, LENS, None)
    split, split_lse = _varlen(q, k, v, LENS, seg)
    e_ns1 = (ns1.double() - ref).abs().max().item()
    e_split = (split.double() - ref).abs().max().item()
    assert e_split <= 1.25 * e_ns1 + 2 ** -8, (label, seg, e_split, e_ns1)
    assert (split_lse.double() - ref_lse).abs().max().item() < 1e-4


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(("hq", "hk", "d"), [(4, 1, 128), (6, 1, 256)])
def test_backward_runs_on_the_combined_lse_and_is_correct(hq, hk, d, path):
    """The trainer's backward is FA3's own on the forward's saved output and the folded LSE."""
    from isoexec.ops.attention.gqa_sm90 import ops

    lens = [3000, 700, 1]
    q, k, v = _qkv(lens, hq, hk, d, seed=8)
    cu = _cu(lens)
    leaves = [t.detach().requires_grad_() for t in (q, k, v)]
    out = ops.varlen_attention(*leaves, cu, cu, max(lens), max(lens), d ** -0.5, seg_tokens=640)
    assert bitwise_equal(out, _varlen(q, k, v, lens, 640, path=path)[0])
    g = torch.randn_like(out)
    out.backward(g)
    ref_leaves = [t.detach().double().requires_grad_() for t in (q, k, v)]
    ref_out, _ = _reference_autograd(*ref_leaves, lens)
    ref_out.backward(g.double())
    for name, a, b in zip(("dq", "dk", "dv"), leaves, ref_leaves):
        err = (a.grad.double() - b.grad).abs().max().item()
        scale = b.grad.abs().max().item()
        assert err <= 2e-2 * scale, (name, err, scale)


def _reference_autograd(q, k, v, lens):
    outs = []
    hq, hk = q.shape[1], k.shape[1]
    for s, n in zip(_starts(lens), lens):
        qs, ks, vs = (t[s:s + n].transpose(0, 1) for t in (q, k, v))
        ks, vs = ks.repeat_interleave(hq // hk, 0), vs.repeat_interleave(hq // hk, 0)
        sc = qs @ ks.transpose(1, 2) * q.shape[-1] ** -0.5
        sc = sc.masked_fill(torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
        outs.append((sc.softmax(-1) @ vs).transpose(0, 1))
    return torch.cat(outs), None


# --------------------------------------------------------------------------------------------- refusals


def test_refusals():
    from isoexec.ops.attention.gqa_sm90 import ops

    lens = [100]
    q, k, v = _qkv(lens, 6, 1, 256, seed=0)
    cu = _cu(lens)
    with pytest.raises(ValueError, match="multiple of 640"):
        ops.varlen_fwd(q, k, v, cu, cu, 100, 100, 0.0625, seg_tokens=1000)
    with pytest.raises(ValueError, match="seg_path"):
        ops.varlen_fwd(q, k, v, cu, cu, 100, 100, 0.0625, seg_tokens=640, seg_path="tree")
    with pytest.raises(RuntimeError, match="path 0"):
        ops._ops().fwd(q, k, v, None, None, None, None, cu, cu, None, None, None, 100, 100, None, None, None, None,
                       None, None, None, None, None, 0.0625, True, -1, 0, 0, 0.0, True, None, -644, None, 0)
    with pytest.raises(RuntimeError, match="proportional split-KV"):
        ops._ops().fwd(q, k, v, None, None, None, None, cu, cu, None, None, None, 100, 100, None, None, None, None,
                       None, None, None, None, None, 0.0625, True, -1, 0, 0, 0.0, True, None, 3, None, 0)
    kc, vc, table = _paged(k, v, lens, 16, seed=0)
    with pytest.raises(ValueError, match="needs max_seqlen_k"):
        ops.paged_fwd(q, kc, vc, cu, torch.tensor(lens, dtype=torch.int32, device="cuda"), table, 100, None, 0.0625,
                      seg_tokens=640)


# --------------------------------------------------------------------------------------------- the cores


def test_the_cores_run_the_segment_split_when_switched_on(monkeypatch):
    """trainer_core / paged_core with CORE_USES_SEGMENT_SPLIT: trainer rows == engine rows, and the trainer's
    autograd backward runs on the segment split's forward."""
    from isoexec.ops.attention.gqa_sm90 import integration, spec

    monkeypatch.setattr(spec, "CORE_USES_SEGMENT_SPLIT", True)
    assert integration.core_seg_tokens() == spec.SEG_TOKENS
    lens = [3 * spec.SEG_TOKENS + 7, 300, 1]
    q, k, v = _qkv(lens, 6, 1, 256, seed=23)
    leaves = [t.detach().requires_grad_() for t in (q, k, v)]
    trainer = integration.trainer_core(*leaves, _cu(lens), max(lens), 256 ** -0.5)
    trainer.float().square().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in leaves)
    want, _ = _varlen(q, k, v, lens, spec.SEG_TOKENS)
    assert bitwise_equal(trainer.detach(), want)
    kc, vc, table = _paged(k, v, lens, 128, seed=4)
    rows = torch.stack([q[s + n - 1] for s, n in zip(_starts(lens), lens)])
    out = torch.empty_like(rows)
    integration.paged_core(rows, kc, vc, cu_seqlens_q=_cu([1] * len(lens)),
                           seqused_k=torch.tensor(lens, dtype=torch.int32, device="cuda"), block_table=table,
                           max_seqlen_q=1, max_seqlen_k=max(lens), scale=256 ** -0.5, out=out)
    assert bitwise_equal(out, torch.stack([want[s + n - 1] for s, n in zip(_starts(lens), lens)]))


@pytest.mark.parametrize(("label", "hq", "hk", "d"), GEOMETRIES, ids=[g[0] for g in GEOMETRIES])
def test_hybrid_splits_and_folds_in_one_launch(label, hq, hk, d):
    """A step of 4 decodes at 40000 keys (split: one tile each, far over the per-SM share) and a 4096-token
    chunk at 6000 keys (folded: many short tiles) in one call: every row equals the packed trainer row, and
    the launch did run the combine (something was split) and the fold (the chunk spans 3 segments)."""
    from torch.profiler import ProfilerActivity, profile

    seg = 2560
    lens = [40000, 40000, 40000, 40000, 6000]
    q, k, v = _qkv(lens, hq, hk, d, seed=31 + d)
    trainer, _ = _varlen(q, k, v, lens, seg)
    kc, vc, table = _paged(k, v, lens, 128, seed=6)
    starts = _starts(lens)
    rows = torch.cat([torch.stack([q[s + n - 1] for s, n in zip(starts[:4], lens[:4])]), q[starts[4] + 1904:starts[4] + 6000]])
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = _paged_call(rows, kc, vc, table, [1] * 4 + [4096], lens, seg, path="hybrid")[0]
        torch.cuda.synchronize()
    want = torch.cat([torch.stack([trainer[s + n - 1] for s, n in zip(starts[:4], lens[:4])]),
                      trainer[starts[4] + 1904:starts[4] + 6000]])
    assert bitwise_equal(out, want)
    assert any("seg_combine_kernel" in e.name for e in prof.events()), "hybrid split nothing: the relation is vacuous"


# --------------------------------------------------------------------------------------------- engine workspace


@pytest.mark.parametrize("batch", (1, 8, 64))
def test_engine_core_captures_without_allocating_and_replays_the_eager_bits(batch):
    """paged_core under CUDA-graph capture (vLLM FULL_DECODE_ONLY): with the workspace reserved from the
    engine limits, capture allocates nothing (scheduler metadata, flags, fold scratch and split partials come
    from the workspace), and replays at new lengths -- many segments -- equal eager calls and the trainer."""
    from isoexec.ops.attention.gqa_sm90 import integration

    max_len, page = 12 * 2560 + 100, 128
    lens = [(7919 * (i + 1)) % (max_len - 1) + 1 for i in range(batch)]
    q, k, v = _qkv(lens, 6, 1, 256, seed=41 + batch)
    kc, vc, table = _paged(k, v, lens, page, seed=batch)
    width = -(-max_len // page)
    table = torch.nn.functional.pad(table, (0, width - table.shape[1]))
    rows = torch.stack([q[s + n - 1] for s, n in zip(_starts(lens), lens)])
    integration.reserve_paged_workspace(device="cuda", max_tokens=256, max_seqs=256, num_heads=6, head_dim_v=256)
    cu = _cu([1] * batch)
    used = torch.ones(batch, dtype=torch.int32, device="cuda")
    out = torch.empty_like(rows)
    call = lambda: integration.paged_core(rows, kc, vc, cu_seqlens_q=cu, seqused_k=used, block_table=table,  # noqa: E731
                                          max_seqlen_q=1, max_seqlen_k=max_len, scale=256 ** -0.5, out=out)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    ws = integration.paged_workspace("cuda")
    before = torch.cuda.memory_allocated()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    assert torch.cuda.memory_allocated() == before, "the captured call allocated"
    assert integration.paged_workspace("cuda") is ws
    used.copy_(torch.tensor(lens, dtype=torch.int32, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    replay = out.clone()
    call()
    assert bitwise_equal(replay, out)
    trainer, _ = _varlen(q, k, v, lens, spec_seg())
    assert bitwise_equal(replay, torch.stack([trainer[s + n - 1] for s, n in zip(_starts(lens), lens)]))


def spec_seg():
    from isoexec.ops.attention.gqa_sm90 import spec

    return spec.SEG_TOKENS


def test_a_capture_larger_than_the_reservation_is_refused_not_allocated(monkeypatch):
    from isoexec.ops.attention.gqa_sm90 import integration

    monkeypatch.setattr(integration, "_WORKSPACE", {})
    lens = [3000]
    q, k, v = _qkv(lens, 6, 1, 256, seed=2)
    kc, vc, table = _paged(k, v, lens, 128, seed=2)
    out = torch.empty_like(q[:1])
    cu, used = _cu([1]), torch.tensor(lens, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="reserve it before capture"):
        with torch.cuda.graph(graph):
            integration.paged_core(q[:1], kc, vc, cu_seqlens_q=cu, seqused_k=used, block_table=table,
                                   max_seqlen_q=1, max_seqlen_k=3000, scale=0.0625, out=out)


def test_a_workspace_too_small_for_partials_folds_with_the_same_bits(monkeypatch):
    """Auto path with no room for split partials runs the fold path: same bits as the split path."""
    from isoexec.ops.attention.gqa_sm90 import ops

    lens = [9000, 20000]
    q, k, v = _qkv(lens, 6, 1, 256, seed=3)
    kc, vc, table = _paged(k, v, lens, 128, seed=3)
    rows = torch.stack([q[s + n - 1] for s, n in zip(_starts(lens), lens)])
    args = (rows, kc, vc, _cu([1, 1]), torch.tensor(lens, dtype=torch.int32, device="cuda"), table, 1, max(lens),
            256 ** -0.5)
    # Exactly the per-call regions (lse, metadata, flags, fold scratch at the d256 tile height of 128 rows) plus
    # 64 floats: nothing left for the partials.
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    small = torch.empty(64 + 64 + 64 + sm * 128 * (256 + 8) + 64, dtype=torch.float32, device="cuda")
    a = ops.paged_fwd(*args, seg_tokens=2560, workspace=small)[0]
    b = ops.paged_fwd(*args, seg_tokens=2560, seg_path="split")[0]
    assert bitwise_equal(a, b)
    with pytest.raises(RuntimeError, match="too small for the split partials"):
        ops.paged_fwd(*args, seg_tokens=2560, seg_path="split", workspace=small)
