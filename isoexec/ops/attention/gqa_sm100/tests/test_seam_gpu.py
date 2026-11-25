"""THE SEAM: trainer rows == engine paged prefill rows == chunked prefill rows == decode rows, raw bits.

The trainer's ``sm100_gqa_segtree_v2_varlen`` core attends packed THD rows; the engine's
``sm100_gqa_segtree_v2_paged`` core attends the same tokens through a paged KV cache with
right-aligned queries (vLLM's layout), including the interleaved K|V pages of vLLM's ``LBNHC``
layout read in place. Both are the same forward program at the recipe's fixed SEG, so the same
(token, head) must give the same bits however many queries share the launch, whatever the page size
or page order, whoever else is in the batch, whether the call is a CUDA-graph replay, and however
the heads are split across tensor-parallel ranks.

Geometries are the release models' attention layers per rank: Qwen3.5-35B-A3B (16q/2kv, d256),
Qwen3.5-9B (16q/4kv, d256), Qwen3.5-0.8B (8q/2kv, d256) and Qwen1.5-MoE-A2.7B (16q/16kv, d128) at
TP 1/2/4/8, where a rank past the KV-head count holds one replicated KV head.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(live_device_arch() != "sm100", reason="requires an SM100 GPU")

#: (label, hq, hk, d): the unsplit layers; per-rank shapes come from ``_rank_heads``.
MODELS = [
    ("qwen3.5-35b-a3b", 16, 2, 256),
    ("qwen3.5-9b", 16, 4, 256),
    ("qwen3.6-27b", 24, 4, 256),
    ("qwen3.5-0.8b", 8, 2, 256),
    ("qwen1.5-moe-a2.7b", 16, 16, 128),
]
TPS = (1, 2, 4, 8)
PAGES = (128, 1152)
LENS = (1, 127, 128, 129, 700, 2049, 2304)


def _bits(t):
    return t.contiguous().view(torch.int16)


def _cu(lens):
    out = torch.zeros(len(lens) + 1, dtype=torch.int32, device="cuda")
    out[1:] = torch.tensor(lens, device="cuda").cumsum(0)
    return out


def _qkv(lens, hq, hk, d, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lens)
    mk = lambda h: torch.randn(T, h, d, device="cuda", generator=g).to(torch.bfloat16)
    return mk(hq), mk(hk), mk(hk)


def _rank_heads(hq, hk, tp, rank):
    """The (q heads, kv heads) slices rank ``rank`` of ``tp`` attends (nn.Attention.set_local_heads)."""
    q_local = hq // tp
    q = slice(rank * q_local, (rank + 1) * q_local)
    if tp <= hk:
        kv_local = hk // tp
        return q, slice(rank * kv_local, (rank + 1) * kv_local)
    kv = (rank * q_local) // (hq // hk)
    return q, slice(kv, kv + 1)


def _paged(k, v, lens, page, seed, *, interleaved):
    """K/V of each sequence in its own pages in a shuffled page order, padding poisoned-finite.

    ``interleaved`` lays the cache out the way vLLM's ``LBNHC`` layout does -- one ``[page, hk, 2D]``
    block holding K|V per (token, head) -- and returns the views ``split_paged_kv_cache`` gives the
    engine, so the kernel reads strided K and V in place.
    """
    from isoexec.ops.attention.paged_kv import split_paged_kv_cache

    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    counts = [(n + page - 1) // page for n in lens]
    nblocks = sum(counts) + 2
    hk, d = k.shape[1:]
    if interleaved:
        storage = torch.empty(nblocks, page, hk, 2 * d, device="cuda", dtype=torch.bfloat16)
        storage[..., :d], storage[..., d:] = 3.0, -2.0
        kc, vc = split_paged_kv_cache(storage.permute(0, 2, 1, 3), num_kv_heads=hk, head_size=d)
        assert not kc.is_contiguous() and kc.stride(0) == page * kc.stride(1)
    else:
        kc = torch.full((nblocks, page, hk, d), 3.0, device="cuda", dtype=torch.bfloat16)
        vc = torch.full_like(kc, -2.0)
    order = torch.randperm(nblocks, device="cuda", generator=g)
    table = torch.zeros(len(lens), max(counts), dtype=torch.int32, device="cuda")
    at = src = 0
    for i, (n, c) in enumerate(zip(lens, counts)):
        blocks = order[at:at + c]
        table[i, :c] = blocks.to(torch.int32)
        for j in range(c):
            take = min(page, n - j * page)
            kc[blocks[j], :take] = k[src + j * page:src + j * page + take]
            vc[blocks[j], :take] = v[src + j * page:src + j * page + take]
        at += c
        src += n
    return kc, vc, table


def _paged_call(q_rows, kc, vc, table, q_lens, k_lens, scale, max_k=None):
    from isoexec.ops.attention.gqa_sm100.integration import paged_core

    out = torch.empty_like(q_rows)
    paged_core(q_rows, kc, vc, cu_seqlens_q=_cu(q_lens),
               seqused_k=torch.tensor(k_lens, dtype=torch.int32, device="cuda"), block_table=table,
               max_seqlen_q=max(q_lens), max_seqlen_k=max_k or max(k_lens), scale=scale, out=out)
    return out


def _trainer(q, k, v, lens, scale):
    from isoexec.ops.attention.gqa_sm100.integration import trainer_core

    return trainer_core(q, k, v, _cu(lens), max(lens), scale).detach()


def _seam(q, k, v, lens, page, scale, *, interleaved, seed):
    """Trainer rows against whole prefill, chunked prefill and decode through one paged cache."""
    trainer = _trainer(q, k, v, lens, scale)
    kc, vc, table = _paged(k, v, lens, page, seed, interleaved=interleaved)
    prefill = _paged_call(q, kc, vc, table, lens, lens, scale)
    assert torch.equal(_bits(prefill), _bits(trainer)), "paged prefill != trainer"

    starts = [0, *torch.tensor(lens).cumsum(0).tolist()[:-1]]
    q_lens = [max(1, n // 3) for n in lens]
    rows = torch.cat([q[s + n - m:s + n] for s, n, m in zip(starts, lens, q_lens)])
    chunk = _paged_call(rows, kc, vc, table, q_lens, lens, scale)
    want = torch.cat([trainer[s + n - m:s + n] for s, n, m in zip(starts, lens, q_lens)])
    assert torch.equal(_bits(chunk), _bits(want)), "chunked prefill != trainer"

    for pos_of in (lambda n: n - 1, lambda n: (n - 1) // 2):
        pos = [pos_of(n) for n in lens]
        rows = torch.stack([q[s + p] for s, p in zip(starts, pos)])
        dec = _paged_call(rows, kc, vc, table, [1] * len(lens), [p + 1 for p in pos], scale)
        want = torch.stack([trainer[s + p] for s, p in zip(starts, pos)])
        assert torch.equal(_bits(dec), _bits(want)), "decode != trainer"
    return trainer


@pytest.mark.parametrize("interleaved", (False, True), ids=("dense-pages", "vllm-lbnhc"))
@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize(("label", "hq", "hk", "d"), MODELS, ids=[m[0] for m in MODELS])
def test_trainer_rows_equal_paged_prefill_chunked_prefill_and_decode(label, hq, hk, d, page, interleaved):
    q, k, v = _qkv(LENS, hq, hk, d, seed=d * 7 + hq + hk)
    _seam(q, k, v, LENS, page, d ** -0.5, interleaved=interleaved, seed=page)


@pytest.mark.parametrize(("label", "hq", "hk", "d"), MODELS, ids=[m[0] for m in MODELS])
def test_bits_do_not_depend_on_the_tensor_parallel_head_split(label, hq, hk, d):
    """Every rank's trainer and decode rows at TP 2/4/8 are the TP=1 rows of the same heads."""
    lens = (300, 1, 2049, 128)
    q, k, v = _qkv(lens, hq, hk, d, seed=3 * hq + d)
    scale = d ** -0.5
    full = _trainer(q, k, v, lens, scale)
    starts = [0, *torch.tensor(lens).cumsum(0).tolist()[:-1]]
    for tp in TPS[1:]:
        for rank in range(tp):
            qs, ks = _rank_heads(hq, hk, tp, rank)
            qr, kr, vr = q[:, qs].contiguous(), k[:, ks].contiguous(), v[:, ks].contiguous()
            got = _seam(qr, kr, vr, lens, 128, scale, interleaved=True, seed=tp * 8 + rank)
            assert torch.equal(_bits(got), _bits(full[:, qs])), f"{label} tp={tp} rank={rank}: heads differ"
            kc, vc, table = _paged(kr, vr, lens, 128, seed=rank, interleaved=False)
            rows = torch.stack([qr[s + n - 1] for s, n in zip(starts, lens)])
            dec = _paged_call(rows, kc, vc, table, [1] * len(lens), list(lens), scale)
            want = torch.stack([full[s + n - 1, qs] for s, n in zip(starts, lens)])
            assert torch.equal(_bits(dec), _bits(want)), f"{label} tp={tp} rank={rank}: decode differs"


@pytest.mark.parametrize("d", (128, 256))
def test_a_row_does_not_depend_on_its_head_ratio(d):
    """A query head's rows are the same whatever ratio packs it (1, 2, 4, and 6, which no tile holds whole and
    which runs as three ratio-2 launches): ratio 6 therefore writes the rows a whole-ratio launch would."""
    lens, hk = (700, 1, 2049, 129), 2
    q, k, v = _qkv(lens, 6 * hk, hk, d, seed=d + 6)
    scale = d ** -0.5
    grouped = q.view(-1, hk, 6, d)
    want = None
    for ratio in (1, 2, 4, 6):
        got = _trainer(grouped[:, :, :ratio].reshape(-1, hk * ratio, d).contiguous(), k, v, lens, scale)
        got = got.view(-1, hk, ratio, d)[:, :, 0]
        want = got if want is None else want
        assert torch.equal(_bits(got), _bits(want)), f"ratio {ratio} changed the bits of head 0"
    kc, vc, table = _paged(k, v, lens, 128, seed=1, interleaved=True)
    full = _trainer(q, k, v, lens, scale)
    assert torch.equal(_bits(_paged_call(q, kc, vc, table, list(lens), list(lens), scale)), _bits(full))


@pytest.mark.parametrize(("label", "hq", "hk", "d"), [MODELS[0], MODELS[3]], ids=[MODELS[0][0], MODELS[3][0]])
def test_a_decode_row_does_not_depend_on_its_batch(label, hq, hk, d):
    """One sequence's decode row alone, in a batch of 64, and in a batch of 512."""
    hq, hk = hq // 8, max(1, hk // 8)  # the TP=8 rank, the engine's geometry
    lens = [3000] + [(37 * i) % 1500 + 1 for i in range(1, 512)]
    q, k, v = _qkv(lens, hq, hk, d, seed=5)
    kc, vc, table = _paged(k, v, lens, 128, seed=9, interleaved=True)
    scale = d ** -0.5
    starts = [0, *torch.tensor(lens).cumsum(0).tolist()[:-1]]
    rows = torch.stack([q[s + n - 1] for s, n in zip(starts, lens)])
    ref = None
    for b in (1, 64, 512):
        out = _paged_call(rows[:b], kc, vc, table[:b], [1] * b, lens[:b], scale)[:1]
        ref = out if ref is None else ref
        assert torch.equal(_bits(out), _bits(ref)), f"{label}: decode row changed with batch {b}"
    t = _trainer(q[:3000], k[:3000], v[:3000], [3000], scale)
    assert torch.equal(_bits(ref[0]), _bits(t[-1]))


@pytest.mark.parametrize("n", (10240, 40000))
def test_long_context_decode_matches_the_trainer_row(n):
    """The production maximum (10240), and a context past SEG_CONTEXT, on the 35B engine geometry."""
    q, k, v = _qkv([n], 2, 1, 256, seed=11)
    scale = 256 ** -0.5
    t = _trainer(q, k, v, [n], scale)
    kc, vc, table = _paged(k, v, [n], 1152, seed=3, interleaved=True)
    for p in (8191, n // 2 + 77, n - 1):
        dec = _paged_call(q[p:p + 1], kc, vc, table, [1], [p + 1], scale)
        assert torch.equal(_bits(dec), _bits(t[p:p + 1]))


@pytest.mark.parametrize(("label", "hq", "hk", "d"), [MODELS[0], MODELS[3]], ids=[MODELS[0][0], MODELS[3][0]])
def test_graph_replay_with_a_capture_time_kv_bound_reproduces_eager_rows(label, hq, hk, d):
    """vLLM captures decode with ``max_seqlen_k = max_model_len``; the plan it sizes changes no bit."""
    hq, hk = hq // 4, max(1, hk // 4)
    lens = [17, 2048, 900, 5000, 1, 4097, 333, 128]
    q, k, v = _qkv(lens, hq, hk, d, seed=13)
    kc, vc, table = _paged(k, v, lens, 128, seed=17, interleaved=True)
    scale = d ** -0.5
    starts = [0, *torch.tensor(lens).cumsum(0).tolist()[:-1]]
    rows = torch.stack([q[s + n - 1] for s, n in zip(starts, lens)])
    eager = _paged_call(rows, kc, vc, table, [1] * len(lens), lens, scale)
    bounded = _paged_call(rows, kc, vc, table, [1] * len(lens), lens, scale, max_k=10240)
    assert torch.equal(_bits(bounded), _bits(eager)), f"{label}: the KV bound changed the bits"

    from isoexec.ops.attention.gqa_sm100.integration import paged_core

    cu_q = _cu([1] * len(lens))
    used = torch.tensor(lens, dtype=torch.int32, device="cuda")
    out = torch.empty_like(rows)
    call = lambda: paged_core(rows, kc, vc, cu_seqlens_q=cu_q, seqused_k=used, block_table=table,
                              max_seqlen_q=1, max_seqlen_k=10240, scale=scale, out=out)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            call()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for step in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(out), _bits(eager)), f"{label}: graph replay {step} != eager"
    # the same graph serves another step's lengths and rows, as an engine replays it
    shorter = [max(1, n - 5) for n in lens]
    used.copy_(torch.tensor(shorter, dtype=torch.int32, device="cuda"))
    rows.copy_(torch.stack([q[s + m - 1] for s, m in zip(starts, shorter)]))
    graph.replay()
    torch.cuda.synchronize()
    again = _paged_call(rows.clone(), kc, vc, table, [1] * len(lens), shorter, scale)
    assert torch.equal(_bits(out), _bits(again)), f"{label}: replay at new lengths != eager"
