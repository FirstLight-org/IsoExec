"""The head-dim-256 decode leaf (``csrc/decode_d256.cu``) is EXECUTION: it must give the general leaf's bits.

Relations, all on the raw bits of out and lse:

* decode leaf == general leaf on the same call: every head ratio, 1..16/ratio query tokens per request,
  page sizes 128 and 1152, permuted pages with NaN-poisoned unwritten slots, vLLM's interleaved ``LBNHC``
  pages, a packed cache over several input populations, and every segment split;
* a row is the same alone and in batches of 1 to 256;
* decode rows == the TRAINER's rows (the packed causal forward over each whole sequence);
* ``paged_core`` takes the decode leaf for a decode call and writes the engine's buffer in place, and a
  CUDA graph of it replays new lengths with the eager bits.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = [pytest.mark.skipif(live_device_arch() != "sm100", reason="requires an SM100 GPU"),
              pytest.mark.usefixtures("declaration")]

from isoexec.ops.attention.gqa_sm100 import ops
from isoexec.ops.attention.gqa_sm100.tests._harness import bit_equal, cu, draw, first_diff, page_kv

D = 256
SCALE = ops.default_scale(D)
MAX_K = 10240
#: Contexts on and off the tile (128) and segment (640 under ``batch1``) boundaries, up to sixteen segments there;
#: test_schedule_boundary_gpu.py covers the declared span's boundaries.
CONTEXTS = (1, 2, 127, 128, 129, 300, 639, 640, 641, 1280, 1281, 2047, 2048, 2049, 4095, 4097, 6000, 8192, 10240)
RATIOS = [(16, 4), (8, 2), (4, 1), (16, 2), (2, 1), (16, 1), (16, 16)]


def _qkv(q_lens, kv_lens, hq, hk, seed, population="normal"):
    q = draw(sum(q_lens), hq, D, seed, population)
    return q, draw(sum(kv_lens), hk, D, seed + 1, population), draw(sum(kv_lens), hk, D, seed + 2, population)


def _call(q, kc, vc, q_lens, kv_lens, leaf, *, table=None, cu_k=None, **kw):
    used = torch.tensor(list(kv_lens), device="cuda", dtype=torch.int32)
    return ops.gqa_sm100_fwd(q, kc, vc, cu(q_lens), used, softmax_scale=SCALE, seg_tokens=ops.segment_span(D),
                             max_seqlen_q=max(q_lens), max_seqlen_k=MAX_K if table is not None else max(kv_lens),
                             block_table=table, cu_seqlens_k=cu_k, leaf=leaf, **kw)


def _same(ref, got, what=""):
    assert bit_equal(ref[0], got[0]), f"{what} out: {first_diff(ref[0], got[0])}"
    assert bit_equal(ref[1], got[1]), f"{what} lse: {first_diff(ref[1], got[1])}"


def _decode_vs_general(q, kc, vc, q_lens, kv_lens, what="", **kw):
    got = _call(q, kc, vc, q_lens, kv_lens, ops.LEAF_DECODE, **kw)
    kw.pop("decode_split", None)
    _same(_call(q, kc, vc, q_lens, kv_lens, ops.LEAF_GENERAL, **kw), got, what)
    return got


@pytest.mark.parametrize("hq,hk", RATIOS)
def test_decode_equals_general_paged(hq, hk):
    for qn in sorted({1, 16 // (hq // hk)}):
        kv_lens = [max(c, qn) for c in CONTEXTS]
        q_lens = [qn] * len(kv_lens)
        q, k, v = _qkv(q_lens, kv_lens, hq, hk, seed=hq * 100 + hk + qn)
        for page in (128, 1152):
            kc, vc, table = page_kv(k, v, kv_lens, page, permute=True, fill=float("nan"))
            _decode_vs_general(q, kc, vc, q_lens, kv_lens, f"q{qn} page{page}", table=table)


def test_decode_equals_general_packed():
    kv_lens = [1, 64, 128, 1000, 2048, 2500, 5000]
    q_lens = [1, 2, 3, 4, 1, 4, 2]
    for population in ("normal", "large", "signed_zero"):
        q, k, v = _qkv(q_lens, kv_lens, 16, 4, seed=7, population=population)
        _decode_vs_general(q, k, v, q_lens, kv_lens, population, cu_k=cu(kv_lens))


def test_decode_lbnhc_interleaved_pages():
    """vLLM's LBNHC cache: K and V interleaved in one page, read in place through strided views."""
    from isoexec.ops.attention.paged_kv import split_paged_kv_cache

    hk, page = 4, 1152
    kv_lens = [3, 1300, 2049, 7000]
    q_lens = [1] * 4
    q, k, v = _qkv(q_lens, kv_lens, 16, hk, seed=11)
    kc, vc, table = page_kv(k, v, kv_lens, page, permute=True)
    cache = torch.stack([kc, vc], dim=3).reshape(kc.shape[0], page, hk, 2 * D)
    ks, vs = split_paged_kv_cache(cache.permute(0, 2, 1, 3), num_kv_heads=hk, head_size=D)
    got = _decode_vs_general(q, ks, vs, q_lens, kv_lens, table=table)
    _same(_call(q, kc, vc, q_lens, kv_lens, ops.LEAF_DECODE, table=table), got, "dense pages")


def test_decode_segment_split():
    """How many CTAs walk a (request, KV head)'s segments is execution: every split gives the same bits."""
    kv_lens = [1, 2048, 2049, 4096, 6000, 9000, 10240, 300]
    q_lens = [1, 4, 2, 1, 3, 1, 4, 1]
    q, k, v = _qkv(q_lens, kv_lens, 16, 4, seed=5)
    kc, vc, table = page_kv(k, v, kv_lens, 1152, permute=True, fill=float("nan"))
    for split in (1, 2, 3, 5):
        _decode_vs_general(q, kc, vc, q_lens, kv_lens, f"split {split}", table=table, decode_split=split)


def test_decode_batch_composition():
    """A row is the same alone and in batches of 1 to 256 (whose automatic splits differ)."""
    g = torch.Generator().manual_seed(0)
    kv_all = torch.randint(1, MAX_K + 1, (256,), generator=g).tolist()
    q, k, v = _qkv([1] * 256, kv_all, 16, 4, seed=13)
    kc, vc, table = page_kv(k, v, kv_all, 128, permute=True, fill=float("nan"))
    full = _decode_vs_general(q, kc, vc, [1] * 256, kv_all, "B256", table=table)
    for batch in (1, 8, 64):
        got = _call(q[:batch].contiguous(), kc, vc, [1] * batch, kv_all[:batch], ops.LEAF_AUTO,
                    table=table[:batch].contiguous())
        assert bit_equal(got[0], full[0][:batch]), f"B{batch}: {first_diff(got[0], full[0][:batch])}"
        assert bit_equal(got[1], full[1][:batch])


@pytest.mark.parametrize("hq,hk", [(16, 4), (8, 2), (24, 4), (12, 2)])
def test_decode_equals_trainer_rows(hq, hk):
    # Ratio 6 (Qwen3.6-27B at TP1 and TP2): the decode leaf serves it in one launch, the trainer as three
    # ratio-2 head groups; the rows must agree.
    kv_lens = [1, 129, 700, 2048, 2049, 4500, 10240]
    for qn in sorted({1, 16 // (hq // hk)}):
        q, k, v = _qkv(kv_lens, kv_lens, hq, hk, seed=qn + hq)
        trainer, trainer_lse = ops.varlen_fwd(q, k, v, cu(kv_lens), max(kv_lens), SCALE)
        q_lens = [min(qn, n) for n in kv_lens]
        idx = torch.cat([torch.arange(e - m, e) for e, m in zip(cu(kv_lens)[1:].tolist(), q_lens)]).cuda()
        kc, vc, table = page_kv(k, v, kv_lens, 1152, permute=True, fill=float("nan"))
        got = _call(q[idx].contiguous(), kc, vc, q_lens, kv_lens, ops.LEAF_DECODE, table=table)
        _same((trainer[idx], trainer_lse[idx]), got, f"q{qn}")


def test_ratio_six_decodes_in_one_launch_with_the_grouped_bits():
    """``paged_fwd`` at ratio 6: one decode launch == the three ratio-2 group launches, bitwise."""
    kv_lens = [1, 300, 2049, 6000, 10240]
    q_lens = [1] * len(kv_lens)
    q, k, v = _qkv(q_lens, kv_lens, 24, 4, seed=21)
    kc, vc, table = page_kv(k, v, kv_lens, 1152, permute=True, fill=float("nan"))
    used = torch.tensor(kv_lens, device="cuda", dtype=torch.int32)
    one = ops.paged_fwd(q, kc, vc, cu(q_lens), used, table, 1, MAX_K, SCALE)
    calls = []

    def grouped(qq):
        calls.append(qq.shape[1])
        return ops.gqa_sm100_fwd(qq, kc, vc, cu(q_lens), used, softmax_scale=SCALE, seg_tokens=ops.segment_span(D),
                                 max_seqlen_q=1, max_seqlen_k=MAX_K, block_table=table)

    three = ops._by_head_groups(q, 4, None, grouped)
    assert calls == [8, 8, 8]
    _same(three, one, "ratio 6")


def test_paged_core_takes_the_decode_leaf_in_place_and_replays():
    """``paged_core`` on a decode call launches the decode leaf (no plan kernel), writes ``out`` in place,
    and a CUDA graph captured at ``max_seqlen_k = MAX_K`` replays new lengths with the eager bits."""
    from torch.profiler import ProfilerActivity, profile

    from isoexec.ops.attention.gqa_sm100.integration import paged_core

    B, hk = 64, 4
    g = torch.Generator().manual_seed(5)
    kv_a = torch.randint(1, MAX_K + 1, (B,), generator=g).tolist()
    kv_b = torch.randint(1, MAX_K + 1, (B,), generator=g).tolist()
    total = [max(a, b) for a, b in zip(kv_a, kv_b)]
    q, k, v = _qkv([1] * B, total, 16, hk, seed=9)
    kc, vc, table = page_kv(k, v, total, 128, permute=True)
    cu_q = cu([1] * B)
    used = torch.tensor(kv_a, device="cuda", dtype=torch.int32)
    out = torch.full_like(q, float("nan"))
    call = lambda: paged_core(q, kc, vc, cu_seqlens_q=cu_q, seqused_k=used, block_table=table,
                              max_seqlen_q=1, max_seqlen_k=MAX_K, scale=SCALE, out=out)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
    assert any("gqa_decode_d256" in n for n in kernels), kernels
    assert not any("plan" in n.lower() or "gqa_leaf" in n for n in kernels), kernels
    assert bit_equal(out, _call(q, kc, vc, [1] * B, kv_a, ops.LEAF_GENERAL, table=table)[0])

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for lens in (kv_b, kv_a):
        used.copy_(torch.tensor(lens, device="cuda", dtype=torch.int32))
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        ref = _call(q, kc, vc, [1] * B, lens, ops.LEAF_GENERAL, table=table)[0]
        assert bit_equal(out, ref), first_diff(out, ref)


def test_decode_leaf_refuses_a_call_it_cannot_serve():
    q, k, v = _qkv([8], [64], 16, 4, seed=1)  # 8 tokens x ratio 4 = 32 packed rows
    with pytest.raises(RuntimeError, match="decode leaf"):
        _call(q, k, v, [8], [64], ops.LEAF_DECODE, cu_k=cu([64]))
