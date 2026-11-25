"""Vocab-parallel sampling head battery (SM90): ``shard_sample`` vs the gathered path, bit for bit.

The gathered path is what the engine ran before: widen the all-gathered bf16 logits to fp32, vLLM's
``gumbel_sample`` (Model Runner V2, fp32 noise), ``full_logprobs`` (leaf8 recipe) gathered at the token,
and vLLM's ``_ranks_kernel``. The shard path is emulated on one GPU (each "rank" a column shard; the
all-gather is a concatenation, the all-reduce a sum), at world 1, 2, 4 and 8.

Run: ``CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. $PY isoexec/ops/logprobs/logprobs_sm90/tests/test_shard_sample_gpu.py``
(needs vLLM >= 0.33 for the reference sampler; the exhaustive noise check takes ~1 min).
"""

from __future__ import annotations

import sys

import pytest
import torch

from isoexec.ops._testing import bitwise_equal

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
                                reason="needs an SM90 device")

DEV = "cuda"
V = 248320  # Qwen3.x vocabulary: 8 leaves of 31040, shards of 62080 at TP4


def _skip() -> bool:
    return not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0)


def _ext():
    from isoexec.ops.logprobs.logprobs_sm90._shard_extension import extension

    return extension()


def _base_ext():
    from isoexec.ops.logprobs.logprobs_sm90._extension import extension

    return extension()


def _vllm_gumbel():
    from vllm.v1.worker.gpu.sample import gumbel

    return gumbel


def _nan_col(vocab):
    from isoexec.runtimes.vllm.vllm_patches import vllm_all_nan_column

    return vllm_all_nan_column(vocab, torch.device(DEV))


def _emulate(x, world, req, rtemp, seed, pos, temperature=None):
    """shard_sample on one GPU: per-shard shard_pass, concatenated payloads, per-shard shard_finish."""
    ext = _ext()
    rows, vocab = x.shape
    nan_col = _nan_col(vocab)
    width, ll = vocab // world, 8 // world
    parts = []
    for r in range(world):
        p = torch.empty(5 * rows * ll, device=DEV)
        ext.shard_pass(x[:, r * width:(r + 1) * width].contiguous(), ll, r * width, req, rtemp, seed, pos, p, temperature)
        parts.append(p)
    gathered = torch.cat(parts)
    outs = []
    for r in range(world):
        s = torch.empty(rows, device=DEV, dtype=torch.int64)
        lp = torch.empty(rows, device=DEV)
        c = torch.empty(rows, device=DEV, dtype=torch.int64)
        ext.shard_finish(gathered, world, ll, x[:, r * width:(r + 1) * width].contiguous(), s, lp, c, temperature,
                         nan_col)
        outs.append((s, lp, c))
    for s, lp, _ in outs[1:]:  # every rank derives the same token and logprob
        assert bitwise_equal(s, outs[0][0]) and bitwise_equal(lp, outs[0][1])
    return outs[0][0], outs[0][1], sum(c for _, _, c in outs)


def _reference(x, req, rtemp, seed, pos, temperature=None):
    """The gathered path: vLLM gumbel_sample + leaf8 full rows + vLLM _ranks_kernel on fp32 logits."""
    from vllm.v1.worker.gpu.sample import logprob as vlp

    from isoexec.ops.logprobs.logprobs_sm90 import ops

    xf = x.float()
    tok = _vllm_gumbel().gumbel_sample(xf, req, rtemp, seed, pos, apply_temperature=False, is_drafting=False)
    rows = ops.full_logprobs(xf, temperature=temperature)
    lp = rows.gather(1, tok[:, None]).squeeze(1)
    ranks = torch.empty(x.shape[0], dtype=torch.int64, device=DEV)
    vlp._ranks_kernel[(x.shape[0],)](ranks, xf, xf.stride(0), tok, x.shape[1], BLOCK_SIZE=8192)
    return tok, lp, ranks


def _batch(rows, *, seed=0, scale=3.0, greedy_every=0, pad_every=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = (torch.randn(rows, V, device=DEV, generator=g) * scale).to(torch.bfloat16)
    nreq = rows + 3
    rtemp = torch.ones(nreq, device=DEV)
    if greedy_every:
        rtemp[::greedy_every] = 0.0
    seeds = torch.randint(-(2 ** 63), 2 ** 63 - 1, (nreq,), device=DEV, generator=g, dtype=torch.int64)
    req = torch.randperm(nreq, device=DEV, generator=g)[:rows].to(torch.int32)
    if pad_every:
        req[::pad_every] = -1
    pos = torch.randint(0, 1 << 20, (rows,), device=DEV, generator=g, dtype=torch.int64)
    return x, req, rtemp, seeds, pos


def test_noise_exhaustive_vs_triton():
    """The Gumbel value as a function of the raw uint32 hash, all 2^32 inputs, vs Triton's compile of
    vLLM's own expression (murmur3_uniform32's tail, _log1p_neg_stable, -log, + logits)."""
    import triton
    import triton.language as tl

    g = _vllm_gumbel()
    log1p_neg_stable = g._log1p_neg_stable

    @triton.jit
    def _ref(out_ptr, first, n, BLOCK: tl.constexpr):
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        random32 = (first + i).to(tl.uint32)
        hi16 = (random32 >> 16).to(tl.int32)
        lo16 = (random32 & 0xFFFF).to(tl.int32)
        u = hi16.to(tl.float32) * 1.52587890625e-05 + (lo16.to(tl.float32) + 0.5) * 2.3283064365386963e-10
        noise = -tl.log(-log1p_neg_stable(u))
        logits = tl.zeros([BLOCK], dtype=tl.float32)
        tl.store(out_ptr + i, logits + noise, mask=i < n)

    ext = _ext()
    chunk = 1 << 28
    a = torch.empty(chunk, device=DEV)
    b = torch.empty(chunk, device=DEV)
    bad = 0
    for first in range(0, 1 << 32, chunk):
        ext.gumbel_range_probe(first, a)
        _ref[(chunk // 1024,)](b, first, chunk, BLOCK=1024)
        bad += int((a.view(torch.int32) != b.view(torch.int32)).sum())
    assert bad == 0, f"{bad} of 2^32 hash values give a different Gumbel value"
    print("  noise: all 2^32 hash outputs bitwise equal to Triton's compile of vLLM's expression")


def test_block_winners_vs_vllm_kernel():
    """vLLM's per-1024-column block winners (value, column) equal gumbel_probe at those columns: checks the
    hash keying (seed incl. negative, pos, global column) and the final add against the real kernel."""
    g = _vllm_gumbel()
    ext = _ext()
    for trial in range(4):
        x, req, rtemp, seeds, pos = _batch(32, seed=100 + trial)
        xf = x.float()
        rows = x.shape[0]
        nb = (V + 1023) // 1024
        la = torch.empty(rows, nb, dtype=torch.int64, device=DEV)
        lm = torch.empty(rows, nb, dtype=torch.float32, device=DEV)
        g._gumbel_sample_kernel[(rows, nb)](
            la, la.stride(0), lm, lm.stride(0), None, 0, 0, None, None, 0, xf, xf.stride(0), req, seeds, pos, rtemp,
            V, BLOCK_SIZE=1024, IS_DRAFTING=False, APPLY_TEMPERATURE=False, USE_FP64=False, PER_TOKEN_COL=False)
        r = req.long()
        mine = ext.gumbel_probe(xf.gather(1, la).reshape(-1), seeds[r][:, None].expand(rows, nb).reshape(-1),
                                pos[:, None].expand(rows, nb).reshape(-1), la.reshape(-1))
        assert bitwise_equal(mine, lm.reshape(-1)), f"trial {trial}: block winner values differ"
    print("  block winners: 4 x 32 rows x 243 blocks bitwise equal to vLLM's _gumbel_sample_kernel")


def test_sample_logprob_rank_vs_gathered_path():
    for world in (1, 2, 4, 8):
        for trial, (scale, greedy, pad) in enumerate([(3.0, 0, 0), (1.0, 5, 0), (8.0, 3, 7), (0.05, 0, 0)]):
            x, req, rtemp, seeds, pos = _batch(48, seed=world * 10 + trial, scale=scale, greedy_every=greedy,
                                              pad_every=pad)
            for temperature in (None, torch.tensor([0.7], device=DEV)):
                ref = _reference(x, req, rtemp, seeds, pos, temperature)
                got = _emulate(x, world, req, rtemp, seeds, pos, temperature)
                for name, a, b in zip(("token", "logprob", "rank"), got, ref):
                    assert bitwise_equal(a, b), (world, trial, temperature, name, (a != b).nonzero()[:5])
    print("  shard path == gathered path (token, logprob, rank) at world 1/2/4/8, greedy and padding rows, "
          "scoring temperature 1 and 0.7")


def test_ties_lowest_column():
    """Constant logits at temperature 0: every column ties; vLLM and the shard path take column 0. A tie
    straddling shards (two equal maxima in different shards) takes the lower column."""
    x, req, rtemp, seeds, pos = _batch(8, seed=7)
    rtemp.zero_()
    x = torch.zeros_like(x)
    x[:, 70000] = 1.0
    x[:, 200000] = 1.0
    for world in (1, 4, 8):
        tok, _, rank = _emulate(x, world, req, rtemp, seeds, pos)
        ref = _reference(x, req, rtemp, seeds, pos)
        assert bitwise_equal(tok, ref[0]) and bool((tok == 70000).all()) and bitwise_equal(rank, ref[2])
    print("  ties: lowest column wins across shards (temperature 0)")


def test_all_nan_row_matches_gathered_path():
    """A row NaN in every column: the gathered path's token (in range), NaN logprob, rank 0 -- bitwise, at every
    world size, greedy and noisy rows, next to ordinary rows."""
    x, req, rtemp, seeds, pos = _batch(8, seed=11, greedy_every=2)
    x[1] = float("nan")
    x[6] = float("nan")
    for world in (1, 2, 4, 8):
        for temperature in (None, torch.tensor([0.7], device=DEV)):
            got = _emulate(x, world, req, rtemp, seeds, pos, temperature)
            ref = _reference(x, req, rtemp, seeds, pos, temperature)
            assert bool(((got[0] >= 0) & (got[0] < V)).all())
            for name, a, b in zip(("token", "logprob", "rank"), got, ref):
                assert bitwise_equal(a, b), (world, temperature, name, a[[1, 6]], b[[1, 6]])
    print(f"  all-NaN rows: token {int(got[0][1])} == gathered path, NaN logprob, rank 0")


def test_leaf_states_unchanged():
    """shard_pass's leaf states are the recipe's leaf_states bits (the observer hook changes nothing)."""
    ext = _ext()
    x, req, rtemp, seeds, pos = _batch(16, seed=3)
    for world in (1, 4):
        width, ll = V // world, 8 // world
        for r in range(world):
            xs = x[:, r * width:(r + 1) * width].contiguous()
            p = torch.empty(5 * 16 * ll, device=DEV)
            ext.shard_pass(xs, ll, r * width, req, rtemp, seeds, pos, p, None)
            m = torch.empty(16, ll, device=DEV)
            s = torch.empty_like(m)
            _base_ext().leaf_states(xs, ll, m, s, None)
            assert bitwise_equal(p[:16 * ll].view(16, ll), m) and bitwise_equal(p[16 * ll:32 * ll].view(16, ll), s)
    print("  leaf states: shard_pass == leaf_states bitwise")


def test_cuda_graph_capture():
    """The kernels capture (no host sync, no allocation in the extension beyond its outputs)."""
    ext = _ext()
    x, req, rtemp, seeds, pos = _batch(16, seed=9)
    ll = 8
    p = torch.empty(5 * 16 * ll, device=DEV)
    s = torch.empty(16, device=DEV, dtype=torch.int64)
    lp = torch.empty(16, device=DEV)
    c = torch.empty(16, device=DEV, dtype=torch.int64)
    graph = torch.cuda.CUDAGraph()
    ext.shard_pass(x, ll, 0, req, rtemp, seeds, pos, p, None)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        ext.shard_pass(x, ll, 0, req, rtemp, seeds, pos, p, None)
        ext.shard_finish(p, 1, ll, x, s, lp, c, None, _nan_col(V))
    graph.replay()
    ref = _reference(x, req, rtemp, seeds, pos)
    assert bitwise_equal(s, ref[0]) and bitwise_equal(lp, ref[1]) and bitwise_equal(c, ref[2])
    print("  graph capture: replay == gathered path")


def main():
    if _skip():
        print("SKIP: no SM90 device")
        return 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(t.__name__)
        t()
    print(f"OK {len(tests)} tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
