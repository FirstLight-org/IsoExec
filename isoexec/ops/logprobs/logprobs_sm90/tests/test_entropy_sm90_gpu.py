"""logprobs_sm90 entropy-from-the-leaf-pass battery (SM90).

The entropy is a metric outside the scoring contract, so it is checked against an fp64 reference within
a tolerance, and bitwise only against itself: run-to-run, and across the TP degrees that hold whole
leaves (1/2/4/8, emulated in one process exactly as the sampled battery emulates the TP gather). The
logprobs the entropy path returns must be the sampled path's bits.

Run: ``CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. $PY isoexec/ops/logprobs/logprobs_sm90/tests/test_entropy_sm90_gpu.py [--bench]``
"""

from __future__ import annotations

import sys
import time

import torch

from isoexec.ops._testing import bitwise_equal

DEV = "cuda"


def _skip() -> bool:
    return not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0)


def _ops():
    from isoexec.ops.logprobs.logprobs_sm90 import ops

    return ops


def _ext():
    from isoexec.ops.logprobs.logprobs_sm90._extension import extension

    return extension()


def _logits(rows, V, dtype, seed, pad=0, scale=4.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = (torch.randn(rows, V, device=DEV, generator=g) * scale).to(dtype)
    if pad:
        x[:, V - pad:] = float("-inf")
    return x


def _ref_entropy(x):
    v = x.double()
    finite = torch.isfinite(v)
    lp = torch.log_softmax(torch.where(finite, v, torch.full_like(v, float("-inf"))), dim=-1)
    p = lp.exp()
    return -(torch.where(finite, p * lp, torch.zeros_like(p))).sum(-1)


def _sim_entropy(x, tp):
    """Per-rank leaf states over this rank's vocab slice (with A), gathered in rank order, then merged."""
    ops, ext = _ops(), _ext()
    rows, V = x.shape
    Vl = V // tp
    ms, ss, as_ = [], [], []
    for r in range(tp):
        m, s, a = ops.leaf_states(x[:, r * Vl:(r + 1) * Vl].contiguous(), 8 // tp, entropy=True)
        ms.append(m), ss.append(s), as_.append(a)
    ent = torch.empty(rows, device=DEV)
    ext.entropy_rows(torch.cat(ms, 1).contiguous(), torch.cat(ss, 1).contiguous(), torch.cat(as_, 1).contiguous(), ent)
    return ent


def test_logprobs_unchanged_and_entropy_accuracy():
    if _skip():
        return
    ops = _ops()
    for dtype in (torch.float32, torch.bfloat16):
        for rows, V, pad in ((7, 512, 0), (33, 4096, 64), (129, 151936, 0), (64, 62080, 100)):
            x = _logits(rows, V, dtype, seed=rows + V, pad=pad)
            t = torch.randint(0, V - pad, (rows,), device=DEV)
            t[0] = -1  # an ignored target
            lp = ops.sampled_logprobs(x, t)
            lp2, ent = ops.sampled_logprobs_entropy(x, t)
            assert bitwise_equal(lp, lp2), f"logprobs changed by the entropy pass {dtype} {rows}x{V}"
            ref = _ref_entropy(x)
            err = (ent.double() - ref).abs().max().item()
            tol = 5e-5 * max(1.0, ref.abs().max().item())
            assert err <= tol, f"entropy {dtype} {rows}x{V}: max err {err:.3e} > {tol:.3e}"


def test_entropy_tp_invariant_and_deterministic():
    if _skip():
        return
    ops = _ops()
    for dtype in (torch.float32, torch.bfloat16):
        x = _logits(257, 8 * 2048, dtype, seed=11, pad=37)
        t = torch.randint(0, 8 * 2048 - 37, (257,), device=DEV)
        _, e1 = ops.sampled_logprobs_entropy(x, t)
        _, e2 = ops.sampled_logprobs_entropy(x, t)
        assert bitwise_equal(e1, e2), "entropy is not run-to-run deterministic"
        for tp in (1, 2, 4, 8):
            assert bitwise_equal(_sim_entropy(x, tp), e1), f"entropy differs at TP={tp}"


def _emulated_world(monkeypatch, x, t, world, entropy):
    """Run the real sharded forward of every rank in one process: dist calls replay the other ranks."""
    ops = _ops()
    rows, V = x.shape
    width = V // world
    shards = [x[:, r * width:(r + 1) * width].contiguous() for r in range(world)]
    planes = []
    for shard in shards:
        st = ops.leaf_states(shard, 8 // world, entropy=entropy)
        planes.append(torch.stack(st))
    raw = torch.full((rows,), -0.0, device=DEV)
    for r, shard in enumerate(shards):
        own = (t >= r * width) & (t < (r + 1) * width)
        val = shard.gather(1, (t - r * width).clamp(0, width - 1)[:, None]).squeeze(1).float()
        raw = raw + torch.where(own, val, torch.full_like(val, -0.0))
    monkeypatch.setattr(ops.dist, "get_world_size", lambda **kw: world)
    monkeypatch.setattr(ops.dist, "all_gather", lambda parts, value, *, group: [p.copy_(q) for p, q in zip(parts, planes)])
    monkeypatch.setattr(ops.dist, "all_reduce", lambda value, op=None, *, group: value.copy_(raw))
    results = []
    for r, shard in enumerate(shards):
        if entropy:
            results.append(ops.sampled_logprobs_entropy(shard, t, vocab_start_index=r * width, group=object()))
        else:
            results.append(ops.sampled_logprobs(shard, t, vocab_start_index=r * width, group=object()))
    return results


def test_sharded_path_bits(monkeypatch):
    """The real TP code path (one all_gather of the stacked leaf planes) at TP 2/4/8: logprobs equal the
    TP1 bits with and without entropy, and the entropy equals the TP1 entropy bit for bit."""
    if _skip():
        return
    ops = _ops()
    x = _logits(65, 8 * 1024, torch.float32, seed=5, pad=200)
    t = torch.randint(0, 8 * 1024 - 200, (65,), device=DEV)
    t[3] = -1
    lp1 = ops.sampled_logprobs(x, t)
    _, e1 = ops.sampled_logprobs_entropy(x, t)
    for world in (2, 4, 8):
        for got in _emulated_world(monkeypatch, x, t, world, entropy=False):
            assert bitwise_equal(got, lp1), f"sampled logprobs changed at TP={world}"
        for lp, ent in _emulated_world(monkeypatch, x, t, world, entropy=True):
            assert bitwise_equal(lp, lp1), f"entropy path logprobs differ at TP={world}"
            assert bitwise_equal(ent, e1), f"entropy differs at TP={world}"


def test_edge_rows():
    if _skip():
        return
    ops = _ops()
    V = 1024
    x = torch.full((4, V), float("-inf"), device=DEV)
    x[0, 5] = 3.0                      # one live column: entropy 0
    x[1] = 0.0                         # uniform: log V
    x[2, :V // 2] = 1.0                # half live (whole leaves absent)
    t = torch.zeros(4, dtype=torch.int64, device=DEV)
    _, ent = ops.sampled_logprobs_entropy(x, t)
    assert ent[0].item() == 0.0
    assert abs(ent[1].item() - torch.log(torch.tensor(float(V))).item()) < 1e-5
    assert abs(ent[2].item() - torch.log(torch.tensor(float(V // 2))).item()) < 1e-5
    assert torch.isnan(ent[3]), "an all -inf row has no distribution"


def bench(rows=11520, V=62080, T=1.0, reps=10):
    """Per micro-batch on one rank's vocab shard: sampled logprobs + entropy, eager entropy vs fused."""
    from isoexec.ops.logprobs.temperature import temperature_divide

    ops = _ops()
    x = _logits(rows, V, torch.float32, seed=1)
    t = torch.randint(0, V, (rows,), device=DEV)

    def eager_entropy(logits):  # engine._vocab_parallel_entropy at world 1
        finite = torch.isfinite(logits)
        local_max = torch.where(finite, logits, torch.full_like(logits, float("-inf"))).amax(dim=-1)
        shifted = logits - local_max.unsqueeze(-1)
        exp = torch.where(finite, torch.exp(shifted), torch.zeros_like(shifted))
        total = exp.sum(dim=-1)
        weighted = torch.where(finite, exp * shifted, torch.zeros_like(shifted)).sum(dim=-1)
        return torch.log(total) - weighted / total

    def before():
        lp = ops.sampled_logprobs(temperature_divide(x, T).contiguous(), t)
        return lp, eager_entropy(x / T)

    def after():
        return ops.sampled_logprobs_entropy(temperature_divide(x, T).contiguous(), t)

    out, peak = {}, {}
    for name, fn in (("before", before), ("after", after)):
        with torch.no_grad():
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            fn()
            torch.cuda.synchronize()
            peak[name] = (torch.cuda.max_memory_allocated() - base) / 2**30
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
        out[name] = (time.perf_counter() - t0) / reps * 1e3
    with torch.no_grad():
        e_ref = eager_entropy(x)
        _, e = ops.sampled_logprobs_entropy(x, t)
    print(f"[entropy bench] rows={rows} V_local={V}: before {out['before']:.2f} ms, after {out['after']:.2f} ms; "
          f"max |fused - eager| = {(e - e_ref).abs().max().item():.3e}; peak extra memory before "
          f"{peak['before']:.2f} GiB, after {peak['after']:.2f} GiB (logits {x.numel() * 4 / 2**30:.2f} GiB)")
    return out


if __name__ == "__main__":
    if _skip():
        print("SKIP: needs an sm90 device")
        sys.exit(0)
    import pytest

    rc = pytest.main(["-q", "-p", "no:cacheprovider", __file__])
    if rc:
        sys.exit(rc)
    if "--bench" in sys.argv:
        bench()
