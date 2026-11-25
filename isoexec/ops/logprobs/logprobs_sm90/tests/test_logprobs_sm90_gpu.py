"""logprobs_sm90 v1 gate battery (SM90). Bit-pattern comparisons only; hazards are witnessed.

Run: ``CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. $PY isoexec/ops/logprobs/logprobs_sm90/tests/test_logprobs_sm90_gpu.py``
"""

from __future__ import annotations

import sys

import torch

from isoexec.ops._testing import assert_hazard_exercised, bitwise_equal

DEV = "cuda"
BF = torch.bfloat16


def _skip() -> bool:
    return not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0)


def _lp():
    from isoexec.ops.logprobs import logprobs_sm90

    return logprobs_sm90


def _ext():
    from isoexec.ops.logprobs.logprobs_sm90._extension import extension

    return extension()


def _shard_sampled(x, t, tp):
    """Single-process emulation of the TP path: per-rank leaf states + exact owner gather, then merge."""
    ext = _ext()
    rows, V = x.shape
    Vl = V // tp
    ms, ss = [], []
    for r in range(tp):
        m = torch.empty(rows, 8 // tp, device=DEV)
        s = torch.empty_like(m)
        ext.leaf_states(x[:, r * Vl : (r + 1) * Vl].contiguous(), 8 // tp, m, s, None)
        ms.append(m)
        ss.append(s)
    raw = torch.full((rows,), -0.0, device=DEV)
    for r in range(tp):
        own = (t >= r * Vl) & (t < (r + 1) * Vl)
        loc = (t - r * Vl).clamp(0, Vl - 1)
        val = x[:, r * Vl : (r + 1) * Vl].gather(1, loc[:, None]).squeeze(1).float()
        raw = raw + torch.where(own, val, torch.full_like(val, -0.0))
    rM, rL = torch.empty(rows, device=DEV), torch.empty(rows, device=DEV)
    lse = torch.empty(rows, device=DEV, dtype=torch.float64)
    out = torch.empty(rows, device=DEV)
    ext.merge_rows(torch.cat(ms, 1), torch.cat(ss, 1), rM, rL, lse, out, None, t.to(torch.int64), raw, V, None)
    return out


def test_accuracy_vs_float64():
    if _skip():
        return
    lp = _lp()
    g = torch.Generator(device=DEV).manual_seed(0)
    for V in (64, 4096, 151680):
        x = torch.randn(17, V, device=DEV, generator=g) * 5
        ref = torch.log_softmax(x.double(), -1)
        full = lp.full_logprobs(x)
        err = (full.double() - ref).abs().max().item()
        assert err < 1e-5, (V, err)
        temp = torch.tensor([0.6], device=DEV)
        reft = torch.log_softmax((x / temp).double(), -1)
        errt = (lp.full_logprobs(x, temperature=temp).double() - reft).abs().max().item()
        assert errt < 1e-5, (V, errt)
        # adversarial order: ascending rows rescale the normalizer at every column
        xs = torch.sort(x, -1).values.contiguous()
        errs = (lp.full_logprobs(xs).double() - torch.log_softmax(xs.double(), -1)).abs().max().item()
        assert errs < 1e-5, (V, errs)
        print(f"  V={V}: max |err| vs float64 {err:.1e} (temperature {errt:.1e}, sorted rows {errs:.1e})")


def test_row_batch_and_view_invariance():
    if _skip():
        return
    lp = _lp()
    g = torch.Generator(device=DEV).manual_seed(1)
    V = 248320
    row = torch.randn(1, V, device=DEV, generator=g) * 4
    solo = lp.full_logprobs(row)[0]
    t = torch.tensor([12345], device=DEV)
    solo_s = lp.sampled_logprobs(row, t)[0]
    assert bitwise_equal(solo[12345], solo_s)
    for N in (2, 7, 32, 128, 513):
        batch = torch.randn(N, V, device=DEV, generator=g) * 4
        pos = (N * 5) // 7
        batch[pos] = row[0]
        full = lp.full_logprobs(batch)
        assert bitwise_equal(full[pos], solo), N
        tt = torch.randint(0, V, (N,), device=DEV, generator=g)
        tt[pos] = 12345
        samp = lp.sampled_logprobs(batch, tt)
        assert bitwise_equal(samp[pos], solo_s), N
        assert bitwise_equal(full.gather(1, tt[:, None]).squeeze(1), samp), N
    big = torch.randn(9, 2 * V, device=DEV, generator=g)
    big[4, ::2] = row[0]
    view = big[:, ::2]  # stride(1) == 2: copied to an aligned buffer, bits unchanged
    assert bitwise_equal(lp.full_logprobs(view)[4], solo)
    rows3 = torch.randn(2, 3, V, device=DEV, generator=g)
    rows3[1, 2] = row[0]
    assert bitwise_equal(lp.full_logprobs(rows3)[1, 2], solo)
    xb = (row * 1).to(BF)
    assert bitwise_equal(lp.full_logprobs(xb), lp.full_logprobs(xb.float()))
    print("  row invariance: N in (1..513), sampled == gathered full row, strided/3D views, bf16 == widened fp32")


def test_tp_shard_invariance():
    if _skip():
        return
    lp = _lp()
    g = torch.Generator(device=DEV).manual_seed(2)
    for V in (248320, 154880, 151680):
        x = torch.randn(33, V, device=DEV, generator=g) * 3
        x[5, : V // 8] = float("-inf")  # an entire leaf masked on one rank
        t = torch.randint(0, V, (33,), device=DEV, generator=g)
        t[3] = -1  # invalid target
        ref = lp.sampled_logprobs(x, t)
        for tp in (2, 4, 8):
            got = _shard_sampled(x, t.clamp(min=0), tp)
            got = torch.where(t >= 0, got, torch.zeros_like(got))
            assert bitwise_equal(got, ref), (V, tp)
        full = lp.full_logprobs(x)
        valid = t >= 0
        assert bitwise_equal(full.gather(1, t.clamp(min=0)[:, None]).squeeze(1)[valid], ref[valid])
    print("  TP invariance: tp in (2,4,8) leaf-owner shards == engine full row == world-1 sampled, 3 vocabularies")


def test_hazards():
    if _skip():
        return
    lp = _lp()
    V = 4096
    g = torch.Generator(device=DEV).manual_seed(3)
    x = torch.randn(12, V, device=DEV, generator=g)
    x[0, 100:3000] = float("-inf")  # masked columns
    x[1, :] = float("-inf")  # all masked -> NaN row
    x[2, 7] = float("nan")  # NaN poisons the row
    x[3, :] = 2.5  # every column ties at the max
    x[4, 5:9] = 1e-40  # subnormal logits
    x[4, 9:12] = -1e-40
    x[5, 3] = float("inf")
    x[6, V // 2 :] = float("-inf")  # the upper four leaves are empty
    x[7] = torch.linspace(-1e4, 1e4, V, device=DEV)  # huge spread, ascending
    x[8, 0] = 3e38
    x[9, :] = -0.0
    x[10, :100] = 1e-45
    out = lp.full_logprobs(x)
    ref = torch.log_softmax(x.double(), -1)
    assert_hazard_exercised("masked_neg_inf", int(torch.isneginf(x).sum()))
    assert_hazard_exercised("ties_at_max", int((x[3] == x[3].max()).sum()) - 1)
    assert_hazard_exercised("subnormal_logits", int(((x.abs() < 1.17549435e-38) & (x != 0)).sum()))
    assert_hazard_exercised("nan_logit", int(torch.isnan(x).sum()))
    assert torch.isnan(out[1]).all() and torch.isnan(out[2]).all() and torch.isnan(out[5]).all()
    finite_rows = [0, 3, 4, 6, 7, 8, 9, 10, 11]
    for r in finite_rows:
        fin = torch.isfinite(ref[r])
        assert torch.equal(torch.isneginf(out[r]), torch.isneginf(ref[r])), r
        err = (out[r][fin].double() - ref[r][fin]).abs().max().item()
        assert err < 1e-4 * max(1.0, ref[r][fin].abs().max().item()), (r, err)
    # every hazard row is row-invariant, including NaN payload bits
    for r in range(12):
        assert bitwise_equal(lp.full_logprobs(x[r : r + 1])[0], out[r]), r
    rev = torch.flip(x, [0]).contiguous()
    assert bitwise_equal(torch.flip(lp.full_logprobs(rev), [0]), out)
    print("  hazards: -inf masks/empty leaves, ties, subnormals, NaN/+inf/all -inf rows, 1e4 spreads; row-invariant bits")


def test_cuda_graph_and_no_host_sync():
    if _skip():
        return
    lp = _lp()
    g = torch.Generator(device=DEV).manual_seed(4)
    V = 151680
    x = torch.randn(8, V, device=DEV, generator=g)
    temp = torch.tensor([0.8], device=DEV)
    t = torch.randint(0, V, (8,), device=DEV, generator=g)
    torch.cuda.set_sync_debug_mode("error")
    try:
        lp.full_logprobs(x, temperature=temp)
        xg = x.clone().requires_grad_(True)
        y = lp.sampled_logprobs(xg, t)
        torch.autograd.grad(y, xg, torch.ones_like(y))
    finally:
        torch.cuda.set_sync_debug_mode(0)
    sx = x.clone()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(2):
            lp.full_logprobs(sx, temperature=temp)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gout = lp.full_logprobs(sx, temperature=temp)
    for trial in range(3):
        nx = torch.randn(8, V, device=DEV, generator=g) * (trial + 1)
        sx.copy_(nx)
        temp.fill_(0.5 + trial * 0.3)
        graph.replay()
        assert bitwise_equal(gout, lp.full_logprobs(nx, temperature=temp)), trial
    print("  CUDA graph replay == eager with changing logits and temperature; fwd/bwd free of host syncs")


def test_backward_vs_float64():
    if _skip():
        return
    lp = _lp()
    g = torch.Generator(device=DEV).manual_seed(5)
    for V, dt in ((4096, torch.float32), (151680, torch.float32), (4096, BF)):
        x = (torch.randn(29, V, device=DEV, generator=g) * 3).to(dt)
        t = torch.randint(0, V, (29,), device=DEV, generator=g)
        t[4] = -100  # ignored target
        t[9] = V + 3
        gy = torch.randn(29, device=DEV, generator=g)
        xg = x.clone().requires_grad_(True)
        (gx,) = torch.autograd.grad(lp.sampled_logprobs(xg, t), xg, gy)
        xd = x.double().requires_grad_(True)
        valid = (t >= 0) & (t < V)
        ref = torch.log_softmax(xd, -1).gather(1, t.clamp(0, V - 1)[:, None]).squeeze(1)
        (gd,) = torch.autograd.grad(ref, xd, torch.where(valid, gy, 0).double())
        rel = ((gx.double() - gd).abs().max() / gd.abs().max()).item()
        tol = 1e-5 if dt == torch.float32 else 1e-2
        assert rel < tol, (V, dt, rel)
        assert gx.dtype == dt
        # simulated shard: the local VJP of a rank equals the corresponding slice of the full VJP
        if dt == torch.float32 and V == 4096:
            ext = _ext()
            lse = torch.logsumexp(x.double(), -1)
            for r in range(2):
                part = x[:, r * V // 2 : (r + 1) * V // 2].contiguous()
                lt = torch.where(valid & (t >= r * V // 2) & (t < (r + 1) * V // 2), t - r * V // 2, torch.full_like(t, -1))
                dx = torch.empty_like(part)
                ext.grad_rows(dx, part, lse, torch.where(valid, gy, 0).float().contiguous(), lt.contiguous(), 4)
                assert ((dx.double() - gd[:, r * V // 2 : (r + 1) * V // 2]).abs().max() / gd.abs().max()).item() < 1e-5
        print(f"  backward V={V} {dt}: rel err {rel:.1e} vs float64 autograd (ignored targets, shard-local VJP)")


def test_degenerate():
    if _skip():
        return
    lp = _lp()
    assert lp.full_logprobs(torch.empty(0, 4096, device=DEV)).shape == (0, 4096)
    x0 = torch.empty(0, 4096, device=DEV, requires_grad=True)
    y0 = lp.sampled_logprobs(x0, torch.empty(0, dtype=torch.long, device=DEV))
    assert y0.shape == (0,)
    try:
        lp.full_logprobs(torch.randn(2, 100, device=DEV))
        raise AssertionError("vocab % 64 != 0 must raise")
    except ValueError:
        pass
    print("  degenerate: zero rows (fwd+autograd), unsupported vocab refused")


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
