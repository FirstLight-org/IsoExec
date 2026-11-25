"""conv_sm90 v1 gate battery (SM90). Bit-pattern comparisons only; hazards are witnessed, not assumed.

Run: ``CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. $PY isoexec/ops/gdn/conv_sm90/tests/test_conv_sm90_gpu.py``
(or under pytest). Skips without an SM90 device.
"""

from __future__ import annotations

import sys

import torch

from isoexec.ops._testing import assert_hazard_exercised, bitwise_equal

BF = torch.bfloat16
DEV = "cuda"


def _skip() -> bool:
    return not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0)


def _ops():
    from isoexec.ops.gdn.conv_sm90 import ops

    return ops


def _cu(lens):
    c = [0]
    for n in lens:
        c.append(c[-1] + n)
    return torch.tensor(c, dtype=torch.int32, device=DEV)


def _eager_recipe(x, w, b, cu, silu=True):
    """The v1 recipe spelled with ATen fp32 ops (separate mul/add launches: no FMA)."""
    T, D = x.shape
    xf, wf = x.float(), w.float()
    y = torch.empty(T, D, dtype=torch.float32, device=DEV)
    for s, e in zip(cu.tolist()[:-1], cu.tolist()[1:]):
        if e == s:
            continue
        xs = torch.cat([torch.zeros(3, D, device=DEV), xf[s:e]])
        acc = wf[:, 0] * xs[0 : e - s]
        for k in range(1, 4):
            acc = acc + wf[:, k] * xs[k : k + e - s]
        if b is not None:
            acc = acc + b.float()
        if silu:
            acc = acc / (1.0 + torch.exp(-acc))
        y[s:e] = acc
    return y.to(x.dtype)


def _float64(x, w, b, cu, silu=True):
    T, D = x.shape
    xd, wd = x.double(), w.double()
    y = torch.zeros(T, D, dtype=torch.float64, device=DEV)
    for s, e in zip(cu.tolist()[:-1], cu.tolist()[1:]):
        xs = torch.cat([torch.zeros(3, D, dtype=torch.float64, device=DEV), xd[s:e]])
        acc = sum(wd[:, k] * xs[k : k + e - s] for k in range(4))
        if b is not None:
            acc = acc + b.double()
        y[s:e] = acc * torch.sigmoid(acc) if silu else acc
    return y


def test_accuracy_vs_float64_and_eager_recipe():
    if _skip():
        return
    ops = _ops()
    g = torch.Generator(device=DEV).manual_seed(0)
    lens = [1, 2, 3, 4, 17, 300, 0, 91]
    cu = _cu(lens)
    T, D = sum(lens), 256
    x = torch.randn(T, D, device=DEV, generator=g)
    w = torch.randn(D, 4, device=DEV, generator=g) * 0.3
    b = torch.randn(D, device=DEV, generator=g) * 0.1
    for bias in (None, b):
        for silu in (True, False):
            y = ops.varlen(x, w, bias, cu_seqlens=cu, activation="silu" if silu else None)
            ref = _float64(x, w, bias, cu, silu)
            err = ((y.double() - ref).abs() / (1 + ref.abs())).max().item()
            assert err < 2e-6, err
            eager = _eager_recipe(x, w, bias, cu, silu)
            # The recipe's scalar graph is ATen's fp32 chain: bitwise unless ATen's exp provider differs.
            same = bitwise_equal(y, eager)
            if not silu:
                assert same, "linear path must equal the ATen mul/add chain bit for bit (FMA leak?)"
            print(f"  accuracy bias={bias is not None} silu={silu}: rel err {err:.2e}, ATen-chain bitwise={same}")


def test_batch_and_plan_invariance():
    if _skip():
        return
    ops = _ops()
    g = torch.Generator(device=DEV).manual_seed(1)
    D = 1024
    target = torch.randn(333, D, device=DEV, generator=g).to(BF)
    w = (torch.randn(D, 4, device=DEV, generator=g) * 0.3).to(BF)
    alone = ops.varlen(target, w, cu_seqlens=_cu([333]))
    plans = [(8, 128), (16, 128), (32, 256)]
    checked = 0
    for pre in (0, 1, 7, 64, 5000):
        for post in (0, 3, 9000):
            parts, lens = [], []
            if pre:
                parts.append(torch.randn(pre, D, device=DEV, generator=g).to(BF))
                lens += [pre // 2, 0, pre - pre // 2]  # includes an empty sequence
            parts.append(target)
            lens.append(333)
            if post:
                parts.append(torch.randn(post, D, device=DEV, generator=g).to(BF))
                lens.append(post)
            x = torch.cat(parts)
            cu = _cu(lens)
            for plan in plans:
                y = ops.varlen(x, w, cu_seqlens=cu, plan=plan)
                assert bitwise_equal(y[pre : pre + 333], alone), (pre, post, plan)
                checked += 1
    assert_hazard_exercised("empty_sequence_lane", checked)
    print(f"  batch/plan invariance: {checked} packings x plans bitwise equal")


def test_split_points_state_roundtrip_and_decode():
    if _skip():
        return
    ops = _ops()
    g = torch.Generator(device=DEV).manual_seed(2)
    D, T = 256, 40
    x = torch.randn(T, D, device=DEV, generator=g).to(BF)
    w = (torch.randn(D, 4, device=DEV, generator=g) * 0.3).to(BF)
    b = (torch.randn(D, device=DEV, generator=g) * 0.1).to(BF)
    ref = ops.varlen(x, w, b, cu_seqlens=_cu([T]))
    layouts = {
        "dim_major": lambda: torch.zeros(9, D, 3, device=DEV, dtype=BF),
        "dim_first": lambda: torch.zeros(9, 3, D, device=DEV, dtype=BF).transpose(1, 2),
        "strided": lambda: torch.zeros(9, D, 6, device=DEV, dtype=BF)[:, :, ::2],
    }
    n_splits = 0
    for name, make in layouts.items():
        for split in list(range(1, T)) + [None]:
            state = make()
            line = 5
            outs = []
            if split is None:  # prefill 1 token, then decode everything
                cut = 1
            else:
                cut = split
            # chunk 1 (fresh), in a batch with a padding sequence at line -1 and one at the null block
            batch = torch.cat([x[:cut], torch.randn(4, D, device=DEV, generator=g).to(BF), torch.randn(2, D, device=DEV, generator=g).to(BF)])
            y = ops.varlen(batch, w, b, cu_seqlens=_cu([cut, 4, 2]), conv_state=state,
                           cache_indices=torch.tensor([line, -1, 0], dtype=torch.int32, device=DEV),
                           has_initial_state=torch.tensor([False, True, True], device=DEV))
            outs.append(y[:cut])
            pos = cut
            if split is not None and pos < T:
                # chunk 2 resumes from the state, in a different position of the pack
                other = torch.randn(7, D, device=DEV, generator=g).to(BF)
                batch = torch.cat([other, x[pos:]])
                y = ops.varlen(batch, w, b, cu_seqlens=_cu([7, T - pos]), conv_state=state,
                               cache_indices=torch.tensor([2, line], dtype=torch.int32, device=DEV),
                               has_initial_state=torch.tensor([False, True], device=DEV))
                outs.append(y[7:])
                pos = T
            while pos < T:
                lanes = torch.randn(3, D, device=DEV, generator=g).to(BF)
                lanes[1] = x[pos]
                idx = torch.tensor([-1, line, 0], dtype=torch.int32, device=DEV)
                before = lanes.clone()
                ops.update(lanes, state, w, b, cache_indices=idx, out=lanes, threads=32 if pos % 2 else 128)
                assert bitwise_equal(lanes[0], before[0]) and bitwise_equal(lanes[2], before[2]), "padding lane written"
                outs.append(lanes[1:2].clone())
                pos += 1
            got = torch.cat(outs)
            assert bitwise_equal(got, ref), (name, split)
            assert bitwise_equal(state[line].transpose(0, 1).contiguous(), x[-3:].contiguous()), (name, split)
            assert torch.count_nonzero(state[0]) == 0 and torch.count_nonzero(state[8]) == 0, "null/pad line written"
            n_splits += 1
    assert_hazard_exercised("chunk_shorter_than_window", n_splits)
    print(f"  split/state round-trip: {n_splits} (layout x split) cases bitwise equal to the trainer forward")


def test_many_sequences_resume_in_shuffled_packs():
    if _skip():
        return
    for seed, plan, maxlen in ((6, None, 90), (7, (8, 128), 6), (8, (16, 128), 12), (9, (32, 256), 40)):
        _resume_case(seed, plan, maxlen)


def _resume_case(seed, plan, maxlen):
    """Short sequences straddle tile boundaries, so spill tokens and state ownership are exercised."""
    ops = _ops()
    g = torch.Generator(device=DEV).manual_seed(seed)
    D, S = 512, 24
    w = (torch.randn(D, 4, device=DEV, generator=g) * 0.3).to(BF)
    lens = torch.randint(1, maxlen, (S,), generator=g, device=DEV).tolist()
    cuts = [int(torch.randint(0, n + 1, (1,), generator=g, device=DEV)) for n in lens]
    seqs = [torch.randn(n, D, device=DEV, generator=g).to(BF) for n in lens]
    refs = [ops.varlen(q, w, cu_seqlens=_cu([q.shape[0]])) for q in seqs]
    state = torch.zeros(S + 1, 3, D, device=DEV, dtype=BF).transpose(1, 2)
    lines = torch.randperm(S, generator=g, device=DEV).add(1).to(torch.int32)
    got = [[] for _ in range(S)]
    for phase in range(2):
        order = torch.randperm(S, generator=g, device=DEV).tolist()
        parts, ls, idx, has = [], [], [], []
        for i in order:
            piece = seqs[i][: cuts[i]] if phase == 0 else seqs[i][cuts[i] :]
            parts.append(piece)
            ls.append(piece.shape[0])
            idx.append(int(lines[i]))
            has.append(phase == 1 and cuts[i] > 0)
        y = ops.varlen(torch.cat(parts), w, cu_seqlens=_cu(ls), conv_state=state,
                       cache_indices=torch.tensor(idx, dtype=torch.int32, device=DEV),
                       has_initial_state=torch.tensor(has, device=DEV), plan=plan)
        off = 0
        for i, n in zip(order, ls):
            got[i].append(y[off : off + n])
            off += n
    for i in range(S):
        assert bitwise_equal(torch.cat(got[i]), refs[i]), (i, lens[i], cuts[i])
    assert_hazard_exercised("zero_length_chunk", sum(1 for i in range(S) if cuts[i] in (0, lens[i])))
    print(f"  plan={plan} maxlen={maxlen}: {S} sequences resumed across shuffled packs (incl. empty chunks) == unsplit")


def test_decode_batch_invariance():
    if _skip():
        return
    ops = _ops()
    g = torch.Generator(device=DEV).manual_seed(3)
    D = 1024
    w = (torch.randn(D, 4, device=DEV, generator=g) * 0.3).to(BF)
    s0 = torch.randn(1, D, 3, device=DEV, generator=g).to(BF)
    x0 = torch.randn(1, D, device=DEV, generator=g).to(BF)
    st = torch.cat([torch.zeros_like(s0), s0])  # line 0 is the null block
    solo = ops.update(x0.clone(), st, w, cache_indices=torch.tensor([1], dtype=torch.int32, device=DEV))
    assert not bitwise_equal(solo, x0)
    for N in (2, 8, 64, 512):
        states = torch.randn(N + 1, D, 3, device=DEV, generator=g).to(BF)
        xs = torch.randn(N, D, device=DEV, generator=g).to(BF)
        pos = N // 2
        states[pos + 1] = s0[0]
        xs[pos] = x0[0]
        idx = torch.arange(1, N + 1, dtype=torch.int32, device=DEV)
        idx[0] = -1  # a CUDA-graph padding lane
        out = ops.update(xs, states, w, cache_indices=idx)
        assert bitwise_equal(out[pos], solo[0]), N
        assert bitwise_equal(states[pos + 1], st[1]), N
    print("  decode batch invariance: N in (2, 8, 64, 512) bitwise equal to N=1")


def test_hazards_nonfinite_signed_zero_subnormal():
    if _skip():
        return
    ops = _ops()
    D, T = 16, 12
    x = torch.randn(T, D, device=DEV)
    w = torch.randn(D, 4, device=DEV) * 0.3
    x[3, 0] = float("inf")
    x[4, 1] = float("nan")
    x[5, 2] = -0.0
    w[3] = 0.0
    x[:, 3] = 0.0
    w[4] = -0.0
    x[:, 5] = 1e-39  # subnormal inputs
    w[5] = torch.tensor([0.5, 0.25, 1.0, 0.75])
    x[:, 6] = -200.0  # exp(-acc) overflows to +inf inside SiLU: y must be -0.0/NaN consistently
    w[6] = 1.0
    ref = ops.varlen(x, w, cu_seqlens=_cu([T]))
    assert_hazard_exercised("subnormal_input", int(((x.abs() < 1.17549435e-38) & (x != 0)).sum()))
    assert_hazard_exercised("nonfinite_input", int((~torch.isfinite(x)).sum()))
    sub = ops.varlen(x, w, cu_seqlens=_cu([T]), activation=None)
    assert_hazard_exercised("subnormal_preactivation", int(((sub.abs() < 1.17549435e-38) & (sub != 0)).sum()))
    neg_zero = torch.signbit(ref) & (ref == 0)
    assert_hazard_exercised("negative_zero_output", int(neg_zero.sum()))
    # decode must reproduce every exceptional pattern of the varlen form
    state = torch.zeros(2, D, 3, device=DEV)
    outs = []
    for t in range(T):
        outs.append(ops.update(x[t : t + 1].clone(), state, w, cache_indices=torch.tensor([1], dtype=torch.int32, device=DEV)))
    assert bitwise_equal(torch.cat(outs), ref)
    exact = _eager_recipe(x, w, None, _cu([T]), silu=False)
    assert bitwise_equal(sub, exact), "non-FTZ / no-FMA linear chain mismatch on subnormals"
    print("  hazards: inf/NaN/-0/subnormal patterns identical across decode and varlen; subnormals kept (no FTZ)")


def test_cuda_graph_replay_and_no_host_sync():
    if _skip():
        return
    ops = _ops()
    g = torch.Generator(device=DEV).manual_seed(4)
    D, T = 512, 300
    w = (torch.randn(D, 4, device=DEV, generator=g) * 0.3).to(BF).requires_grad_(False)
    x = torch.randn(T, D, device=DEV, generator=g).to(BF)
    cu = _cu([100, 150, 50])
    state = torch.zeros(4, D, 3, device=DEV, dtype=BF)
    idx = torch.tensor([1, 2, 3], dtype=torch.int32, device=DEV)
    has = torch.tensor([False, False, False], device=DEV)
    xs = torch.randn(3, D, device=DEV, generator=g).to(BF)
    torch.cuda.set_sync_debug_mode("error")
    try:
        ops.varlen(x, w, cu_seqlens=cu, conv_state=state, cache_indices=idx, has_initial_state=has)
        ops.update(xs.clone(), state, w, cache_indices=idx)
        xg = x.float().requires_grad_(True)
        wg = w.float().requires_grad_(True)
        y = ops.conv(xg, wg, cu_seqlens=cu)
        torch.autograd.grad(y, (xg, wg), torch.ones_like(y))
    finally:
        torch.cuda.set_sync_debug_mode(0)
    # capture varlen + update in one graph, replay with new inputs
    sx, sxs, sstate = x.clone(), xs.clone(), state.clone()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(2):
            ops.varlen(sx, w, cu_seqlens=cu, conv_state=sstate, cache_indices=idx, has_initial_state=has)
            ops.update(sxs.clone(), sstate, w, cache_indices=idx)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gy = ops.varlen(sx, w, cu_seqlens=cu, conv_state=sstate, cache_indices=idx, has_initial_state=has)
        gz = ops.update(sxs, sstate, w, cache_indices=idx, out=sxs)
    for trial in range(3):
        nx = torch.randn(T, D, device=DEV, generator=g).to(BF)
        nxs = torch.randn(3, D, device=DEV, generator=g).to(BF)
        sx.copy_(nx)
        sxs.copy_(nxs)
        sstate.zero_()
        graph.replay()
        e_state = torch.zeros_like(state)
        ey = ops.varlen(nx, w, cu_seqlens=cu, conv_state=e_state, cache_indices=idx, has_initial_state=has)
        ez = ops.update(nxs.clone(), e_state, w, cache_indices=idx)
        assert bitwise_equal(gy, ey) and bitwise_equal(gz, ez) and bitwise_equal(sstate, e_state), trial
    print("  CUDA graph replay == eager (3 fresh inputs); fwd/update/fwd+bwd ran under sync_debug_mode=error")


def test_backward_vs_float64():
    if _skip():
        return
    ops = _ops()
    g = torch.Generator(device=DEV).manual_seed(5)
    # Covers both VJP layouts: bwd2 (T*D <= 12M) and the per-tile bwd (T*D = 14M -> 64-token tiles, 25M -> 128).
    cases = [([1, 2, 3, 4, 5, 22], 16), ([3, 97], 24), ([100, 0, 1500, 7, 441], 64), ([5000, 3000, 191], 32),
             ([7000, 6, 7000], 1024), ([12000, 1, 12500], 1024)]
    for lens, D in cases:
        cu = _cu(lens)
        T = sum(lens)
        for dt in (torch.float32, BF):
            x = torch.randn(T, D, device=DEV, generator=g).to(dt)
            w = (torch.randn(D, 1, 4, device=DEV, generator=g) * 0.3).to(dt)
            b = (torch.randn(D, device=DEV, generator=g) * 0.1).to(dt)
            dy = torch.randn(T, D, device=DEV, generator=g).to(dt)
            big = T * D > 12_000_000
            for bias in ((None,) if big else (None, b)):
                for act in (("silu",) if big else ("silu", None)):
                    xg, wg = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
                    bg = bias.clone().requires_grad_(True) if bias is not None else None
                    y = ops.conv(xg, wg, bg, cu_seqlens=cu, activation=act)
                    leaves = [xg, wg] + ([bg] if bg is not None else [])
                    grads = torch.autograd.grad(y, leaves, dy)
                    xd, wd = x.double().requires_grad_(True), w.double().squeeze(1).requires_grad_(True)
                    bd = bias.double().requires_grad_(True) if bias is not None else None
                    ys = []
                    for s, e in zip(cu.tolist()[:-1], cu.tolist()[1:]):
                        xs = torch.cat([torch.zeros(3, D, dtype=torch.float64, device=DEV), xd[s:e]])
                        acc = sum(wd[:, k] * xs[k : k + e - s] for k in range(4))
                        if bd is not None:
                            acc = acc + bd
                        ys.append(acc * torch.sigmoid(acc) if act else acc)
                    ref = torch.autograd.grad(torch.cat(ys), [xd, wd] + ([bd] if bd is not None else []), dy.double())
                    tol = 2e-6 if dt == torch.float32 else 1e-2  # bf16 dx carries one bf16 rounding
                    for name, got, want in zip(("dx", "dw", "db"), grads, ref):
                        got = got.double().reshape(want.shape)
                        rel = (got - want).abs().max().item() / max(want.abs().max().item(), 1e-30)
                        lim = tol  # bf16 grads are returned in the bf16 parameter dtype
                        assert rel < lim, (lens, dt, bias is not None, act, name, rel)
    print(f"  backward: dx/dw/db match float64 autograd over {len(cases)} ragged packs x dtypes x bias x act")


def test_degenerate_shapes_and_layouts():
    if _skip():
        return
    ops = _ops()
    D = 64
    w = torch.randn(D, 4, device=DEV).to(BF)
    empty = ops.varlen(torch.empty(0, D, device=DEV, dtype=BF), w, cu_seqlens=_cu([]))
    assert empty.shape == (0, D)
    empty2 = ops.varlen(torch.empty(0, D, device=DEV, dtype=BF), w, cu_seqlens=_cu([0, 0]))
    assert empty2.shape == (0, D)
    upd = ops.update(torch.empty(0, D, device=DEV, dtype=BF), torch.zeros(1, D, 3, device=DEV, dtype=BF), w,
                     cache_indices=torch.empty(0, dtype=torch.int32, device=DEV))
    assert upd.shape == (0, D)
    base = torch.randn(11, D + 1, device=DEV).to(BF)
    x_nc = base[1:, 1:]  # non-contiguous and misaligned
    x_c = x_nc.contiguous()
    cu = _cu([4, 6])
    assert bitwise_equal(ops.varlen(x_nc, w, cu_seqlens=cu), ops.varlen(x_c, w, cu_seqlens=cu))
    raw = torch.empty(11 * D * 2 + 2, device=DEV, dtype=torch.uint8)
    x_off = torch.as_strided(torch.empty(1, device=DEV, dtype=BF), (0,), (1,))
    del raw, x_off
    try:
        ops.varlen(torch.randn(3, 12, device=DEV).to(BF), torch.randn(12, 4, device=DEV).to(BF), cu_seqlens=_cu([3]))
        raise AssertionError("D % 8 != 0 must raise")
    except ValueError:
        pass
    print("  degenerate: T=0, empty sequences, non-contiguous/misaligned views, D%8 refusal")


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
