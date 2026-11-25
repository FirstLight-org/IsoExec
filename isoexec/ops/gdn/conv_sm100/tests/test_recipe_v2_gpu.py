"""conv_sm100 **recipe v2** gate battery (SM100/B200)."""

from __future__ import annotations

import sys

import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.gdn.conv_sm100.tests.test_conv_sm100_gpu import _cu, _float64, _ops, _skip

BF = torch.bfloat16
DEV = "cuda"
V2 = 2

#: The launch configs ``csrc/conv_fwd.cu`` instantiates for recipe 2 (one block width, all three
#: vector widths, plus a narrow block), with token tiles that are not multiples of each other.
V2_PLANS = [(4, 128, 4), (8, 128, 8), (16, 128, 2), (12, 64, 8), (128, 128, 4), (5, 128, 2)]


def _round32(t: torch.Tensor) -> torch.Tensor:
    """One RNE rounding of an exact float64 value to fp32, kept in float64 for the next step."""
    return t.float().double()


def _emulate(x: torch.Tensor, w: torch.Tensor, recipe: int) -> torch.Tensor:
    """The declared scalar program for one sequence, evaluated in float64 with one explicit rounding at each declared rounding point."""
    T, D = x.shape
    xd, wd = x.double(), w.double()
    xs = torch.cat([torch.zeros(3, D, dtype=torch.float64, device=DEV), xd])
    taps = [xs[k : k + T] for k in range(4)]  # oldest history first, current input last
    if recipe == 1:  # mul.rn then add.rn, every product rounded before it is added
        acc = _round32(wd[:, 0] * taps[0])
        for k in range(1, 4):
            acc = _round32(acc + _round32(wd[:, k] * taps[k]))
    else:  # one mul.rn, then three fma.rn: the product is not rounded before the add
        acc = _round32(wd[:, 0] * taps[0])
        for k in range(1, 4):
            acc = _round32(wd[:, k] * taps[k] + acc)
    return acc.float()


def _unit_operands(T, D, seed):
    """fp32 operands with magnitudes in ``[0.5, 2)``: the range that makes ``_emulate`` exact."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    def draw(*shape):
        m = torch.rand(*shape, generator=g, device=DEV) * 1.5 + 0.5
        s = torch.where(torch.rand(*shape, generator=g, device=DEV) < 0.5, -1.0, 1.0)
        return (m * s).float()
    return draw(T, D), draw(D, 4)


def test_v2_rounding_schedule_is_bitwise_the_declared_program():
    """The kernel's bits ARE ``recipe_v2.cuh``'s four roundings, in the declared tap order."""
    if _skip():
        return
    ops = _ops()
    T, D = 96, 256
    x, w = _unit_operands(T, D, 20)
    cu = _cu([T])
    for plan in ((4, 128, 4), (16, 128, 2), (12, 64, 8)):
        y2 = ops.varlen(x, w, cu_seqlens=cu, activation=None, plan=plan, recipe=V2)
        assert bitwise_equal(y2, _emulate(x, w, 2)), ("v2 schedule", plan)
    y1 = ops.varlen(x, w, cu_seqlens=cu, activation=None)
    assert bitwise_equal(y1, _emulate(x, w, 1)), "v1 schedule drifted"
    print("  v2 rounding schedule: bitwise equal to the declared fma.rn program (3 plans); v1 unmoved")


def test_v2_is_a_different_schedule_than_v1():
    """The contraction is real: on the same inputs the two recipes disagree, and v2 -- which rounds
    three fewer times per token -- is the more accurate of the two against float64."""
    if _skip():
        return
    ops = _ops()
    T, D = 512, 512
    x, w = _unit_operands(T, D, 21)
    cu = _cu([T])
    ref = _float64(x, w, None, cu, silu=False)
    y1 = ops.varlen(x, w, cu_seqlens=cu, activation=None)
    y2 = ops.varlen(x, w, cu_seqlens=cu, activation=None, recipe=V2)
    assert not bitwise_equal(y1, y2), "recipe=2 produced recipe v1's bits: the fma path is not live"
    frac = (y1.view(torch.int32) != y2.view(torch.int32)).float().mean().item()
    e1 = ((y1.double() - ref).abs() / (1 + ref.abs())).max().item()
    e2 = ((y2.double() - ref).abs() / (1 + ref.abs())).max().item()
    assert e2 <= e1, (e1, e2)
    print(f"  v2 vs v1: {frac:.1%} of elements differ; rel err vs float64 {e2:.2e} (v2) <= {e1:.2e} (v1)")


def test_v2_accuracy_vs_float64():
    """Ragged packs, bias on and off, SiLU on and off, both dtypes: v2 tracks float64."""
    if _skip():
        return
    ops = _ops()
    g = torch.Generator(device=DEV).manual_seed(22)
    lens = [1, 2, 3, 4, 17, 300, 0, 91]
    cu = _cu(lens)
    T, D = sum(lens), 256
    x32 = torch.randn(T, D, device=DEV, generator=g)
    w32 = torch.randn(D, 4, device=DEV, generator=g) * 0.3
    b32 = torch.randn(D, device=DEV, generator=g) * 0.1
    for dt, tol in ((torch.float32, 2e-6), (BF, 1e-2)):
        x, w, b = x32.to(dt), w32.to(dt), b32.to(dt)
        for bias in (None, b):
            for silu in (True, False):
                y = ops.varlen(x, w, bias, cu_seqlens=cu, activation="silu" if silu else None,
                               recipe=V2)
                ref = _float64(x, w, bias, cu, silu)
                err = ((y.double() - ref).abs() / (1 + ref.abs())).max().item()
                assert err < tol, (dt, bias is not None, silu, err)
    print("  v2 accuracy: ragged packs x dtypes x bias x activation within tolerance of float64")


def test_v2_batch_and_plan_invariance():
    """One sequence's v2 output is independent of what shares the launch, where it sits in the pack,
    and which execution plan runs -- the same claim v1 makes, re-proved for this recipe."""
    if _skip():
        return
    ops = _ops()
    g = torch.Generator(device=DEV).manual_seed(23)
    D = 1024
    target = torch.randn(333, D, device=DEV, generator=g).to(BF)
    w = (torch.randn(D, 4, device=DEV, generator=g) * 0.3).to(BF)
    alone = ops.varlen(target, w, cu_seqlens=_cu([333]), recipe=V2)
    # (v2 and v1 need not differ *here*: these operands are bf16, so an 8-bit output mantissa hides
    # most of a 24-bit schedule difference. The schedule claim is made on fp32 operands in
    # `test_v2_is_a_different_schedule_than_v1`; this test is about invariance.)
    checked = 0
    for pre in (0, 1, 7, 5000):
        for post in (0, 3, 9000):
            parts, lens = [], []
            if pre:
                parts.append(torch.randn(pre, D, device=DEV, generator=g).to(BF))
                lens.append(pre)
            parts.append(target)
            lens.append(333)
            if post:
                parts.append(torch.randn(post, D, device=DEV, generator=g).to(BF))
                lens.append(post)
            x = torch.cat(parts)
            for plan in V2_PLANS:
                y = ops.varlen(x, w, cu_seqlens=_cu(lens), plan=plan, recipe=V2)
                assert bitwise_equal(y[pre : pre + 333], alone), (pre, post, plan)
                checked += 1
    print(f"  v2 batch/plan invariance: {checked} packings x plans bitwise equal")


def test_v2_split_points_state_roundtrip_and_decode():
    """Chunked v2 prefill with carried state, then v2 decode, reproduces the unsplit v2 forward bit
    for bit at every split point and in all three conv-state layouts."""
    if _skip():
        return
    ops = _ops()
    g = torch.Generator(device=DEV).manual_seed(24)
    D, T = 256, 40
    x = torch.randn(T, D, device=DEV, generator=g).to(BF)
    w = (torch.randn(D, 4, device=DEV, generator=g) * 0.3).to(BF)
    b = (torch.randn(D, device=DEV, generator=g) * 0.1).to(BF)
    ref = ops.varlen(x, w, b, cu_seqlens=_cu([T]), recipe=V2)
    layouts = {
        "dim_major": lambda: torch.zeros(9, D, 3, device=DEV, dtype=BF),
        "dim_first": lambda: torch.zeros(9, 3, D, device=DEV, dtype=BF).transpose(1, 2),
        "strided": lambda: torch.zeros(9, D, 6, device=DEV, dtype=BF)[:, :, ::2],
    }
    n = 0
    for name, make in layouts.items():
        for split in list(range(1, T, 3)) + [None]:
            state = make()
            line = 5
            outs = []
            cut = 1 if split is None else split
            y = ops.varlen(x[:cut], w, b, cu_seqlens=_cu([cut]), conv_state=state,
                           cache_indices=torch.tensor([line], dtype=torch.int32, device=DEV),
                           has_initial_state=torch.tensor([False], device=DEV), recipe=V2)
            outs.append(y)
            pos = cut
            if split is not None and pos < T:
                other = torch.randn(7, D, device=DEV, generator=g).to(BF)
                y = ops.varlen(torch.cat([other, x[pos:]]), w, b, cu_seqlens=_cu([7, T - pos]),
                               conv_state=state,
                               cache_indices=torch.tensor([2, line], dtype=torch.int32, device=DEV),
                               has_initial_state=torch.tensor([False, True], device=DEV), recipe=V2)
                outs.append(y[7:])
                pos = T
            while pos < T:
                lanes = torch.randn(3, D, device=DEV, generator=g).to(BF)
                lanes[1] = x[pos]
                idx = torch.tensor([-1, line, 0], dtype=torch.int32, device=DEV)
                before = lanes.clone()
                ops.update(lanes, state, w, b, cache_indices=idx, out=lanes, recipe=V2,
                           threads=32 if pos % 2 else 128, cpt=8 if pos % 2 else 2)
                assert bitwise_equal(lanes[0], before[0]) and bitwise_equal(lanes[2], before[2])
                outs.append(lanes[1:2].clone())
                pos += 1
            assert bitwise_equal(torch.cat(outs), ref), (name, split)
            assert bitwise_equal(state[line].transpose(0, 1).contiguous(), x[-3:].contiguous())
            assert torch.count_nonzero(state[0]) == 0 and torch.count_nonzero(state[8]) == 0
            n += 1
    print(f"  v2 split/state round-trip: {n} (layout x split) cases bitwise equal to the unsplit v2 forward")


def test_v2_hazards_and_decode_agreement():
    """Non-finite, signed-zero and subnormal patterns are reproduced identically by v2's decode and
    v2's prefill (FTZ is off in this schedule too -- ``fma.rn`` is a rounding mode, not fast math)."""
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
    x[:, 5] = 1e-39  # subnormal inputs
    x[:, 6] = -200.0  # exp(-acc) overflows inside SiLU
    ref = ops.varlen(x, w, cu_seqlens=_cu([T]), recipe=V2)
    lin = ops.varlen(x, w, cu_seqlens=_cu([T]), activation=None, recipe=V2)
    assert int(((lin.abs() < 1.17549435e-38) & (lin != 0)).sum()) > 0, "no subnormal pre-activation"
    state = torch.zeros(2, D, 3, device=DEV)
    outs = [ops.update(x[t : t + 1].clone(), state, w, recipe=V2,
                       cache_indices=torch.tensor([1], dtype=torch.int32, device=DEV))
            for t in range(T)]
    assert bitwise_equal(torch.cat(outs), ref)
    print("  v2 hazards: inf/NaN/-0/subnormal patterns identical across v2 decode and v2 varlen")


def test_v2_repeat_and_cuda_graph_bit_equality():
    """Two launches of one call agree, and a captured graph replays v2's bits on fresh inputs."""
    if _skip():
        return
    ops = _ops()
    g = torch.Generator(device=DEV).manual_seed(25)
    D, T = 512, 300
    w = (torch.randn(D, 4, device=DEV, generator=g) * 0.3).to(BF)
    x = torch.randn(T, D, device=DEV, generator=g).to(BF)
    cu = _cu([100, 150, 50])
    assert bitwise_equal(ops.varlen(x, w, cu_seqlens=cu, recipe=V2),
                         ops.varlen(x, w, cu_seqlens=cu, recipe=V2)), "repeat launch moved bits"
    state = torch.zeros(4, D, 3, device=DEV, dtype=BF)
    idx = torch.tensor([1, 2, 3], dtype=torch.int32, device=DEV)
    has = torch.tensor([False, False, False], device=DEV)
    xs = torch.randn(3, D, device=DEV, generator=g).to(BF)
    sx, sxs, sstate = x.clone(), xs.clone(), state.clone()
    # The v2 entry points must reach the device without a host sync of their own (the engine seam
    # captures them). Checked in eager mode; capture itself synchronizes, so it runs outside.
    torch.cuda.set_sync_debug_mode("error")
    try:
        ops.varlen(sx, w, cu_seqlens=cu, conv_state=sstate, cache_indices=idx,
                   has_initial_state=has, recipe=V2)
        ops.update(sxs.clone(), sstate, w, cache_indices=idx, recipe=V2)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(2):
            ops.varlen(sx, w, cu_seqlens=cu, conv_state=sstate, cache_indices=idx,
                       has_initial_state=has, recipe=V2)
            ops.update(sxs.clone(), sstate, w, cache_indices=idx, recipe=V2)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gy = ops.varlen(sx, w, cu_seqlens=cu, conv_state=sstate, cache_indices=idx,
                        has_initial_state=has, recipe=V2)
        gz = ops.update(sxs, sstate, w, cache_indices=idx, out=sxs, recipe=V2)
    for trial in range(3):
        nx = torch.randn(T, D, device=DEV, generator=g).to(BF)
        nxs = torch.randn(3, D, device=DEV, generator=g).to(BF)
        sx.copy_(nx)
        sxs.copy_(nxs)
        sstate.zero_()
        graph.replay()
        e_state = torch.zeros_like(state)
        ey = ops.varlen(nx, w, cu_seqlens=cu, conv_state=e_state, cache_indices=idx,
                        has_initial_state=has, recipe=V2)
        ez = ops.update(nxs.clone(), e_state, w, cache_indices=idx, recipe=V2)
        assert bitwise_equal(gy, ey) and bitwise_equal(gz, ez) and bitwise_equal(sstate, e_state), trial
    print("  v2 repeat launch == itself; graph replay == eager (3 fresh inputs); no host sync in eager")


def test_v2_refuses_a_config_it_does_not_carry():
    """v2 is instantiated for a stated subset of launch configs; anything else is a refusal, never a
    silent fallback to the other recipe's schedule."""
    if _skip():
        return
    ops = _ops()
    D, T = 256, 64
    x = torch.randn(T, D, device=DEV).to(BF)
    w = (torch.randn(D, 4, device=DEV) * 0.3).to(BF)
    try:
        ops.varlen(x, w, cu_seqlens=_cu([T]), plan=(8, 256, 4), recipe=V2)
        raise AssertionError("an uninstantiated recipe-2 config must raise")
    except RuntimeError as exc:
        assert "recipe 2" in str(exc), str(exc)[:200]
    try:
        ops.varlen(x, w, cu_seqlens=_cu([T]), recipe=3)
        raise AssertionError("recipe=3 must raise")
    except ValueError:
        pass
    print("  v2 refusals: uninstantiated launch config and unknown recipe both refused")


def main():
    if _skip():
        print("SKIP: no SM100 device")
        return 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(t.__name__)
        t()
    print(f"OK {len(tests)} tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
