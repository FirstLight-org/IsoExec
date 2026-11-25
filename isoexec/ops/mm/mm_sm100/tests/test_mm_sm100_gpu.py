"""Collected GPU checks for the SM100 dense GEMM."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from isoexec.core.arch import live_device_arch  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or live_device_arch() != "sm100",
    reason="the sm100 mm kernel runs on a Blackwell device only",
)


def _bits(t):
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


@pytest.fixture(scope="module")
def mm():
    from isoexec.ops.mm.mm_sm100 import ops

    ops.require_sm100()
    return ops


def operands(M, N, K, seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn(M, K, generator=g, device="cuda") * 0.05).bfloat16().contiguous()
    w = (torch.randn(N, K, generator=g, device="cuda") * 0.05).bfloat16().contiguous()
    return a, w.t()


@pytest.mark.parametrize("M,N,K", [(1, 1152, 2048), (32, 1152, 2048), (512, 4096, 4096)])
def test_accuracy_against_fp64(mm, M, N, K):
    a, b = operands(M, N, K)
    ref = (a.double() @ b.double()).float()
    cublas = torch.matmul(a, b)
    out = mm.mm_sm100(a, b)
    torch.cuda.synchronize()
    err = (out.float() - ref).abs().max().item()
    cerr = (cublas.float() - ref).abs().max().item()
    assert err <= max(4.0 * cerr, 1e-6), f"{err:.3e} vs cuBLAS {cerr:.3e}"


@pytest.mark.parametrize("M,N,K", [(1, 1152, 2048), (512, 1152, 2048)])
def test_repeat_launches_are_bit_identical(mm, M, N, K):
    a, b = operands(M, N, K)
    first = mm.mm_sm100(a, b)
    for _ in range(3):
        assert torch.equal(_bits(mm.mm_sm100(a, b)), _bits(first))


def test_row_bits_do_not_depend_on_the_batch(mm):
    """Row 0 of an M=1 call equals row 0 of the same operand inside M=512."""
    N, K = 1152, 2048
    a, b = operands(512, N, K)
    big = mm.mm_sm100(a, b)
    one = mm.mm_sm100(a[:1].contiguous(), b)
    torch.cuda.synchronize()
    assert torch.equal(_bits(one), _bits(big[:1]))


def test_unsplit_weights_are_bit_identical_to_cublas(mm):
    """The evidence that the K order this kernel consumes is the tensor core's canonical one."""
    from isoexec.ops.mm.mm_sm100 import segments as seg

    N, K = 4096, 4096
    if seg.segment_count(K, torch.bfloat16) != 1:
        pytest.skip("the schedule splits this weight, so the bits differ by design")
    a, b = operands(64, N, K)
    assert torch.equal(_bits(mm.mm_sm100(a, b)), _bits(torch.matmul(a, b)))


def test_segments_per_cta_does_not_change_the_bits(mm):
    """One CTA per segment (tree in a combine kernel) vs one CTA per tile (tree in TMEM)."""
    from isoexec.ops.mm.mm_sm100 import segments as seg

    N, K = 1152, 2048
    plans = seg.plans_at_split(512, K, N, 2)  # an execution split: the policy gives S = 1
    spcs = {p.spc for p in plans}
    if len(spcs) < 2:
        pytest.skip("this shape admits one cut of the tree only")
    a, b = operands(512, N, K)
    base = mm.run_plan(a, b, plans[0]).clone()
    for plan in plans[1:]:
        assert torch.equal(_bits(mm.run_plan(a, b, plan)), _bits(base)), plan


def test_persistent_grid_width_does_not_change_the_bits(mm):
    from isoexec.ops.mm.mm_sm100 import segments as seg

    N, K = 1152, 2048
    a, b = operands(32, N, K)
    plan = seg.plan_for(32, K, N, torch.bfloat16)
    base = mm.run_plan(a, b, plan).clone()
    for slots in (1, 5, 37, 74):
        assert torch.equal(_bits(mm.run_plan(a, b, plan, slots=slots)), _bits(base)), slots


def test_shapes_that_are_not_multiples_of_the_tile(mm):
    """Rows past M and columns past N are absent, not zero: TMA bounds stop both ends."""
    for M in (1, 7, 63, 65, 129, 1023):
        a, b = operands(M, 1152, 2048, seed=M)
        ref = (a.double() @ b.double()).float()
        out = mm.mm_sm100(a, b)
        cublas = torch.matmul(a, b)
        torch.cuda.synchronize()
        err = (out.float() - ref).abs().max().item()
        cerr = (cublas.float() - ref).abs().max().item()
        assert out.shape == (M, 1152)
        assert err <= max(4.0 * cerr, 1e-6), f"M={M}: {err:.3e} vs {cerr:.3e}"


def test_declines_rather_than_serves_what_it_cannot(mm):
    a, b = operands(8, 1152, 2048)
    assert mm.supported_shape(a, b)
    # fp16 is a different recipe; a non-unit inner stride is a layout the TMA map cannot express.
    assert not mm.supported_shape(a.half(), b.half())
    assert not mm.supported_shape(a[:, ::2], b)
    with pytest.raises(mm.UnsupportedShape):
        mm.mm(a, b, bias=torch.zeros(1152, device="cuda", dtype=torch.bfloat16))


@pytest.mark.parametrize("K,N", [(4096, 64), (4096, 128), (5120, 512)])
def test_narrow_weights_are_plan_and_batch_invariant(K, N):
    """On the narrowest audited weights (the ones policy v3 split): the production plan, every other
    legal plan at the published S, and the fp32-root publisher agree bitwise, and a row's bits do not
    read M (rows of an M = 1 / 23 call equal the same rows of an M = 2048 call)."""
    from isoexec.ops.mm.mm_sm100 import ops as mm_ops
    from isoexec.ops.mm.mm_sm100 import segments as seg

    torch.manual_seed(K + N)
    W = (torch.randn(N, K, device="cuda") * 0.05).to(torch.bfloat16)
    b = W.t()
    big = torch.randn(2048, K, device="cuda").to(torch.bfloat16)
    ref = mm_ops.mm_sm100(big, b)
    bits = ref.view(torch.int16)
    for M in (1, 23, 88, 1024):
        a = big[:M].contiguous()
        got = mm_ops.mm_sm100(a, b)
        assert torch.equal(got.view(torch.int16), bits[:M]), (M, seg.plan_for(M, K, N, torch.bfloat16))
        for p in seg.legal_plans(M, K, N, torch.bfloat16):
            assert torch.equal(mm_ops.run_plan(a, b, p).view(torch.int16), bits[:M]), (M, p.config)
        if M >= 16:   # the fp32-root publishers need a 16-row TMA pitch
            whole = seg.root_plan_for(M, K, N, torch.bfloat16)
            assert whole.spc == whole.split
            root = mm_ops.run_plan(a, b, whole, root_dtype=torch.float32)
            assert torch.equal(root.to(torch.bfloat16).view(torch.int16), bits[:M]), (M, whole.config)


def test_the_schedule_policy_is_not_on_the_per_call_path(monkeypatch):
    """The policy is evaluated when a (shape, strides, device) key is first planned; repeated calls take the host
    fast path of ``ops.mm_sm100`` and never reach ``schedule_policy`` or the planner again."""
    from isoexec.ops import schedule_policy
    from isoexec.ops.mm.mm_sm100 import ops
    from isoexec.ops.mm.mm_sm100 import segments as seg
    from isoexec.ops.mm.sm100 import mm_linear

    calls = {"reduce": 0, "plan_for": 0}
    real_reduce, real_plan = schedule_policy.reduce, seg.plan_for

    def counted_reduce(*a, **k):
        calls["reduce"] += 1
        return real_reduce(*a, **k)

    def counted_plan(*a, **k):
        calls["plan_for"] += 1
        return real_plan(*a, **k)

    monkeypatch.setattr(schedule_policy, "reduce", counted_reduce)
    monkeypatch.setattr(seg, "plan_for", counted_plan)
    for M, K, N in ((3, 768, 328), (130, 1280, 200)):        # shapes no other test of this module plans
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        ops._FAST.pop((x.shape, w.t().shape, x.stride(), w.t().stride(), x.dtype, w.dtype, x.device, w.device), None)
        first = mm_linear(x, w)
        after_first = dict(calls)
        assert after_first["plan_for"] >= 1 and after_first["reduce"] >= 1
        for _ in range(50):
            assert torch.equal(mm_linear(x, w).view(torch.int16), first.view(torch.int16))
        assert calls == after_first, (calls, after_first)
        calls.update(reduce=0, plan_for=0)
