"""gqa_sm90 kernels: run-to-run determinism, backward policy, autograd wiring, refusals, graph replay."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires an SM90 GPU")

LENS = (1, 300, 2049, 4000, 5000)


def _bits(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def _cu(lens):
    out = torch.zeros(len(lens) + 1, dtype=torch.int32, device="cuda")
    out[1:] = torch.tensor(lens, device="cuda").cumsum(0)
    return out


def _inputs(hq, hk, d, dv=None, lens=LENS, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lens)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g).to(torch.bfloat16)  # noqa: E731
    return r(T, hq, d), r(T, hk, d), r(T, hk, dv or d), r(T, hq, dv or d), _cu(lens), max(lens)


@pytest.fixture
def policy():
    from isoexec.ops.attention.gqa_sm90 import ops

    yield ops
    ops.require_deterministic_backward(False)


@pytest.mark.parametrize(("hq", "hk", "d"), [(4, 1, 256), (2, 1, 256), (16, 2, 128), (8, 8, 64)])
def test_forward_is_run_to_run_deterministic(hq, hk, d):
    from isoexec.ops.attention.gqa_sm90 import ops

    q, k, v, _, cu, m = _inputs(hq, hk, d)
    o0, l0 = ops.varlen_fwd(q, k, v, cu, cu, m, m, d ** -0.5)
    for _ in range(3):
        o, lse = ops.varlen_fwd(q, k, v, cu, cu, m, m, d ** -0.5)
        assert torch.equal(_bits(o), _bits(o0)) and torch.equal(_bits(lse), _bits(l0))


@pytest.mark.parametrize(("hq", "hk", "d"), [(16, 2, 128), (4, 1, 128), (8, 8, 64), (4, 1, 64)])
def test_backward_is_run_to_run_deterministic_where_fa3_has_a_deterministic_kernel(hq, hk, d, policy):
    """FA3's deterministic backward (head dim <= 128): dq/dk/dv bit-identical over 5 runs, GQA included."""
    policy.require_deterministic_backward(True)
    q, k, v, do, cu, m = _inputs(hq, hk, d, seed=1)
    o, lse = policy.varlen_fwd(q, k, v, cu, cu, m, m, d ** -0.5)
    ref = policy.varlen_bwd(do, q, k, v, o, lse, cu, cu, m, m, d ** -0.5)
    for _ in range(4):
        got = policy.varlen_bwd(do, q, k, v, o, lse, cu, cu, m, m, d ** -0.5)
        for name, a, b in zip(("dq", "dk", "dv"), got, ref):
            assert torch.equal(_bits(a), _bits(b)), name


def test_head_dim_256_backward_under_required_determinism(policy):
    """FA3 has no deterministic backward at 256: a process that requires determinism gets torch's FA2
    deterministic backward (bit-repeatable), FA3's atomic kernel otherwise; d != dv is refused."""
    q, k, v, do, cu, m = _inputs(4, 1, 256, seed=6)
    o, lse = policy.varlen_fwd(q, k, v, cu, cu, m, m, 0.0625)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        policy.require_deterministic_backward(True)
        ref = policy.varlen_bwd(do, q, k, v, o, lse, cu, cu, m, m, 0.0625)
        for _ in range(3):
            got = policy.varlen_bwd(do, q, k, v, o, lse, cu, cu, m, m, 0.0625)
            assert all(torch.equal(_bits(a), _bits(b)) for a, b in zip(got, ref))
        atomic = policy.varlen_bwd(do, q, k, v, o, lse, cu, cu, m, m, 0.0625, allow_nondeterministic=True)
        rel = max(((a.float() - b.float()).norm() / b.float().norm()).item() for a, b in zip(atomic, ref))
        assert rel < 1e-2, rel
        q2, k2, v2, do2, cu2, m2 = _inputs(4, 4, 192, 128, lens=(300,))
        o2, l2 = policy.varlen_fwd(q2, k2, v2, cu2, cu2, m2, m2, 0.07)
        with pytest.raises(RuntimeError, match="neither FA3 nor FA2"):
            policy.varlen_bwd(do2, q2, k2, v2, o2, l2, cu2, cu2, m2, m2, 0.07)
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize(("hq", "hk", "d", "dv"), [(4, 1, 256, 256), (16, 2, 128, 128), (4, 4, 192, 128)])
def test_autograd_core_is_the_raw_forward_and_backward(hq, hk, d, dv):
    from isoexec.ops.attention.gqa_sm90 import ops

    q, k, v, do, cu, m = _inputs(hq, hk, d, dv, seed=2)
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]
    y = ops.varlen_attention(*leaves, cu, cu, m, m, d ** -0.5)
    y.backward(do)
    o, lse = ops.varlen_fwd(q, k, v, cu, cu, m, m, d ** -0.5)
    assert torch.equal(_bits(y.detach()), _bits(o))
    if ops.deterministic_backward(d, dv):  # the atomic 256 backward has no run-to-run identity to compare
        dq, dk, dv_ = ops.varlen_bwd(do, q, k, v, o, lse, cu, cu, m, m, d ** -0.5)
        for a, b in zip((leaves[0].grad, leaves[1].grad, leaves[2].grad), (dq, dk, dv_)):
            assert torch.equal(_bits(a), _bits(b))
    for leaf in leaves:
        assert leaf.grad is not None and torch.isfinite(leaf.grad).all()


def test_backward_matches_a_float64_reference():
    from isoexec.ops.attention.gqa_sm90 import ops

    lens = (5, 130, 257)
    q, k, v, do, cu, m = _inputs(4, 2, 128, lens=lens, seed=3)
    scale = 128 ** -0.5
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]
    ops.varlen_attention(*leaves, cu, cu, m, m, scale).backward(do)
    ref_leaves = [t.double().requires_grad_(True) for t in (q, k, v)]
    outs, s = [], 0
    for n in lens:
        qq, kk, vv = (t[s:s + n] for t in ref_leaves)
        kk, vv = kk.repeat_interleave(2, 1), vv.repeat_interleave(2, 1)
        a = torch.einsum("thd,shd->hts", qq, kk) * scale
        a = a.masked_fill(torch.ones(n, n, device="cuda", dtype=torch.bool).triu(1), float("-inf")).softmax(-1)
        outs.append(torch.einsum("hts,shd->thd", a, vv))
        s += n
    torch.cat(outs).backward(do.double())
    for got, ref in zip(leaves, ref_leaves):
        rel = (got.grad.double() - ref.grad).norm() / ref.grad.norm()
        assert rel < 1e-2, rel.item()


def test_refusals_before_any_launch():
    from isoexec.ops.attention.gqa_sm90 import ops

    q, k, v, _, cu, m = _inputs(4, 1, 128, lens=(10,))
    with pytest.raises(RuntimeError, match="bf16"):
        ops.varlen_fwd(q.half(), k.half(), v.half(), cu, cu, m, m, 0.1)
    with pytest.raises(RuntimeError, match="not a multiple"):
        ops.varlen_fwd(q[:, :3], k[:, :1].expand(-1, 2, -1), v[:, :1].expand(-1, 2, -1), cu, cu, m, m, 0.1)
    with pytest.raises(RuntimeError, match="no FA3 forward kernel"):
        ops.varlen_fwd(q, k, torch.cat((v, v, v, v), -1), cu, cu, m, m, 0.1)
    with pytest.raises(TypeError):
        ops.varlen_attention(q, k, v, cu.long(), cu, m, m, 0.1)


def test_paged_decode_replays_bit_identically_under_cuda_graph():
    """vLLM captures decode in FULL_DECODE_ONLY graphs; replays must equal eager calls."""
    from isoexec.ops.attention.gqa_sm90.integration import paged_core

    g = torch.Generator(device="cuda").manual_seed(4)
    b, page, nblocks = 8, 16, 64
    kc = torch.randn(nblocks, page, 1, 256, device="cuda", generator=g).to(torch.bfloat16)
    vc = torch.randn_like(kc)
    table = torch.randperm(nblocks, device="cuda", generator=g)[: b * 8].view(b, 8).to(torch.int32)
    q = torch.randn(b, 2, 256, device="cuda", generator=g).to(torch.bfloat16)
    cu = torch.arange(b + 1, device="cuda", dtype=torch.int32)
    used = torch.randint(1, 8 * page, (b,), device="cuda", generator=g, dtype=torch.int32)
    out = torch.empty_like(q)

    def call():
        paged_core(q, kc, vc, cu_seqlens_q=cu, seqused_k=used, block_table=table, max_seqlen_q=1,
                   max_seqlen_k=8 * page, scale=0.0625, out=out)

    call()
    eager = out.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(eager))
    used.copy_(torch.randint(1, 8 * page, (b,), device="cuda", generator=g, dtype=torch.int32))
    graph.replay()
    replay = out.clone()
    call()
    assert torch.equal(_bits(replay), _bits(out))
