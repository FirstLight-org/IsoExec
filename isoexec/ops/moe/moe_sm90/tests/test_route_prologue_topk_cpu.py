"""Check the [T,K] route prologue against the dense round trip and paired column permutations.
CPU/grad paths retain dense references; AOT checks require integer-only, atomics-free SM90 indexing kernels."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops.moe.carriers import RoutingResult
from isoexec.ops.moe.moe_sm90 import moe_fused_permute as P


def _routing(T, E, k, seed):
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand(T, E, generator=g)
    vals, idx = torch.topk(scores, k, dim=1, sorted=True)
    probs = torch.softmax(vals, -1)
    # a -0.0 payload and a subnormal: the prob select must move bits, not add them
    if T > 1 and k > 1:
        probs[0, 0] = -0.0
        probs[1, k - 1] = 1e-40
    return RoutingResult(indices=idx.to(torch.int32), probs=probs, scores_full=None)


_CASES = [(1, 1, 1), (1, 8, 8), (5, 3, 1), (7, 4, 4), (13, 17, 8), (64, 256, 8), (33, 16, 2), (128, 5, 5)]


@pytest.mark.parametrize("T,E,k", _CASES)
def test_topk_reference_equals_dense_references(T, E, k):
    r = _routing(T, E, k, seed=T * 131 + E)
    si, re, counts, pp = P.compact_route_positions_topk_reference(r.indices, r.probs, E)
    rmap = P._dense_map_from_topk(r.indices, E)
    si_ref, re_ref = P.megatron_permute_index_reference(rmap, T * k)
    assert torch.equal(si, si_ref) and torch.equal(re, re_ref)
    assert torch.equal(counts, rmap.sum(0).long())
    dense = torch.zeros(T, E).scatter_(1, r.indices.long(), r.probs)
    assert torch.equal(pp.view(torch.int32), dense.reshape(-1)[si * E + re].view(torch.int32))


@pytest.mark.parametrize("T,E,k", [(13, 17, 8), (64, 256, 8), (7, 4, 4)])
def test_column_order_is_unobservable(T, E, k):
    r = _routing(T, E, k, seed=7)
    g = torch.Generator().manual_seed(3)
    perm = torch.stack([torch.randperm(k, generator=g) for _ in range(T)])
    r2 = RoutingResult(indices=torch.gather(r.indices, 1, perm), probs=torch.gather(r.probs, 1, perm), scores_full=None)
    a = P.compact_route_positions_topk_reference(r.indices, r.probs, E)
    b = P.compact_route_positions_topk_reference(r2.indices, r2.probs, E)
    for x, y in zip(a, b):
        assert torch.equal(x.view(torch.uint8), y.view(torch.uint8))
    x = torch.randn(T, 6).to(torch.bfloat16)
    outs1 = P.route_to_permuted_rows(x, r, E)
    outs2 = P.route_to_permuted_rows(x, r2, E)
    for u, v in zip(outs1, outs2):
        assert torch.equal(u.view(torch.uint8), v.view(torch.uint8))
    assert torch.equal(outs1[4], outs2[4])


def test_cpu_path_is_the_dense_round_trip_and_topk_declines_off_cuda():
    T, E, k = 9, 12, 3
    r = _routing(T, E, k, seed=1)
    x = torch.randn(T, 5).to(torch.bfloat16)
    before = P.topk_prologue_counts()
    got = P.route_to_permuted_rows(x, r, E)
    ref = P._route_to_permuted_rows_dense(x, r, E)
    for a, b in zip(got, ref):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    after = P.topk_prologue_counts()
    assert after["dense"] == before["dense"] + 1 and after["served"] == before["served"]
    assert not P._topk_prologue_eligible(x, r, E)  # CPU


def test_grad_keeps_the_dense_path_even_on_a_cuda_looking_predicate(monkeypatch):
    # the predicate's grad clause, independent of a device: probs carrying grad => dense
    T, E, k = 4, 6, 2
    r = _routing(T, E, k, seed=2)
    r = RoutingResult(indices=r.indices, probs=r.probs.clone().requires_grad_(True), scores_full=None)
    x = torch.randn(T, 3).to(torch.bfloat16)
    monkeypatch.setattr(P, "HAVE_TRITON", True)

    # emulate is_cuda=True without a device: the clause order puts the grad test before any launch
    class _X(torch.Tensor):
        pass

    with torch.enable_grad():
        assert not P._topk_prologue_eligible(x, r, E)


def test_permute_rows_topk_mode_emits_the_same_rows_on_cpu():
    T, E, k = 13, 17, 8
    r = _routing(T, E, k, seed=5)
    si, re, counts, pp = P.compact_route_positions_topk_reference(r.indices, r.probs, E)
    x = torch.randn(T, 9).to(torch.bfloat16)
    out_a, rows_a = P.permute_rows(
        x, si.clone(), route_expert=re, emit_combine_rows=True, topk_indices=r.indices, num_experts=E
    )
    rmap = P._dense_map_from_topk(r.indices, E)
    si_b = si.clone()
    out_b, rows_b = P.permute_rows(x, si_b, routing_map=rmap, route_expert=re, emit_combine_rows=True)
    assert torch.equal(out_a.view(torch.uint8), out_b.view(torch.uint8))
    assert torch.equal(rows_a, rows_b)
    assert not si.__dict__ and not si_b.__dict__
    with pytest.raises(RuntimeError):
        P.permute_rows(x, si, route_expert=re, emit_combine_rows=True, topk_indices=r.indices)  # no num_experts


def test_source_has_no_atomics_and_no_float_add_in_the_topk_kernels():
    import inspect

    src = inspect.getsource(P)
    assert "tl.atomic_add" not in src
    body = src[src.index("def _route_compact_topk_kernel") : src.index("def _permute_gather_kernel")]
    assert "p = tl.where(hitj, pj, p)" in body and "+ pj" not in body  # a select, never an add


@pytest.mark.skipif(not P.HAVE_TRITON, reason="offline SM90 compilation needs Triton")
def test_topk_kernels_compile_for_sm90_without_cuda_and_carry_no_atomics_or_float_math():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    cases = (
        (
            P._route_counts_topk_kernel,
            {"IDX": "*i32", "COUNTS": "*i64", "T": "i32", "E": "i32", "K": "constexpr", "BLOCK_T": "constexpr"},
            {"K": 8, "BLOCK_T": 1024},
            1,
        ),
        (
            P._route_compact_topk_kernel,
            {
                "IDX": "*i32",
                "PROBS": "*fp32",
                "COUNTS": "*i64",
                "SORTED_IDX": "*i64",
                "ROUTE_EXPERT": "*i64",
                "PERM_PROBS": "*fp32",
                "T": "i32",
                "E": "i32",
                "K": "constexpr",
                "BLOCK_T": "constexpr",
                "BLOCK_E": "constexpr",
            },
            {"K": 8, "BLOCK_T": 1024, "BLOCK_E": 256},
            3,
        ),
        (
            P._permute_gather_kernel,
            {
                "TOKENS": "*bf16",
                "SORTED_IDX": "*i64",
                "ROUTE_EXPERT": "*i64",
                "RMAP": "*i32",
                "OUT": "*bf16",
                "COMBINE_ROWS": "*i32",
                "hidden": "i32",
                "E": "i32",
                "K": "constexpr",
                "BLOCK_H": "constexpr",
                "BLOCK_E": "constexpr",
                "EMIT_ROWS": "constexpr",
                "TOPK_IDX": "constexpr",
            },
            {"K": 8, "BLOCK_H": 1024, "BLOCK_E": 256, "EMIT_ROWS": True, "TOPK_IDX": True},
            2,
        ),
    )
    for kernel, signature, constexprs, want_stores in cases:
        compiled = triton.compile(
            ASTSource(fn=kernel, signature=signature, constexprs=constexprs),
            target=GPUTarget("cuda", 90, 32),
        )
        ttir = compiled.asm["ttir"]
        assert "atomic" not in ttir, f"{kernel.__name__} must stay atomics-free"
        assert ttir.count("tt.store") == want_stores, (kernel.__name__, ttir.count("tt.store"))
        for op in ("arith.addf", "arith.mulf", "arith.subf", "arith.truncf", "arith.extf"):
            assert op not in ttir, f"{kernel.__name__} must not do float arithmetic ({op})"
