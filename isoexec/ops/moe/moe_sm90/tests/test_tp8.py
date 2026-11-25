"""TP8 dispatches preserve the arithmetic across routing, storage and tile changes."""

import pytest
import torch

from isoexec.ops.moe.carriers import RoutingResult
from isoexec.ops._testing import bitwise_equal
from isoexec.ops.moe.moe_sm90 import ops
from isoexec.ops.moe.moe_sm90.integration import expert_mlp
from isoexec.ops.moe.moe_sm90.moe_fused_permute import route_to_permuted_rows

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="SM90 required"
)


@pytest.mark.parametrize("experts", [3, 256, 513])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_counts_plan_matches_offsets_and_graph_replay(experts, dtype):
    torch.manual_seed(12)
    counts = torch.randint(0, 33, (experts,), device="cuda", dtype=dtype)
    counts[0] = 0
    rows = int(counts.sum())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = ops.make_plan_from_counts(counts, rows)
    for _ in range(3):
        counts.copy_(counts.roll(1))
        graph.replay()
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0))).int()
        ref = ops.make_plan(offsets, rows)
        assert bitwise_equal(got.offsets, ref.offsets)
        assert bitwise_equal(got.total, ref.total)
        n = int(ref.total)
        assert bitwise_equal(got.tiles[:n], ref.tiles[:n])
    empty = ops.make_plan_from_counts(torch.zeros_like(counts), 0)
    assert int(empty.total) == 0 and not bool(empty.offsets.any())


@pytest.mark.parametrize("n", [64, 97, 129])
@pytest.mark.parametrize("offset", [0, 1])
def test_single_leaf_output_tails_and_unaligned_output(n, offset):
    torch.manual_seed(23)
    counts = torch.tensor([0, 1, 49], device="cuda", dtype=torch.int64)
    plan = ops.make_plan_from_counts(counts, 50)
    x = torch.randn(50, 64, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(3, n, 64, device="cuda", dtype=x.dtype) * 0.1
    storage = torch.empty(50 * n + offset, device="cuda", dtype=torch.float32)
    out = storage[offset:].view(50, n)
    ops.fc2(x, w, plan, leaves=1, out=out)
    # Compare each output against a singleton, crossing BM16/BM64 dispatches.
    for row, expert in [(0, 1), (1, 2), (17, 2), (49, 2)]:
        one = ops.make_plan_from_counts(torch.tensor([1], device="cuda", dtype=torch.int32), 1)
        ref = ops.fc2(x[row : row + 1], w[expert : expert + 1], one, leaves=1)
        assert bitwise_equal(out[row : row + 1], ref)


@pytest.mark.parametrize("tokens", [1, 8, 33, 128])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_routed_fc1_and_compact_root_match_materialized_path(tokens, index_dtype):
    torch.manual_seed(79)
    E, H, I = 8, 2048, 64
    x = torch.randn(tokens, H, device="cuda", dtype=torch.bfloat16) * 0.1
    w1 = torch.randn(E, 2 * I, H, device="cuda", dtype=x.dtype) * 0.02
    w2 = torch.randn(E, H, I, device="cuda", dtype=x.dtype) * 0.02
    idx = torch.randn(tokens, E, device="cuda").topk(2, dim=-1).indices.to(index_dtype)
    routing = RoutingResult(idx, torch.rand(tokens, 2, device="cuda"), None)
    with torch.no_grad():
        perm, probs, counts, indices, rows = route_to_permuted_rows(x, routing, E)
        plan = ops.make_plan_from_counts(counts, perm.size(0))
        ref = ops.fc2(ops.fc1_glu(perm, w1, plan, probs), w2, plan, leaves=1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = expert_mlp(x, routing, w1, w2, n_leaves=1, compact_root=True)
        graph.replay()
        assert got.rows.dtype == torch.bfloat16
        assert bitwise_equal(got.rows.float(), ref)
        assert bitwise_equal(got.extra, indices)
        assert bitwise_equal(got.combine_rows, rows)
        # All tokens now select the same two experts; plans/maps must be rebuilt.
        idx[:, 0] = 1
        idx[:, 1] = 3
        graph.replay()
        perm, probs, counts, indices, rows = route_to_permuted_rows(x, routing, E)
        plan = ops.make_plan_from_counts(counts, perm.size(0))
        ref = ops.fc2(ops.fc1_glu(perm, w1, plan, probs), w2, plan, leaves=1)
        assert bitwise_equal(got.rows.float(), ref)
        assert bitwise_equal(got.combine_rows, rows)


def test_compact_combine_is_fp32_and_preserves_negative_zero():
    roots = torch.tensor([[1.0], [0.00390625], [-1.0]], device="cuda", dtype=torch.bfloat16)
    rows = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    got = ops.combine_compact_roots(roots, rows)
    assert bitwise_equal(got, torch.tensor([[0.00390625]], device="cuda", dtype=roots.dtype))
    zero = torch.tensor([[-0.0]], device="cuda", dtype=roots.dtype)
    assert bitwise_equal(ops.combine_compact_roots(zero, rows[:, :1]), zero)


def test_default_consumer_and_training_keep_fp32_roots():
    x = torch.randn(3, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w1 = torch.randn(2, 128, 2048, device="cuda", dtype=x.dtype, requires_grad=True) * 0.02
    w2 = torch.randn(2, 128, 64, device="cuda", dtype=x.dtype, requires_grad=True) * 0.02
    p = torch.rand(3, 1, device="cuda", requires_grad=True)
    routing = RoutingResult(torch.tensor([[0], [1], [0]], device="cuda", dtype=torch.int32), p, None)
    got = expert_mlp(x, routing, w1, w2, n_leaves=1, compact_root=True)
    assert got.rows.dtype == torch.float32
    grads = torch.autograd.grad(got.rows.sum(), (x, p, w1, w2))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    with torch.no_grad():
        assert expert_mlp(x, routing, w1, w2, n_leaves=1).rows.dtype == torch.float32


@pytest.mark.parametrize("tokens", [1, 8, 32, 129])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_standalone_routed_path_preserves_duplicate_routes(tokens, dtype):
    torch.manual_seed(31)
    x = torch.randn(tokens, 2048, device="cuda", dtype=dtype) * 0.1
    w1 = torch.randn(8, 128, 2048, device="cuda", dtype=dtype) * 0.02
    w2 = torch.randn(8, 128, 64, device="cuda", dtype=dtype) * 0.02
    ids = torch.randint(8, (tokens, 2), device="cuda", dtype=torch.int32)
    ids[0, 1] = ids[0, 0]
    probs = torch.rand(tokens, 2, device="cuda")
    probs[0, 0] = -0.0
    with torch.no_grad():
        plan = ops.route(ids, 8)
        permuted, pp = ops.dispatch(x, probs, plan)
        expected = ops.combine(ops.experts(permuted, w1, w2, plan.grouped, pp, leaves=1), plan, dtype=dtype)
        got = ops.moe(x, w1, w2, ids, probs, leaves=1)
        assert bitwise_equal(got, expected)


@pytest.mark.parametrize("routes", [1, 8, 63, 64, 65, 255, 256])
def test_fused_small_metadata_matches_stable_integer_oracle(routes):
    torch.manual_seed(52)
    ids = torch.randint(256, (routes, 1), device="cuda", dtype=torch.int64)
    probs = torch.randn(routes, 1, device="cuda")
    probs[0] = -0.0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan, indices, pp, rows = ops.small_route_plan(ids, probs, 256)
    for _ in range(2):
        ids.copy_(ids.roll(1, 0))
        graph.replay()
        order = ids.flatten().argsort(stable=True)
        counts = torch.bincount(ids.flatten(), minlength=256)
        expected_plan = ops.make_plan_from_counts(counts, routes)
        assert bitwise_equal(indices, order)
        assert bitwise_equal(pp, probs.flatten()[order])
        assert bitwise_equal(plan.offsets, expected_plan.offsets)
        assert bitwise_equal(plan.total, expected_plan.total)
        assert bitwise_equal(plan.tiles[: int(plan.total)], expected_plan.tiles[: int(plan.total)])
        restored = torch.empty_like(order)
        restored[order] = torch.arange(routes, device="cuda")
        assert bitwise_equal(rows.flatten(), restored.int())


def test_routed_fc1_finite_row_is_invariant_to_nan_neighbors():
    torch.manual_seed(91)
    x = torch.randn(33, 2048, device="cuda", dtype=torch.bfloat16) * 0.1
    w1 = torch.randn(8, 128, 2048, device="cuda", dtype=x.dtype) * 0.02
    w2 = torch.randn(8, 128, 64, device="cuda", dtype=x.dtype) * 0.02
    ids = torch.tensor([[1, 3]], device="cuda", dtype=torch.int32).repeat(33, 1)
    probs = torch.rand(33, 2, device="cuda")
    with torch.no_grad():
        expected = ops.moe(x[:1], w1, w2, ids[:1], probs[:1], leaves=1)
        x[1:] = float("nan")
        got = ops.moe(x, w1, w2, ids, probs, leaves=1)
    assert torch.isnan(x[1:]).all() and torch.isfinite(got[:1]).all()
    assert bitwise_equal(got[:1], expected)
