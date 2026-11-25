"""SM90 WGMMA arithmetic version 1: numerical and bitwise invariance gates."""

import pytest
import torch

from isoexec.ops._testing import assert_hazard_exercised, bitwise_equal
from isoexec.ops.moe.moe_sm90 import combine, dispatch, fc1, fc1_glu, fc2, make_plan, moe, route

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires H100/SM90"
)


def plan(counts):
    return make_plan(
        torch.tensor([0] + counts, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32), sum(counts)
    )


def rand(*shape, dtype=torch.bfloat16):
    return torch.randn(shape, device="cuda", dtype=dtype) * 0.2


def tree(xs):
    while len(xs) > 1:
        xs = [xs[i] + xs[i + 1] for i in range(0, len(xs), 2)]
    return xs[0]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("k,n", [(8, 7), (96, 129), (256, 64), (768, 193)])
def test_fc1_accuracy(dtype, k, n):
    torch.manual_seed(7)
    counts = [0, 1, 17, 65, 0, 129]
    p = plan(counts)
    x, w = rand(sum(counts), k, dtype=dtype), rand(len(counts), n, k, dtype=dtype)
    y = fc1(x, w, p)
    expected = torch.cat(
        [
            x[a:b].float() @ w[e].float().T
            for e, (a, b) in enumerate(zip(p.offsets.tolist()[:-1], p.offsets.tolist()[1:]))
        ]
    ).to(dtype)
    torch.testing.assert_close(y, expected, atol=0.008 if dtype == torch.bfloat16 else 0.001, rtol=0.01)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("leaves,leaf_k", [(1, 8), (2, 24), (4, 96), (8, 96), (8, 128), (8, 192)])
def test_fc2_bitwise_leaf_round_and_tree(dtype, leaves, leaf_k):
    torch.manual_seed(11)
    p = plan([3, 0, 73])
    x, w = rand(76, leaves * leaf_k, dtype=dtype), rand(3, 97, leaves * leaf_k, dtype=dtype)
    y = fc2(x, w, p, leaves=leaves)
    parts = [
        fc2(x[:, i * leaf_k : (i + 1) * leaf_k], w[:, :, i * leaf_k : (i + 1) * leaf_k], p, leaves=1)
        for i in range(leaves)
    ]
    assert y.shape == (76, 97) and y.dtype == torch.float32
    assert bitwise_equal(y, tree(parts))
    if leaves > 1:
        assert_hazard_exercised("leaf rounding changes result", not bitwise_equal(y, fc2(x, w, p, leaves=1)))
    for ranks in (2, 4, 8):
        if ranks <= leaves:
            width = x.size(1) // ranks
            roots = [
                fc2(x[:, r * width : (r + 1) * width], w[:, :, r * width : (r + 1) * width], p, leaves=leaves // ranks)
                for r in range(ranks)
            ]
            assert bitwise_equal(y, tree(roots))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("op", [fc1, fc2, fc1_glu])
def test_bitwise_shape_invariance(op, dtype):
    torch.manual_seed(23)
    k, n = 768, 192
    row, w = rand(1, k, dtype=dtype), rand(3, n, k, dtype=dtype)
    expected = op(row, w, plan([0, 1, 0]))
    for before, after in [(1, 0), (7, 56), (31, 97), (63, 65), (64, 193), (128, 7)]:
        x = rand(before + after + 1 + 17, k, dtype=dtype)
        x[17 + before] = row[0]
        y = op(x, w, plan([17, before + after + 1, 0]))
        assert bitwise_equal(y[17 + before : 18 + before], expected)
    # Output-channel shards also retain the arithmetic (GLU pairs halves, so skip here).
    if op != fc1_glu:
        y = op(row, w[:, 64:129], plan([0, 1, 0]))
        assert bitwise_equal(y, expected[:, 64:129])


def test_route_dispatch_ordered_combine_and_replay():
    torch.manual_seed(5)
    t, e, k, h = 137, 9, 4, 128
    ids = torch.randint(e, (t, k), dtype=torch.int32, device="cuda")
    probs, x = torch.rand(t, k, device="cuda"), rand(t, h)
    rp = route(ids, e)
    gx, gp = dispatch(x, probs, rp)
    assert bitwise_equal(gx[rp.restore.long()], x[:, None].expand(t, k, h))
    assert bitwise_equal(gp[rp.restore.long()], probs)
    rows = rand(t * k, h, dtype=torch.float32)
    expected = []
    for token in range(t):
        slots = torch.argsort(ids[token], stable=True)
        rr = rp.restore[token, slots].long()
        v = rows[rr[0]]
        for j in range(1, k):
            v = v + rows[rr[j]]
        expected.append(v)
    assert bitwise_equal(combine(rows, rp, dtype=torch.float32), torch.stack(expected))
    assert_hazard_exercised("expert ties", bool((ids.sort(1).values[:, 1:] == ids.sort(1).values[:, :-1]).any()))


def test_moe_batch_invariance_and_graph_dynamic_routes():
    torch.manual_seed(41)
    t, e, h, inter, topk = 129, 5, 128, 192, 3
    x, gu, down = rand(t, h), rand(e, 2 * inter, h), rand(e, h, inter)
    ids = torch.randint(e, (t, topk), device="cuda", dtype=torch.int32)
    probs = torch.rand(t, topk, device="cuda")
    expected = moe(x, gu, down, ids, probs)
    for idx in (0, 31, 64, 128):
        single = moe(x[idx : idx + 1], gu, down, ids[idx : idx + 1], probs[idx : idx + 1])
        assert bitwise_equal(single, expected[idx : idx + 1])
    # Separate expert boundaries/row placements on every replay, with all storage captured.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            moe(x, gu, down, ids, probs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = moe(x, gu, down, ids, probs)
    torch.cuda.current_stream().wait_stream(stream)
    for selected in (ids.clone(), torch.zeros_like(ids), (ids + 2) % e):
        ids.copy_(selected)
        graph.replay()
        assert bitwise_equal(captured, moe(x, gu, down, ids, probs))


def test_empty_strides_out_and_grad_refusal():
    p = plan([0, 0])
    x, w = rand(0, 128), rand(2, 67, 128)
    assert fc1(x, w, p).shape == (0, 67)
    assert fc2(x, w, p).shape == (0, 67)
    assert moe(
        rand(0, 128),
        rand(2, 384, 128),
        rand(2, 128, 192),
        torch.empty((0, 2), device="cuda", dtype=torch.int32),
        torch.empty((0, 2), device="cuda"),
    ).shape == (0, 128)
    p = plan([1, 1])
    x = rand(4, 128)[::2]
    out = torch.empty((2, 67), device="cuda", dtype=x.dtype)
    assert fc1(x, w, p, out=out) is out
    assert bitwise_equal(out, fc1(x.contiguous(), w, p))
    with pytest.raises(ValueError, match="forward-only"):
        fc1(x.requires_grad_(), w, p)
    with torch.no_grad():
        assert bitwise_equal(out, fc1(x, w, p))
    with pytest.raises(ValueError, match="K must be contiguous"):
        fc1(rand(2, 256)[:, ::2], w, p)
    with pytest.raises(ValueError, match="leaves"):
        fc2(rand(2, 128), w, p, leaves=3)


def test_subnormal_and_signed_zero_hazards():
    p = plan([1])
    x = torch.zeros((1, 128), device="cuda", dtype=torch.bfloat16)
    w = torch.ones((1, 64, 128), device="cuda", dtype=x.dtype)
    x[0, 0] = 2.0**-130
    assert_hazard_exercised("bf16 subnormal", 0 < x[0, 0].item() < torch.finfo(x.dtype).tiny)
    y = fc2(x, w, p, leaves=8)
    parts = [fc2(x[:, i : i + 16], w[:, :, i : i + 16], p, leaves=1) for i in range(0, 128, 16)]
    assert bitwise_equal(y, tree(parts))
    rp = route(torch.zeros((1, 1), device="cuda", dtype=torch.int32), 1)
    rows = torch.full((1, 64), -0.0, device="cuda")
    assert_hazard_exercised("negative zero", bool(torch.signbit(rows).all()))
    assert bitwise_equal(combine(rows, rp, dtype=torch.float32), rows)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("leaf_k", [64, 96, 128, 192])
def test_wide_gemm_shape_invariance_and_shard_tree(dtype, leaf_k):
    torch.manual_seed(102)
    counts = [1, 127, 257]
    p = plan(counts)
    x, w = rand(sum(counts), leaf_k * 8, dtype=dtype), rand(3, 257, leaf_k * 8, dtype=dtype)
    out = fc2(x, w, p)
    parts = [
        fc2(x[:, i * leaf_k : (i + 1) * leaf_k], w[:, :, i * leaf_k : (i + 1) * leaf_k], p, leaves=1) for i in range(8)
    ]
    assert bitwise_equal(out, tree(parts))
    for row, expert in [(0, 0), (1, 1), (64, 1), (128, 2), (192, 2), (256, 2), (384, 2)]:
        singleton = plan([int(i == expert) for i in range(3)])
        assert bitwise_equal(out[row : row + 1], fc2(x[row : row + 1], w, singleton))
        assert bitwise_equal(fc1(x, w, p)[row : row + 1], fc1(x[row : row + 1], w, singleton))
    # Independent, high-precision numerical oracle for the full rounded tree.
    refs = []
    start = 0
    for e, count in enumerate(counts):
        leaves = [
            (
                x[start : start + count, i * leaf_k : (i + 1) * leaf_k].double()
                @ w[e, :, i * leaf_k : (i + 1) * leaf_k].double().T
            )
            .to(dtype)
            .float()
            for i in range(8)
        ]
        refs.append(tree(leaves))
        start += count
    torch.testing.assert_close(out, torch.cat(refs), rtol=0.01, atol=0.01)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_fc1_rounding_and_probabilities(dtype):
    torch.manual_seed(204)
    p = plan([3, 129, 257])
    x, gu = rand(p.rows, 256, dtype=dtype), rand(3, 384, 256, dtype=dtype)
    probs = torch.rand(p.rows, device="cuda")
    raw = fc1(x, gu, p)
    expected = (torch.nn.functional.silu(raw[:, :192]) * raw[:, 192:] * probs[:, None]).to(dtype)
    result = fc1_glu(x, gu, p, probs)
    torch.testing.assert_close(result, expected, rtol=0.01, atol=0.001)
    from isoexec.ops.moe.moe_sm90._extension import extension

    standalone = torch.empty_like(result)
    extension().activation(raw, probs, standalone)
    assert bitwise_equal(result, standalone)
    for row, expert in [(0, 0), (3, 1), (131, 1), (132, 2), (388, 2)]:
        singleton = plan([int(i == expert) for i in range(3)])
        assert bitwise_equal(result[row : row + 1], fc1_glu(x[row : row + 1], gu, singleton, probs[row : row + 1]))


def test_zero_width_combine_and_output_alias_refusal():
    rp = route(torch.zeros((2, 1), device="cuda", dtype=torch.int32), 1)
    assert combine(torch.empty((2, 0), device="cuda"), rp).shape == (2, 0)
    x, w = rand(2, 128), rand(1, 128, 128)
    with pytest.raises(ValueError, match="alias"):
        fc1(x, w, plan([2]), out=x)


def test_graph_observes_weight_updates_and_subnormal_batchmates():
    x = torch.zeros((1, 768), device="cuda", dtype=torch.bfloat16)
    x[0, 0] = 2.0**-130
    w = torch.ones((1, 67, 768), device="cuda", dtype=x.dtype)
    p = plan([1])
    expected = fc2(x, w, p)
    batch = rand(193, 768)
    batch[128] = x[0]
    batch_plan = plan([193])
    assert bitwise_equal(fc2(batch, w, batch_plan)[128:129], expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fc1(batch, w, batch_plan)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = fc1(batch, w, batch_plan)
    torch.cuda.current_stream().wait_stream(stream)
    w.mul_(2)
    graph.replay()
    assert bitwise_equal(result, fc1(batch, w, batch_plan))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_expert_permutation_and_nonfinite_batchmates(dtype):
    torch.manual_seed(303)
    counts = [0, 3, 128, 257, 1]
    x, w = rand(sum(counts), 768, dtype=dtype), rand(5, 129, 768, dtype=dtype)
    reference = fc2(x, w, plan(counts))
    order = [4, 2, 0, 3, 1]
    chunks = x.split(counts)
    permuted = fc2(torch.cat([chunks[i] for i in order]), w[order], plan([counts[i] for i in order]))
    assert bitwise_equal(permuted, torch.cat([reference.split(counts)[i] for i in order]))
    row = rand(1, 768, dtype=dtype)
    batch = torch.full((257, 768), float("nan"), device="cuda", dtype=dtype)
    batch[128] = row[0]
    for op in (fc1, fc2):
        assert bitwise_equal(op(row, w[:1], plan([1])), op(batch, w[:1], plan([257]))[128:129])


def test_route_plan_multiple_scan_chunks():
    torch.manual_seed(404)
    ids = torch.randint(513, (2049, 2), device="cuda", dtype=torch.int32)
    rp = route(ids, 513)
    counts = torch.bincount(ids.flatten().long(), minlength=513).to(torch.int32)
    expected = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int32), counts.cumsum(0, dtype=torch.int32)))
    assert torch.equal(rp.grouped.offsets, expected)
    assert rp.grouped.total.item() == ((counts + 15) // 16).sum().item()
    assert torch.equal(rp.inverse[rp.restore.flatten().long()], torch.arange(ids.numel(), device="cuda"))
