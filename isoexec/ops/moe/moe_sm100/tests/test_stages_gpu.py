"""A fast stage gate on a B200: shapes, finiteness, fp64 accuracy and the structural
relations that do not need the full battery. Collected by pytest, skipped elsewhere.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or live_device_arch() != "sm100",
    reason="moe_sm100 requires an SM100 device",
)

# Qwen3.5-35B-A3B and GLM-4.7-Flash MoE geometry, plus a small adversarial one.
GEOMETRY = [
    pytest.param(2048, 256, 512, 8, 8, id="qwen35"),
    pytest.param(2048, 64, 1536, 4, 8, id="glm47"),
    pytest.param(512, 8, 128, 2, 2, id="small"),
]


@pytest.fixture(scope="module")
def ops():
    from isoexec.ops.moe.moe_sm100 import ops as module

    return module


def _weights(E, H, I, dtype, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    gu = (torch.randn(E, 2 * I, H, generator=g, dtype=torch.float32) / 16).to("cuda", dtype)
    dn = (torch.randn(E, H, I, generator=g, dtype=torch.float32) / 16).to("cuda", dtype)
    rw = (torch.randn(E, H, generator=g, dtype=torch.float32) / 8).to("cuda", dtype)
    return gu, dn, rw


@pytest.mark.parametrize("H,E,I,k,leaves", GEOMETRY)
@pytest.mark.parametrize("T", [1, 33, 512])
def test_routed_forward_matches_fp64(ops, H, E, I, k, leaves, T):
    from isoexec.ops.moe.moe_sm100.tests._program import canonical_plan, combine_reference

    dtype = torch.bfloat16
    gu, dn, rw = _weights(E, H, I, dtype, seed=H + E + I)
    x = (torch.randn(T, H, generator=torch.Generator().manual_seed(T)) / 4).to("cuda", dtype)
    with torch.no_grad():
        idx, probs = ops.route_mechanics(ops.route_score(x, rw), k)
        plan = ops.route(idx, E)
        gx, gp = ops.dispatch(x, probs, plan)
        rows = ops.fc2(ops.fc1_glu(gx, gu, plan.grouped, gp), dn, plan.grouped, leaves=leaves)
        y = ops.combine(rows, plan, dtype=dtype)

        assert torch.isfinite(rows).all()
        assert torch.isfinite(y).all()
        # The plan is the stable counting sort, and the combine is the fp32 k-order chain.
        _, offsets, _inverse, restore, combine_rows = canonical_plan(idx, E)
        assert torch.equal(plan.grouped.offsets, offsets)
        assert torch.equal(plan.restore, restore)
        assert torch.equal(plan.combine_rows, combine_rows)
        want = combine_reference(rows, plan.combine_rows, dtype)
        assert torch.equal(y.view(torch.int16), want.view(torch.int16))

        ref = torch.zeros(T, H, dtype=torch.float64, device="cuda")
        xs = x.double()
        for t in range(T):
            for j in range(k):
                e = int(idx[t, j])
                g = xs[t] @ gu[e].double().t()
                gate, up = g[:I], g[I:]
                h = (gate / (1 + torch.exp(-gate))) * up * probs[t, j].double()
                ref[t] += h @ dn[e].double().t()
        scale = max(ref.abs().max().item(), 1e-12)
        assert (y.double() - ref).abs().max().item() / scale < 4e-2


@pytest.mark.parametrize("H,E,I,k,leaves", GEOMETRY)
def test_fc2_is_the_segment_tree(ops, H, E, I, k, leaves):
    """The structural claim, stated without a tolerance: the FC2 root IS the balanced tree
    over the L separately rounded segment roots."""
    from isoexec.ops.moe.moe_sm100.tests._program import biteq, segment_tree

    dtype = torch.bfloat16
    gu, dn, rw = _weights(E, H, I, dtype, seed=H * 3 + E)
    x = (torch.randn(256, H, generator=torch.Generator().manual_seed(5)) / 4).to("cuda", dtype)
    with torch.no_grad():
        idx, probs = ops.route_mechanics(ops.route_score(x, rw), k)
        plan = ops.route(idx, E)
        gx, gp = ops.dispatch(x, probs, plan)
        h = ops.fc1_glu(gx, gu, plan.grouped, gp)
        rows = ops.fc2(h, dn, plan.grouped, leaves=leaves)
        seg = I // leaves
        leaves_fp32 = [
            ops.fc1(h[:, j * seg : (j + 1) * seg], dn[:, :, j * seg : (j + 1) * seg], plan.grouped).float()
            for j in range(leaves)
        ]
        assert biteq(rows, segment_tree(leaves_fp32))


@pytest.mark.parametrize("H,E,I,k", [(2048, 256, 512, 8), (2048, 64, 1536, 4)])
def test_fc1_glu_is_the_epilogue_of_the_projection(ops, H, E, I, k):
    from isoexec.ops.moe.moe_sm100.tests._program import biteq

    dtype = torch.bfloat16
    gu, _down, rw = _weights(E, H, I, dtype, seed=H + 7 * E)
    x = (torch.randn(256, H, generator=torch.Generator().manual_seed(11)) / 4).to("cuda", dtype)
    with torch.no_grad():
        idx, probs = ops.route_mechanics(ops.route_score(x, rw), k)
        plan = ops.route(idx, E)
        gx, gp = ops.dispatch(x, probs, plan)
        raw = ops.fc1(gx, gu, plan.grouped)
        got = ops.fc1_glu(gx, gu, plan.grouped, gp)
        # ops.activation IS the kernel's epilogue, applied to the stored projection.
        assert biteq(got, ops.activation(raw, gp))


def test_router_matches_the_dense_reference(ops):
    from isoexec.ops.moe.moe_sm100.tests._program import biteq, selected_sum_reference
    from isoexec.ops.moe.routing import softmax_rows, stable_topk

    logits = (torch.randn(128, 256, generator=torch.Generator().manual_seed(2))).to("cuda")
    with torch.no_grad():
        idx, probs = ops.route_mechanics(logits, 8)
    vals, ref_idx = stable_topk(logits, 8)
    assert torch.equal(idx, ref_idx)
    # the selected-set sum is the schedule policy's rule-2 chain (ascending expert id, from +0), not
    # ``softmax_rows``' torch.sum (the half fold in selection order, this kernel's v1 program)
    e = torch.exp(vals - torch.amax(vals, dim=-1, keepdim=True))
    assert biteq(probs, (e / selected_sum_reference(e, ref_idx)).contiguous())
    torch.testing.assert_close(probs, softmax_rows(vals), rtol=1e-6, atol=1e-7)


def test_forward_only(ops):
    x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="forward-only"):
        ops.route_score(x, w)


def test_refuses_a_non_power_of_two_top_k(ops):
    logits = torch.randn(4, 64, device="cuda")
    with torch.no_grad(), pytest.raises(ValueError, match="power of two"):
        ops.route_mechanics(logits, 6)


@pytest.mark.parametrize("leaves", [1, 2])
def test_segment_that_is_not_a_tma_atom_is_the_zero_padded_program(ops, leaves):
    """A K segment that is not a multiple of 64 is the program of the same segment zero-padded to the atom (exact zero
    products at the end of each leaf chain): one leaf reads K's tail through zero-filled TMA boxes, several leaves pad
    each declared leaf. K not a multiple of 8 per leaf is refused."""
    with torch.no_grad():
        g = torch.Generator(device="cpu").manual_seed(96 + leaves)
        h = torch.randn(128, 96, generator=g).to("cuda", torch.bfloat16)
        dn = torch.randn(2, 128, 96, generator=g).to("cuda", torch.bfloat16)
        offsets = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
        plan = ops.make_plan(offsets, 128)
        seg = 96 // leaves
        pad = -seg % 64
        hp = torch.nn.functional.pad(h.view(128, leaves, seg), (0, pad)).flatten(1)
        dnp = torch.nn.functional.pad(dn.view(2, 128, leaves, seg), (0, pad)).flatten(2)
        got = ops.fc2(h, dn, plan, leaves=leaves)
        want = ops.fc2(hp, dnp, plan, leaves=leaves)
        assert torch.equal(got.view(torch.int32), want.view(torch.int32))
        with pytest.raises(ValueError, match="multiple of"):
            ops.fc2(h[:, :92], dn[:, :, :92], plan, leaves=1)


@pytest.mark.parametrize("T,E,k", [(255, 289, 9), (2049, 289, 9), (999, 13, 3), (4097, 64, 6), (300, 7, 7)])
def test_plan_when_top_k_does_not_divide_the_route_block(ops, T, E, k):
    """Regression: with k not dividing the 1024-route emit block, a token's routes straddle block
    boundaries; every combine slot must still be written, in canonical order (ascending expert id,
    then slot), on both the single-CTA and the multi-block plan paths."""
    from isoexec.ops.moe.moe_sm100.tests._program import canonical_plan

    g = torch.Generator(device="cpu").manual_seed(T + E + k)
    scores = torch.rand(T, E, generator=g)
    idx = scores.topk(k, -1).indices.to(torch.int32)
    idx[1::3] = idx[1::3].flip(-1)                       # selection order need not be ascending
    idx = idx.cuda().contiguous()
    for _ in range(3):
        junk = torch.full((T * k * 4,), -7, dtype=torch.int32, device="cuda")   # dirty the allocator
        del junk
        plan = ops.route(idx, E)
        _, offsets, inverse, restore, combine_rows = canonical_plan(idx, E)
        assert torch.equal(plan.grouped.offsets, offsets)
        assert torch.equal(plan.restore, restore) and torch.equal(plan.inverse, inverse)
        assert torch.equal(plan.combine_rows, combine_rows)
