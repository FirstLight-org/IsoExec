"""Actual Gloo expert movement with independent CPU arithmetic; not CUDA proof."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn

from isoexec.ops.moe.carriers import ExpertOutputs, RoutingResult
from isoexec.ops.moe.expert_parallel import expert_dispatch, expert_reduce
from isoexec.ops.moe.reference import ref_combine, ref_expert_mlp
from isoexec.schema import Axis, Placement


class _GlooTree(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, group):
        peers = [torch.empty_like(rows) for _ in range(2)]
        dist.all_gather(peers, rows, group=group)
        return peers[0] + peers[1]

    @staticmethod
    def backward(ctx, *grad_outputs):
        (gradient,) = grad_outputs
        return gradient, None


def _expert_reference(x, routing, gate_up, down, **kwargs):
    output = ref_expert_mlp(x, routing, gate_up, down, **kwargs)
    assert routing.indices.shape[1] == 1
    return ExpertOutputs(output.rows[:, 0], "permuted", routing,
                         extra=torch.arange(x.shape[0], dtype=torch.int32))


def _combine(routed, shared):
    assert routed.combine_rows is not None
    return ref_combine(ExpertOutputs(routed.rows[routed.combine_rows.long()], "per_slot", routed.routing), shared)


def _check_ep(rank, group):
    generator = torch.Generator().manual_seed(100)
    inputs = (torch.randn(8, 16, generator=generator) * 0.1).bfloat16()
    weights1 = (torch.randn(4, 32, 16, generator=generator) * 0.1).bfloat16()
    weights2 = (torch.randn(4, 16, 16, generator=generator) * 0.1).bfloat16()
    indices = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3]], dtype=torch.int32).repeat(2, 1)
    probabilities = torch.tensor([[0.625, 0.25]], dtype=torch.float32).repeat(8, 1)
    local = slice(rank * 4, rank * 4 + 4)
    experts = slice(rank * 2, rank * 2 + 2)
    x, p, w1, w2 = [value.clone().requires_grad_() for value in
                    (inputs[local], probabilities[local], weights1[experts], weights2[experts])]
    one = Axis(1, 0)
    placement = Placement(tp=one, dp=Axis(2, rank, group), ep=Axis(2, rank, group),
                          etp=one, pp=one, cp=one, side="trainer", leaves=8)
    dispatch = expert_dispatch(_expert_reference, ep=2, ep_rank=rank, ep_group=group,
                               combine=_combine, placement=placement)
    actual = dispatch(x, RoutingResult(indices[local], p, None), w1, w2, weight_after_fc2=True).rows
    xx, pp, ww1, ww2 = [value.clone().requires_grad_() for value in
                        (inputs, probabilities, weights1, weights2)]
    expected = ref_combine(ref_expert_mlp(xx, RoutingResult(indices, pp, None), ww1, ww2,
                                        weight_after_fc2=True), None)
    assert torch.equal(actual, expected[local])
    actual.square().sum().backward()
    expected.square().sum().backward()
    for value, reference, region in ((x, xx, local), (p, pp, local), (w1, ww1, experts), (w2, ww2, experts)):
        assert value.grad is not None and reference.grad is not None
        torch.testing.assert_close(value.grad, reference.grad[region], rtol=0, atol=0)


def _check_etp(rank, group, deferred):
    from isoexec.ops.moe import moe_batch_invariant
    from isoexec.ops.moe.moe_sm90.moe_fused_permute import canonical_combine_rows_reference
    from isoexec.ops.moe.moe_sm90.moe_pik_combine_owner import engine_combine_owner

    # Replace only the CUDA transport primitive with an actual Gloo tree.
    setattr(moe_batch_invariant, "_PikTreeAllReduce", _GlooTree)
    parts = (torch.tensor([[1.003, 0.007], [0.241, 2.014], [1.991, 0.019], [0.52, -0.12]]),
             torch.tensor([[0.003, 0.009], [0.003, 0.003], [0.007, 0.011], [0.007, -0.019]]))
    partial = parts[rank].clone().requires_grad_()
    p = torch.tensor([[0.71, 0.13], [0.54, 0.27]], requires_grad=True)
    indices = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    tokens = torch.tensor([0, 1, 0, 1])
    rows = canonical_combine_rows_reference(torch.ones(2, 2, dtype=torch.bool),
                                             tokens, torch.tensor([0, 0, 1, 1]))
    calls = []

    def inner(x, routing, gate_up, down, *, weight_after_fc2, defer_weight_after_fc2):
        assert weight_after_fc2 and defer_weight_after_fc2
        return ExpertOutputs(partial, "permuted", routing, extra=tokens, combine_rows=rows)

    reduced = expert_reduce(inner, etp=2, group=group, deferred=deferred,
                            tp_copy=lambda value: calls.append(value) or value)
    output = reduced(torch.ones(2, 2, dtype=torch.bfloat16), RoutingResult(indices, p, None), None, None,
                     weight_after_fc2=True)
    actual = engine_combine_owner(_combine)(output, None, owner_group=group) if deferred else _combine(output, None)
    assert len(calls) == 1, "Replicated post-FC2 probability gradients must not be TP-summed twice"
    ref_parts = [part.clone().requires_grad_() for part in parts]
    ref_p = p.detach().clone().requires_grad_()
    # Independently locate each selected (token, expert) in the expert-major
    # rows. Do not reuse combine_rows to associate probabilities in the oracle.
    selected_rows = indices.long() * 2 + torch.arange(2)[:, None]
    full = (ref_parts[0] + ref_parts[1])[selected_rows].bfloat16()
    weighted = (full * ref_p[..., None].bfloat16()).float()
    ordered = weighted.gather(1, indices.argsort(dim=1)[..., None].expand_as(weighted))
    expected = ordered[:, 0] + ordered[:, 1]
    assert torch.equal(actual, expected)
    # This fixture distinguishes full-root weighting from weighting rank partials.
    wrong_parts = [(part[selected_rows].bfloat16() * ref_p[..., None].bfloat16()).float() for part in ref_parts]
    wrong = wrong_parts[0] + wrong_parts[1]
    assert not torch.equal(wrong[:, 0] + wrong[:, 1], expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(partial.grad, ref_parts[rank].grad, rtol=0, atol=0)
    torch.testing.assert_close(p.grad, ref_p.grad, rtol=0, atol=0)


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        _check_ep(rank, dist.group.WORLD)
        for deferred in (False, True):
            _check_etp(rank, dist.group.WORLD, deferred)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")
def test_after_fc2_ep2_and_etp2_forward_and_gradients(tmp_path):
    spawn(_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def test_dispatch_row_adjoint_folds_router_slots_in_order():
    """The EP dispatch's token-row gather sums a token's k route gradients in one fp32 chain over
    router slots and rounds once -- never index_select's atomic scatter-add in the operand dtype."""
    from isoexec.ops.moe.expert_parallel import _RouteRows

    g = torch.Generator().manual_seed(7)
    tl, k, h = 5, 4, 8
    x = torch.randn(tl, h, generator=g).bfloat16().requires_grad_()
    owner = torch.randint(0, 3, (tl * k,), generator=g)
    order = torch.argsort(owner, stable=True)
    tok = torch.div(order, k, rounding_mode="floor")
    restore = torch.empty_like(order).index_copy_(0, order, torch.arange(tl * k)).view(tl, k)
    rows = _RouteRows.apply(x, tok, restore)
    assert torch.equal(rows, x.detach().index_select(0, tok))
    upstream = torch.randn(tl * k, h, generator=g).bfloat16()
    rows.backward(upstream)
    by_route = torch.empty_like(upstream).index_copy_(0, order, upstream).view(tl, k, h)
    want = by_route[:, 0].float()
    for j in range(1, k):
        want = want + by_route[:, j].float()
    assert x.grad is not None
    assert torch.equal(x.grad, want.bfloat16())
