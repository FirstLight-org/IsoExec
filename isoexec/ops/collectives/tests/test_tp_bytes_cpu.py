"""Tensor-parallel forward/backward tests using real CPU process groups."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops.collectives.tests._gloo_harness import run_world

WORLD = 4
T, N = 5, 3
LO, HI = 1, 3  # the select window claim (2) and claim (3) are stated against


def _placement(tp: int, rank: int, group):
    from isoexec.schema import Axis, Placement

    t, one = Axis(tp, rank, group), Axis(1, 0, None)
    return Placement(tp=t, ep=one, dp=one, pp=one, etp=t, cp=one, side="trainer", leaves=8, leaf_dtype="bf16")


def _install_cpu_shims():
    """The two shims described in the module docstring. Returns the patched pik module."""
    import torch.distributed as dist

    from isoexec.ops.collectives.pik import allreduce as PA
    from isoexec.ops.collectives.pik import plan as PP

    def _tree_reduce_cpu(stacked: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        c = stacked.shape[0]
        if c == 1:
            out.copy_(stacked[0])
            return out
        slots = [stacked[i].clone() for i in range(c)]
        for dst, lhs, rhs in PP.combine_order(c):
            slots[dst] = slots[lhs] + slots[rhs]
        out.copy_(slots[0])
        return out

    PA._tree_reduce = _tree_reduce_cpu

    orig = dist.all_gather_into_tensor

    def _ag_flat(out, inp, group=None, **kw):
        world = dist.get_world_size(group)
        inp = inp.contiguous()
        return orig(out.view(world * inp.shape[0], *inp.shape[1:]), inp, group=group, **kw)

    PA.dist.all_gather_into_tensor = _ag_flat
    return PA


def _body(rank: int, world: int, out: dict, init_method: str) -> None:
    import torch.distributed as dist

    from isoexec.ops.collectives.tp_bytes import (
        bind_kv_gather,
        make_all_gather,
        make_kv_gather,
        make_tp_copy,
    )

    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world)
    _install_cpu_shims()
    group = dist.group.WORLD
    pl = _placement(world, rank, group)
    ag = make_all_gather(pl)
    res: dict = {}

    # (1) rank-ordered concat: rank r's block is exactly r*100 + the local column index
    base = torch.arange(T * N, dtype=torch.float32).reshape(T, N)
    want = torch.cat([base + r * 100 for r in range(world)], dim=1)
    x = (base + rank * 100).requires_grad_(True)
    y = ag(x)
    res["concat_rank_order"] = bool(torch.equal(y.detach(), want))
    res["concat_shape"] = list(y.shape)

    # (2) select keeps exactly those rank slots; kv_gather picks this rank's head window
    res["select_is_slot_slice"] = bool(torch.equal(ag(x, select=(LO, HI)).detach(), want[:, LO * N : HI * N]))
    u = rank // 2  # world=4, 2 kv heads -> step 2
    res["kv_gather_head"] = bool(torch.equal(make_kv_gather(pl, 2)(x).detach(), want[:, u * 2 * N : (u + 1) * 2 * N]))
    # The TP-group entry native trainers bind selects the same head window and sums only its peers.
    from isoexec.ops import _dispatch

    _dispatch.ARCH = "sm90"  # production bindings admit the declared architecture; the collective is CPU-safe
    bound = bind_kv_gather(tp_group=group, kv_heads=2, site="t.kv_gather")
    res["bound_kv_gather"] = (bound.impl, dict(bound.inputs)) == (
        "nccl_bytes_tree_rs_bwd", {"tp": world, "side": "trainer", "kv_heads": 2}
    ) and bool(torch.equal(bound.fn(x).detach(), want[:, u * 2 * N : (u + 1) * 2 * N]))
    x3 = torch.ones(T, N, requires_grad=True)
    (bound.fn(x3) * float(rank + 1)).sum().backward()
    res["bound_kv_gather_adjoint"] = bool(torch.equal(x3.grad, torch.full((T, N), float(4 * u + 3))))
    res["whole_kv_heads_bind_no_movement"] = bind_kv_gather(tp_group=group, kv_heads=world, site="t").impl == "none"

    # Check the backward adjoint as well as forward reconstruction.
    x2 = torch.ones(T, N, requires_grad=True)
    (ag(x2, select=(LO, HI)) * float(rank + 1)).sum().backward()
    inside = float(sum(p + 1 for p in range(world))) if LO <= rank < HI else 0.0
    res["adjoint_is_the_window_sum"] = bool(torch.equal(x2.grad, torch.full((T, N), inside)))
    res["adjoint_value"] = float(x2.grad[0, 0])

    # (4) tp_copy: view forward, group SUM backward
    tc = make_tp_copy(pl)
    z = torch.full((T, N), float(rank + 1), requires_grad=True)
    coeff = torch.arange(1, T * N + 1, dtype=torch.float32).reshape(T, N)
    (tc(z) * coeff).sum().backward()
    res["tp_copy_backward_is_sum"] = bool(torch.equal(z.grad, coeff * world))
    v = tc(z)
    res["tp_copy_forward_is_view"] = bool(v.data_ptr() == z.data_ptr() and v is not z)
    gs = [torch.empty_like(z.grad) for _ in range(world)]
    dist.all_gather(gs, z.grad.contiguous(), group=group)
    res["tp_copy_backward_rank_identical"] = all(torch.equal(gs[0], g) for g in gs[1:])
    # (5) T == 0: empty on every rank, no collective (a rank-dependent skip would hang here)
    e = torch.zeros(0, N, requires_grad=True)
    o = ag(e, select=(0, 2))
    res["t_zero_shape"] = list(o.shape)
    o.sum().backward()
    res["t_zero_grad_shape"] = list(e.grad.shape)

    out[rank] = res
    dist.barrier(group=group)
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return run_world(_body, world=WORLD, rendezvous=tmp_path_factory.mktemp("tp-bytes") / "store")


def test_all_gather_is_the_rank_ordered_concat(results):
    for r, res in results.items():
        assert res["concat_rank_order"], r
        assert res["concat_shape"] == [T, WORLD * N], r


def test_select_is_a_slot_slice_and_kv_gather_picks_this_ranks_head(results):
    for r, res in results.items():
        assert res["select_is_slot_slice"], r
        assert res["kv_gather_head"], r


def test_native_kv_gather_binding_matches_the_placement_gather(results):
    for r, res in results.items():
        assert res["bound_kv_gather"], r
        # Ranks 2u and 2u + 1 share head u; their upstream scales are 2u + 1 and 2u + 2.
        assert res["bound_kv_gather_adjoint"], r
        assert res["whole_kv_heads_bind_no_movement"], r


def test_all_gather_adjoint_sums_exactly_the_peers_that_share_the_window(results):
    """A wrong slot map would put another head's gradient into this head's -- invisible forward."""
    for r, res in results.items():
        expected = float(sum(p + 1 for p in range(WORLD))) if LO <= r < HI else 0.0
        assert res["adjoint_is_the_window_sum"], (r, res["adjoint_value"], expected)
        assert res["adjoint_value"] == expected, (r, res["adjoint_value"], expected)


def test_tp_copy_is_a_view_forward_and_the_group_sum_backward(results):
    for r, res in results.items():
        assert res["tp_copy_forward_is_view"], r
        assert res["tp_copy_backward_is_sum"], r
        assert res["tp_copy_backward_rank_identical"], r


def test_zero_tokens_short_circuits_without_a_collective(results):
    for r, res in results.items():
        assert res["t_zero_shape"] == [0, 2 * N], r
        assert res["t_zero_grad_shape"] == [0, N], r
