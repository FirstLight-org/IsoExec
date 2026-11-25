"""Sequence-parallel forward/backward tests using real CPU process groups."""

from __future__ import annotations

import sys
import types

import pytest
import torch

from isoexec.ops.collectives.tests._gloo_harness import assert_claim, run_world

WORLD = 4
TL, N = 3, 2  # tokens per rank, feature width
K_LOCAL, N_OUT = 2, 5


def _placement(tp: int, rank: int, group):
    from isoexec.schema import Axis, Placement

    t, one = Axis(tp, rank, group), Axis(1, 0, None)
    return Placement(
        tp=t, ep=one, dp=one, pp=one, etp=t, cp=one, side="trainer", leaves=8, leaf_dtype="bf16", sp=True
    )


def _install_cpu_shims():
    import torch.distributed as dist

    from isoexec.ops.collectives.pik import allreduce as PA
    from isoexec.ops.collectives.pik import plan as PP

    def _tree_reduce_cpu(stacked: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        c = stacked.shape[0]
        if c == 1:
            out.copy_(stacked[0])
            return out
        slots = [stacked[i].float().clone() for i in range(c)]
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


def _install_fake_pik_linear(PA):
    """Plain-matmul twins of pik's two row-parallel folds, over the real trees."""
    lin = types.ModuleType("pik.linear")

    def row_parallel_linear(x, w, bias, *, plan, tp_size, tp_rank, k_full, group, out_dtype):
        return PA.tree_all_reduce(x.float() @ w.float().t(), group=group).to(out_dtype)

    def row_parallel_linear_rs(x, w, *, plan, tp_size, tp_rank, k_full, group, out_dtype):
        return PA.tree_reduce_scatter(x.float() @ w.float().t(), group=group).to(out_dtype)

    lin.row_parallel_linear, lin.row_parallel_linear_rs = row_parallel_linear, row_parallel_linear_rs
    pkg = types.ModuleType("pik")
    pkg.linear = lin
    sys.modules["pik"], sys.modules["pik.linear"] = pkg, lin


def _body(rank: int, world: int, out: dict, init_method: str) -> None:
    import torch.distributed as dist

    from isoexec.ops.collectives import tp_tree
    from isoexec.ops.collectives.tp_bytes import (
        TpBytesError,
        make_seq_all_gather,
        make_seq_all_gather_replicated,
        make_seq_scatter,
    )

    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world)
    PA = _install_cpu_shims()
    group = dist.group.WORLD
    pl = _placement(world, rank, group)
    res: dict = {}
    T = world * TL

    # (1) rank-ordered first-dim concat; also with autograd off
    base = torch.arange(TL * N, dtype=torch.float32).reshape(TL, N)
    want = torch.cat([base + r * 100 for r in range(world)], dim=0)
    gather = make_seq_all_gather(pl)
    x = (base + rank * 100).requires_grad_(True)
    res["gather_is_rank_ordered_dim0"] = bool(torch.equal(gather(x).detach(), want))
    with torch.no_grad():
        res["gather_moves_data_without_autograd"] = bool(torch.equal(gather(base + rank * 100), want))

    # (2) adjoint: fixed-tree SUM of every rank's gradient, this rank's chunk. Integer-valued
    #     gradients make the expected sum exact whatever the tree order.
    coeff = torch.arange(1, T * N + 1, dtype=torch.float32).reshape(T, N)
    (gather(x) * (coeff * float(rank + 1))).sum().backward()
    total = coeff * float(sum(p + 1 for p in range(world)))
    res["gather_adjoint_is_tree_sum_chunk"] = bool(torch.equal(x.grad, total[rank * TL : (rank + 1) * TL]))

    # (3) replicated flavour: same forward, slice backward (no sum)
    rep = make_seq_all_gather_replicated(pl)
    x3 = (base + rank * 100).requires_grad_(True)
    y3 = rep(x3)
    res["replicated_forward_matches"] = bool(torch.equal(y3.detach(), want))
    (y3 * coeff).sum().backward()
    res["replicated_adjoint_is_own_slice"] = bool(torch.equal(x3.grad, coeff[rank * TL : (rank + 1) * TL]))

    # (4) scatter: this rank's chunk, a copy; backward all-gathers the chunk gradients (bytes)
    scatter = make_seq_scatter(pl)
    full = want.clone().requires_grad_(True)
    chunk = scatter(full)
    res["scatter_is_this_ranks_chunk"] = bool(torch.equal(chunk.detach(), want[rank * TL : (rank + 1) * TL]))
    res["scatter_output_is_a_copy"] = bool(chunk.data_ptr() != full.data_ptr() + rank * TL * N * 4 or rank == -1)
    g = torch.full((TL, N), float(rank + 1))
    g[0, 0] = -0.0  # a signed zero must survive the byte movement
    chunk.backward(g)
    rows = [torch.full((TL, N), float(p + 1)) for p in range(world)]
    for r_ in rows:
        r_[0, 0] = -0.0
    wanted_grad = torch.cat(rows, dim=0)
    res["scatter_adjoint_is_all_gather"] = bool(torch.equal(full.grad, wanted_grad))
    res["scatter_adjoint_keeps_negative_zero"] = bool(torch.signbit(full.grad[rank * TL, 0]).item())

    # (5) divisibility refusal, before any collective (every rank refuses the same way)
    try:
        scatter(torch.zeros(T + 1, N))
        res["nondivisible_refused"] = False
    except TpBytesError as exc:
        res["nondivisible_refused"] = "do not divide" in str(exc)

    # (6) row-parallel: reduce-scatter form == all-reduce form under reconstruction; same grads
    _install_fake_pik_linear(PA)
    tp_tree._reduction_plan = lambda placement: object()  # the twins ignore the plan
    torch.manual_seed(7)  # identical on every rank: the replicated input stream and upstream grads
    xin = torch.randint(-3, 4, (T, K_LOCAL * world)).float()
    wfull = torch.randint(-2, 3, (N_OUT, K_LOCAL * world)).float()
    dy_full = torch.randint(-2, 3, (T, N_OUT)).float()
    xk = xin[:, rank * K_LOCAL : (rank + 1) * K_LOCAL].contiguous()
    wk = wfull[:, rank * K_LOCAL : (rank + 1) * K_LOCAL].contiguous()

    import dataclasses

    pl_off = dataclasses.replace(pl, sp=False)
    rp, rs = tp_tree.make_row_parallel(pl_off), tp_tree.make_row_parallel_rs(pl)
    xa, wa = xk.clone().requires_grad_(True), wk.clone().requires_grad_(True)
    ya = rp(xa, wa)
    ya.backward(dy_full)
    xb, wb = xk.clone().requires_grad_(True), wk.clone().requires_grad_(True)
    yb = rs(xb, wb)
    parts = [torch.empty_like(yb) for _ in range(world)]
    dist.all_gather(parts, yb.detach().contiguous(), group=group)
    res["rs_reconstructs_the_all_reduce_output"] = bool(torch.equal(torch.cat(parts, dim=0), ya.detach()))
    res["rs_output_is_this_ranks_chunk"] = bool(torch.equal(yb.detach(), ya.detach()[rank * TL : (rank + 1) * TL]))
    res["all_reduce_output_is_the_full_product"] = bool(torch.equal(ya.detach(), xin @ wfull.t()))
    yb.backward(dy_full[rank * TL : (rank + 1) * TL].contiguous())
    res["rs_dx_matches"] = bool(torch.equal(xb.grad, xa.grad))
    res["rs_dw_matches"] = bool(torch.equal(wb.grad, wa.grad))
    with torch.no_grad():
        res["rs_forward_without_autograd"] = bool(torch.equal(rs(xk, wk), yb.detach()))

    out[rank] = res
    dist.barrier(group=group)
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return run_world(_body, world=WORLD, rendezvous=tmp_path_factory.mktemp("seq-parallel") / "store")


CLAIMS = (
    "gather_is_rank_ordered_dim0",
    "gather_moves_data_without_autograd",
    "gather_adjoint_is_tree_sum_chunk",
    "replicated_forward_matches",
    "replicated_adjoint_is_own_slice",
    "scatter_is_this_ranks_chunk",
    "scatter_output_is_a_copy",
    "scatter_adjoint_is_all_gather",
    "scatter_adjoint_keeps_negative_zero",
    "nondivisible_refused",
    "rs_reconstructs_the_all_reduce_output",
    "rs_output_is_this_ranks_chunk",
    "all_reduce_output_is_the_full_product",
    "rs_dx_matches",
    "rs_dw_matches",
    "rs_forward_without_autograd",
)


@pytest.mark.parametrize("claim", CLAIMS)
def test_sequence_parallel_wire(results, claim):
    assert_claim(results, claim, WORLD)


def test_placement_requires_a_tp_axis_for_sequence_parallelism():
    from isoexec.schema import Axis, Placement

    one = Axis(1, 0, None)
    two = Axis(2, 0, object())
    for side in ("trainer", "engine"):
        with pytest.raises(ValueError, match="tp.size >= 2"):
            Placement(tp=one, ep=one, dp=one, pp=one, etp=one, cp=one, side=side, leaves=8, sp=True)
        assert Placement(tp=two, ep=one, dp=one, pp=one, etp=two, cp=one, side=side, leaves=8, sp=True).sp
