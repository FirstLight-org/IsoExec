"""Every SM100 path folds a token's expert terms with one chain: ascending expert id from +0 (the native combine
kernel; ``_program.combine_reference`` is its declared program). moe_sm90's folds start from the first term, so
they give -0 where this chain gives +0 on a channel whose terms are all -0. The declarations say so, and the
SM100 engine binds no sm90 fold: with deferred fc2 trees (EP = 1, ETP > 1) it composes them with the same native
kernel whatever ``ISOEXEC_MOE_AFTER_FC2_OWNER`` says. No CUDA: the owner kernel is its generated source,
interpreted (``moe_sm90/tests/test_after_fc2_owner_cpu.run_emitted``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isoexec.ops.moe.carriers import ExpertOutputs, RoutingResult
from isoexec.ops.moe.moe_sm90 import moe_pik_combine_owner as OC
from isoexec.ops.moe.moe_sm90.tests.test_after_fc2_owner_cpu import run_emitted, tree_sum
from isoexec.ops.moe.moe_sm100 import spec
from isoexec.ops.moe.moe_sm100.tests._program import combine_reference


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def test_the_seed_difference_from_sm90_is_declared():
    assert spec.SM100_SELECTED_SUM == "ascending_expert_chain_from_zero"
    assert spec.SM100_COMBINE_MATCHES_SM90 is False
    # the class on the records is the on-arch one: both members are this package's, one native kernel
    members = {i.impl_id for i in spec.IMPLS if i.twin_class == spec.KORDER}
    assert members == {"sm100_gather_korder_v2", "sm100_after_fc2_combine_v2"}


@pytest.mark.parametrize("world,topk,tokens", [(2, 1, 3), (2, 3, 4), (8, 8, 9)])
def test_the_sm90_owner_fold_is_this_chain_except_on_an_all_negative_zero_channel(world, topk, tokens):
    """Why the exchange is not bound here: the same terms, the same order, and one channel of other bits."""
    torch.manual_seed(world * 100 + topk)
    hidden = 8
    indices = torch.stack([torch.randperm(16)[:topk] for _ in range(tokens)])
    probs = torch.rand(tokens, topk).float()
    rows = torch.randperm(tokens * topk).reshape(tokens, topk).int()
    partials = (torch.randn(world, tokens * topk, hidden) * 7).bfloat16().float()
    partials[:, :, 0] = -0.0          # every selected expert's term is -0 in channel 0
    partials[0, :, 1] = 0.0           # channel 1: a +0 and a -0 partial per row, so each term is +0
    partials[1:, :, 1] = -0.0
    routed = ExpertOutputs(partials[0], "permuted_partial", RoutingResult(indices, probs, None), combine_rows=rows,
                           weight_after_fc2=True)
    ordered = OC._owner_route_probabilities(routed)
    by_row = torch.empty(tokens * topk).scatter(0, rows.flatten().long(), ordered.flatten())
    # expert_parallel's ``finish``: the complete fc2 tree, one bf16 rounding, the bf16 routing weight
    terms = (tree_sum(partials).bfloat16() * by_row[:, None].bfloat16()).float()
    assert bool((terms[:, 0] == 0).all() and torch.signbit(terms[:, 0]).all())
    native = combine_reference(terms, rows, torch.bfloat16)      # the SM100 combine's program (+0 seed)
    owner = run_emitted(partials, rows, ordered)                 # the sm90 owner kernel (first-term seed)
    differ = _bits(native) != _bits(owner)
    assert differ[:, 0].all() and not differ[:, 1:].any()
    assert (_bits(owner)[:, 0] == -0x8000).all() and (_bits(native)[:, 0] == 0).all()   # -0 there, +0 here


def _deferred():
    return ExpertOutputs(torch.ones(2, 8), "permuted_partial",
                         RoutingResult(torch.tensor([[2, 1]]), torch.ones(1, 2), None),
                         reduce=lambda rows: rows, weight_after_fc2=True)


@pytest.mark.parametrize("flag", [None, "0", "1"])
def test_a_combine_bound_without_the_exchange_always_composes_the_reference(monkeypatch, capsys, flag):
    modes = []
    monkeypatch.setattr(OC, "_engine_reference", lambda routed, shared, eager_fn: ("reference", eager_fn))
    monkeypatch.setattr(OC, "_engine_owner_dispatch", lambda *args: pytest.fail("the owner exchange ran"))
    monkeypatch.setattr(OC, "_negotiate_owner_mode", lambda mode, group, device: modes.append(mode))
    if flag is None:
        monkeypatch.delenv("ISOEXEC_MOE_AFTER_FC2_OWNER", raising=False)
    else:
        monkeypatch.setenv("ISOEXEC_MOE_AFTER_FC2_OWNER", flag)

    def eager(routed, shared):
        return "eager"

    combine = OC.engine_combine_owner(eager, owner_exchange=False)
    group = object()
    assert combine(_deferred(), None, owner_group=group) == ("reference", eager)
    assert combine(_deferred(), None, owner_group=group) == ("reference", eager)
    # every rank negotiates the same mode: no exchange, weights after fc2
    assert modes == [(False, True)]
    # without deferred pieces it is the eager combine, and half a seam is still refused
    assert combine(ExpertOutputs(torch.ones(2, 8), "permuted", _deferred().routing), None) == "eager"
    with pytest.raises(RuntimeError, match="half a seam"):
        combine(_deferred(), None)
    said = capsys.readouterr().out
    assert said.count("ISOEXEC_MOE_AFTER_FC2_OWNER=1 is not served") == (1 if flag == "1" else 0)


@pytest.mark.parametrize("flag", ["0", "1"])
def test_the_sm100_engine_combine_keeps_the_owner_seam_and_never_runs_the_sm90_fold(monkeypatch, flag):
    from isoexec.nn import _seams
    from isoexec.nn.moe import SEAM_OWNER_GROUP
    from isoexec.ops.moe import sm100

    monkeypatch.setattr(OC, "_engine_reference", lambda routed, shared, eager_fn: eager_fn)
    monkeypatch.setattr(OC, "_engine_owner_dispatch", lambda *args: pytest.fail("the owner exchange ran"))
    monkeypatch.setattr(OC, "_negotiate_owner_mode", lambda *args: None)
    monkeypatch.setenv("ISOEXEC_MOE_AFTER_FC2_OWNER", flag)
    # the binder's choice only: the registry of a CPU process holds the sm90 records, not this arch's
    monkeypatch.setattr(sm100, "resolution", lambda op, impl, fn, **kw: SimpleNamespace(impl=impl, fn=fn))
    engine = sm100.resolve_sm100("moe.combine", site="test.moe", routing="softmax_renorm", side="engine",
                                 weight_after_fc2=True)
    trainer = sm100.resolve_sm100("moe.combine", site="test.moe", routing="softmax_renorm", side="trainer",
                                  weight_after_fc2=True)
    assert engine.impl == trainer.impl == "sm100_after_fc2_combine_v2"
    assert trainer.fn is sm100.combine_trainable
    # production engine ETP needs the seam (nn.MoE._combine_owns_reductions)
    assert _seams.accepts_kw(engine.fn, SEAM_OWNER_GROUP)
    # the deferred composition folds with the trainer's own callable: the native kernel
    assert engine.fn(_deferred(), None, owner_group=object()) is sm100.combine_trainable
