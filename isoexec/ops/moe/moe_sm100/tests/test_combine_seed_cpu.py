"""Every SM100 path folds a token's expert terms with one chain: canonical slot order (ascending expert id),
starting from the first term taken as is (the native combine kernel; ``_program.combine_reference`` is its
declared program). That is moe_sm90's fold too (``fused_combine``, the owner exchange's ``_unified_owner_core``),
so a channel whose terms are all -0 sums to -0 on every path, and the SM100 engine binds the owner exchange: with
deferred fc2 trees (EP = 1, ETP > 1) ``ISOEXEC_MOE_AFTER_FC2_OWNER=1`` serves the same bits as the reference
composition. No CUDA: the owner kernel is its generated source, interpreted
(``moe_sm90/tests/test_after_fc2_owner_cpu.run_emitted``); the native kernel's source is pinned by
``test_contract_cpu.py`` and its bits by the GPU batteries."""

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


def _chain(terms: torch.Tensor, rows: torch.Tensor, seed) -> torch.Tensor:
    """An fp32 chain over the slots in order, written out term by term: ``seed`` None starts from the first
    term as is, a float starts from that value."""
    out = torch.empty(rows.shape[0], terms.shape[1], dtype=torch.float32)
    for t in range(rows.shape[0]):
        acc = terms[int(rows[t, 0])].clone() if seed is None else torch.full((terms.shape[1],), seed)
        for j in range(0 if seed is not None else 1, rows.shape[1]):
            acc = acc + terms[int(rows[t, j])]
        out[t] = acc
    return out.bfloat16()


def test_the_combine_is_declared_as_sm90s_and_the_weights_are_not():
    assert spec.SM100_COMBINE_MATCHES_SM90 is True
    assert spec.SM100_COMBINE_ORDER == "ascending_expert_then_slot"
    # the routing weights' selected-set sum is another program than sm90's half fold, and says so
    assert spec.SM100_SELECTED_SUM == "ascending_expert_chain_from_zero"
    assert "sm100_fused_topk_v2" in {i.impl_id for i in spec.IMPLS}
    # the class on the records: both members are this package's, one native kernel, the base's ids
    members = {i.impl_id for i in spec.IMPLS if i.twin_class == spec.KORDER}
    assert members == {"sm100_gather_korder_v1", "sm100_after_fc2_combine_v1"}
    assert {i.version for i in spec.IMPLS if i.twin_class == spec.KORDER} == {1}


@pytest.mark.parametrize("world,topk,tokens", [(2, 1, 3), (2, 2, 5), (2, 3, 4), (4, 4, 6), (2, 6, 5), (8, 8, 9)])
def test_every_fold_is_the_first_term_chain_also_on_an_all_negative_zero_channel(world, topk, tokens):
    """The declared program, the sm90 owner kernel and a chain written out term by term give the same bits on
    every channel: random terms, all -0 terms (-0), and +0 / -0 mixtures (+0)."""
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
    # channel 2: -0 in every slot but each token's LAST one, which is +0 (a mixed-sign-zero sum is +0)
    terms[:, 2] = -0.0
    terms[rows[:, -1].long(), 2] = 0.0
    native = combine_reference(terms, rows, torch.bfloat16)      # the SM100 combine's declared program
    first = _chain(terms, rows, None)
    assert torch.equal(_bits(native), _bits(first))
    assert torch.equal(_bits(native), _bits(_chain(terms, rows, -0.0)))    # -0 is the additive identity
    assert (_bits(native)[:, 0] == -0x8000).all()                           # all -0 terms sum to -0
    assert (_bits(native)[:, 2] == 0).all()                                 # a +0 among the -0 terms: +0
    # a chain seeded with +0 is another program on exactly the all -0 channel: this test can tell them apart
    differ = _bits(_chain(terms, rows, 0.0)) != _bits(native)
    assert differ[:, 0].all() and not differ[:, 1:].any()
    # the sm90 owner kernel (first-term start) on the deferred pieces: the same bits everywhere
    terms_owner = (tree_sum(partials).bfloat16() * by_row[:, None].bfloat16()).float()
    owner = run_emitted(partials, rows, ordered)
    assert torch.equal(_bits(owner), _bits(combine_reference(terms_owner, rows, torch.bfloat16)))
    assert (_bits(owner)[:, 0] == -0x8000).all() and (_bits(owner)[:, 1] == 0).all()


def _deferred():
    return ExpertOutputs(torch.ones(2, 8), "permuted_partial",
                         RoutingResult(torch.tensor([[2, 1]]), torch.ones(1, 2), None),
                         reduce=lambda rows: rows, weight_after_fc2=True)


@pytest.mark.parametrize("flag", [None, "0", "1"])
def test_the_sm100_engine_combine_binds_the_owner_exchange(monkeypatch, flag):
    from isoexec.nn import _seams
    from isoexec.nn.moe import SEAM_OWNER_GROUP
    from isoexec.ops.moe import sm100

    modes = []
    monkeypatch.setattr(OC, "_engine_reference", lambda routed, shared, eager_fn: ("reference", eager_fn))
    monkeypatch.setattr(OC, "_engine_owner_dispatch",
                        lambda routed, shared, group, eager_fn: ("exchange", eager_fn))
    monkeypatch.setattr(OC, "_negotiate_owner_mode", lambda mode, group, device: modes.append(mode))
    if flag is None:
        monkeypatch.delenv("ISOEXEC_MOE_AFTER_FC2_OWNER", raising=False)
    else:
        monkeypatch.setenv("ISOEXEC_MOE_AFTER_FC2_OWNER", flag)
    # the binder's choice only: the registry of a CPU process holds the sm90 records, not this arch's
    monkeypatch.setattr(sm100, "resolution", lambda op, impl, fn, **kw: SimpleNamespace(impl=impl, fn=fn))
    engine = sm100.resolve_sm100("moe.combine", site="test.moe", routing="softmax_renorm", side="engine",
                                 weight_after_fc2=True)
    trainer = sm100.resolve_sm100("moe.combine", site="test.moe", routing="softmax_renorm", side="trainer",
                                  weight_after_fc2=True)
    assert engine.impl == trainer.impl == "sm100_after_fc2_combine_v1"
    assert trainer.fn is sm100.combine_trainable
    # production engine ETP needs the seam (nn.MoE._combine_owns_reductions)
    assert _seams.accepts_kw(engine.fn, SEAM_OWNER_GROUP)
    # deferred pieces: the owner exchange when opted in, else the reference composition; either way the eager
    # fold handed down is the trainer's own callable, the native kernel
    group = object()
    path = "exchange" if flag == "1" else "reference"
    assert engine.fn(_deferred(), None, owner_group=group) == (path, sm100.combine_trainable)
    assert engine.fn(_deferred(), None, owner_group=group) == (path, sm100.combine_trainable)
    assert modes == [(flag == "1", True)]          # negotiated once per group: (opted in, weights after fc2)
    with pytest.raises(RuntimeError, match="half a seam"):
        engine.fn(_deferred(), None)
