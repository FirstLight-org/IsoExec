"""Check decoder-block registration order, residual bindings and independent child placement.
Whole-model tests cover numerical parity.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from isoexec.nn import MLP, DecoderBlock, inventory, place, resolved_ops
from isoexec.nn._slots import TEST_PROVIDER
from isoexec.ops.table import OpTable
from isoexec.schema import replicated

from isoexec.nn.tests import _fixtures as F

INTERMEDIATE = 512


class _Sublayer(nn.Module):
    """A stand-in mixer/MLP: the block does not ask what its sublayers are, and neither does this."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(F.HIDDEN, dtype=torch.bfloat16))

    def tensor_axes(self):
        return {"weight": replicated()}

    def forward(self, h, seq=None, aux=None):
        del seq, aux
        return h * self.weight


def _block(device: str = "meta") -> DecoderBlock:
    return DecoderBlock(
        mixer=F.component(F.spec(), device=device),
        mlp=MLP(hidden=F.HIDDEN, intermediate=INTERMEDIATE, device=device),
        hidden=F.HIDDEN,
        eps=F.EPS,
        device=device,
    )


def test_the_row_registers_its_tensors_in_the_logical_order():
    """``pre_attn_norm``, ``mixer``, ``pre_mlp_norm``, ``mlp`` -- the order the inventory reads off
    ``named_parameters()`` and the order the checkpoint vocabulary names them in."""
    groups = [n.split(".")[0] for n in dict(_block().named_parameters())]
    assert [g for i, g in enumerate(groups) if i == 0 or g != groups[i - 1]] == [
        "pre_attn_norm",
        "mixer",
        "pre_mlp_norm",
        "mlp",
    ]


def test_the_row_holds_no_tensors_of_its_own_and_the_walk_still_covers_them_all():
    """The composite inherits ``tensor_axes() == {}``; every tensor under it is a child's."""
    block = _block()
    assert dict(block.tensor_axes()) == {}
    rows = inventory(block, prefix=F.NS)
    assert set(rows) == {f"{F.NS}.{n}" for n, _ in block.named_parameters()}


def test_the_residual_adds_are_resolved_ops_and_not_test_substitutions():
    """Require both residual slots to contain resolved operation bindings."""
    block = _block()
    place(block, F.placement(), prefix=F.NS, op_table=OpTable(mode="reference"))
    seams = resolved_ops(block)[F.NS]
    assert [r.op for r in seams.values()] == ["residual.add.attn", "residual.add.mlp"]
    assert {r.provider for r in seams.values()} == {"ops.reference"}
    assert TEST_PROVIDER not in {r.provider for r in seams.values()}


def test_the_row_does_not_place_its_children_the_walk_does():
    """A composite that placed its own children would be refused for re-entry when the walk
    arrived. Each child is placed once, under its own FQN, by the one walk."""
    block = _block()
    sites = place(block, F.placement(), prefix=F.NS, op_table=OpTable(mode="reference"))
    assert sites[0] == F.NS
    assert f"{F.NS}.pre_attn_norm" in sites and f"{F.NS}.mixer" in sites and f"{F.NS}.mlp" in sites
    assert block.pre_mlp_norm.site == f"{F.NS}.pre_mlp_norm"


def test_facts_state_the_row_and_leave_the_sublayers_to_their_own_sites():
    block = _block()
    assert dict(block.facts()) == {
        "kind": "decoder_block",
        "hidden": F.HIDDEN,
        "eps": F.EPS,
        "convention": "rms_zero_centered",
    }
    assert block.pre_attn_norm.facts()["role"] == "pre_attn"
    assert block.pre_mlp_norm.facts()["role"] == "pre_mlp"


def test_a_sublayer_that_is_not_a_module_is_refused_at_construction():
    """A bare callable registers no parameters, so the row's tensors would be missing from the
    inventory and from every shard plan built off it -- with nothing anywhere to notice."""
    with pytest.raises(TypeError, match="mixer must be an nn.Module"):
        DecoderBlock(mixer=lambda h, seq: h, mlp=_Sublayer(), hidden=F.HIDDEN, eps=F.EPS)


def test_the_row_runs_both_seams_in_order():
    """Two joins, in order, each reading the slot: the composition, on real numbers."""
    block = DecoderBlock(mixer=_Sublayer(), mlp=_Sublayer(), hidden=F.HIDDEN, eps=F.EPS, device="cpu")
    place(block, F.placement(), prefix=F.NS, op_table=OpTable(mode="reference"))
    with torch.no_grad():
        block.pre_attn_norm.weight.fill_(-1.0)  # zero-centered gamma of -1 is an exact zero gain
        block.pre_mlp_norm.weight.fill_(-1.0)
        block.mixer.weight.fill_(1.0)
        block.mlp.weight.fill_(1.0)
        h = torch.randn(4, F.HIDDEN, dtype=torch.bfloat16)
        out = block(h, None)
    assert torch.equal(F.bits(out), F.bits(h))  # both sublayers contributed exactly zero
