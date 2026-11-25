"""Check MoE communication bindings across TP, EP, ETP and sequence-parallel layouts. Negative
controls ensure reference identities cannot match qualified SM90 kernels.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.nn import SeamError, Seams
from isoexec.nn._slots import BoundOp
from isoexec.nn.component import PlacementError
from isoexec.ops import table as _resolve
from isoexec.ops.collectives.spec import SP_SEAM_VERSION
from isoexec.ops.table import OpTable

from isoexec.nn.tests import _moe_fixtures as F

#: The impl id a seam records where the placement asks for no movement at all.
NONE = "none"


def _double(monkeypatch) -> None:
    """Wrap reference bindings with TP seam keywords for placement tests.

    The double preserves reference arithmetic while exposing the host wiring contract;
    substitution remains visible in the binding record.
    """
    real = _resolve.OpTable.bind

    def fake(self, op, *, site, inputs=None, convention=None, facts=None):
        r = real(self, op, site=site, inputs=inputs, convention=convention, facts=facts)
        if op == "moe.expert_mlp":

            def experts(x, routing, gate_up, down, *, n_leaves=None):
                return r.fn(x, routing, gate_up, down)

            return BoundOp(r.op, "double.experts", None, "test", r.inputs, experts)
        if op == "moe.shared_expert":

            def shared(x, gate, up, down, expert_gate, *, row_parallel=None, tp_copy=None):
                return r.fn(x, gate, up, down, expert_gate)

            return BoundOp(r.op, "double.shared", None, "test", r.inputs, shared)
        return r

    monkeypatch.setattr(_resolve.OpTable, "bind", fake)


def _placed(*, tp=1, ep=1, sp=False, shared=True, leaves=8, side="trainer"):
    moe = F.component(F.spec(shared_intermediate=F.SHARED_INTERMEDIATE if shared else 0))
    moe.apply_placement(F.placement(tp=tp, ep=ep, sp=sp, leaves=leaves, side=side), site=F.SITE, op_table=OpTable(mode="reference"))
    return moe




@pytest.mark.parametrize(("slot", "sm90_impls"), [
    ("combine", ("pik_leaf_tree_ep_owner_fold", "pik_leaf_tree")),
    ("expert_mlp", ("cuda_wgmma_v1_ep_owner_a2a", "cuda_wgmma_v1")),
])
def test_the_reference_stages_at_ep_two_keep_the_reference_identity(slot, sm90_impls):
    """EP wrapping must preserve reference identity, so a reference composition cannot claim an
    SM90 peer's implementation.
    """
    bound = _placed(ep=2, shared=False).slots.resolved()[slot]
    assert bound.provider == _resolve.REFERENCE_PROVIDER
    assert bound.impl not in sm90_impls


def test_the_ep_wrappers_keep_the_resolved_impl_id_and_record_the_movement():
    """Preserve the selected implementation identity and record EP transport separately."""
    moe = _placed(ep=2, shared=False)
    base = {op: _resolve.OpTable(mode="reference").bind(op, site=F.SITE, inputs={}).impl for op in ("moe.expert_mlp", "moe.combine")}
    experts, combine = moe.slots.resolved()["expert_mlp"], moe.slots.resolved()["combine"]
    assert (experts.impl, combine.impl) == (base["moe.expert_mlp"], base["moe.combine"])
    assert experts.inputs["ep"] == combine.inputs["ep"] == 2
    assert (experts.inputs["ep_movement"], combine.inputs["ep_movement"]) == ("owner_a2a", "owner_fold")




def test_the_sp_seams_are_installed_only_under_sequence_parallelism():
    """Install gather/scatter only for SP and preserve their declared versions."""
    moe = _placed(tp=2, ep=2, sp=True, shared=False)
    gather, scatter = moe.slots.resolved()["sp_gather"], moe.slots.resolved()["sp_scatter"]
    assert (gather.op, gather.impl) == ("collectives.sp_gather", "seq_all_gather_replicated_fwd_slice_bwd")
    assert (scatter.op, scatter.impl) == ("collectives.sp_scatter", "seq_scatter_fwd_all_gather_bwd")
    assert dict(gather.inputs) == {"tp": 2, "sp": True, "rank": 0}
    # Both SP seams must report the version declared by ops/collectives/spec.py.
    assert gather.version == scatter.version == SP_SEAM_VERSION
    assert moe.slots.sp_gather.__name__ == "seq_all_gather_replicated_tp2"
    assert moe.slots.sp_scatter.__name__ == "seq_scatter_tp2"


@pytest.mark.parametrize(("tp", "ep"), [(1, 1), (2, 2)])
def test_without_sp_the_seams_record_no_movement_rather_than_nothing(tp, ep):
    """A recorded "no seam here" and an absent entry read the same to a human and differently to a
    peer comparison; only one of them is a statement."""
    moe = _placed(tp=tp, ep=ep, shared=False)
    assert {moe.slots.resolved()[s].impl for s in ("sp_gather", "sp_scatter")} == {NONE}
    assert moe.slots.sp_gather(torch.ones(2)).tolist() == [1.0, 1.0]  # the identity, callable


def test_the_fc1_gradient_wire_is_the_plain_tp_copy_even_under_sp(monkeypatch):
    """Inside the gathered mixture region, fc1 needs only the TP gradient-sum wire."""
    _double(monkeypatch)
    moe = _placed(tp=2, sp=True)
    wire = moe.slots.resolved()["fc1_tp_copy"]
    assert (wire.op, wire.impl) == ("collectives.tp_copy", "identity_fwd_tree_sum_bwd")
    assert moe.slots.fc1_tp_copy.__name__ == "tp_copy_tp2"  # not seq_all_gather_tp2
    assert moe.slots.sp_gather.__name__ == "seq_all_gather_replicated_tp2"
    assert "tp_copy" in moe.slots.resolved()["shared_expert"].inputs["seams"]


def test_an_engine_wires_no_fc1_gradient_sum():
    assert _placed(tp=2, ep=2, shared=False, side="engine").slots.resolved()["fc1_tp_copy"].impl == NONE




def test_at_tp_one_the_shared_expert_owns_its_own_fc2_and_the_slot_says_so():
    waiting = _placed().slots.unfilled()
    assert set(waiting) == {"shared_row_parallel"}
    assert "owns its whole fc2 contraction" in waiting["shared_row_parallel"].fill_verb


def test_a_row_with_no_shared_expert_says_nothing_fills_its_shared_slots():
    moe = _placed(shared=False)
    assert set(moe.slots.unfilled()) == {"shared_expert", "shared_row_parallel"}
    assert all("no shared expert" in u.fill_verb for u in moe.slots.unfilled().values())


def test_the_shared_expert_takes_both_wires_at_tp_above_one(monkeypatch):
    _double(monkeypatch)
    moe = _placed(tp=2)
    assert moe.slots.resolved()["shared_expert"].inputs["seams"] == ("row_parallel", "tp_copy")
    rp = moe.slots.resolved()["shared_row_parallel"]
    # The reference leaf tree, because this host serves no accelerator: see
    # ``test_a_sequence_parallel_reduce_scatter_has_no_plain_torch_statement`` for the other arm.
    assert (rp.op, rp.impl, rp.provider) == (
        "collectives.row_parallel", "ref.leaf_tree", _resolve.REFERENCE_PROVIDER,
    )




def test_the_etp_tree_runs_inside_the_experts_and_records_the_split(monkeypatch):
    """Record local leaf ownership while applying the qualified ETP reduction."""
    _double(monkeypatch)
    moe = _placed(tp=4, shared=False)
    experts = moe.slots.resolved()["expert_mlp"]
    assert (experts.inputs["etp"], experts.inputs["fc2_leaves_local"]) == (4, 2)  # leaves=8 over etp=4
    assert experts.inputs["seams"] == ("n_leaves",)
    assert "deferred" not in experts.inputs, "the owner layout is not built here; see moe.py"
    assert moe.slots.expert_mlp.__name__ == "expert_mlp_etp4"
    assert int(experts.inputs["leaves"]) == 8, "the GLOBAL leaf plan stays on the record"


def test_an_expert_leaf_plan_the_degree_does_not_divide_is_refused(monkeypatch):
    _double(monkeypatch)
    with pytest.raises(PlacementError, match="not divisible by etp="):
        _placed(tp=8, leaves=4, shared=False)




def test_a_sequence_parallel_reduce_scatter_has_no_plain_torch_statement():
    """Reject a reference SP reduce-scatter instead of recording a qualified accelerator identity."""
    placement = F.placement(tp=2, sp=True)
    with pytest.raises(SeamError, match="no plain-torch statement"):
        Seams(placement, F.SITE, op_table=OpTable(mode="reference")).row_parallel()
    # Inside a region something already gathered, the same placement is the plain fixed tree, and
    # that has a reference. The mixture's shared expert is the shipped case.
    inside = Seams(placement, F.SITE, op_table=OpTable(mode="reference")).row_parallel(inside_sp_gather=True)
    assert (inside.impl, inside.provider) == ("ref.leaf_tree", _resolve.REFERENCE_PROVIDER)
