"""Declarations only, no CUDA: what this package promises and where its proof lives."""

from __future__ import annotations

import os

import pytest

from isoexec.core.registry import TWIN_PROOFS, OneOf, Registry, runnable_batteries
from isoexec.ops.moe.moe_sm100 import spec

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 5))
BATTERY = "isoexec/ops/moe/moe_sm100/tests/moe_relations_gpu.py"


#: Every impl this package declares: all of them are its own, because ``ops/_dispatch``'s selector
#: refuses a selected package that declares anything its arch does not support.
OURS = spec.IMPLS

#: Every id in this package that claims membership of an equivalence class ON the record. Named
#: here because a twin claim needs a colocated test that names the claiming impl
#: (core/tests/test_op_gate_alignment_cpu.py), and because writing them out makes an id added to
#: a class without a test for it fail here rather than three directories away.
TWIN_CLAIMS = {
    "sm100_tcgen05_v1": spec.SEGMENT_TREE_GLU,
    "sm100_tcgen05_unfused_act_v1": spec.SEGMENT_TREE_GLU,
    # Two on-arch members of one native fold, proven EP-invariant by EP_BATTERY.
    "sm100_gather_korder_v2": spec.KORDER,
    "sm100_after_fc2_combine_v2": spec.KORDER,
}

#: The EP 1/2/4 battery for ``KORDER`` on SM100 (NCCL, the 35B layer).
EP_BATTERY = "isoexec/ops/moe/moe_sm100/tests/test_ep_korder_gpu.py"

#: Classes whose other member is SM90's, recorded in TWIN_PROOFS instead of on the record. None since
#: ``sm100_fused_topk_v2``: its selected-set sum is the schedule policy's ascending-id chain, sm90's is
#: the half fold, so the two routers' weights are not one program and the claim was withdrawn.
CROSS_ARCH_CLAIMS: dict = {}


def test_package_declares_sm100_only():
    assert spec.ARCH == "sm100"
    assert OURS, "a row is a place to put a proof; this package now has one"
    for impl in OURS:
        assert impl.supported_archs == frozenset({"sm100"}), impl.impl_id
        # The id IS the recipe, and it says which version of it.
        assert impl.version == int(impl.impl_id.rsplit("_v", 1)[1]), impl.impl_id
        assert impl.impl_id.startswith("sm100_"), impl.impl_id


def test_every_owed_op_is_declared():
    owed = {
        "moe.route_score",
        "moe.route_mechanics",
        "moe.expert_mlp",
        "moe.combine",
        "moe.shared_expert",
    }
    assert {impl.op for impl in OURS} == owed


def test_no_sm90_impl_id_is_reused():
    """The id IS the recipe. tcgen05's K step gives a different fp32 accumulation history
    inside a segment, so an sm90 id on sm100 would claim a proof that does not travel."""
    from isoexec.ops.moe.moe_sm90 import spec as sm90

    sm90_ids = {(i.op, i.impl_id) for i in sm90.IMPLS}
    assert not {(i.op, i.impl_id) for i in OURS} & sm90_ids


def test_the_cross_arch_classes_are_claimed_in_twin_proofs_not_on_the_record():
    """Each cross-arch class's other member is SM90's, and an arch-selected package cannot declare
    it (``_assert_proven`` refuses any record whose ``supported_archs`` omits the live arch), so a
    ``twin_class`` field here would read as a one-member class. The claim lives in TWIN_PROOFS.
    spec.WHY_NO_TWIN_FIELD says the same thing where the records are."""
    from isoexec.ops.moe.moe_sm90 import spec as sm90

    assert spec.WHY_NO_TWIN_FIELD
    for impl_id, cls in CROSS_ARCH_CLAIMS.items():
        impl = next(i for i in OURS if i.impl_id == impl_id)
        assert not impl.twin_class, f"{impl_id} may not carry {cls} on the record"
        assert BATTERY in TWIN_PROOFS[cls], f"{cls} does not name this package's battery"
        # The sibling exists and still claims the class on its own arch.
        siblings = [i for i in sm90.IMPLS if i.twin_class == cls]
        assert siblings, f"{cls} has lost its SM90 members; the cross-arch claim has no counterparty"
        assert all(i.supported_archs == frozenset({"sm90"}) for i in siblings)


def test_each_twin_claim_is_named_here_and_carries_this_battery():
    """The colocated evidence for every equivalence claim this package makes."""
    claimed = {i.impl_id: i.twin_class for i in OURS if i.twin_class}
    assert not set(claimed) & set(CROSS_ARCH_CLAIMS)
    assert claimed == TWIN_CLAIMS, (
        "an impl joined or left an equivalence class without this test following it: "
        f"declared {sorted(claimed.items())}, named here {sorted(TWIN_CLAIMS.items())}"
    )
    for impl_id, cls in TWIN_CLAIMS.items():
        assert BATTERY in TWIN_PROOFS[cls], f"{impl_id} claims {cls} whose proofs omit this battery"


def test_ep_admits_against_the_two_sm100_korder_members():
    """``runtimes/parallel.combine_twin_members`` reads the class off the records; on a B200 the
    moe registry holds only this package, so both members must be ours and the EP battery must
    be named for the class."""
    from isoexec.runtimes.parallel import COMBINE_TWIN_CLASS

    assert COMBINE_TWIN_CLASS == spec.KORDER
    members = {i.impl_id for i in OURS if i.op == "moe.combine" and i.twin_class == spec.KORDER}
    assert members == {"sm100_gather_korder_v2", "sm100_after_fc2_combine_v2"}
    assert {i.impl_id for i in OURS if i.op == "moe.combine"} == members
    assert EP_BATTERY in runnable_batteries(spec.KORDER)
    assert os.path.isfile(os.path.join(_REPO_ROOT, EP_BATTERY))
    with open(os.path.join(_REPO_ROOT, EP_BATTERY)) as fh:
        body = fh.read()
    assert all(i in body for i in members), "the EP battery must name both members it proves"


def test_the_two_expert_mlp_members_are_the_fused_and_unfused_epilogue():
    """``sm100_tcgen05_unfused_act_v1`` is the raw FC1 projection plus the standalone activation kernel; the battery proves it bitwise equal to the fused GLU epilogue, which is the equivalence ``moe.expert_mlp.segment_tree_glu`` names."""
    ids = {i.impl_id for i in OURS if i.op == "moe.expert_mlp" and i.twin_class == spec.SEGMENT_TREE_GLU}
    assert ids == {"sm100_tcgen05_v1", "sm100_tcgen05_unfused_act_v1"}
    fused, unfused = (next(i for i in OURS if i.impl_id == x) for x in sorted(ids))
    assert fused.inputs == unfused.inputs, "the two members take the same caller inputs"
    with open(os.path.join(_REPO_ROOT, BATTERY)) as fh:
        body = fh.read()
    assert "activation(fc1)" in body or "fc1_glu == activation" in body, (
        "the battery must state the fused == unfused relation these two members claim"
    )


def test_declarations_build_into_a_registry():
    registry = Registry.from_impls(spec.IMPLS)
    assert not registry.lint()


def test_expert_mlp_declares_its_caller_inputs():
    for impl in (i for i in OURS if i.op == "moe.expert_mlp"):
        assert set(impl.inputs) == {"leaves", "leaf_dtype", "act"}, impl.impl_id
        leaves = impl.inputs["leaves"]
        assert isinstance(leaves, OneOf)
        assert leaves.values == (1, 2, 4, 8), impl.impl_id


@pytest.mark.parametrize(
    "cls", [spec.KORDER, spec.SEGMENT_TREE_GLU]
)
def test_every_twin_class_names_this_package_battery(cls):
    assert cls in TWIN_PROOFS, cls
    assert BATTERY in runnable_batteries(cls), cls
    assert os.path.isfile(os.path.join(_REPO_ROOT, BATTERY))


def test_no_twin_class_name_carries_an_arch():
    """The arch is the key a binder resolves on; the class name says what is equal."""
    for cls in (spec.SORTED_TOPK, spec.KORDER, spec.SEGMENT_TREE_GLU):
        assert "sm100" not in cls and "sm90" not in cls, cls


def test_the_one_score_program_difference_is_declared():
    """A reader of the declarations must be able to see that ``moe.route_score`` is NOT sm90's program."""
    assert spec.SM100_SCORE_MATCHES_SM90 is False
    score = [i for i in spec.IMPLS if i.impl_id == "sm100_fp32_chain_v1"]
    assert len(score) == 1 and score[0].impl_id == "sm100_fp32_chain_v1"
    assert not score[0].twin_class, "a different score program joins no equality class"


def test_after_fc2_has_distinct_recipe_and_trainable_router_bindings(monkeypatch):
    from isoexec.ops.moe.resolve import resolve
    from isoexec.ops import _dispatch

    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    records = {impl.impl_id: impl for impl in spec.IMPLS}
    assert not records["sm100_tcgen05_after_fc2_v1"].twin_class
    assert resolve("moe.route_score", site="router", routing="softmax").fn.__name__ == "route_score_fp32_vendored"
    assert resolve("moe.route_mechanics", site="router", routing="softmax").fn.__name__ == "route_mechanics_softmax"
    for op in ("moe.expert_mlp", "moe.combine", "moe.shared_expert"):
        binding = resolve(op, site="moe", side="trainer", weight_after_fc2=True)
        assert binding.impl in records


@pytest.mark.parametrize("side", ["trainer", "engine"])
def test_after_fc2_combine_uses_native_provider_and_preserves_gradients(monkeypatch, side):
    """Exercise the resolved binding without compiling a CUDA extension on CPU."""
    from types import SimpleNamespace

    import torch

    from isoexec.ops import _dispatch
    from isoexec.ops.moe.carriers import ExpertOutputs, RoutingResult
    from isoexec.ops.moe.moe_sm100 import ops
    from isoexec.ops.moe.resolve import resolve

    calls = []

    def native_combine(rows, inverse, output):
        calls.append(True)
        output.copy_(rows[inverse[:, 0].long()] + rows[inverse[:, 1].long()])

    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    monkeypatch.setattr(ops, "extension", lambda: SimpleNamespace(combine=native_combine))
    rows = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], requires_grad=True)
    shared = torch.tensor([[0.5, 0.25], [0.75, 1.]], requires_grad=True)
    routing = RoutingResult(torch.tensor([[1, 0], [0, 1]]), torch.ones(2, 2), None)
    routed = ExpertOutputs(rows, "permuted", routing, extra=torch.tensor([0, 1, 0, 1]),
                           combine_rows=torch.tensor([[0, 2], [1, 3]], dtype=torch.int32))
    binding = resolve("moe.combine", site="moe", side=side, weight_after_fc2=True)
    output = binding.fn(routed, shared)
    expected = rows[[0, 1]] + rows[[2, 3]] + shared
    assert calls == [True]
    assert torch.equal(output, expected)
    gradient = torch.tensor([[0.25, -0.5], [1., 2.]])
    output.backward(gradient)
    assert rows.grad is not None and shared.grad is not None
    assert torch.equal(rows.grad, gradient[[0, 1, 0, 1]])
    assert torch.equal(shared.grad, gradient)


def test_declared_constants_state_the_recipe():
    assert spec.SM100_COMBINE_ORDER == "ascending_expert_then_slot"
    assert spec.SM100_DISPATCH_ORDER == "canonical_token_major"
    assert spec.SM100_SEGMENT_MULTIPLE == 64
    assert spec.FC1_GLU_ACT == "silu"


def test_the_combine_order_agrees_with_the_family_constant():
    from isoexec.ops.moe import spec as family

    assert family.COMBINE_ORDER == "ascending_expert"
    assert spec.SM100_COMBINE_ORDER.startswith(family.COMBINE_ORDER)
    assert family.COMBINE_ACCUM == "fp32"
    assert family.ROUTER_DTYPE == "fp32"


def test_the_expert_mlp_seam_is_wired_to_the_declared_id():
    """``ops/moe/resolve.py`` reads ``IMPL``/``ACT``/``expert_mlp`` off the arch-selected package, so this is the one seam that resolves on sm100 with no edit outside the package."""
    from isoexec.ops.moe.moe_sm100 import integration

    declared = {i.impl_id for i in spec.IMPLS if i.op == "moe.expert_mlp"}
    assert integration.IMPL in declared
    assert integration.ACT.startswith("sm100_")
    assert getattr(integration.expert_mlp, "supports_compact_root") is False
    assert integration.FC1_GLU_ACT == spec.FC1_GLU_ACT == "silu"


def test_importing_the_package_builds_nothing():
    """The extension is built on the first call, never on import. A fresh interpreter: in this one an earlier
    test of the session may have built it."""
    import subprocess
    import sys

    code = ("import isoexec.ops.moe.moe_sm100, isoexec.ops.moe.moe_sm100.ops;"
            "import isoexec.ops.moe.moe_sm100._extension as ext;"
            "assert ext.extension.cache_info().currsize == 0")
    subprocess.run([sys.executable, "-c", code], check=True)


# ------------------------------------------------------------ the schedule is the policy's output

_MOE_FIXTURES = (
    "isoexec/models/qwen3_5_moe/tests/fixtures/qwen3_5_35b_a3b.json",
    "isoexec/models/qwen3_moe/tests/fixtures/qwen3_30b_a3b.json",
    "isoexec/models/qwen2_moe/tests/fixtures/qwen1_5_moe_a2_7b.json",
)


def _deployed_moe_shapes():
    """(hidden, routed intermediate, shared intermediate or 0, top_k) of every MoE model fixture in the tree."""
    import json

    out = []
    for rel in _MOE_FIXTURES:
        with open(os.path.join(_REPO_ROOT, rel)) as fh:
            raw = json.load(fh)
        cfg = raw.get("text_config", raw)
        out.append((int(cfg["hidden_size"]), int(cfg["moe_intermediate_size"]),
                    int(cfg.get("shared_expert_intermediate_size") or 0), int(cfg["num_experts_per_tok"])))
    return out


def test_the_schedule_is_the_schedule_policy_on_the_deployed_shapes():
    """Fails if a segment count, a segment length or a merge of this operator differs from
    ``isoexec/ops/schedule_policy.py`` on the models the tree ships."""
    from isoexec.ops import schedule_policy as P
    from isoexec.ops.moe.moe_sm100 import ops
    from isoexec.runtimes.trainer_installs import PIK_LEAVES

    assert ops.REDUCER_BLOCK == 16 and ops.REDUCER_SHORTEST is None
    # rule 2: the selected-set sum and the combine
    assert ops.SELECTED == P.selected() == P.Segments(1, None, "chain")
    shapes = _deployed_moe_shapes()
    assert shapes
    for hidden, routed, shared, top_k in shapes:
        # rule 3, the hidden axis (route_score, fc1): never sharded, one segment
        fc1 = ops.projection_schedule(hidden)
        assert fc1 == P.reduce(hidden, b=16) == P.Segments(hidden, 1, "tree"), hidden
        # rule 3, the intermediate axis (fc2): d_max shards of one segment each; the tree's declared d_max
        # is the leaf count (ISOEXEC_PIK_LEAVES / Placement.leaves, 8 by default)
        for inter in filter(None, (routed, shared)):
            for d_max in (1, 2, 4, 8):
                if inter % (8 * d_max):
                    continue
                shard = ops.projection_schedule(inter, d_max)
                assert shard == P.reduce(inter // d_max, b=16) and shard.G == 1, (inter, d_max)
                assert ops.fc2_leaves(inter, d_max) == d_max
        assert ops.fc2_leaves(routed, PIK_LEAVES) == PIK_LEAVES == 8
        assert 1 <= top_k <= 32
    import inspect

    for fn in (ops.fc2, ops.experts, ops.moe, ops.routed_forward):
        assert inspect.signature(fn).parameters["leaves"].default == PIK_LEAVES, fn.__name__
    with pytest.raises(ValueError):
        ops.projection_schedule(512, 3)            # a degree that is not a power of two is refused


def test_the_kernel_source_runs_the_rule_2_chain():
    """``csrc/route.cu``: the selected-set sum and the combine are chains in ascending expert id seeded with
    +0, and the half fold of v1 is gone. A source check, because the order is code and not a parameter; the
    bits are gated by ``moe_relations_gpu.py`` (router phase, combine phase)."""
    with open(os.path.join(_REPO_ROOT, "isoexec/ops/moe/moe_sm100/csrc/route.cu")) as fh:
        src = fh.read()
    assert "halffold_sum" not in src and "chain_sum" not in src
    start = src.index("__device__ __forceinline__ float selected_sum(")
    body = src[start:src.index("//: THE SELECTED-SET WEIGHTS", start)]
    assert "lower += pick[l] < pick[j]" in body and "float acc = 0.f;" in body and "__fadd_rn(acc, term)" in body
    assert src.count("selected_sum<K, BK>(sel, pick)") == 2          # softmax_renorm and sigmoid
    kernel = src[src.index("void combine_kernel("):src.index("void combine_kernel(") + 3200]
    assert "acc[i] = v[0][i]" not in kernel and "load(0, acc)" not in kernel   # no first-slot special case
    assert kernel.count("acc[i] = 0.f;") == 2 and "for (int j = 0; j < KT; ++j)" in kernel and "int j = 0;" in kernel
    assert spec.SM100_SELECTED_SUM == "ascending_expert_chain_from_zero"
    assert spec.SM100_COMBINE_ORDER == "ascending_expert_then_slot"
