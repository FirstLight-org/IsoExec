"""CPU checks of accelerator binding selection, declared identities and unsupported inputs. These
override the reference arch pin without executing GPU kernels.
"""

from __future__ import annotations

import pytest

from isoexec.core.arch import NON_ACCELERATOR_ARCH, is_accelerator_arch
from isoexec.nn import SeamError, Seams, _seams
from isoexec.nn.tests import _fixtures as F
from isoexec.ops import table as _resolve
from isoexec.ops.table import OpResolutionError

#: The arch of the node these proofs run on. Any real accelerator tag would do; this is the one the
#: fleet has, and ``ops/_dispatch.FAMILIES`` declares a package for it on every dispatched family.
ACCELERATOR = "sm90"

# Cover both architecture-dispatched families and process-wide implementations.
DISPATCHED: tuple[tuple[str, dict, str], ...] = (
    ("norms.rms.pre_attn", {"convention": "rms_zero_centered", "facts": {"width": 256}}, "sm90_rms_v1"),
    ("norms.gated_out", {"convention": "gated_rms_zero_centered", "facts": {"width": 128}}, "sm90_gated_v1"),
    ("moe.combine", {}, "pik_leaf_tree"),
    ("conv.short", {}, "sm90_cuda_v1"),
    ("gdn.core", {}, "fla_native_recurrent_v1"),  # sm90 default: ops/gdn/gdn_recurrent_sm90
    ("logprobs.log_softmax", {}, "sm90_online_leaf8_v1"),
    ("moe.expert_mlp", {}, "cuda_wgmma_v1"),
)
UNTABLED: tuple[tuple[str, dict, str], ...] = (
    ("mm", {}, "triton_batch_invariant"),
    ("attention.core.gqa", {}, "sm90_fa3_segsplit_v1_varlen"),
    ("attention.gate", {"facts": {"gate": "sigmoid"}}, "compiled_sigmoid_gate_fp32"),
    ("residual.add.attn", {}, "eager"),
    ("embed.lookup", {}, "eager"),
    ("mlp.act", {}, "sm90_fused_v1"),
    ("logprobs.head", {}, "eager"),
)


#: Ops whose resolved recipe is past its first version (gqa_sm90: 1 was ns1, 2 the fixed segment split).
RECIPE_VERSIONS = {"attention.core.gqa": 2}


@pytest.fixture
def on_accelerator(monkeypatch):
    """Undo ``conftest.py``'s pin for one test: resolve as the 8xH100 node resolves.

    Patching the same one attribute the fixture patched, so this file exercises the seam it is
    asserting rather than a second path to the same answer.
    """
    from isoexec.ops import _dispatch

    monkeypatch.setattr(_dispatch, "ARCH", ACCELERATOR)
    # mm resolves to the provider object actually installed. On a real H100 an earlier test in the same
    # process may have installed the sm90 segment-tree wrapper; resolve against the stock provider here.
    from isoexec.ops.mm import mm_batch_invariant as V

    monkeypatch.setattr(V, "matmul_persistent", getattr(V, "_isoexec_stock_matmul_persistent", V.matmul_persistent))
    return ACCELERATOR


def test_the_folder_pin_is_in_effect_and_has_exactly_one_attribute():
    """Verify the CPU reference arch pin before overriding it for accelerator resolution."""
    from isoexec.ops import _dispatch

    assert _resolve.live_arch() == _dispatch.ARCH == NON_ACCELERATOR_ARCH
    assert not is_accelerator_arch(_resolve.live_arch())
    assert _resolve.accelerator_provider("mm") is None


@pytest.mark.parametrize(("op", "asked", "impl"), DISPATCHED + UNTABLED, ids=[o for o, _, _ in DISPATCHED + UNTABLED])
def test_every_op_a_component_resolves_takes_a_real_kernel_on_an_accelerator(op, asked, impl, on_accelerator):
    """The resolution, on both paths. This is what the folder pin removes."""
    assert _resolve.accelerator_provider(op) is not None
    resolved = _resolve.OpTable().bind(op, site="lm.layers.0.mixer", **asked)
    assert resolved.op == op
    assert resolved.impl == impl
    # Never the plain-torch statement, and never a fabricated version: the id is declared, so the
    # registry has a version for it, and a record with `version=None` could not be a kernel's.
    assert resolved.provider != _resolve.REFERENCE_PROVIDER
    assert resolved.provider.startswith("ops.")
    assert resolved.version == RECIPE_VERSIONS.get(op, 1)
    assert callable(resolved.fn)


def test_the_declared_version_is_read_off_the_registry_and_not_invented(on_accelerator):
    """The resolved implementation and version must match the registry declaration."""
    from isoexec.core.registry_build import build_registry
    from isoexec.ops._resolve import ResolutionError, declared_version

    registry = build_registry()
    for op, asked, impl in DISPATCHED + UNTABLED:
        resolved = _resolve.OpTable().bind(op, site="lm.layers.0.mixer", **asked)
        assert registry.get_op(op).impls[resolved.impl].version == resolved.version
    with pytest.raises(ResolutionError, match="no ops/\\*/spec.py declares"):
        declared_version("norms.rms.pre_attn", "invented_here", site="lm.layers.0.pre_attn_norm")


def test_a_whole_component_resolves_at_apply_placement(on_accelerator):
    """Not per-op luck: placing a component on the accelerator arch fills every slot with a kernel."""
    attn = F.component(F.spec())
    attn.apply_placement(F.placement(), site=F.SITE)
    resolved = attn.slots.resolved()
    assert {slot for slot in resolved} == {
        "mm", "qk_rope", "core", "gate", "row_parallel", "tp_copy", "kv_gather", "cp_gather"
    }
    assert all(r.provider != _resolve.REFERENCE_PROVIDER for r in resolved.values())
    # The fused q/k-norm-plus-rope kernel owns both op ids, so the q/k norm has no slot of its own
    # and the stage says so -- the same shape `rope_sm90/integration.bind` gives a host table.
    stage = resolved["qk_rope"]
    assert stage.op == "norms.qk" and stage.impl == "sm90_qk_norm_rope_v2"
    assert stage.owns == {"norms.qk": "sm90_qk_norm_rope_v2", "rope.apply": "sm90_qk_norm_rope_v2"}
    assert attn.slots.unfilled()["qk_norm"].fill_verb.startswith("nothing does:")


@pytest.mark.parametrize("convention", ("rms_zero_centered", "rms"))
def test_an_rms_norm_takes_the_kernel_on_the_accelerator_and_the_reference_off_it(convention, monkeypatch):
    """Check that architecture selection reaches the component's norm binding."""
    from isoexec.nn import RMSNorm
    from isoexec.ops import _dispatch

    def place_fresh(arch: str):
        monkeypatch.setattr(_dispatch, "ARCH", arch)
        norm = RMSNorm(hidden=256, eps=1e-6, convention=convention, role="pre_attn")
        norm.apply_placement(F.placement(), site="lm.layers.0.pre_attn_norm",
                             op_table=_resolve.OpTable(mode="reference" if arch == NON_ACCELERATOR_ARCH else "production"))
        return norm.slots.resolved()["norm"]

    kernel = place_fresh(ACCELERATOR)
    assert (kernel.impl, kernel.version) == ("sm90_rms_v1", 1)
    assert kernel.provider == "ops.norms.norms_sm90"
    assert dict(kernel.inputs) == {"eps": 1e-6, "convention": convention}

    reference = place_fresh(NON_ACCELERATOR_ARCH)
    expected = {"rms_zero_centered": "ref.rms_zero_centered", "rms": "ref.rms"}[convention]
    assert (reference.impl, reference.version, reference.provider) == (
        expected, None, _resolve.REFERENCE_PROVIDER
    )
    assert dict(reference.inputs) == dict(kernel.inputs), "the record is the same; the kernel is not"
    assert reference.impl != kernel.impl, "the two arms must not be able to compare equal"


def test_a_row_with_no_qk_norm_takes_the_standalone_rotation(on_accelerator):
    """The other branch of the same statement: with nothing to fuse, the rope seam's table recipe
    binds ``rope.apply`` and the q/k norm slot is empty because the row has none."""
    attn = F.component(F.spec(qk_norm=None))
    attn.apply_placement(F.placement(), site=F.SITE)
    stage = attn.slots.resolved()["qk_rope"]
    assert (stage.op, stage.impl) == ("rope.apply", "sm90_table_v1")
    assert stage.owns == {"rope.apply": "sm90_table_v1"}
    assert attn.slots.unfilled()["qk_norm"].fill_verb == "nothing does: this row has no q/k norm"


@pytest.mark.parametrize("side", ["engine", "trainer"])
def test_single_rank_row_parallel_binds_the_tree_on_sm100(monkeypatch, side):
    """TP=1 still runs the fixed leaf tree (no communication), so a one-GPU sm100 host must bind it."""
    from isoexec.ops import _dispatch

    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    placement = F.placement(side=side)
    bound = Seams(placement, F.SITE).row_parallel()
    assert (bound.op, bound.impl, bound.provider, bound.version) == ("collectives.row_parallel", "pik_tree", _seams.PROVIDER, 1)
    assert bound.inputs == {"tp": 1, "leaves": 8, "leaf_dtype": placement.leaf_dtype}


def test_the_row_parallel_seam_chooses_the_kernel_the_moment_an_arch_serves_it(on_accelerator):
    """Resolve the declared accelerator tree when available and the distinct reference tree
    otherwise.
    """
    from isoexec.ops import _dispatch

    placement = F.placement()
    live = Seams(placement, F.SITE).row_parallel()
    assert (live.impl, live.provider) == ("pik_tree", _seams.PROVIDER)
    assert live.version == 1

    monkey = _dispatch.ARCH
    assert monkey == ACCELERATOR
    _dispatch.ARCH = NON_ACCELERATOR_ARCH
    try:
        pinned = Seams(placement, F.SITE, op_table=_resolve.OpTable(mode="reference")).row_parallel()
    finally:
        _dispatch.ARCH = monkey
    assert (pinned.impl, pinned.provider, pinned.version) == ("ref.leaf_tree", _resolve.REFERENCE_PROVIDER, None)
    assert pinned.impl != live.impl, "the two arms must not be able to compare equal"


def test_the_sequence_parallel_form_of_the_tree_is_the_kernel_here_and_a_refusal_there(on_accelerator):
    """SP requires the qualified reduce-scatter tree; no plain-torch reference form is available."""
    from isoexec.ops import _dispatch

    from isoexec.nn.tests import _moe_fixtures as MF

    placement = MF.placement(tp=2, sp=True)  # the SP degree lives on the mixture fixture's placement
    live = Seams(placement, F.SITE).row_parallel()
    assert (live.impl, live.provider, live.version) == ("pik_tree_rs", _seams.PROVIDER, 1)

    monkey = _dispatch.ARCH
    _dispatch.ARCH = NON_ACCELERATOR_ARCH
    try:
        with pytest.raises(SeamError, match="no plain-torch statement"):
            Seams(placement, F.SITE, op_table=_resolve.OpTable(mode="reference")).row_parallel()
    finally:
        _dispatch.ARCH = monkey




@pytest.mark.parametrize("convention,zero_centered", [("rms_zero_centered", True), ("rms", False)])
def test_each_qk_gamma_convention_selects_its_own_kernel_form(on_accelerator, monkeypatch, convention, zero_centered):
    """``rms`` and ``rms_zero_centered`` read the same gamma bytes two ways: the convention is recorded
    on the binding and is what reaches the fused kernel's ``zero_centered`` argument."""
    from types import SimpleNamespace

    import torch

    from isoexec.ops.attention.rope_sm90 import integration as sm90, ops as sm90_ops

    calls = []

    def fake_op(q, k, wq, wk, positions, table, eps, *, zero_centered):
        calls.append(zero_centered)
        return q, k

    monkeypatch.setattr(sm90_ops, "qk_norm_rope", fake_op)
    monkeypatch.setattr(sm90._TableSet, "get", lambda self, *args, **kwargs: None)
    bound = _resolve.OpTable().bind("norms.qk", site=F.SITE, inputs={"eps": 1e-6}, convention=convention)
    assert bound.impl == "sm90_qk_norm_rope_v2" and bound.inputs["convention"] == convention
    q = torch.zeros(2, 1, 64)
    bound.fn(q, q, q[0, 0], q[0, 0], 1e-6, seq=SimpleNamespace(positions=torch.zeros(2, dtype=torch.long)),
             rotary=SimpleNamespace(style="neox", theta=1e6, rotary_dim=64),
             spec=SimpleNamespace(mrope_section=None, mrope_interleaved=False))
    assert calls == [zero_centered]


def test_a_gamma_convention_nothing_on_this_arch_states_refuses_by_name(on_accelerator):
    """The +-1 checkpoint-mapping bug class. A convention no package on this arch states for the q/k
    norm refuses rather than running a kernel over gamma it was not declared for."""
    with pytest.raises(OpResolutionError) as exc:
        _resolve.OpTable().bind("norms.qk", site=F.SITE, inputs={"eps": 1e-6}, convention="gated_rms")
    message = str(exc.value)
    assert "norms.qk" in message and "'gated_rms'" in message and "rms_zero_centered" in message
    assert F.SITE in message, "the refusal names the site that asked"


def test_plain_gamma_qk_norm_refuses_on_sm100_until_verified_there(monkeypatch):
    """rope_sm100 declares the zero-centred q/k norm only; a Qwen3 (plain-gamma) row refuses by name."""
    from isoexec.ops import _dispatch

    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    with pytest.raises(OpResolutionError) as exc:
        _resolve.OpTable().bind("norms.qk", site=F.SITE, inputs={"eps": 1e-6}, convention="rms")
    message = str(exc.value)
    assert "norms.qk" in message and "'rms'" in message and "rope_sm100" in message and F.SITE in message


def test_a_width_the_recipe_has_no_forward_for_refuses_by_name(on_accelerator):
    """The same refusal ``norms_sm90/integration.bind`` raises when it checks a model's widths:
    a 32-wide GDN out-norm has no SM90 norm, and the eager torch statement is a different rounding
    schedule rather than a fallback."""
    with pytest.raises(OpResolutionError) as exc:
        _resolve.OpTable().bind(
            "norms.gated_out", site=F.SITE, convention="gated_rms_zero_centered", facts={"width": 32}
        )
    message = str(exc.value)
    assert "norms.gated_out" in message and "32" in message and "no forward for width" in message


def test_an_op_id_no_family_serves_refuses_and_names_the_families_that_do(on_accelerator):
    with pytest.raises(OpResolutionError, match="no ops family resolves op id"):
        _resolve.OpTable().bind("quant.block_fp8", site=F.SITE)


def test_a_fact_a_family_does_not_take_refuses_rather_than_being_ignored(on_accelerator):
    """A fact nobody reads is a decision nobody made. Silently dropping it is how a caller believes
    it selected something it did not."""
    with pytest.raises(OpResolutionError) as exc:
        _resolve.OpTable().bind("residual.add.attn", site=F.SITE, facts={"rounding": "fp32"})
    assert "rounding" in str(exc.value)


def test_the_reference_arm_is_untouched_by_any_of_this():
    """The pin is still in effect for this test, and the reference composition still resolves --
    the accelerator arm is an addition, not a replacement."""
    resolved = _resolve.OpTable(mode="reference").bind(
        "norms.rms.pre_attn", site=F.SITE, inputs={"eps": 1e-6}, convention="rms_zero_centered",
        facts={"width": 256},
    )
    assert (resolved.impl, resolved.provider, resolved.version) == (
        "ref.rms_zero_centered", _resolve.REFERENCE_PROVIDER, None
    )
    assert dict(resolved.inputs) == {"eps": 1e-6, "convention": "rms_zero_centered"}




def test_the_engine_mixture_owns_its_fc2_trees_at_the_combine(on_accelerator):
    """Deferred owner reductions keep the trainer's combine and shared-expert kernels."""
    from isoexec.nn.tests import _moe_fixtures as MF

    def placed(side):
        moe = MF.component(MF.spec())
        moe.apply_placement(MF.placement(tp=4, side=side), site=MF.SITE)
        return moe

    trainer, engine = placed("trainer"), placed("engine")
    t_slots, e_slots = trainer.slots.resolved(), engine.slots.resolved()

    assert t_slots["route_select"].impl == "deterministic_sorted_topk"
    assert e_slots["route_select"].impl == "fused_o2"
    for slot in ("combine", "shared_expert"):
        assert e_slots[slot].impl == t_slots[slot].impl, slot

    # The trainer keeps the in-place trees; the engine hands both to the combine.
    assert not getattr(t_slots["shared_row_parallel"].fn, "partial_only", False)
    assert getattr(e_slots["shared_row_parallel"].fn, "partial_only", False)
    assert "owner_group" not in (t_slots["combine"].inputs.get("seams") or ())
    assert e_slots["combine"].inputs["seams"] == ("owner_group",)
    assert not t_slots["expert_mlp"].fn.__name__.endswith("|deferred")
    assert e_slots["expert_mlp"].fn.__name__.endswith("|deferred")



if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_engine_etp_refuses_a_combine_without_its_owner_seam(on_accelerator, monkeypatch):
    """A malformed production binding cannot silently select trainer reductions."""
    from dataclasses import replace

    from isoexec.nn import PlacementError
    from isoexec.nn.tests import _moe_fixtures as MF

    real = _resolve.OpTable.bind

    def bind(self, op, **kw):
        got = real(self, op, **kw)
        if op == "moe.combine":
            def missing_owner(experts, shared):
                raise AssertionError("placement must refuse before execution")
            return replace(got, fn=missing_owner)
        return got

    monkeypatch.setattr(_resolve.OpTable, "bind", bind)
    moe = MF.component(MF.spec())
    with pytest.raises(PlacementError, match="production engine ETP requires.*owner_group"):
        moe.apply_placement(MF.placement(tp=4, side="engine"), site=MF.SITE)
