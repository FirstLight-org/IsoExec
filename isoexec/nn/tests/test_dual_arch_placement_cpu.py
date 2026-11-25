"""Exercise public placement across architecture rows without CUDA execution."""

import pytest

from isoexec import nn
from isoexec.nn.tests import _fixtures as F
from isoexec.ops import _dispatch
from isoexec.ops.table import OpResolutionError, OpTable


@pytest.mark.parametrize("arch,provider", [
    ("sm90", "ops.norms.norms_sm90"),
    ("sm100", "ops.norms.norms_sm100"),
])
def test_fresh_placement_selects_arch_proven_norm(monkeypatch, arch, provider):
    monkeypatch.setattr(_dispatch, "ARCH", arch)
    norm = nn.RMSNorm(hidden=256)
    assert norm.weight.device.type == "meta"
    nn.place(norm, F.placement(), prefix="model.norm")
    bound = nn.resolved_ops(norm)["model.norm"]["norm"]
    assert bound.provider == provider
    assert bound.impl.startswith(arch + "_")


def test_architecture_switch_requires_a_fresh_table_and_component(monkeypatch):
    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    old_table = OpTable()
    old = nn.RMSNorm(hidden=256)
    nn.place(old, F.placement(), prefix="model.old", op_table=old_table)
    assert old.slots.resolved()["norm"].provider == "ops.norms.norms_sm90"

    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    with pytest.raises(OpResolutionError, match="active architecture"):
        old_table.bind("norms.rms.final", site="model.old", facts={"width": 256}, convention="rms")
    fresh = nn.RMSNorm(hidden=256)
    nn.place(fresh, F.placement(), prefix="model.fresh")
    assert fresh.slots.resolved()["norm"].provider == "ops.norms.norms_sm100"
    assert old.slots.resolved()["norm"].provider == "ops.norms.norms_sm90"


def test_live_current_device_drives_selection_without_arch_override(monkeypatch):
    from isoexec.core import arch as core_arch

    monkeypatch.delattr(_dispatch, "ARCH", raising=False)
    monkeypatch.setattr(core_arch, "live_device_arch", lambda: "sm90")
    hopper = OpTable()
    assert hopper.arch == "sm90"
    monkeypatch.setattr(core_arch, "live_device_arch", lambda: "sm100")
    blackwell = OpTable()
    assert blackwell.arch == "sm100"
    with pytest.raises(OpResolutionError, match="active architecture"):
        hopper.bind("norms.rms.final", site="model.norm", facts={"width": 256}, convention="rms")


@pytest.mark.parametrize("op,facts,provider", [
    ("conv.short", {}, "ops.gdn.conv_sm100"),
    ("gdn.core", {}, "ops.gdn.gdn_cpr_sm100"),
    ("logprobs.log_softmax", {}, "ops.logprobs.logprobs_sm100"),
    ("moe.expert_mlp", {}, "ops.moe.moe_sm100"),
    ("norms.rms.final", {"width": 256, "convention": "rms"}, "ops.norms.norms_sm100"),
    ("norms.qk", {"convention": "rms_zero_centered"}, "ops.attention.rope_sm100"),
])
def test_six_family_bindings_name_sm100_provider(monkeypatch, op, facts, provider):
    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    bound = OpTable().bind(op, site="model.op", facts=facts)
    assert bound.provider == provider
    assert bound.version is not None
    assert callable(bound.fn)


def test_cpu_production_refuses_instead_of_choosing_reference(monkeypatch):
    monkeypatch.setattr(_dispatch, "ARCH", "cpu")
    with pytest.raises(OpResolutionError, match="no production implementation"):
        nn.place(nn.RMSNorm(hidden=256), F.placement(), prefix="model.norm")
