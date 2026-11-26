"""Declarations, the arch seam and the table bindings -- all CPU, no extension build."""

from __future__ import annotations

import pytest
import torch

from isoexec.core.registry import Registry
from isoexec.ops.norms.norms_sm100 import integration as I
from isoexec.ops.norms.norms_sm100 import ops as O
from isoexec.ops.norms.norms_sm100.spec import (
    ADD_RMS_IMPL,
    ADD_RMS_IMPL_V2,
    GATED_IMPL,
    IMPLS,
    RECIPE_VERSION,
    RECIPE_VERSION_V2,
    RMS_IMPL,
    RMS_IMPL_V2,
)


def test_declarations():
    reg = Registry.from_impls(IMPLS)
    assert set(reg.ops) == {
        "norms.rms.pre_attn",
        "norms.rms.pre_mlp",
        "norms.rms.final",
        "norms.mla",
        "norms.gated_out",
    }
    SM100 = {RMS_IMPL: RECIPE_VERSION, GATED_IMPL: RECIPE_VERSION, RMS_IMPL_V2: RECIPE_VERSION_V2,
             ADD_RMS_IMPL: RECIPE_VERSION, ADD_RMS_IMPL_V2: RECIPE_VERSION_V2}
    for op in reg.ops.values():
        for impl in op.impls.values():
            assert impl.supported_archs == frozenset({"sm100"})
            assert impl.impl_id in SM100
            assert impl.version == SM100[impl.impl_id]
            assert not impl.twin_class


def test_both_forward_recipes_are_declared_at_every_rms_op():
    reg = Registry.from_impls(IMPLS)
    for op in ("norms.rms.pre_attn", "norms.rms.pre_mlp", "norms.rms.final"):
        # the fused residual join + RMSNorm is declared once per recipe, at the residual stages only
        assert set(reg.ops[op].impls) == {RMS_IMPL, RMS_IMPL_V2, ADD_RMS_IMPL, ADD_RMS_IMPL_V2}, op
    assert set(reg.ops["norms.mla"].impls) == {RMS_IMPL, RMS_IMPL_V2}
    assert set(reg.ops["norms.gated_out"].impls) == {GATED_IMPL}


def test_the_v2_impl_id_is_its_own_name():
    assert RMS_IMPL_V2 == "sm100_rms_v2" and RMS_IMPL_V2 != RMS_IMPL
    assert RECIPE_VERSION_V2 == 2 and RECIPE_VERSION_V2 != RECIPE_VERSION


def test_impl_ids_are_never_the_sm90_spelling():
    """Impl ids follow the sm90 spelling with the new arch; the sm90 id itself is never reused."""
    assert RMS_IMPL == "sm100_rms_v1" and GATED_IMPL == "sm100_gated_v1"
    from isoexec.ops.norms.norms_sm90.spec import ADD_RMS_IMPL as SM90_ADD
    from isoexec.ops.norms.norms_sm90.spec import GATED_IMPL as SM90_GATED
    from isoexec.ops.norms.norms_sm90.spec import RMS_IMPL as SM90_RMS

    assert {RMS_IMPL, GATED_IMPL, ADD_RMS_IMPL, ADD_RMS_IMPL_V2}.isdisjoint({SM90_RMS, SM90_GATED, SM90_ADD})
    assert I.add_rms_impl(4096) == ADD_RMS_IMPL_V2 and I.add_rms_impl(2048) == ADD_RMS_IMPL


def test_the_recipe_version_has_one_definition():
    """``spec.py`` declares it and ``ops.py`` imports it -- see norms_sm90's identical test."""
    assert O.RECIPE_VERSION is RECIPE_VERSION


def test_the_mla_norm_is_declared_here():
    reg = Registry.from_impls(IMPLS)
    assert set(reg.ops["norms.mla"].impls) == {RMS_IMPL, RMS_IMPL_V2}


def test_recipe_widths():
    assert [O.supported_width(n) for n in (2048, 4096, 5120, 768, 512, 128, 384)] == [True] * 7
    assert not any(O.supported_width(n) for n in (0, 100, 130, 1000, 4095))


def test_the_arch_selects_the_family(monkeypatch):
    from isoexec.ops import _dispatch

    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    assert _dispatch.selected("norms") == "norms.norms_sm100"
    monkeypatch.setattr(_dispatch, "ARCH", "cpu")
    assert _dispatch.selected("norms") is None  # no accelerator: the reference impls


def test_cpu_tensors_are_rejected_before_any_kernel_runs():
    x = torch.zeros(4, 2048)
    with pytest.raises(RuntimeError) as caught:
        O.rms_norm(x, torch.zeros(2048), 1e-6, True)
    assert "cpu" in str(caught.value).lower(), caught.value


def test_integration_exports_the_same_impl_ids():
    assert I.RMS_IMPL == RMS_IMPL and I.GATED_IMPL == GATED_IMPL
