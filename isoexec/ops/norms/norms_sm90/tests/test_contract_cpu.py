"""Declarations, the arch seam and the table bindings -- all CPU, no extension build."""

from __future__ import annotations

import pytest
import torch

from isoexec.core.registry import Registry
from isoexec.ops.norms.norms_sm90 import ops as O
from isoexec.ops.norms.norms_sm90.spec import ADD_RMS_IMPL, GATED_IMPL, IMPLS, RECIPE_VERSION, RMS_IMPL


def test_declarations():
    reg = Registry.from_impls(IMPLS)
    assert set(reg.ops) == {
        "norms.rms.pre_attn",
        "norms.rms.pre_mlp",
        "norms.rms.final",
        "norms.mla",
        "norms.gated_out",
    }
    SM90 = {RMS_IMPL, GATED_IMPL, ADD_RMS_IMPL}
    for op in reg.ops.values():
        for impl in op.impls.values():
            assert impl.supported_archs == frozenset({"sm90"})
            assert impl.impl_id in SM90
            assert impl.version == RECIPE_VERSION
            # Recipe v1 reproduces no other implementation's bits.
            assert not impl.twin_class


def test_the_recipe_version_has_one_definition():
    """``spec.py`` declares it and ``ops.py`` imports it. Two copies of the version is the one
    drift a reader cannot see: the number is the only declared signal that the rounding schedule
    changed, so a stale second copy reads as agreement."""
    assert O.RECIPE_VERSION is RECIPE_VERSION


def test_the_mla_norm_is_declared_here():
    """``norms.mla`` is ``norms.rms`` at latent width -- eps and gamma convention, a norm's facts --
    so its implementation is declared with the norms."""
    reg = Registry.from_impls(IMPLS)
    assert list(reg.ops["norms.mla"].impls) == [RMS_IMPL]


def test_recipe_widths():
    assert [O.supported_width(n) for n in (2048, 4096, 5120, 768, 512, 128, 384)] == [True] * 7
    assert not any(O.supported_width(n) for n in (0, 100, 130, 1000, 4095))


def test_the_arch_selects_the_family(monkeypatch):
    from isoexec.ops import _dispatch

    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    assert _dispatch.selected("norms") == "norms.norms_sm90"
    monkeypatch.setattr(_dispatch, "ARCH", "cpu")
    assert _dispatch.selected("norms") is None  # no accelerator: the reference impls


def test_cpu_tensors_are_rejected_before_any_kernel_runs():
    """A bare ``raises(Exception)`` here is satisfied by the extension failing to import, which is
    the opposite of the claim: it would report "rejected" for a call that never reached the entry
    point. The rejection this pins is the device guard's, so it is a ``RuntimeError`` that names
    the device -- an ImportError or a missing attribute no longer passes for it."""
    x = torch.zeros(4, 2048)
    with pytest.raises(RuntimeError) as caught:
        O.rms_norm(x, torch.zeros(2048), 1e-6, True)
    assert "cpu" in str(caught.value).lower(), caught.value
