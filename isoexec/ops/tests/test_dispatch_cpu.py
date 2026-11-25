"""Check architecture selection, declaration proof coverage and refusal of unsupported accelerators."""

from __future__ import annotations

import pytest

from isoexec.core.arch import NON_ACCELERATOR_ARCH
from isoexec.ops import _dispatch

SM90_PACKAGES = {
    "conv": "gdn.conv_sm90",
    "gdn": "gdn.gdn_recurrent_sm90",
    "gqa": "attention.gqa_sm90",
    "logprobs": "logprobs.logprobs_sm90",
    "moe": "moe.moe_sm90",
    "norms": "norms.norms_sm90",
    "rope": "attention.rope_sm90",
}
SM100_PACKAGES = {name: package.replace("sm90", "sm100") for name, package in SM90_PACKAGES.items()}
# SM90 GDN rides the native recurrent core (gdn_recurrent_sm90); SM100 and the CPU reference keep the CPR core.
SM100_PACKAGES["gdn"] = "gdn.gdn_cpr_sm100"
REFERENCE_PACKAGES = {**SM90_PACKAGES, "gdn": "gdn.gdn_cpr_sm90"}


def test_every_seam_has_a_cpu_row_and_both_proven_packages():
    assert set(_dispatch.FAMILIES) == set(SM90_PACKAGES)
    for name, fam in _dispatch.FAMILIES.items():
        assert fam.archs[NON_ACCELERATOR_ARCH] is None, name
        assert fam.archs["sm90"] == SM90_PACKAGES[name], name
        assert fam.archs["sm100"] == SM100_PACKAGES[name], name
        reference = REFERENCE_PACKAGES[name]
        assert fam.reference_package == reference, name
        assert set(fam.verbs) <= {"bind", "install"}, name


def test_a_seam_row_is_not_a_registry_family():
    """The two sets used to coincide and no longer do."""
    from isoexec.core.registry_build import _FAMILIES

    assert {"rope", "conv"} & set(_FAMILIES) == set()
    assert {"gdn", "norms", "logprobs", "moe"} <= set(_FAMILIES)


def test_sm90_selects_every_package(monkeypatch):
    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    monkeypatch.delenv("ISOEXEC_GDN_KERNEL", raising=False)
    assert _dispatch.resolved() == SM90_PACKAGES


def test_sm100_selects_every_package(monkeypatch):
    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    monkeypatch.delenv("ISOEXEC_GDN_KERNEL", raising=False)
    assert _dispatch.resolved() == {name: package or "legacy" for name, package in SM100_PACKAGES.items()}


def test_no_accelerator_selects_the_reference_impls(monkeypatch):
    monkeypatch.setattr(_dispatch, "ARCH", NON_ACCELERATOR_ARCH)
    assert _dispatch.resolved() == {name: "legacy" for name in _dispatch.FAMILIES}


@pytest.mark.parametrize("arch", ["sm80", "sm120"])
def test_an_arch_with_no_row_refuses_and_names_itself(monkeypatch, arch):
    monkeypatch.setattr(_dispatch, "ARCH", arch)
    for name in _dispatch.FAMILIES:
        with pytest.raises(_dispatch.ArchSelectionError) as e:
            _dispatch.selected(name)
        assert arch in str(e.value) and "refuses" in str(e.value)


def test_a_package_that_declares_nothing_cannot_pass_the_arch_proof(monkeypatch):
    import types

    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    monkeypatch.setitem(_dispatch.FAMILIES["norms"].archs, "sm90", "norms.empty_test")
    original = _dispatch.importlib.import_module

    def load(name, package=None):
        if name == "isoexec.ops.norms.empty_test.spec":
            return types.SimpleNamespace(IMPLS=())
        return original(name, package)

    monkeypatch.setattr(_dispatch.importlib, "import_module", load)
    _dispatch._assert_proven.cache_clear()
    try:
        with pytest.raises(_dispatch.ArchSelectionError, match="declares no impl"):
            _dispatch.selected("norms")
    finally:
        _dispatch._assert_proven.cache_clear()


def test_a_row_without_the_declarations_to_back_it_still_refuses(monkeypatch):
    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    monkeypatch.setitem(_dispatch.FAMILIES["norms"].archs, "sm100", "norms.norms_sm90")
    with pytest.raises(_dispatch.ArchSelectionError) as e:
        _dispatch.selected("norms")
    assert "supported_archs" in str(e.value) and "sm90_rms_v1" in str(e.value)


def test_the_row_is_the_only_switch(monkeypatch):
    """A bisect back to legacy on sm90 edits a row; no environment key does it any more."""
    import pathlib

    src = pathlib.Path(_dispatch.__file__).read_text()
    assert "environ" not in src and "getenv" not in src
    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    monkeypatch.setitem(_dispatch.FAMILIES["norms"].archs, "sm90", None)
    assert _dispatch.resolved()["norms"] == "legacy"


@pytest.mark.parametrize(("kernel", "package"), [
    (None, "gdn.gdn_recurrent_sm90"),
    ("recurrent", "gdn.gdn_recurrent_sm90"),
    ("cpr", "gdn.gdn_cpr_sm90"),
])
def test_sm90_gdn_kernel_flag_selects_the_cpr_alternative(monkeypatch, kernel, package):
    """The native recurrent core stays the sm90 default; ISOEXEC_GDN_KERNEL=cpr selects the CPR core, and
    the registry answers for the package selected now, not the one cached first."""
    from isoexec.ops import _resolve

    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    if kernel is None:
        monkeypatch.delenv("ISOEXEC_GDN_KERNEL", raising=False)
    else:
        monkeypatch.setenv("ISOEXEC_GDN_KERNEL", kernel)
    assert _dispatch.selected("gdn") == package
    impl = "cuda_cpr_v1" if package.endswith("cpr_sm90") else "fla_native_recurrent_v1"
    assert set(_resolve.registry().get_op("gdn.core").impls) == {impl}


@pytest.mark.parametrize(("kernel", "package"), [
    (None, "gdn.gdn_cpr_sm100"),
    ("cpr", "gdn.gdn_cpr_sm100"),
    ("recurrent", "gdn.gdn_recurrent_sm100"),
])
def test_sm100_gdn_kernel_flag_selects_the_recurrent_alternative(monkeypatch, kernel, package):
    """CPR stays the sm100 default; ISOEXEC_GDN_KERNEL=recurrent selects the native recurrent core, and
    the registry answers for the package selected now, not the one cached first."""
    from isoexec.ops import _resolve

    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    if kernel is None:
        monkeypatch.delenv("ISOEXEC_GDN_KERNEL", raising=False)
    else:
        monkeypatch.setenv("ISOEXEC_GDN_KERNEL", kernel)
    assert _dispatch.selected("gdn") == package
    impl = "fla_native_recurrent_v1" if package.endswith("recurrent_sm100") else "sm100_cpr_v1"
    assert set(_resolve.registry().get_op("gdn.core").impls) == {impl}
