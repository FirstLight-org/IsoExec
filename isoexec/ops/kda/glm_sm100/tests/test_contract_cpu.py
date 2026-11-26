"""Import-light contract of the package (no GPU, no build)."""

from __future__ import annotations

import importlib
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def test_spec_declares_one_sm100_impl_cpu():
    spec = importlib.import_module("isoexec.ops.kda.glm_sm100.spec")
    (impl,) = spec.IMPLS
    assert impl.op == "kda.glm" and impl.impl_id == "sm100_kda_v1" and impl.version == spec.RECIPE_VERSION
    assert impl.supported_archs == frozenset({"sm100"})
    assert spec.CHUNK == 64 and spec.HEAD_DIM == 128


def test_importing_ops_and_integration_does_not_build_cpu():
    import sys

    importlib.import_module("isoexec.ops.kda.glm_sm100.integration")
    assert "isoexec_kda_glm_sm100_cuda" not in sys.modules


def test_recipe_header_is_the_numerical_contract_cpu():
    text = (PKG / "csrc" / "recipe.cuh").read_text()
    for needle in ("sm100_kda_v1", "CHAIN64_i^b(j)", "DIAG(a_j, k_i)", "X = (I + L)^-1", "E(x) = ex2.approx.f32"):
        assert needle in text, needle


def test_build_uses_the_shared_numerics_profile_cpu():
    from isoexec.ops.build_cache import NUMERICS_FLAGS, sm100_cuda_flags

    flags = sm100_cuda_flags()
    assert all(f in flags for f in NUMERICS_FLAGS)
    src = (PKG / "_extension.py").read_text()
    assert "build_sm100_extension" in src
