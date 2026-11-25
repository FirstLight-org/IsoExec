"""CPU gates: registry declarations, flag classification, table construction on the arch that selects it."""

from __future__ import annotations

import pytest

from isoexec.core.registry import OneOf, Registry
from isoexec.ops.attention.rope_sm90 import reference as R
from isoexec.ops.attention.rope_sm90 import spec


def test_registers_impls_on_existing_ops():
    """Fused q/k norm/RoPE extends both op declarations without replacing existing norm implementations."""
    from isoexec.ops.norms.norms_sm90.spec import IMPLS as NORMS_IMPLS

    for first, second in ((NORMS_IMPLS, spec.IMPLS), (spec.IMPLS, NORMS_IMPLS)):
        reg = Registry.from_impls((*first, *second))
        rope = reg.get_op("rope.apply").impls[spec.ROPE_IMPL]
        qk = reg.get_op("norms.qk").impls[spec.QK_IMPL]
        assert rope.version == 1 and qk.version == 2 and rope.supported_archs == frozenset({"sm90"})
        assert not rope.twin_class and not qk.twin_class, "a new recipe: no twin claim against eager/fused"
        # The fused kernel is declared on BOTH ops it computes, under its own impl id.
        fused_rope = reg.get_op("rope.apply").impls[spec.QK_IMPL]
        assert fused_rope.version == 2 and not fused_rope.twin_class and not fused_rope.inputs
        domain = qk.inputs["convention"]
        assert set(qk.inputs) == {"convention"} and isinstance(domain, OneOf) and "rms_zero_centered" in domain
        assert not rope.inputs
        # Fused q/k normalization is declared by the rope package alone.
        assert set(reg.get_op("norms.qk").impls) == {spec.QK_IMPL}
        assert set(reg.get_op("rope.apply").impls) == {spec.ROPE_IMPL, spec.QK_IMPL}
        assert not reg.lint()


def test_registry_build_includes_the_family():
    from isoexec.core.registry_build import build_registry

    reg = build_registry()
    assert spec.ROPE_IMPL in reg.get_op("rope.apply").impls
    assert spec.QK_IMPL in reg.get_op("norms.qk").impls


def test_flags_are_classified():
    from isoexec.core import flags

    for name in ("ISOEXEC_ROPE_SM90_MAX_POSITIONS", "ISOEXEC_ROPE_SM90_BUILD_DIR", "ISOEXEC_ROPE_SM90_BUILD_VERBOSE"):
        assert name in flags.SIDE_LOCAL_SWITCHES


def test_mrope_masks_follow_the_qwen35_column_layout():
    m1, m2 = R.mrope_masks(64, (11, 11, 10), True)
    axis = [((m1 >> j) & 1) + 2 * ((m2 >> j) & 1) for j in range(32)]
    assert axis[:7] == [0, 1, 2, 0, 1, 2, 0] and axis.count(1) == 11 and axis.count(2) == 10
    m1, m2 = R.mrope_masks(64, (16, 8, 8), False)
    axis = [((m1 >> j) & 1) + 2 * ((m2 >> j) & 1) for j in range(32)]
    assert axis == [0] * 16 + [1] * 8 + [2] * 8
    assert R.mrope_masks(64, None, False) == (0, 0)


def test_inv_freq_is_host_double():
    inv = R.inv_freq_double(1e7, 64)
    assert inv[0] == 1.0 and inv[1] == 1.0 / (1e7 ** (2 / 64)) and len(inv) == 32


def test_mrope_refuses_columns_beyond_native_mask_width_before_cuda(monkeypatch):
    from isoexec.ops.attention.rope_sm90 import ops

    def unexpected_extension():
        pytest.fail("Invalid MRoPE must refuse before compiling or allocating CUDA")

    monkeypatch.setattr(ops, "extension", unexpected_extension)
    # These small masks fit uint64; the later columns would still shift by >=64.
    with pytest.raises(ValueError, match="64-bit column masks"):
        ops.build_table(1e7, 256, 4, "cpu", mrope_section=(100, 14, 14), mrope_interleaved=True)
    assert R.mrope_masks(256, None, False) == (0, 0)
    assert R.mrope_masks(256, (128, 0, 0), False) == (0, 0)
