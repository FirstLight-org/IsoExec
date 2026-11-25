"""CPU gates: declarations only, no CUDA."""

from __future__ import annotations

from pathlib import Path

from isoexec.ops.attention.rope_sm100 import spec


def test_rotary_and_fused_norm_recipes_are_declared():
    """Both fused ownership records and the standalone rotation admit only SM100."""
    assert {(i.op, i.impl_id) for i in spec.IMPLS} == {
        ("rope.apply", spec.ROPE_IMPL), ("norms.qk", spec.QK_IMPL), ("rope.apply", spec.QK_IMPL)
    }
    assert all(i.supported_archs == frozenset({"sm100"}) for i in spec.IMPLS)
    assert spec.ARCH == "sm100"


def test_impl_ids_follow_the_sm90_spelling_under_the_new_arch():
    from isoexec.ops.attention.rope_sm90 import spec as sm90_spec

    assert spec.RECIPE_VERSION == sm90_spec.RECIPE_VERSION == 1
    assert spec.ROPE_IMPL == sm90_spec.ROPE_IMPL.replace("sm90", "sm100")
    # The SM100 forward is SM90's v1 forward recipe; its backward is SM100's own fixed-order
    # recipe (no float atomics), so the id stays separately named rather than aliasing SM90 v2.
    assert spec.QK_IMPL == "sm100_qk_norm_rope_v1"
    assert sm90_spec.QK_IMPL == "sm90_qk_norm_rope_v2"
    # Never rename an sm90 impl id: the two are different recipes-under-arch, not aliases.
    assert spec.ROPE_IMPL != sm90_spec.ROPE_IMPL and spec.QK_IMPL != sm90_spec.QK_IMPL


def test_hazards_match_sm90():
    from isoexec.ops.attention.rope_sm90 import spec as sm90_spec

    assert set(spec.HAZARDS) == set(sm90_spec.HAZARDS)


def test_reference_recipe_is_sm90s_recipe_verbatim():
    """The scalar graph is architecture-neutral (docs/sm100-bringup.md section 4)."""
    import inspect

    from isoexec.ops.attention.rope_sm100 import reference as sm100_ref
    from isoexec.ops.attention.rope_sm90 import reference as sm90_ref

    for name in sm100_ref.__all__:
        assert getattr(sm100_ref, name) is getattr(sm90_ref, name), name
        assert inspect.getsource(getattr(sm90_ref, name))  # importable and has a body


def test_csrc_scalar_numerics_are_byte_identical_to_sm90():
    """Keep rounding-sensitive SM100 CUDA source aligned with SM90."""
    sm100_dir = Path(__file__).parent.parent / "csrc"
    sm90_dir = sm100_dir.parent.parent / "rope_sm90" / "csrc"

    def normalize(text: str) -> str:
        return text.replace("rope_sm100", "rope_sm90").replace("SM100", "SM90").replace("sm100", "sm90")

    # common.cuh must be identical modulo the namespace/name substitution alone.
    a = normalize((sm100_dir / "common.cuh").read_text())
    b = (sm90_dir / "common.cuh").read_text()
    # The sm100 file carries a longer header comment explaining the shared identity; compare bodies.
    a_body = a.split("namespace rope_sm90 {", 1)[1]
    b_body = b.split("namespace rope_sm90 {", 1)[1]
    assert a_body == b_body


def test_gpu_battery_files_exist():
    tests_dir = Path(__file__).parent
    for name in ("test_forward_gpu.py", "test_backward_gpu.py"):
        assert (tests_dir / name).is_file(), name
