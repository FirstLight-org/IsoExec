"""CPU gates: declarations, dispatch wiring, the build profile, head-dim kernel selection, lazy loading."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from isoexec.ops.attention.gqa_sm90 import spec
from isoexec.ops.attention.gqa_sm90._extension import FA3_DIR, SOURCES


def test_the_package_declares_both_cores_in_one_twin_class():
    from isoexec.core.registry import TWIN_PROOFS, Registry, runnable_batteries

    reg = Registry.from_impls(spec.IMPLS)
    assert set(reg.ops) == {"attention.core.gqa"}
    impls = reg.ops["attention.core.gqa"].impls
    assert set(impls) == {spec.TRAINER_IMPL, spec.PAGED_IMPL}
    for impl in impls.values():
        assert impl.version == spec.RECIPE_VERSION == 2
        assert impl.supported_archs == frozenset({"sm90"})
        assert impl.twin_class == spec.TWIN_CLASS
        assert not impl.inputs  # num_splits, window and dtype are the kernel's, not a caller's
    root = Path(__file__).resolve().parents[5]
    assert all((root / p).is_file() for p in runnable_batteries(spec.TWIN_CLASS)), TWIN_PROOFS[spec.TWIN_CLASS]


def test_the_family_registry_takes_the_gqa_core_from_this_package():
    from isoexec.core.registry_build import build_registry

    impls = build_registry().get_op("attention.core.gqa").impls
    assert set(impls) == {spec.TRAINER_IMPL, spec.PAGED_IMPL}


def test_the_dispatch_seam_selects_this_package_on_sm90_only():
    from isoexec.ops import _dispatch

    fam = _dispatch.FAMILIES["gqa"]
    assert fam.archs == {"sm90": "attention.gqa_sm90", "sm100": "attention.gqa_sm100", "cpu": None}
    assert fam.reference_package == "attention.gqa_sm90" and fam.verbs == ()


def test_resolution_binds_the_trainer_core_on_sm90_and_refuses_elsewhere(monkeypatch):
    from isoexec.ops import _dispatch
    from isoexec.ops._resolve import ResolutionError, resolve
    from isoexec.ops.attention.gqa_sm90 import integration

    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    bound = resolve("attention.core.gqa", site="t")
    assert (bound.impl, bound.version, bound.provider) == (spec.TRAINER_IMPL, 2, "ops.attention.gqa_sm90")
    assert bound.fn is integration.trainer_core
    monkeypatch.setattr(_dispatch, "ARCH", "cpu")
    with pytest.raises(ResolutionError, match="selects no kernel package"):
        resolve("attention.core.gqa", site="t")


def test_the_build_keeps_split_kv_for_the_segment_split_and_targets_sm90a():
    assert "FLASHATTENTION_DISABLE_SPLIT" not in spec.BUILD_DEFINES
    assert "--use_fast_math" in spec.NVCC_FLAGS and "-gencode=arch=compute_90a,code=sm_90a" in spec.NVCC_FLAGS
    # FA3's proportional split boundary is gone; the fixed segment replaces it (block.h).
    block = (FA3_DIR / "block.h").read_text()
    assert "num_n_blocks_per_split" not in block and "split_idx_actual * seg_n_blocks" in block
    api = (FA3_DIR / "flash_api_stable.cpp").read_text()
    assert "proportional split-KV is not built" in api and "run_isoexec_seg_combine(params, stream);" in api


def test_seg_is_a_constant_every_kv_tile_divides():
    """SEG must be a multiple of the KV tile of every kernel it serves: 80 (head dim 256), 128 (64, 128)."""
    assert spec.SEG_TOKENS % 640 == 0 and spec.SEG_TOKENS > 0
    assert spec.SEG_RECIPE == "sm90_fa3_segsplit_v1"
    tiles = (FA3_DIR / "tile_size.h").read_text()
    assert "return {128, is_local ? 64 : 80, true, true};" in tiles  # bf16 head dim 256
    assert "return {128, use_blockN_128 ? 128 : 176, true, true};" in tiles  # bf16 head dim 128 (causal: 128)


def test_num_splits_argument():
    from isoexec.ops.attention.gqa_sm90.ops import num_splits_arg

    assert num_splits_arg(None) == 1 and num_splits_arg(2560) == -2560 and num_splits_arg(640) == -640
    assert num_splits_arg(2560, "split") == -2561 and num_splits_arg(2560, "fold") == -2562
    assert num_splits_arg(2560, "hybrid") == -2563
    for bad in (0, -640, 1000, 2560.0, True):
        with pytest.raises(ValueError):
            num_splits_arg(bad)
    with pytest.raises(ValueError):
        num_splits_arg(640, "flat")


def test_sources_are_the_setup_py_instantiations_of_the_profile():
    fwd = {f"instantiations/flash_fwd_hdim{d}_bf16{v}_sm90.cu"
           for d in ("64", "128", "256", "64_256", "64_512") for v in ("", "_packgqa", "_paged")}
    split = {f"instantiations/flash_fwd_hdim{d}_bf16{v}_sm90.cu" for d in (64, 128, 256) for v in ("_split", "_paged_split")}
    bwd = {f"instantiations/flash_bwd_hdim{d}_bf16_sm90.cu" for d in (64, 128, 256)}
    assert set(SOURCES) == fwd | split | bwd | {"flash_api_stable.cpp", "flash_prepare_scheduler.cu", "seg_combine.cu"}
    for src in SOURCES:
        text = (FA3_DIR / src).read_text()
        for guard in re.findall(r"#ifndef (FLASHATTENTION_DISABLE_\w+)", text):
            if src.startswith("instantiations/"):
                assert guard not in spec.BUILD_DEFINES, f"{src} is compiled out by {guard}"


@pytest.mark.parametrize(("d", "dv", "kernel"), [
    (64, 64, (64, 64)), (32, 32, (64, 64)), (64, 128, (64, 256)), (64, 256, (64, 256)), (64, 512, (64, 512)),
    (96, 96, (128, 128)), (128, 128, (128, 128)), (192, 128, (256, 256)), (256, 256, (256, 256)),
])
def test_forward_kernel_follows_fa3_dispatch_in_this_build(d, dv, kernel):
    from isoexec.ops.attention.gqa_sm90.ops import forward_kernel

    assert forward_kernel(d, dv) == kernel


@pytest.mark.parametrize(("d", "dv"), [(320, 320), (128, 512), (256, 512), (60, 60)])
def test_forward_kernel_refuses_a_pair_no_built_kernel_covers(d, dv):
    from isoexec.ops.attention.gqa_sm90.ops import forward_kernel

    with pytest.raises(RuntimeError, match="no FA3 forward kernel"):
        forward_kernel(d, dv, where="test")


def test_deterministic_backward_where_fa3_has_one():
    from isoexec.ops.attention.gqa_sm90.ops import deterministic_backward

    assert deterministic_backward(64, 64) and deterministic_backward(128, 128) and deterministic_backward(96, 96)
    assert not deterministic_backward(256, 256) and not deterministic_backward(192, 128)
    # FA3: ``Deterministic = Deterministic_ && kHeadDim < 256`` and mha_bwd refuses deterministic at 256.
    launch = (FA3_DIR / "flash_bwd_launch_template.h").read_text()
    assert "Deterministic_ && kHeadDim < 256" in launch
    assert spec.DETERMINISTIC_BWD_MAX_HEAD_DIM == 128


def test_every_entry_pins_the_split_recipe_and_the_causal_window():
    """The one call site passes 1 or -SEG (num_splits_arg), and the causal window (-1, 0)."""
    src = (Path(__file__).resolve().parents[1] / "ops.py").read_text()
    tree = ast.parse(src)
    fwd_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "fwd"]
    assert len(fwd_calls) == 1
    args = [ast.unparse(a) for a in fwd_calls[0].args]
    assert args[24:28] == ["True", "-1", "0", "0"]  # is_causal, window left/right, attention_chunk
    assert args[31] == "num_splits" and args[32] == "None" and args[33] == "0"  # pack_gqa auto, sm_margin 0
    assert args[34] == "workspace"
    assert "num_splits = num_splits_arg(seg_tokens, seg_path)" in src
    assert spec.NUM_SPLITS == 1


def test_importing_the_package_builds_nothing(monkeypatch):
    import importlib
    import sys

    import torch.utils.cpp_extension as cpp

    monkeypatch.setattr(cpp, "load", lambda *a, **k: pytest.fail("import must not build"))
    for name in [m for m in sys.modules if m.startswith("isoexec.ops.attention.gqa_sm90")]:
        monkeypatch.delitem(sys.modules, name)
    pkg = importlib.import_module("isoexec.ops.attention.gqa_sm90")
    importlib.import_module("isoexec.ops.attention.gqa_sm90.integration")
    assert callable(pkg.varlen_attention)
    with pytest.raises(AttributeError):
        pkg.not_an_export


def test_backward_policy_requires_a_bool():
    from isoexec.ops.attention.gqa_sm90 import ops

    with pytest.raises(TypeError):
        ops.require_deterministic_backward(1)


def test_install_refuses_a_worker_without_an_sm90_device(monkeypatch):
    import torch

    from isoexec.ops.attention.gqa_sm90 import integration

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires an SM90 CUDA device"):
        integration.install()


def test_the_trainer_install_selects_attention_through_the_dispatch_table(monkeypatch):
    from isoexec.ops import _dispatch
    from isoexec.runtimes import trainer_installs

    monkeypatch.setattr(_dispatch, "ARCH", "cpu")
    with pytest.raises(_dispatch.ArchSelectionError, match="seam 'gqa'"):
        trainer_installs.install_attention(False)


def test_the_cores_run_the_segment_split_recipe():
    from isoexec.ops.attention.gqa_sm90 import integration

    assert spec.CORE_USES_SEGMENT_SPLIT and integration.core_seg_tokens() == spec.SEG_TOKENS == 2560
    assert (spec.TRAINER_IMPL, spec.PAGED_IMPL, spec.TWIN_CLASS) == (
        "sm90_fa3_segsplit_v1_varlen", "sm90_fa3_segsplit_v1_paged", "attention.gqa_sm90_segsplit_v1")
    assert spec.SEG_RECIPE in spec.TRAINER_IMPL and spec.SEG_RECIPE in spec.PAGED_IMPL


def test_workspace_floats_is_monotone_and_covers_the_lse():
    from isoexec.ops.attention.gqa_sm90.ops import workspace_floats

    a = workspace_floats(256, 6, 256, 256, 132)
    assert a >= 6 * 256 + 132 * 128 * (256 + 8)
    assert workspace_floats(8192, 6, 256, 256, 132) > a and workspace_floats(256, 6, 256, 256, 132, partial_floats=10) == a + 10
