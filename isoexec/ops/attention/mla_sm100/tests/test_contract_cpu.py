"""Check MLA declarations and constants without CUDA."""

from __future__ import annotations

import re
from pathlib import Path

from isoexec.ops.attention.mla_sm100 import spec
from isoexec.ops.attention.mla_sm100 import fwd as fwd_mod
from isoexec.ops.attention.mla_sm100 import _extension as ext_mod

CSRC = Path(spec.__file__).resolve().parent / "csrc" / "fwd"


def _cpp_constant(path: str, name: str) -> str:
    text = (CSRC / path).read_text()
    m = re.search(rf"{name}\s*=\s*([^;]+);", text)
    assert m, f"{name} not found in {path}"
    return m.group(1).strip()


def test_the_page_and_tile_constants_agree_across_the_two_languages():
    """KV tile == page == 64 is the recipe; a change to either side is a version bump."""
    assert spec.PAGE == 64
    assert _cpp_constant("config.h", "PAGE_BLOCK_SIZE") == "64"
    assert _cpp_constant("config.h", "BLOCK_SIZE_M") == "64"
    assert _cpp_constant("config.h", "HEAD_DIM_K") == "576"
    assert _cpp_constant("config.h", "HEAD_DIM_V") == "512"
    assert fwd_mod.HEAD_DIM_K == 576 and fwd_mod.HEAD_DIM_V == 512
    assert fwd_mod.M_BLOCK == 64
    # 520, not 516: 516 floats is 2064 B a row and 2064 % 32 == 16, so odd rows' 32-byte lane
    # groups straddle two sectors in every node load, node store and published partial store.
    # Execution only -- the partial record still holds O[512], L, M -- and worth 2.6% of a call.
    assert fwd_mod.PARTIAL_WIDTH == 520
    assert fwd_mod.PARTIAL_WIDTH * 4 % 32 == 0, "a partial row must start on a 32-byte sector"
    assert _cpp_constant("params.h", "PARTIAL_WIDTH") == "HEAD_DIM_V + 8"


def test_the_empty_and_mask_sentinels_are_the_declared_finite_ones():
    """dtir section 2/3: the empty sentinel is finite, and the mask value must scale below the
    running-max seed so that max(seed, mask*scale_log2) == seed."""
    assert _cpp_constant("leaf.cuh", "MAX_INIT_VAL_SM") == "-1e30f"
    assert _cpp_constant("leaf.cuh", "MAX_INIT_VAL") == "-1e33f"
    assert _cpp_constant("combine.cu", "MAX_INIT_VAL_SM") == "-1e30f"


def test_the_declared_nvcc_profile_disables_fma_contraction():
    """One profile for every kernel of the op: the leaf, the plan and the combine are separate
    translation units, and contraction differing between them would break bit equality inside
    one architecture."""
    flags = ext_mod.profile_flags()
    assert "--fmad=false" in flags
    assert "--ftz=true" in flags and "--prec-div=false" in flags
    assert "-gencode=arch=compute_100a,code=sm_100a" in flags
    assert not any("use_fast_math" in f for f in flags)
    # the digest is stable and reflects flags + CUTLASS identity + source bytes
    assert len(fwd_mod.profile_digest()) == 16
    assert fwd_mod.profile_digest() == ext_mod.profile_digest()


def test_the_execution_overrides_are_execution_only_and_all_keys_are_known():
    """The plan refuses an unknown key, so a variant can never be silently ignored."""
    known = {"granularity", "reverse", "item_overhead", "fold_overhead", "num_parts",
             "combine_config", "combine_single_group", "no_pdl"}
    for ov in fwd_mod.EXEC_OVERRIDE_VARIANTS:
        assert set(ov) <= known, ov
        assert "seg_tokens" not in ov and "seg" not in ov
    assert {"granularity": "single"} in fwd_mod.EXEC_OVERRIDE_VARIANTS
    assert {"granularity": "whole"} in fwd_mod.EXEC_OVERRIDE_VARIANTS


def test_the_leaf_states_the_blackwell_schedule_that_the_traits_encode():
    """The TMEM map is what makes a 512-wide fp32 accumulator fit without a CTA pair; if the
    interleave assumption is removed, this test is the place that notices."""
    traits = (CSRC / "traits.h").read_text()
    assert "TMEM_HALF_DP = 16" in traits
    assert _cpp_constant("traits.h", "TMEM_COLS") == "512"
    assert _cpp_constant("config.h", "PV_N") == "256"
    # Round 5: Q lives in TMEM, so the 72 KB it occupied in SMEM is a third KV stage.
    assert _cpp_constant("config.h", "KV_STAGES") == "3"
    assert _cpp_constant("config.h", "S_BUFFERS") == "2"
    # ... and the TMEM map that makes room for Q: S buffers and Q on the DP+0 half (an A operand on the
    # DP+16 half faults), both O halves on DP+16.  288 = 36 K16 steps x 8 columns.
    assert _cpp_constant("traits.h", "TMEM_Q") == "Config::S_BUFFERS * PAGE_BLOCK_SIZE"
    assert "TMEM_Q + Q_COLS <= TMEM_COLS" in traits
    leaf = (CSRC / "leaf.cuh").read_text()
    # the two arithmetic hazards the Blackwell schedule introduces are stated in the source
    assert "0 * NaN is NaN" in leaf                    # OOB V must be zeroed before PV
    assert "ScaleOut::Zero" in leaf                    # a segment's first PV overwrites O
    assert "x*1.0f == x" in leaf                       # the free correction skip


def test_this_package_keeps_its_constants_and_the_family_owns_the_declaration():
    assert spec.ARCH == "sm100"
    ids = getattr(spec, "IMPL_ID", None)
    if ids is not None:
        assert "sm100" in ids and "sm90" not in ids


def test_the_family_declares_a_distinct_recipe_with_local_batteries():
    """SM100 MLA has no bitwise twin; its own correctness batteries remain runnable."""
    from isoexec.ops.attention.spec import _CORE

    mine = [i for i in _CORE
            if i.op == "attention.core.mla" and i.impl_id == "sm100_absorbed_segtree"]
    assert len(mine) == 1
    assert mine[0].supported_archs == frozenset({"sm100"})
    assert not mine[0].twin_class
    assert not spec.TWIN_CLASS
    here = Path(__file__).resolve().parent
    assert (here / "test_conformance_gpu.py").is_file()
    assert (here / "conformance.py").is_file()
