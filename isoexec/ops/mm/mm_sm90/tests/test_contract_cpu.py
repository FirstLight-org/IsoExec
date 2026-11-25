"""Declarations, the segment policy and the reference program of the sm90 mm recipe (no CUDA, no build)."""

from __future__ import annotations

import ast
import builtins
import inspect
import os
import re
import textwrap
from pathlib import Path

import pytest
import torch

from isoexec.ops.mm.mm_sm90 import reference as R
from isoexec.ops.mm.mm_sm90 import segments as seg
from isoexec.ops.mm.mm_sm90 import spec

PKG = Path(seg.__file__).resolve().parent


def test_package_declares_sm90_and_a_new_recipe_id():
    assert spec.ARCH == "sm90"
    assert spec.RECIPE_VERSION == "sm90_wgmma_segtree_v1"
    assert spec.OPS == ("mm",)
    # Never a reused id: the sm100 recipe is a different program (tcgen05 k32), and the old sm90 ids
    # define no segment structure.
    assert spec.IMPL_ID not in ("triton_batch_invariant", "cublaslt_pinned", "sm100_tcgen05_segtree")
    for impl in spec.IMPLS:
        assert impl.supported_archs == frozenset({"sm90"}) and impl.op == "mm"
        assert impl.version == spec.IMPL_VERSION


def test_registered_in_the_mm_family_and_twin_proofs():
    from isoexec.core.registry import TWIN_PROOFS
    from isoexec.ops.mm import spec as family

    assert spec.IMPL_ID in {i.impl_id for i in family.IMPLS}
    # the providers it replaces are retired from the registry, not left selectable beside it
    assert "cublaslt_pinned" not in {i.impl_id for i in family.IMPLS}
    proofs = TWIN_PROOFS[spec.TWIN_CLASS]
    for p in proofs:
        assert (PKG.parents[3] / p).is_file(), p


_POLICY_INPUTS = {"K", "dtype_name", "ValueError", "LEGAL_SPLITS"}


def test_segment_count_reads_only_K_and_dtype():
    tree = ast.parse(textwrap.dedent(inspect.getsource(seg.segment_count_by_policy)))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    extra = names - _POLICY_INPUTS - set(dir(builtins))
    assert not extra, f"segment_count_by_policy reads undeclared names: {sorted(extra)}"
    for fn in (seg.segment_count_by_policy, seg.segment_count, seg.segment_k):
        params = list(inspect.signature(fn).parameters)
        assert params[0] == "K" and "N" not in params and "M" not in params, fn.__name__


def test_segments_are_aligned_and_divide_K():
    for K in (512, 1536, 4352, 5120, 6144, 17408, 2048, 4096):
        S = seg.segment_count(K, "bf16")
        assert S in seg.LEGAL_SPLITS and K % S == 0 and (K // S) % seg.SEGMENT_ALIGN == 0


def test_table_digest_covers_rows_and_policy():
    a = {"policy_version": 1, "rows": [[2048, "bf16", 2]]}
    assert seg.table_digest(a) == seg.table_digest({"policy_version": 1, "rows": [[2048, "bf16", 2]]})
    assert seg.table_digest(a) != seg.table_digest({"policy_version": 1, "rows": [[2048, "bf16", 4]]})
    assert seg.table_digest(a) != seg.table_digest({"policy_version": 2, "rows": [[2048, "bf16", 2]]})
    t = seg.published_table()
    assert t.get("policy_version") == seg.POLICY_VERSION
    for K, d, S in t.get("rows", ()):
        assert seg.segment_count_by_policy(K, d) == S


def _compiled_table() -> dict[str, tuple]:
    """Parse csrc/mm_sm90_table.cuh without a compiler."""
    text = (PKG / "csrc" / "mm_sm90_table.cuh").read_text()
    rows = {}
    for m in re.finditer(r"^\s*X\((\w+), (\d+), (\d+), (\d+), (\d+), (\w+), ([\w:]+), (\d), (\d), (\d), (\d+)\)",
                         text, re.M):
        name, bm, bn, cm, cn, sched, ed, f32, lv, sw, _st = m.groups()
        rows[name] = (int(bm), int(bn), int(cm), int(cn), {"WS": "ws", "PP": "pingpong", "CO": "cooperative"}[sched],
                      int(f32), int(lv), int(sw), ed)
    return rows


def test_planner_mirrors_the_compiled_table_both_ways():
    rows = _compiled_table()
    assert set(rows) == set(seg._CONFIGS)
    for name, (bm, bn, cm, cn, sched, f32, lv, sw, ed) in rows.items():
        assert seg._CONFIGS[name] == seg.ConfigSpec(bm, bn, cm, cn, sched, f32, lv, sw), name
        assert (ed == "float") == bool(f32), name
    tus = len(re.findall(r"#define ISOEXEC_MM_SM90_TU\d+\(X\)", (PKG / "csrc" / "mm_sm90_table.cuh").read_text()))
    from isoexec.ops.mm.mm_sm90._extension import SOURCES

    assert sorted(SOURCES) == sorted(["mm_sm90_bind.cu", *(f"mm_sm90_tu{i}.cu" for i in range(tus))])


def test_every_plan_at_a_split_shares_the_segmentation():
    for K, N, S in ((1536, 5120, 2), (4352, 5120, 2), (6144, 5120, 8), (17408, 5120, 8), (5120, 3584, 1)):
        for M in (1, 32, 320, 4096):
            plans = seg.plans_at_split(M, K, N, S)
            assert plans, (K, N, S)
            for p in plans:
                assert p.split == S and p.segment_k * S == K
                if p.mode == seg.ONCHIP and S > 1:
                    assert (1 << p.spec.levels) >= S
                if p.mode == seg.SPLIT:
                    assert p.spec.out_fp32 and p.spec.levels == 0
    assert seg.plans_at_split(32, 4352, 5120, 3) == []  # not a power-of-two tree


def test_the_tile_chooser_only_ever_picks_a_bf16_single_chain_config():
    for N in (8, 3584, 4120, 5120, 8704, 14336, 62080, 248320):
        for M in (1, 7, 8, 32, 64, 65, 128, 256, 320, 512, 1024, 2048, 8192, 16384):
            p = seg.plan_for(M, 5120, N, torch.bfloat16)
            assert p.split == 1 and p.spec.out_fp32 == 0 and p.spec.levels == 0, (M, N, p)


def test_reference_tree_is_pik_combine_order():
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik

    pik = ensure_pik()
    for g in (1, 2, 4, 8, 16):
        assert R.combine_order(g) == pik.combine_order(g)


def test_reference_program_structure_on_cpu():
    torch.manual_seed(0)
    x = torch.randn(5, 512).bfloat16()
    w = torch.randn(24, 512).bfloat16()
    # S = 1 is one leaf, one round
    assert torch.equal(R.segtree(x, w, 1), R.fp64_leaf(x, w).to(torch.bfloat16))
    # S = 4 is ((l0 + l1) + (l2 + l3)) in fp32, one round
    ls = [R.fp64_leaf(x[:, i * 128:(i + 1) * 128], w[:, i * 128:(i + 1) * 128]) for i in range(4)]
    want = ((ls[0] + ls[1]) + (ls[2] + ls[3])).to(torch.bfloat16)
    assert torch.equal(R.segtree(x, w, 4), want)
    # bf16 leaves round each segment root first
    lb = [t.to(torch.bfloat16).float() for t in ls]
    assert torch.equal(R.segtree(x, w, 4, round_leaf=True, out_dtype=torch.float32), (lb[0] + lb[1]) + (lb[2] + lb[3]))
    # a row's result does not read any other row
    assert torch.equal(R.segtree(x[2:3], w, 4), R.segtree(x, w, 4)[2:3])


def test_same_bits_mod_nan():
    a = torch.tensor([1.0, float("nan"), -0.0]).bfloat16()
    b = torch.tensor([1.0, float("nan"), -0.0]).bfloat16()
    b.view(torch.int16)[1] = 0x7FFF
    assert R.same_bits_mod_nan(a, b)
    c = torch.tensor([1.0, float("nan"), 0.0]).bfloat16()
    assert not R.same_bits_mod_nan(a, c)


def test_flag_is_classified_as_a_both_sides_recipe_selector():
    from isoexec.core import flags

    f = flags.get(spec.FLAG)
    assert "both" in f.sides


def test_live_impl_id_reads_the_installed_wrapper():
    from isoexec.ops.mm.trainer import _live_impl_id

    def fake(a, b, bias=None):
        return None

    fake._isoexec_sm90_segtree = True
    assert _live_impl_id(fake) == spec.IMPL_ID


#: A local Qwen3.8-27B checkpoint directory (config.json is all this reads); the test skips without one.
QWEN38 = Path(os.environ.get("ISOEXEC_TEST_QWEN38_PATH", "/nonexistent"))


@pytest.mark.skipif(not (QWEN38 / "config.json").is_file(), reason="set ISOEXEC_TEST_QWEN38_PATH to a Qwen3.8-27B checkpoint")
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_every_census_shape_of_the_target_model_is_served(tp):
    from isoexec.integrations.verl.model import source_config
    from isoexec.runtimes.trainer_installs import model_gemm_census

    source, raw = source_config(str(QWEN38))
    census = model_gemm_census(source, raw)
    for site in census.sites(tp, tp):
        assert seg.supported(site.K, site.N), site
    # row-parallel sites (pik leaves of the FULL K at G = 8): o_proj / gdn out (6144), fc2 (17408)
    for k_full in (6144, 17408):
        leaf_k = k_full // 8
        assert seg.execution_supported(k_full // tp, 5120, leaf_k), (tp, k_full)
