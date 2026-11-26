"""Declarations and the K schedule, with no CUDA and no build."""

from __future__ import annotations

from pathlib import Path

import pytest

from isoexec.ops.mm.mm_sm100 import segments as seg
from isoexec.ops.mm.mm_sm100 import spec

PKG = Path(seg.__file__).resolve().parent


def test_package_declares_sm100_and_a_recipe_version():
    assert spec.ARCH == "sm100"
    assert spec.RECIPE_VERSION == "sm100_tcgen05_segtree_v2"
    assert spec.OPS == ("mm",)
    # The sm90 ids are different recipes and must never be reused for this kernel.
    assert spec.IMPL_ID not in ("triton_batch_invariant", "cublaslt_pinned")


def test_impls_are_declared_only_after_the_battery_passes():
    """Until then the honest state is 'no impl', which is a gap and not a pass."""
    for impl in spec.IMPLS:
        assert impl.supported_archs == frozenset({"sm100"}), impl
        assert impl.op == "mm"
        assert impl.version == spec.IMPL_VERSION == 2, impl


# ------------------------------------------------------------ the schedule is the policy's output

def _deployed_weights():
    """Every (K, N) the example models present at their deployed TP degrees, plus the SkyRL model sites."""
    from isoexec.ops.mm.mm_sm100.tests._helpers import RL_CENSUS

    return sorted({(K, N) for N, K, _label in RL_CENSUS} | set(_MODEL_SITES))


#: The audited model forward weights (K, N) at their deployed shard (README section 4.32).
_MODEL_SITES = (
    (4096, 64), (4096, 128), (5120, 512), (5120, 1280), (1280, 4096), (1536, 4096),
    (8192, 4096), (4096, 8192), (512, 32768), (1280, 32768), (6144, 25600), (4096, 154880),
)


def test_the_kernel_schedule_is_the_schedule_policy_on_every_deployed_shape():
    """Fails if a segment count or length the kernel runs differs from ``schedule_policy`` rule 3."""
    from isoexec.ops import schedule_policy as P

    assert seg.REDUCER_BLOCK == 16 and seg.REDUCER_SHORTEST is None
    weights = _deployed_weights()
    assert len(weights) > 40
    for K, N in weights:
        if K % seg.REDUCER_BLOCK:
            with pytest.raises(ValueError):
                seg.schedule(K)
            continue
        want = P.reduce(K, b=16)                      # no published constant: G = 1, S = K
        assert want.G == 1 and want.S == K and want.merge == "tree"
        assert seg.schedule(K) == want
        assert seg.segment_count(K, "bf16") == want.G
        assert seg.segment_k(K, "bf16") == want.S
        if not seg.supported(K, N):
            continue
        for M in (1, 8, 32, 128, 256, 4096):
            plan = seg.plan_for(M, K, N, "bf16")
            assert (plan.split, plan.segment_k) == (want.G, want.S), (M, K, N, plan)
            assert plan.groups * plan.spc == want.G
            for other in seg.legal_plans(M, K, N, "bf16"):
                assert (other.split, other.segment_k) == (want.G, want.S), (M, K, N, other)


def test_a_row_parallel_weight_is_d_max_shards_of_the_policys_segments():
    """k sharded: one tree over the d_max shards, each shard the policy's G segments of K / d_max."""
    from isoexec.ops import schedule_policy as P

    for K in (2048, 4096, 5120, 10240, 12288, 14336, 3072):
        for d_max in (1, 2, 4, 8):
            want = P.reduce(K // d_max, b=16)
            assert seg.schedule(K, d_max) == want
            assert seg.row_parallel_leaves(K, d_max) == d_max * want.G == d_max
    with pytest.raises(ValueError):
        seg.schedule(11008, 32)       # 344-term shards are not whole 16-term blocks
    with pytest.raises(ValueError):
        seg.schedule(4096, 3)         # degrees that are not powers of two are refused
    with pytest.raises(ValueError):
        seg.schedule(2056)            # not whole 16-term blocks


def test_the_schedule_reads_only_K_and_d_max():
    """S is a function of the contraction length and the declared degree, never of the weight's width."""
    import ast
    import builtins
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(seg.schedule)))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    allowed = {"K", "d_max", "REDUCER_BLOCK", "REDUCER_SHORTEST", "schedule_policy", "reduce", "Segments"}
    extra = names - allowed - set(dir(builtins))
    assert not extra, f"schedule reads undeclared names: {sorted(extra)}"
    for fn in (seg.schedule, seg.segment_count, seg.segment_k, seg.row_parallel_leaves):
        assert list(inspect.signature(fn).parameters)[0] == "K", fn.__name__
        assert "N" not in inspect.signature(fn).parameters, fn.__name__


def test_no_segment_table_is_left():
    """The table, its loader, its digest and its policy version are gone; the schedule is a function."""
    assert not (PKG / "segments.json").exists()
    for name in ("published_table", "table_digest", "write_table", "regenerate", "POLICY_VERSION",
                 "LEGAL_SPLITS", "segment_count_by_policy", "MEASURED_SPLITS", "default_k_set"):
        assert not hasattr(seg, name), name


def test_the_schedule_digest_covers_everything_the_schedule_reads(monkeypatch):
    base = seg.schedule_digest()
    assert seg.schedule_inputs() == {"policy": "7.3", "rule": 3, "block": 16, "shortest": None,
                                     "merge": "tree", "root": "bf16_rne"}
    monkeypatch.setattr(seg, "REDUCER_SHORTEST", 1024)
    assert seg.schedule_digest() != base
    monkeypatch.setattr(seg, "REDUCER_SHORTEST", None)
    monkeypatch.setattr(seg.schedule_policy, "VERSION", "0")
    assert seg.schedule_digest() != base


def test_the_installed_schedule_policy_is_the_canonical_module():
    """``isoexec/ops/schedule_policy.py`` is installed unchanged (policy_validation/port/schedule_policy.py)."""
    import hashlib

    from isoexec.ops import schedule_policy as P

    assert P.VERSION == "7.3"
    assert hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest() == \
        "14f043d9d504f1b55e6fdda6df8b013eef5e42c79cb4cb0e20556afd35bf2607"


def test_every_plan_for_a_shape_shares_one_schedule():
    """Plans differ in ownership only: same split, same segment size, groups * spc == split."""
    for K, N in ((2048, 1152), (4096, 4096), (5120, 14336), (14336, 5120), (2048, 9216)):
        S = seg.segment_count(K, "bf16")
        for M in (1, 32, 512, 4096):
            for plan in seg.legal_plans(M, K, N, "bf16"):
                assert plan.split == S
                assert plan.segment_k == K // S
                assert plan.groups * plan.spc == S
                assert plan.segment_k * S == K
        chosen = seg.plan_for(M, K, N, "bf16")
        assert chosen.split == S


def test_both_cuts_of_the_tree_are_available_and_the_split_survives_either():
    """One CTA per segment and one CTA per tile are both compiled and both legal at an execution split."""
    K, N, S = 2048, 1152, 2
    for M in (1, 32, 512, 4096):
        plans = seg.plans_at_split(M, K, N, S)
        assert all(p.split == S for p in plans)
        cut = [p for p in plans if p.spc == 1]
        whole = [p for p in plans if p.spc == S]
        assert cut, f"M={M}: no one-CTA-per-segment plan"
        assert whole, f"M={M}: no one-CTA-per-tile plan"
        assert all(p.groups == S and p.publishes_partials for p in cut)
        assert all(p.groups == 1 and not p.publishes_partials for p in whole)
        assert cut[0].workspace_floats(M, N) == S * M * N
        assert whole[0].workspace_floats(M, N) == 0


def test_the_a_fetch_height_always_covers_M():
    """A short A fetch is only admissible while it covers every real row of the tile."""
    for M in (1, 7, 8, 9, 31, 32, 33, 64, 128, 129, 512, 4096):
        rows = seg._a_rows_for(M)
        assert rows == 0 or M <= rows, (M, rows)
        for plan in seg.legal_plans(M, 2048, 1152, "bf16"):
            height = next(
                a for (_f, a, _spc, _mode), n in seg._CONFIGS.items() if n == plan.config
            )
            assert height == 0 or M <= height, (M, plan.config)


def _tree(values):
    """The declared merge, in Python: adjacent pairs, unpaired node carried verbatim."""
    v = list(values)
    while len(v) > 1:
        out = [v[i] + v[i + 1] for i in range(0, len(v) - 1, 2)]
        if len(v) & 1:
            out.append(v[-1])
        v = out
    return v[0]


def test_the_declared_tree_is_balanced_adjacent_pairs():
    assert _tree(["a"]) == "a"
    assert _tree(["a", "b"]) == "ab"
    assert _tree(["a", "b", "c"]) == "abc"        # (a+b) then carry c, then (ab)+c
    assert _tree(["a", "b", "c", "d"]) == "abcd"  # (a+b)+(c+d)
    # The decomposition the two plans rely on: the tree over [0,8) is the tree over its aligned
    # halves, which is what makes segments-per-CTA free.
    leaves = list("01234567")
    assert _tree(leaves) == _tree([_tree(leaves[:4]), _tree(leaves[4:])])
    assert _tree(leaves) == _tree([_tree(leaves[i:i + 2]) for i in range(0, 8, 2)])
    assert _tree(leaves) == _tree([_tree(leaves[i:i + 4]) for i in range(0, 8, 4)])


def test_the_tree_is_not_the_chain():
    """Different identities. If these ever agreed the plan relations would prove nothing."""
    def chain(values):
        acc = values[0]
        for v in values[1:]:
            acc = acc + v
        return acc

    # Python floats are FP64, so the magnitudes have to straddle 2**53 for the grouping to show.
    # The kernel's partials are FP32 and straddle 2**24; the shape of the counterexample is the
    # same one.
    v = [2.0**54, 1.0, -(2.0**54), 1.0]
    assert _tree(v) == 0.0
    assert chain(v) == 1.0


def test_unsupported_shapes_are_refused_not_padded():
    """A K the segment alignment cannot express is declined, so nobody pads a reduction."""
    assert seg.supported(2048, 1152)
    assert not seg.supported(2048 + 64, 1152)   # K not a multiple of the alignment
    assert not seg.supported(2048, 63)          # narrower than one store subtile
    assert not seg.supported(2048, 1153)        # odd N: the combine pairs columns
    assert not seg.supported(1024, 1028)        # a 2056-byte output row: TMA pitches are 16 bytes


def test_config_names_cover_every_plan_the_planner_can_emit():
    for K, N in ((2048, 1152), (4096, 4096), (5120, 14336), (14336, 5120), (2048, 9216),
                 (4096, 24576), (2048, 12352)):
        for M in (1, 32, 128, 129, 512, 4096):
            plan = seg.plan_for(M, K, N, "bf16")
            assert plan.config in seg._CONFIGS.values()
            assert plan.spc <= seg._MAX_SPC[plan.config.split("_")[0]]
            assert plan.split % plan.spc == 0


def test_the_declared_twin_class_names_a_battery():
    """A twin claim is a test that passes, not a string -- so the registry must name the battery."""
    from pathlib import Path

    from isoexec.core.registry import TWIN_PROOFS

    from .. import spec as pkg_spec

    assert pkg_spec.TWIN_CLASS in TWIN_PROOFS
    proofs = TWIN_PROOFS[pkg_spec.TWIN_CLASS]
    assert proofs, f"{pkg_spec.TWIN_CLASS} names no battery"
    root = Path(__file__).resolve().parents[5]
    for rel in proofs:
        assert (root / rel).is_file(), rel
    # And the battery this package actually ships is one of them.
    assert any(r.endswith("mm_sm100/tests/mm_sm100_battery_gpu.py") for r in proofs)


def test_the_family_spec_carries_this_impl_and_does_not_rename_the_sm90_ones():
    from isoexec.ops.mm import spec as family

    from .. import spec as pkg_spec

    ids = {i.impl_id: i for i in family.IMPLS}
    assert pkg_spec.IMPL_ID in ids, "ops/mm/spec.py does not declare the sm100 impl"
    mine = ids[pkg_spec.IMPL_ID]
    assert mine.supported_archs == frozenset({"sm100"})
    # The class is recorded in TWIN_PROOFS, not on the record: it has one declarable member and a
    # one-member class is refused by core/tests/test_op_gate_alignment_cpu.py. See
    # spec.WHY_NO_TWIN_FIELD.
    assert not mine.twin_class, pkg_spec.WHY_NO_TWIN_FIELD
    # The sm90 recipes are different recipes and keep their ids and their arch.
    for legacy in ("triton_batch_invariant", "sm90_wgmma_segtree"):
        assert ids[legacy].supported_archs == frozenset({"sm90"})
        assert not ids[legacy].twin_class, f"{legacy} is not a twin of the sm100 recipe"


def test_the_live_impl_predicate_answers_for_the_installed_provider():
    """``mm_impl_id`` is a predicate on the object, so it cannot attest a provider it does not run."""
    from isoexec.ops.mm.trainer import _live_impl_id

    class _Plain:
        pass

    class _Sm90:
        _isoexec_sm90_segtree = True

    class _Sm100:
        _isoexec_sm100 = True

    class _Both:  # a wrapper over a wrapper: the sm100 marker is the one that answers
        _isoexec_sm90_segtree = True
        _isoexec_sm100 = True

    from .. import spec as pkg_spec

    assert _live_impl_id(_Plain()) == "triton_batch_invariant"
    assert _live_impl_id(_Sm90()) == "sm90_wgmma_segtree"
    assert _live_impl_id(_Sm100()) == pkg_spec.IMPL_ID
    assert _live_impl_id(_Both()) == pkg_spec.IMPL_ID


def test_the_battery_manifest_names_this_battery_with_a_claim():
    from isoexec.checks.tests.test_gpu_battery_manifest_cpu import BATTERIES

    rows = [b for b in BATTERIES
            if b.path == "isoexec/ops/mm/mm_sm100/tests/mm_sm100_battery_gpu.py"]
    assert len(rows) == 1, "the sm100 mm battery is not registered exactly once"
    assert rows[0].claim.strip()


def test_the_record_a_peer_compares_pins_the_schedule_and_not_only_the_id():
    """The impl id does not pin the arithmetic: what the schedule reads travels as a digest too."""
    from .. import integration

    rec = integration.record()
    assert rec["impl_id"] == "sm100_tcgen05_segtree"
    assert rec["recipe_version"] == seg_recipe()
    assert rec["schedule_digest"] == seg.schedule_digest()
    assert rec["schedule_policy_version"] == "7.3"
    assert "segment_table_digest" not in rec and "segment_policy_version" not in rec


def seg_recipe() -> str:
    from .. import spec as pkg_spec

    return pkg_spec.RECIPE_VERSION


def test_the_provider_is_off_unless_the_flag_is_set(monkeypatch):
    from .. import integration

    monkeypatch.delenv("ISOEXEC_MM_SM100", raising=False)
    assert integration.mm_sm100_enabled() is False
    assert integration.install_mm_sm100() is False
    monkeypatch.setenv("ISOEXEC_MM_SM100", "1")
    assert integration.mm_sm100_enabled() is True


# ------------------------------------------------------------ round 5: the compiled table mirror

def _compiled_entry_names() -> list[str]:
    import re

    src = (PKG / "csrc" / "mm_sm100_gemm.cu").read_text()
    return re.findall(r'ENTRY\("([a-z0-9_]+)"', src)


def test_the_compiled_table_and_the_planner_name_the_same_configurations():
    """ENTRY names in the .cu and `_CONFIGS` + `_MEASUREMENT_ONLY` agree in both directions."""
    names = _compiled_entry_names()
    assert len(names) == len(set(names)), "a configuration name is compiled twice"
    planned = set(seg._CONFIGS.values()) | set(seg._MEASUREMENT_ONLY)
    assert set(names) == planned, (sorted(set(names) - planned), sorted(planned - set(names)))
    for fam, _rows, spc, _mode in seg._CONFIGS:
        assert fam in seg._MAX_SPC, fam
        assert spc <= seg._MAX_SPC[fam] or _mode != seg.ONCHIP, (fam, spc)


# ------------------------------------------------------------ round 5: the collectives entry points

def test_the_execution_alignment_is_the_umma_k_and_the_dispatch_alignment_does_not_move():
    """Section 4.17 (d): an EXECUTION segment is any multiple of 16 (the reducer's block); the
    production dispatch serves K % 256 == 0."""
    assert seg.EXECUTION_ALIGN == seg.REDUCER_BLOCK == 16
    assert seg.SEGMENT_ALIGN == 256
    # the dispatch still declines what it declined
    assert not seg.supported(5504, 4096) and not seg.supported(2752, 4096)
    assert not seg.supported(128, 2048)
    # the execution surface serves pik's shipped leaves
    assert seg.execution_supported(5504, 4096, 1376)
    assert seg.execution_supported(2752, 4096, 1376)
    assert seg.execution_supported(128, 2048, 64)
    assert seg.execution_supported(64, 2048, 64)
    assert not seg.execution_supported(2048, 1152, 24)     # not a multiple of the UMMA K
    assert not seg.execution_supported(2048, 1152, 1376)   # does not divide K


def test_plans_at_split_serve_sub_256_segments_and_refuse_sub_16_ones():
    for M, K, N, split in ((512, 5504, 4096, 4), (32, 2752, 4096, 2), (1, 128, 2048, 2),
                           (4096, 4160, 2048, 4)):
        plans = seg.plans_at_split(M, K, N, split)
        assert plans, (M, K, N, split)
        assert {p.segment_k for p in plans} == {K // split}
        assert all(p.groups * p.spc == split for p in plans)
        # every tile family with a whole tree or a cut at that split is offered, whatever its K block
        fams = {seg._KEY_OF[p.config][0] for p in plans}
        assert {"narrow", "wide"} <= fams, fams
    assert seg.plans_at_split(512, 2048, 1152, 3) == []      # 2048 / 3 is not an integer
    assert seg.plans_at_split(512, 2056, 1152, 1) == []      # 2056 is not a multiple of 16


def test_every_prefill_family_has_a_publishing_twin_at_the_policys_split_and_an_execution_split():
    """`run_plan(..., root_dtype=float32)` must work on whatever plan the production rule picks."""
    for fam in seg._TILE_GEOM:
        for split in (seg.schedule(4096).G, 2):  # the policy's G and an execution split
            if seg._MAX_SPC.get(fam, 0) < split or (fam, 0, split, seg.ONCHIP) not in seg._CONFIGS:
                continue
            plan = seg.Plan(config=seg._CONFIGS[(fam, 0, split, seg.ONCHIP)],
                            segment_k=4096 // split, split=split, spc=split)
            twin = seg.publishing_twin(plan)
            assert twin.mode == seg.COMBINE and twin.groups == 1 and twin.spc == split
            # the tile's own twin where one is compiled; any whole-tree publisher otherwise
            if (fam, 0, split, seg.COMBINE) in seg._CONFIGS:
                assert seg._KEY_OF[twin.config][:3] == (fam, 0, split)
    # and on every production plan of the relation shapes, decode included
    for K, N in ((2048, 1152), (4096, 4096), (5120, 14336), (2048, 9216), (4096, 24576)):
        for M in (1, 8, 32, 128, 512, 4096, 16384):
            prod = seg.plan_for(M, K, N, "bf16")
            twin = seg.publishing_twin(prod)
            assert twin.split == prod.split and twin.segment_k == prod.segment_k
            rows = seg._KEY_OF[twin.config][1]
            assert rows == 0 or M <= rows, (M, twin.config)
    # a cut plan has no single-CTA root
    cut = next(p for p in seg.plans_at_split(512, 2048, 1152, 2) if p.spc == 1)
    with pytest.raises(ValueError):
        seg.publishing_twin(cut)


def test_run_plan_exposes_the_four_entry_points_with_the_old_behaviour_by_default():
    import inspect

    import torch

    from isoexec.ops.mm.mm_sm100 import ops

    sig = inspect.signature(ops.run_plan)
    assert sig.parameters["root_dtype"].default is torch.bfloat16
    assert sig.parameters["combine"].default is True
    assert sig.parameters["leaf_dtype"].default is torch.float32
    for name in ("root_dtype", "combine", "leaf_dtype", "out", "workspace", "slots"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    # the binding's two new arguments are declared in the source it is built from
    src = (PKG / "csrc" / "mm_sm100_gemm.cu").read_text()
    assert 'py::arg("combine") = true' in src and 'py::arg("publish_bf16") = false' in src


# ------------------------------------------------------------ the deployed census

def test_every_deployed_census_weight_is_served_unless_the_dispatch_declines_it():
    from isoexec.ops.mm.mm_sm100.tests._helpers import RL_CENSUS

    for N, K, label in RL_CENSUS:
        if seg.supported(K, N):
            assert seg.plan_for(1, K, N, "bf16").split == seg.schedule(K).G, label
        else:
            # mimo.mlp.fc2@tp4, K = 2752: the dispatch granularity (SEGMENT_ALIGN) declines it
            assert (K, N) == (2752, 4096), label
            assert seg.execution_supported(K, N, 1376)   # an execution plan can still serve it


def test_the_model_tile_table_only_names_compiled_whole_tree_plans_that_cover_their_band():
    """Round 8 (README sections 4.31, 4.32): the model tile table is scheduling only."""
    import torch

    inv = {v: k for k, v in seg._CONFIGS.items()}
    assert seg._MODEL_TILES, "the table is empty; nothing is asserted"
    for (N, K), bands in seg._MODEL_TILES.items():
        split = seg.segment_count(K, "bf16")
        prev = 0
        for lo, hi, cfg in bands:
            assert lo > prev and lo <= hi, (N, K, bands)
            prev = hi
            fam, rows, spc, mode = inv[cfg]
            assert (spc == split and mode == seg.ONCHIP) or \
                (split % spc == 0 and split // spc in (2, 4, 8) and mode == seg.COMBINE), (N, K, cfg)
            assert not rows or hi <= rows, (N, K, cfg)
            for M in (lo, hi):
                assert seg.plan_for(M, K, N, torch.bfloat16).config == cfg
                # an explicit override still reaches the rule, never the table
                assert seg.plan_for(M, K, N, torch.bfloat16, spc=split).config in seg._CONFIGS.values()


@pytest.mark.parametrize("supported,has_bias", [(False, False), (True, True)])
def test_installed_provider_refuses_unrecorded_recipes(monkeypatch, supported, has_bias):
    from .. import integration

    monkeypatch.setenv("ISOEXEC_MM_SM100", "1")
    monkeypatch.setattr(integration, "_WRAPPER", None)
    monkeypatch.setattr(integration, "_LOG_ONCE", True)
    monkeypatch.setattr(integration.mm_ops, "require_sm100", lambda: None)
    monkeypatch.setattr(integration.mm_ops, "config_ids", lambda: {})
    monkeypatch.setattr(integration.mm_ops, "supported_shape", lambda a, b: supported)
    monkeypatch.setattr(integration.seam, "install_vendored_mm", lambda: None)
    def unexpected(*args, **kw):
        raise AssertionError("an unsupported shape reached the incumbent")
    monkeypatch.setattr(integration.V, "matmul_persistent", unexpected)
    monkeypatch.setattr(integration.V, "_isoexec_stock_matmul_persistent", unexpected, raising=False)
    assert integration.install_mm_sm100()
    with pytest.raises(integration.mm_ops.UnsupportedShape, match="declined"):
        integration.V.matmul_persistent(object(), object(), bias=object() if has_bias else None)


def test_the_general_kernels_k_schedule_is_the_schedule_policy():
    """``mm_sm100/general`` (the backward layouts): its segment count and length are rule 3 on the constants the
    kernel publishes (min_segment = 8192, max_segments 16 for an MN-major operand; nothing for K-major x
    K-major), and that is the schedule the kernel has always run."""
    from isoexec.ops import schedule_policy as P
    from isoexec.ops.mm.mm_sm100.general import ops as G

    assert (G.SEGMENT_BLOCK, G.SEGMENT_MIN_K, G.MAX_SPLIT, G.SEGMENT_OCCUPANCY) == (64, 8192, 16, 1)
    for K in (64, 4608, 5120, 8192, 16320, 16384, 32768, 65536, 131072, 262144, 100032):
        assert G.schedule(K, False, False) == P.reduce(K, b=64) and G.split_for(K, False, False) == 1
        want = P.reduce(K, b=64, min_segment=8192, p=P.workers(148, 1), max_segments=16)
        for layout in ((True, False), (False, True), (True, True)):
            assert G.schedule(K, *layout) == want and G.split_for(K, *layout) == want.G
        # the rule the kernel ran before the policy: largest power of two <= 16 with K / S >= 8192, ceil-sized
        s = 1
        while s * 2 <= 16 and K // (s * 2) >= 8192:
            s *= 2
        assert (want.G, want.S) == (s, -(-(K // 64) // s) * 64), K
        assert G.plan(256, 512, K, 1, True, False)[0] == want.G
