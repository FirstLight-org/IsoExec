"""Require legal narrow-leaf schedules with BLOCK_K dividing LEAF_K, including cached and fallback configs.
A larger BLOCK_K makes the floor-divided reduction loop execute zero times, so unsupported Triton leaves must refuse."""

from __future__ import annotations

import importlib
import json
import pathlib
import sys
import types

import pytest

triton = pytest.importorskip("triton")

_PIK = pathlib.Path(__file__).resolve().parents[1] / "pik"


def _mod(name: str):
    """Import a pik submodule without running ``pik/__init__.py``, which asks for a CUDA device.

    A namespace-package stub whose ``__path__`` is the vendored pik directory is registered as ``pik``
    first, so the submodules' relative imports resolve normally and nothing touches the GPU.
    """
    if "pik" not in sys.modules or not hasattr(sys.modules["pik"], "__path__"):
        stub = types.ModuleType("pik")
        stub.__path__ = [str(_PIK)]  # type: ignore[attr-defined]
        sys.modules.setdefault("pik", stub)
    # gemm.py builds its autotune space AT IMPORT (`@triton.autotune(configs=_configs())`), and
    # `_configs` asks `arch.current()` which device this is. Pin it to Hopper for the CPU run; every
    # assertion below feeds the arch's config LIST in explicitly, so the pin decides nothing.
    a = importlib.import_module("pik.arch")
    a.current = lambda device=None: a.HOPPER  # type: ignore[assignment]
    return importlib.import_module(f"pik.{name}")


def _inject_exact(monkeypatch, gemm, available: bool):
    """Stand in for the exact leaf producer (``sm90seg``): available or not, without a GPU."""
    monkeypatch.setattr(gemm, "_sm90seg_ok", lambda m_leaves, leaf_k: available)


@pytest.fixture(scope="module")
def gemm():
    return _mod("gemm")


@pytest.fixture(scope="module")
def arch():
    return _mod("arch")


def _space(arch_configs):
    """The autotune space as ``_configs()`` builds it, without asking the GPU which arch we are on."""
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8}, num_warps=w, num_stages=s)
        for bm, bn, bk, w, s in arch_configs
    ]


# Every leaf width a shipped ReductionPlan can produce: K_full / G for the row-parallel sites of the
# tiny (H=256), 9B (H=4096) and 35B (H=2048) models at G in {8, 16}, plus the degenerate narrow ones.
_LEAF_WIDTHS = [4, 8, 16, 32, 64, 128, 256, 512, 768, 1536]


@pytest.mark.parametrize("arch_name", ["hopper", "blackwell"])
@pytest.mark.parametrize("leaf_k", _LEAF_WIDTHS)
def test_prune_never_returns_a_block_k_that_does_not_tile_the_leaf(gemm, arch, arch_name, leaf_k):
    space = _space(arch._HOPPER_CONFIGS if arch_name == "hopper" else arch._BLACKWELL_CONFIGS)
    if not gemm.triton_leaf_ok(leaf_k):
        with pytest.raises(ValueError, match="no legal BLOCK_K"):
            gemm._prune(space, {"LEAF_K": leaf_k})
        return
    out = gemm._prune(space, {"LEAF_K": leaf_k})
    assert out, f"LEAF_K={leaf_k} on {arch_name}: pruner returned no config at all"
    for c in out:
        bk = c.kwargs["BLOCK_K"]
        # The load-bearing assertion: LEAF_K // BLOCK_K must be at least one k-loop trip.
        assert leaf_k % bk == 0, f"{arch_name} LEAF_K={leaf_k}: BLOCK_K={bk} does not tile the leaf"
        assert leaf_k // bk >= 1, f"{arch_name} LEAF_K={leaf_k}: BLOCK_K={bk} gives a ZERO-trip k-loop"


def test_the_narrow_leaves_are_exactly_the_ones_that_used_to_be_silently_zero(gemm, arch):
    """Regression witness: both arch spaces start at BLOCK_K=64, so LEAF_K in {8, 16} reached the
    fallback and got BLOCK_K=32 -> 16 // 32 == 0 trips. 32 reached it too and was fine (1 trip)."""
    for space in (arch._HOPPER_CONFIGS, arch._BLACKWELL_CONFIGS):
        assert min(bk for _, _, bk, _, _ in space) >= 64
    assert 16 // 32 == 0  # the arithmetic that produced the zeros
    assert gemm._fallback_config(32).kwargs["BLOCK_K"] == 32  # unchanged: no bits move at LEAF_K=32
    assert gemm._fallback_config(16).kwargs["BLOCK_K"] == 16  # repaired: one trip instead of zero
    assert gemm.triton_leaf_ok(16) and gemm.triton_leaf_ok(32)
    assert not gemm.triton_leaf_ok(8) and not gemm.triton_leaf_ok(4)


def test_sched_candidates_offers_no_triton_schedule_at_a_leaf_with_no_legal_block_k(gemm, monkeypatch):
    _inject_exact(monkeypatch, gemm, True)
    narrow = gemm._sched_candidates(8, 8)
    assert narrow == ["sm90seg"], narrow
    wide = gemm._sched_candidates(8, 64)
    assert "sm90seg" in wide and "fused" in wide and "split8" in wide  # s == m is full split-K

    _inject_exact(monkeypatch, gemm, False)
    with pytest.raises(ValueError, match="no Triton schedule"):
        gemm._sched_candidates(8, 8)


def test_bench_fallback_stays_inside_the_candidate_list(gemm):
    """A schedule that computes nothing is the FASTEST; the picker must never reach one that
    ``_sched_candidates`` deliberately withheld. The old ``best or "splitk"`` did exactly that."""
    src = pathlib.Path(gemm.__file__).read_text()
    assert "return best or cands[0]" in src
    assert 'return best or "splitk"' not in src


def test_autotune_cache_drops_entries_an_older_pruner_wrote():
    ac = _mod("autotune_cache")
    poisoned = {"kwargs": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, "num_warps": 4, "num_stages": 3}
    legal = {"kwargs": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, "num_warps": 4, "num_stages": 3}
    assert not ac._cfg_legal((230, 256, 16, 8), poisoned)  # 16 % 32 -> zero-trip
    assert not ac._cfg_legal((230, 256, 8, 2), poisoned)
    assert ac._cfg_legal((230, 256, 32, 8), poisoned)  # 32 % 32 == 0
    assert ac._cfg_legal((230, 256, 512, 8), legal)
    assert ac._cfg_legal(("not", "a", "leaf", "key"), legal)  # other kernels pass through
    assert ac._cfg_legal((1, 2), {"kwargs": {}, "num_warps": 4, "num_stages": 3})


# CPU stubs test schedule admission so a fast but incorrect zero-trip kernel cannot win tuning.

torch = pytest.importorskip("torch")


def _fake_dispatch(table):
    """A `_dispatch` stand-in that writes a per-schedule answer supplied by the test."""

    def _d(x, w, out, leaf_k, m_leaves, sched, leaf_dtype=None):
        v = table[sched]
        if isinstance(v, Exception):
            raise v
        out.copy_(v)
        return out

    return _d


def _operands(m=4, n=3, leaf_k=16, m_leaves=2):
    torch.manual_seed(0)
    x = torch.randn(m, leaf_k * m_leaves)
    w = torch.randn(n, leaf_k * m_leaves)
    return x, w, torch.zeros(m, n)


def test_a_schedule_that_computes_nothing_is_rejected_before_it_can_win_the_bench(gemm, monkeypatch):
    """The whole point: exact zeros lose, even though they would win every wall-clock bench."""
    _inject_exact(monkeypatch, gemm, True)
    x, w, out = _operands()
    truth = torch.randn_like(out)
    monkeypatch.setattr(
        gemm,
        "_dispatch",
        _fake_dispatch({"sm90seg": truth, "fused": torch.zeros_like(out), "split1": truth, "split2": truth}),
    )
    admitted = gemm._admit_schedules(x, w, out, 16, 2, ["sm90seg", "fused", "split1", "split2"], torch.float32)
    assert admitted == ["sm90seg", "split1", "split2"]  # order preserved, only the zero producer dropped
    assert torch.equal(out, torch.zeros(4, 3))  # `out` is scratch-free: the admission never writes it


def test_when_every_candidate_agrees_the_candidate_list_is_returned_unchanged(gemm, monkeypatch):
    """Bit-neutrality, stated as a test: on a shape where the schedules already agree -- which is every
    shape that works today -- the bench receives exactly the list it received before this check existed,
    in the same order, so it makes exactly the same pick and the op returns exactly the same bits."""
    _inject_exact(monkeypatch, gemm, True)
    x, w, out = _operands()
    truth = torch.randn_like(out)
    cands = ["sm90seg", "fused", "split1", "split2", "split4"]
    monkeypatch.setattr(gemm, "_dispatch", _fake_dispatch({c: truth for c in cands}))
    assert gemm._admit_schedules(x, w, out, 64, 4, list(cands), torch.float32) == cands


def test_one_bit_of_difference_is_a_rejection_when_the_exact_producer_is_the_reference(gemm, monkeypatch):
    _inject_exact(monkeypatch, gemm, True)
    x, w, out = _operands()
    truth = torch.randn_like(out)
    off = truth.clone()
    off.view(-1)[0] = torch.tensor(off.view(-1)[0].item()).view(torch.int32).add(1).view(torch.float32)
    monkeypatch.setattr(gemm, "_dispatch", _fake_dispatch({"sm90seg": truth, "fused": off}))
    assert gemm._admit_schedules(x, w, out, 64, 2, ["sm90seg", "fused"], torch.float32) == ["sm90seg"]


def test_a_schedule_that_raises_is_a_rejection_not_a_silent_skip(gemm, monkeypatch):
    _inject_exact(monkeypatch, gemm, True)
    x, w, out = _operands()
    truth = torch.randn_like(out)
    monkeypatch.setattr(gemm, "_dispatch", _fake_dispatch({"sm90seg": truth, "fused": RuntimeError("no such kernel")}))
    assert gemm._admit_schedules(x, w, out, 64, 2, ["sm90seg", "fused"], torch.float32) == ["sm90seg"]


def test_no_candidate_agreeing_raises_rather_than_picking_one_by_wall_clock(gemm, monkeypatch):
    _inject_exact(monkeypatch, gemm, False)
    x, w, out = _operands()
    monkeypatch.setattr(
        gemm, "_dispatch", _fake_dispatch({"fused": torch.zeros_like(out), "split1": torch.zeros_like(out)})
    )
    with pytest.raises(ValueError, match="no schedule agrees"):
        gemm._admit_schedules(x, w, out, 16, 2, ["fused", "split1"], torch.float32)


def test_without_the_exact_producer_the_eager_leaf_tree_is_the_reference_and_still_rejects_zeros(gemm, monkeypatch):
    """No exact producer means no bitwise reference exists, so the fallback is numeric -- loose enough to
    forgive backend rounding, decisive against a kernel that computes nothing (relative error 1.0)."""
    _inject_exact(monkeypatch, gemm, False)
    x, w, out = _operands(leaf_k=16, m_leaves=2)
    eager = gemm._eager_leaf_reference(x, w, 16, 2, torch.float32)
    monkeypatch.setattr(
        gemm,
        "_dispatch",
        _fake_dispatch({"fused": eager + 1e-6, "split1": torch.zeros_like(out), "split2": eager}),
    )
    assert gemm._admit_schedules(x, w, out, 16, 2, ["fused", "split1", "split2"], torch.float32) == [
        "fused",
        "split2",
    ]


def test_the_eager_reference_is_the_leaf_tree_with_the_plan_s_rounding_point(gemm):
    x, w, _ = _operands(m=5, n=7, leaf_k=8, m_leaves=4)
    got = gemm._eager_leaf_reference(x, w, 8, 4, torch.float32)
    lv = [x[:, j * 8 : (j + 1) * 8].float() @ w[:, j * 8 : (j + 1) * 8].float().T for j in range(4)]
    assert torch.equal(got, (lv[0] + lv[1]) + (lv[2] + lv[3]))  # balanced binary tree, not a running sum
    # bf16 leaves: the leaf is the rounding point the plan pins; the tree above it stays fp32
    bf = gemm._eager_leaf_reference(x, w, 8, 4, torch.bfloat16)
    lb = [t.bfloat16().float() for t in lv]
    assert torch.equal(bf, (lb[0] + lb[1]) + (lb[2] + lb[3]))
    assert not torch.equal(bf, got)


def test_pick_schedule_admits_before_it_benches(gemm, monkeypatch):
    """Order matters: admission must gate the bench, not filter its winner afterwards."""
    _inject_exact(monkeypatch, gemm, True)
    x, w, out = _operands()
    calls = []
    monkeypatch.setattr(gemm, "_SCHED_CACHE", {})
    monkeypatch.setattr(gemm, "_sched_candidates", lambda m, lk: ["sm90seg", "fused"])
    monkeypatch.setattr(
        gemm,
        "_admit_schedules",
        lambda *a, **k: (calls.append("admit"), ["sm90seg"])[1],
    )
    monkeypatch.setattr(
        gemm, "_bench_schedules", lambda x, w, o, lk, m, c, ld: (calls.append(("bench", tuple(c))), c[0])[1]
    )
    assert gemm._pick_schedule(x, w, out, 16, 2) == "sm90seg"
    assert calls == ["admit", ("bench", ("sm90seg",))]

    # m == 1 degenerates to a single sm90 segment-tree GEMM: it IS the reference, so there is nothing to
    # admit and nothing to time.
    calls.clear()
    monkeypatch.setattr(gemm, "_SCHED_CACHE", {})
    assert gemm._pick_schedule(x, w, out, 16, 1) == "sm90seg"
    assert calls == []


def test_benchmark_preserves_bf16_leaf_plan_with_fp32_output(gemm, monkeypatch):
    """A BF16 leaf tree has FP32 internal nodes; benchmark the same arithmetic as execution."""
    import torch
    import triton.testing

    x, w, out = _operands()
    assert out.dtype == torch.float32
    dispatched = []
    monkeypatch.setattr(gemm, "_dispatch", lambda *args: dispatched.append(args[-1]))

    def bench(fn, **kwargs):
        fn()
        return 1.0

    monkeypatch.setattr(triton.testing, "do_bench", bench)
    assert gemm._bench_schedules(x, w, out, 16, 2, ["fused"], torch.bfloat16) == "fused"
    assert dispatched == [torch.bfloat16]


def test_schedule_reuse_isolated_by_device_and_leaf_plan(gemm, monkeypatch):
    """Same shape on another device or rounding plan must pass its own admission and timing."""
    import torch

    x, w, out = _operands()
    monkeypatch.setattr(gemm, "_SCHED_CACHE", {})
    monkeypatch.setattr(gemm, "_sched_candidates", lambda m, lk: ["fused"])
    admitted, timed = [], []

    def admit(x, w, out, lk, m, candidates, dtype):
        admitted.append((x.device, dtype))
        return candidates

    def bench(x, w, out, lk, m, candidates, dtype):
        timed.append((x.device, dtype))
        return candidates[0]

    monkeypatch.setattr(gemm, "_admit_schedules", admit)
    monkeypatch.setattr(gemm, "_bench_schedules", bench)
    for device in (torch.device("cuda", 0), torch.device("cuda", 1)):
        # Geometry-only operands avoid allocating CUDA memory in a CPU regression.
        operand = types.SimpleNamespace(shape=x.shape, dtype=x.dtype, device=device)
        for dtype in (torch.bfloat16, torch.float32):
            for _ in range(2):
                assert gemm._pick_schedule(operand, w, out, 16, 2, dtype) == "fused"
    expected = [(torch.device("cuda", i), dtype) for i in (0, 1) for dtype in (torch.bfloat16, torch.float32)]
    assert admitted == expected
    assert timed == expected


# Disk configs bypass early_config_prune; validate the complete admission rule on cache load and repair stale entries.


def _fresh_cache(tmp_path, monkeypatch, payload):
    ac = _mod("autotune_cache")
    f = tmp_path / "autotune.json"
    f.write_text(json.dumps(payload))
    monkeypatch.setattr(ac, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(ac, "_CACHE_FILE", f)
    monkeypatch.setattr(ac, "_LOADED", {})
    monkeypatch.setattr(ac, "_dropped_on_read", 0)
    monkeypatch.setattr(ac, "_REGISTRY", {})
    return ac, f


_POISONED = {"kwargs": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, "num_warps": 4, "num_stages": 3}
_LEGAL = {"kwargs": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, "num_warps": 4, "num_stages": 3}
_SUBMIN = {"kwargs": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 8, "GROUP_M": 8}, "num_warps": 4, "num_stages": 3}


def test_min_block_k_does_not_drift_from_the_kernels(gemm):
    """`autotune_cache` cannot import `gemm` (gemm imports IT, and calls `register` from its own
    decorators while half-initialized), so the constant is mirrored. This is the pin that keeps the
    mirror honest."""
    ac = _mod("autotune_cache")
    assert ac._MIN_BLOCK_K == gemm.MIN_BLOCK_K


def test_cfg_legal_is_the_pruners_whole_admission_not_only_divisibility():
    ac = _mod("autotune_cache")
    assert not ac._cfg_legal((230, 256, 16, 8), _POISONED)  # 16 % 32 -> zero trips
    assert ac._cfg_legal((230, 256, 32, 8), _POISONED)  # 32 % 32 == 1 trip, unchanged
    # divides exactly, and is still a config no pruner here would emit: tl.dot will not take BLOCK_K < 16
    assert not ac._cfg_legal((230, 256, 8, 2), _SUBMIN)
    assert not ac._cfg_legal((230, 256, 8, 2), _LEGAL)  # the LEAF itself is below MIN_BLOCK_K
    assert ac._cfg_legal((230, 256, 512, 8), _LEGAL)


def test_the_filter_is_applied_where_the_file_enters_the_process(tmp_path, monkeypatch):
    """Not at each call site: a reader added later must not be able to forget to ask."""
    ac, _ = _fresh_cache(
        tmp_path,
        monkeypatch,
        {
            "_fused_tree_m8": {"[230, 256, 16, 8]": _POISONED, "[230, 256, 512, 8]": _LEGAL},
            "_split_tree_g2": {"[230, 256, 32, 8]": _POISONED},
        },
    )
    got = ac._read()
    assert got == {
        "_fused_tree_m8": {"[230, 256, 512, 8]": _LEGAL},
        "_split_tree_g2": {"[230, 256, 32, 8]": _POISONED},  # legal at LEAF_K=32; nothing moves there
    }
    assert ac._dropped_on_read == 1


def test_a_poisoned_file_is_repaired_on_disk_even_when_nothing_new_was_tuned(tmp_path, monkeypatch):
    """`save` used to re-filter what `_read` handed it, so the repair rode on that pass. With the filter
    moved to `_read`, `save` has to be told the file needed repairing -- otherwise a process that tunes
    nothing writes nothing and the poison outlives it."""
    ac, f = _fresh_cache(tmp_path, monkeypatch, {"_fused_tree_m8": {"[230, 256, 16, 8]": _POISONED}})
    ac.save()
    assert json.loads(f.read_text()) == {"_fused_tree_m8": {}}


def test_a_clean_file_is_left_alone(tmp_path, monkeypatch):
    ac, f = _fresh_cache(tmp_path, monkeypatch, {"_fused_tree_m8": {"[230, 256, 512, 8]": _LEGAL}})
    before = f.stat().st_mtime_ns
    ac.save()
    assert f.stat().st_mtime_ns == before  # `changed` stayed False; no rewrite, no atomic replace


def test_register_seeds_only_what_the_current_pruner_would_have_produced(tmp_path, monkeypatch):
    ac, _ = _fresh_cache(
        tmp_path,
        monkeypatch,
        {"k": {"[230, 256, 16, 8]": _POISONED, "[230, 256, 128, 8]": _LEGAL}},
    )
    monkeypatch.setattr(ac, "_BUCKET_FLOOR", 1)

    class _Autotuner(triton.runtime.Autotuner):
        def __init__(self):  # a bare shell; only `.cache` is exercised
            self.cache = {}

    kern = ac.register("k", _Autotuner())
    assert [k for k in kern.cache] == [(256, 256, 128, 8)]
