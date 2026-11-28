"""CPU contract: the declarations, the dispatch wiring and the numerical structure this package owns.

Nothing here builds a kernel or touches a device. What it pins:

* the recipe's declarations, its twin class and the batteries that prove it;
* the dispatch row selecting this package on SM100 and the trainer core it resolves to;
* the recipe's fixed SEG (the schedule policy's; test_schedule_policy_cpu.py pins how it is derived);
* the canonical adjacent-pair tree and ``node_size_at``, against an independent recomputation, so a
  change to the tree shape fails here before it reaches a GPU;
* that importing the package builds nothing and loads no vendored backward.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from isoexec.ops.attention import gqa_sm100
from isoexec.ops.attention.gqa_sm100 import ops, reference, spec


def test_the_package_declares_both_cores_in_one_twin_class():
    from isoexec.core.registry import TWIN_PROOFS, Registry, runnable_batteries

    reg = Registry.from_impls(spec.IMPLS)
    assert set(reg.ops) == {"attention.core.gqa"}
    impls = reg.ops["attention.core.gqa"].impls
    assert set(impls) == {spec.TRAINER_IMPL, spec.PAGED_IMPL}
    for impl in impls.values():
        assert impl.version == spec.RECIPE_VERSION == 3
        assert impl.supported_archs == frozenset({"sm100"})
        assert impl.twin_class == spec.TWIN_CLASS
        assert not impl.inputs  # SEG is the recipe's (spec's declarations), not a caller's
    root = Path(__file__).resolve().parents[5]
    batteries = runnable_batteries(spec.TWIN_CLASS)
    assert all((root / p).is_file() for p in batteries), TWIN_PROOFS[spec.TWIN_CLASS]
    assert "isoexec/ops/attention/gqa_sm100/tests/test_seam_gpu.py" in batteries
    assert spec.RECIPE == "sm100_gqa_segtree_v2" and spec.ARCH == "sm100"


def test_the_dispatch_seam_selects_this_package_on_sm100():
    from isoexec.ops import _dispatch

    fam = _dispatch.FAMILIES["gqa"]
    assert fam.archs["sm100"] == "attention.gqa_sm100"


def test_resolution_binds_the_trainer_core_on_sm100(monkeypatch):
    from isoexec.core.registry_build import build_registry
    from isoexec.ops import _dispatch
    from isoexec.ops._resolve import resolve
    from isoexec.ops.attention.gqa_sm100 import integration

    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    assert set(build_registry().get_op("attention.core.gqa").impls) == {spec.TRAINER_IMPL, spec.PAGED_IMPL}
    bound = resolve("attention.core.gqa", site="t")
    assert (bound.impl, bound.version, bound.provider) == (spec.TRAINER_IMPL, 3, "ops.attention.gqa_sm100")
    assert bound.fn is integration.trainer_core
    assert integration.PAGED_IMPL == spec.PAGED_IMPL


def test_the_engine_cache_contract():
    from isoexec.ops.attention.gqa_sm100 import integration

    assert integration.PAGE_MULTIPLE == spec.PAGE_MULTIPLE == ops.KEY_TILE == 128
    assert integration.KV_CACHE_LAYOUTS == ("LBNHC",)


def test_the_recipe_reads_one_seg_per_head_dim():
    assert spec.HEAD_DIMS == ops.HEAD_DIMS == (128, 256)
    assert {d: ops.segment_span(d) for d in spec.HEAD_DIMS} == {128: 10240, 256: 10240}
    assert ops.segment_span(256) == ops.seg_for(256, spec.LONGEST_CONTEXT)


def test_arch_and_tile_width():
    assert ops.KEY_TILE == 128
    assert gqa_sm100.__doc__


def test_leaf_choices_mirror_the_kernel():
    import re

    params = (Path(ops.__file__).parent / "csrc" / "params.h").read_text()
    kernel = {name: int(value) for name, value in re.findall(r"\b(LEAF_\w+|DECODE_ROWS) = (\d+)", params)}
    assert kernel == {"LEAF_AUTO": ops.LEAF_AUTO, "LEAF_GENERAL": ops.LEAF_GENERAL, "LEAF_DECODE": ops.LEAF_DECODE,
                      "DECODE_ROWS": ops.DECODE_ROWS}


def test_segment_bounds_are_anchored_at_key_zero():
    seg = 512
    for border in (1, 511, 512, 513, 1024, 2048, 2049):
        bounds = reference.segment_bounds(border, seg)
        assert bounds[0][0] == 0, "the first segment starts at key 0, never at a chunk offset"
        assert bounds[-1][1] == border
        for k, (lo, hi) in enumerate(bounds):
            assert lo == k * seg
            assert hi == min((k + 1) * seg, border)
        assert len(bounds) == ops.segment_count(border, seg) == reference.segment_count(border, seg)


def _independent_tree(n: int) -> list[tuple[int, int, int]]:
    """The adjacent-pair tree built top-down, to check the bottom-up definition against."""
    merges: list[tuple[int, int, int]] = []

    def build(lo: int, size: int):
        if size == 1:
            return
        half = size // 2
        if lo + half >= n:
            # the right child is empty: the left subtree is carried up verbatim
            build(lo, half)
            return
        build(lo, half)
        build(lo + half, half)
        merges.append((lo, lo + half, half))

    full = 1
    while full < n:
        full *= 2
    build(0, full)
    return merges


@pytest.mark.parametrize("n", list(range(1, 40)) + [64, 100, 128, 255, 256])
def test_canonical_tree_matches_an_independent_construction(n):
    got = reference.tree_merges(n)
    want = _independent_tree(n)
    assert sorted(got) == sorted(want), n
    assert len(got) == n - 1, "a tree over n leaves performs exactly n-1 merges"
    ids = [right - 1 for _, right, _ in got]
    assert len(set(ids)) == len(ids), n


@pytest.mark.parametrize("n", [1, 2, 3, 5, 6, 7, 8, 13, 16, 17, 31, 64])
@pytest.mark.parametrize("run", [1, 2, 4, 8])
def test_node_size_at_only_ever_names_nodes_of_the_canonical_tree(n, run):
    nodes = []
    k = 0
    while k < n:
        z = reference.node_size_at(k, n, run)
        assert z >= 1 and k + z <= n
        nodes.append((k, z))
        k += z
    assert sum(z for _, z in nodes) == n, "the published nodes tile the row's leaves exactly"
    for k, z in nodes:
        if z > 1:
            assert (z & (z - 1)) == 0, (n, run, k, z)
            assert k % z == 0, (n, run, k, z)
    tail = [z for _, z in nodes if z != run]
    assert tail == sorted(tail, reverse=True), (n, run, nodes)


def test_default_scale_is_one_over_root_head_dim():
    for d in (128, 256):
        assert ops.default_scale(d) == 1.0 / math.sqrt(d)


def test_seg_tokens_must_be_a_multiple_of_the_tile_width_at_the_call():
    import torch

    with pytest.raises(ValueError):
        ops.gqa_sm100_fwd(torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0),
                          torch.empty(0), softmax_scale=1.0, seg_tokens=100, max_seqlen_q=1,
                          max_seqlen_k=1)


def test_importing_the_package_builds_nothing_and_loads_no_backward(monkeypatch):
    import importlib

    import torch.utils.cpp_extension as cpp

    monkeypatch.setattr(cpp, "load", lambda *a, **k: pytest.fail("import must not build"))
    for name in [m for m in sys.modules if m.startswith(("isoexec.ops.attention.gqa_sm100", "isoexec.ops._vendor"))]:
        monkeypatch.delitem(sys.modules, name)
    pkg = importlib.import_module("isoexec.ops.attention.gqa_sm100")
    importlib.import_module("isoexec.ops.attention.gqa_sm100.integration")
    assert callable(pkg.varlen_attention)
    assert not [m for m in sys.modules if m.startswith("isoexec.ops._vendor")]
    with pytest.raises(AttributeError):
        _ = pkg.not_an_export


def test_install_refuses_a_worker_without_an_sm100_device(monkeypatch):
    import torch

    from isoexec.ops.attention.gqa_sm100 import integration

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires an SM100 CUDA device"):
        integration.install()


def test_the_backward_is_the_vendored_fa4_and_not_an_installed_one():
    import ast

    tree = ast.parse((Path(__file__).resolve().parents[1] / "ops.py").read_text())
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "isoexec.ops._vendor.flash_attn_cute.interface_bwd_sm100" in modules
    assert not [m for m in modules if m.split(".")[0] in ("flash_attn", "quack")]
    assert spec.FA4_COMMIT == "0bbb25a3a5ad3c58c029b3d287d6c9af56a5cad5"


def test_a_cute_dsl_install_that_mixes_compiler_builds_is_refused(monkeypatch, tmp_path):
    import base64
    import hashlib
    from importlib import metadata
    from types import SimpleNamespace

    def record(path, data):
        path.write_bytes(data)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        return SimpleNamespace(hash=SimpleNamespace(mode="sha256", value=digest), __str__=lambda self=None: path.name)

    good = record(tmp_path / "a.py", b"cu13")
    other = record(tmp_path / "b.py", b"cu13 too")
    dist = SimpleNamespace(version="4.5.2", files=[good, other], locate_file=lambda f: tmp_path / ("a.py" if f is good else "b.py"))

    def distribution(name):
        if name == "nvidia-cutlass-dsl-libs-cu13":
            return dist
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", distribution)
    ops.require_consistent_cute_dsl.cache_clear()
    try:
        assert ops.require_consistent_cute_dsl() == "nvidia-cutlass-dsl-libs-cu13==4.5.2"
        ops.require_consistent_cute_dsl.cache_clear()
        (tmp_path / "b.py").write_bytes(b"written by the base build")
        with pytest.raises(RuntimeError, match="mixes compiler builds"):
            ops.require_consistent_cute_dsl()
    finally:
        ops.require_consistent_cute_dsl.cache_clear()
