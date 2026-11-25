"""Every dense GEMM site of the released Qwen3.5 geometries is bitwise invariant to the TP degree.

One B200 plays every rank of TP 1/2/4/8. A column-parallel rank multiplies the row blocks of the
logical weight that placement gives it (``runtimes.parallel.colwise_ranges``), concatenated across
the site's members exactly as the module packs them, through the production ``mm`` binding; its
output must be the TP=1 output's columns, bit for bit. A row-parallel rank runs pik's leaf GEMM on
its K slice, and the rank partials are folded by pik's own tree kernel (``allreduce._tree_reduce``,
the arithmetic of the NCCL transport, which only moves bytes); the root must be the TP=1
``row_parallel_linear`` output. Logical shapes are the models' own; values are random, with
activations spread over 2^16 so that any change of K grouping shows in the bits.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from isoexec.core.arch import live_device_arch  # noqa: E402
from isoexec.ops.mm.mm_sm100.tests import _tp_sites as T  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="the sm100 mm kernel runs on a Blackwell device only")

#: Decode, an odd count, and a prefill-shaped M: the plans (tile family, A height, grid) differ.
MS = (1, 23, 384)

SITES = [s for g in T.GEOMETRIES for s in T.sites(g) if s.kind in ("column", "row")]


def _id(site):
    return f"{site.geometry}:{site.name}"


def _bits(t):
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


def _equal(a, b) -> bool:
    return bool(torch.equal(_bits(a), _bits(b)))


def _activations(M, K, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, K, generator=g, device="cuda")
    x *= 2.0 ** torch.randint(-8, 9, (M, K), generator=g, device="cuda").float()
    return x.bfloat16()


def _weight(shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(shape, generator=g, device="cuda", dtype=torch.bfloat16).mul_(0.05)


@pytest.fixture(scope="module")
def mm():
    from isoexec.ops.mm.mm_sm100 import ops
    from isoexec.ops.mm.resolve import resolve

    ops.require_sm100()
    bound = resolve("mm", site="tests.tp_invariance")
    assert bound.impl == "sm100_tcgen05_segtree"
    return bound.fn


@pytest.mark.parametrize("site", [s for s in SITES if s.kind == "column"], ids=_id)
def test_column_parallel_ranks_hold_the_tp1_columns(mm, site):
    weights = {m.role: _weight(m.shape, seed=17 + i) for i, m in enumerate(site.members)}
    full = weights[site.members[0].role] if len(site.members) == 1 else \
        torch.cat([weights[m.role] for m in site.members])
    offsets, at = [], 0
    for m in site.members:
        offsets.append(at)
        at += m.shape[0]
    for M in MS:
        x = _activations(M, site.K, seed=M)
        ref = mm(x, full)
        for tp in T.TPS[1:]:
            for rank in range(tp):
                blocks, expect = [], []
                for m, off in zip(site.members, offsets):
                    for lo, hi in T.column_ranges(m, tp, rank):
                        blocks.append(weights[m.role][lo:hi])
                        expect.append(ref[:, off + lo:off + hi])
                local = blocks[0] if len(blocks) == 1 else torch.cat(blocks)
                out = mm(x, local)
                want = expect[0] if len(expect) == 1 else torch.cat(expect, dim=1)
                assert _equal(out, want), f"{_id(site)} tp={tp} rank={rank} M={M} N_local={local.shape[0]}"
                del out, local, want
        del x, ref
    del weights, full
    torch.cuda.empty_cache()


@pytest.mark.parametrize("site", [s for s in SITES if s.kind == "row"], ids=_id)
def test_row_parallel_ranks_fold_to_the_tp1_root(site):
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik
    from isoexec.ops.collectives.pik_tp_invariant import get_plan

    ensure_pik()
    from pik.allreduce import _tree_reduce  # type: ignore
    from pik.gemm import ti_gemm  # type: ignore
    from pik.linear import row_parallel_linear  # type: ignore

    plan = get_plan()
    (member,) = site.members
    w = _weight(member.shape, seed=29)
    K = site.K
    for M in MS:
        x = _activations(M, K, seed=1000 + M)
        ref = row_parallel_linear(x, w, None, plan=plan, tp_size=1, tp_rank=0, k_full=K, group=None,
                                  out_dtype=torch.bfloat16)
        for tp in T.TPS[1:]:
            parts = []
            for rank in range(tp):
                lo, hi = T.row_range(site, tp, rank)
                parts.append(ti_gemm(x[:, lo:hi].contiguous(), w[:, lo:hi].contiguous(), plan=plan,
                                     tp_size=tp, tp_rank=rank, k_full=K).float())
            stacked = torch.stack(parts)
            root = _tree_reduce(stacked, torch.empty_like(stacked[0])).to(torch.bfloat16)
            assert _equal(root, ref), f"{_id(site)} tp={tp} M={M}"
            del parts, stacked, root
        del x, ref
    del w


def test_a_k_split_would_be_seen(mm):
    """Non-vacuity: had a rank been given another K segmentation (what the v3 rule did to a narrow
    shard), these activations show it in the bits."""
    from isoexec.ops.mm.mm_sm100 import ops
    from isoexec.ops.mm.mm_sm100 import segments as seg

    K, N = 1024, 2048  # Qwen3.5-0.8B attention q, which v3 split at TP 1 and not at TP 8
    w = _weight((N, K), seed=5)
    x = _activations(23, K, seed=23)
    split = [p for p in seg.plans_at_split(23, K, N, 2) if p.spc == 2][0]
    assert not _equal(ops.run_plan(x, w.t(), split), mm(x, w))
