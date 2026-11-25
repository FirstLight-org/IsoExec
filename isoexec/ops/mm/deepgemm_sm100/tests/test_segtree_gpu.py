"""Bitwise gate of the DeepGEMM segment-tree GEMMs (``segtree``) against the incumbent ``mm_sm100`` programs at
the same segment count: the composition's ``mm`` shapes (compressor, index weights / keys, ``wq_a``, ``wkv``),
``wo_b``'s fixed leaf tree at TP 1 / 2 / 4 / 8 (BF16 and FP32 roots; the cut form and the on-chip leaf-tree
epilogue), over M = 1..2048."""
import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

from isoexec.ops.mm.deepgemm_sm100 import segtree  # noqa: E402
from isoexec.ops.mm.mm_sm100 import segments as seg  # noqa: E402

MS = [1, 2, 3, 7, 8, 9, 16, 31, 64, 100, 127, 128, 129, 255, 256, 511, 1024, 2048]


def _bits(t):
    return t.contiguous().view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


def _incumbent_mm(a, w, root_fp32):
    """The composition's ``mm`` without its segtree rows: the mm_sm100 plans at the published S."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops
    saved = ops.SEGTREE_MM_ROWS, ops.SEGTREE_MM_NARROW_ROWS
    ops.SEGTREE_MM_ROWS = ops.SEGTREE_MM_NARROW_ROWS = 0
    try:
        return ops.mm(a, w, root_fp32=root_fp32)
    finally:
        ops.SEGTREE_MM_ROWS, ops.SEGTREE_MM_NARROW_ROWS = saved


# (N, K): wq_a, wkv, compressor (ratio 4 / 128), indexer weights_proj, indexer wk, the router; hidden 4096 and the
# released / debug checkpoints' 5120
@pytest.mark.parametrize("n,k", [(1280, 4096), (512, 4096), (1024, 4096), (384, 4096), (64, 4096), (128, 4096),
                                 (256, 4096), (1280, 5120), (512, 5120), (1024, 5120), (64, 5120), (256, 5120)])
@pytest.mark.parametrize("root_fp32", [False, True])
def test_segment_tree_is_mm_sm100(n, k, root_fp32):
    s = seg.segment_count(k, torch.bfloat16)
    if not segtree.serves(n, k, s):
        pytest.skip("not a DeepGEMM operand set")
    g = torch.Generator(device="cuda").manual_seed(n + k + s)
    w = (torch.randn(n, k, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    for m in MS:
        x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
        want = _incumbent_mm(x, w, root_fp32)
        got = segtree.linear(x, w, s, out_dtype=torch.float32 if root_fp32 else torch.bfloat16)
        assert torch.equal(_bits(got), _bits(want)), (m, s)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_wo_b_leaves_are_the_on_chip_tree(tp):
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops, spec
    leaves = spec.TP_LEAVES // tp
    n, k = 4096, 8192 // tp
    g = torch.Generator(device="cuda").manual_seed(tp)
    w = (torch.randn(n, k, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    saved = ops.SEGTREE_WOB_ROWS, ops.ONCHIP_WOB_ROWS
    for m in MS:
        y = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
        ops.SEGTREE_WOB_ROWS = ops.ONCHIP_WOB_ROWS = 0                # the mm_sm100 on-chip tree
        try:
            want = ops._wo_b_forward(w, y, tp)
        finally:
            ops.SEGTREE_WOB_ROWS, ops.ONCHIP_WOB_ROWS = saved
        out_dtype = torch.float32 if tp > 1 else torch.bfloat16
        assert torch.equal(_bits(segtree.linear(y, w, leaves, out_dtype=out_dtype)), _bits(want)), m
        assert torch.equal(_bits(segtree.leaf_linear(y, w, leaves, out_dtype=out_dtype)), _bits(want)), m
        assert torch.equal(_bits(ops._wo_b_forward(w, y, tp)), _bits(want)), m      # the selected program


def _hard_inputs(m, k, g):
    """Rows with signed zeros, cancelling pairs and large magnitudes beside ordinary values."""
    y = torch.randn(m, k, generator=g, device="cuda")
    y[:, :64] = 0.0
    y[:, 64:128] = -0.0
    y[0::3, 128:256] *= 1e4
    y[1::3, 256:384] = -y[1::3, 384:512]
    return y.to(torch.bfloat16)


@pytest.mark.parametrize("n", [5120, 640])
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_progressive_leaf_fold_is_the_tree(n, tp):
    """The on-chip leaf tree with the progressive fold (pair nodes + binary counter) == the leaves computed one CTA each
    and folded by leaf_tree after the fact, at the wo_b shapes (TP 1 / 2 / 4 / 8 rank subtrees: L = 8 / 4 / 2 / 1 leaves,
    BF16 root at TP 1, FP32 subtree root otherwise), M up to 8192, hard inputs; and the TP relation: the rank subtree
    roots folded by leaf_tree in rank order == the TP 1 root."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops, spec
    leaves, K = spec.TP_LEAVES // tp, 8192
    k = K // tp
    g = torch.Generator(device="cuda").manual_seed(100 * tp + n)
    w = (torch.randn(n, K, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    out_dtype = torch.float32 if tp > 1 else torch.bfloat16
    for m in (1, 15, 16, 17, 255, 256, 257, 512, 1000, 4096, 8192):
        y = _hard_inputs(m, K, g)
        got = segtree.leaf_linear(y[:, :k].contiguous(), w[:, :k].contiguous(), leaves, out_dtype=out_dtype)
        want = segtree.linear(y[:, :k].contiguous(), w[:, :k].contiguous(), leaves, out_dtype=out_dtype)
        assert torch.equal(_bits(got), _bits(want)), (m, tp)
        if tp > 1:
            roots = torch.stack([segtree.leaf_linear(y[:, r * k:(r + 1) * k].contiguous(),
                                                     w[:, r * k:(r + 1) * k].contiguous(), leaves,
                                                     out_dtype=torch.float32) for r in range(tp)])
            folded = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
            ops.extension().leaf_tree(roots.contiguous(), folded)
            full = segtree.leaf_linear(y, w, spec.TP_LEAVES, out_dtype=torch.bfloat16)
            assert torch.equal(_bits(folded), _bits(full)), (m, tp)


@pytest.mark.parametrize("n", [5120, 640])
@pytest.mark.parametrize("tp", [1, 2, 4])
def test_register_partials_are_the_tree(n, tp, monkeypatch):
    """The leaf tree with the counter partials in the registers of two epilogue warpgroups (segtree.LEAF_REGS) ==
    the leaves computed one CTA each and folded by leaf_tree, at the wo_b shapes (L = 8 / 4 / 2), M up to 8192, hard
    inputs, BF16 and FP32 roots: the partials' storage location changes no tree node."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import spec
    leaves, K = spec.TP_LEAVES // tp, 8192
    k = K // tp
    g = torch.Generator(device="cuda").manual_seed(300 * tp + n)
    w = (torch.randn(n, k, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    for out_dtype in (torch.bfloat16, torch.float32):
        for m in (1, 15, 16, 17, 255, 256, 257, 512, 1000, 4096, 8192):
            y = _hard_inputs(m, K, g)[:, :k].contiguous()
            want = segtree.linear(y, w, leaves, out_dtype=out_dtype)
            monkeypatch.setattr(segtree, "LEAF_REGS", True)
            got = segtree.leaf_linear(y, w, leaves, out_dtype=out_dtype)
            monkeypatch.setattr(segtree, "LEAF_REGS", False)
            assert torch.equal(_bits(got), _bits(want)), (m, tp, out_dtype)
