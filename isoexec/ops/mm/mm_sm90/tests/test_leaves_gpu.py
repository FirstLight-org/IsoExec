"""pik leaf mode of the sm90 segment-tree GEMM (the producer for pik's LEAF and NODE wires).

A leaf is one segment's ascending wgmma chain over its leaf_k slice, rounded to bf16 (pik's bf16-leaf
contract) or kept fp32; the node is the balanced tree over a rank's m leaves in fp32. Checked against the
reference program (independent Triton chain), against pik's own leaf semantics (the pik Triton leaf kernel), for every configuration, M and m, and across
tensor-parallel degrees: the rank nodes of TP 2/4/8 combine to the TP 1 node bit for bit.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from isoexec.core.arch import live_device_arch  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or live_device_arch() != "sm90",
    reason="the sm90 mm kernel runs on a Hopper device only",
)

G = 8  # pik's leaf count (ISOEXEC_PIK_LEAVES)


def _bits(t):
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


@pytest.fixture(scope="module")
def mm():
    from isoexec.ops.mm.mm_sm90 import ops

    ops.require_sm90()
    return ops


def operands(M, N, K, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, K, generator=g, device="cuda").bfloat16()
    w = (torch.randn(N, K, generator=g, device="cuda")
         * torch.exp(torch.randn(N, K, generator=g, device="cuda") * 2) * 0.02).bfloat16()
    return x, w


def ref_leaves(x, w, leaf_k):
    from isoexec.ops.mm.mm_sm90 import reference as R

    m = x.shape[1] // leaf_k
    return [R.chain_leaf_triton(x[:, j * leaf_k:(j + 1) * leaf_k], w[:, j * leaf_k:(j + 1) * leaf_k])
            for j in range(m)]


# (K_local, N, leaf_k): Qwen3.8-27B o_proj / gdn out and fc2 at TP4 and TP1, plus small odd ones
SHAPES = [(1536, 5120, 768), (4352, 5120, 2176), (6144, 5120, 768), (17408, 5120, 2176), (512, 72, 64),
          (1024, 136, 256),
          # leaves that are not whole 64-wide K blocks (zero-padded tails): MiMo-shaped 1376, the tiny
          # oracle model's narrow leaves at G = 16
          (2752, 512, 1376), (64, 256, 8), (128, 128, 16), (96, 64, 48)]
MS = (1, 5, 32, 64, 129, 320, 1024)


@pytest.mark.parametrize("K,N,leaf_k", SHAPES)
def test_leaf_wire_every_config_every_M(mm, K, N, leaf_k):
    from isoexec.ops.mm.mm_sm90 import segments as seg

    m = K // leaf_k
    if m > 2 and K > 8192:
        ms = (1, 64, 320)
    else:
        ms = MS
    x, w = operands(max(ms), N, K, seed=K + N)
    want = torch.stack(ref_leaves(x, w, leaf_k)).to(torch.bfloat16)
    for M in ms:
        for name in seg.configs(out_fp32=0, exact_levels=0):
            out = torch.full((m, M, N), float("nan"), dtype=torch.bfloat16, device="cuda")
            mm.leaves(x[:M], w, leaf_k, out, config=name)
            assert torch.equal(_bits(out), _bits(want[:, :M])), (M, name)
        out = torch.empty((m, M, N), dtype=torch.bfloat16, device="cuda")
        mm.leaves(x[:M], w, leaf_k, out)  # the production choice
        assert torch.equal(_bits(out), _bits(want[:, :M])), M


@pytest.mark.parametrize("K,N,leaf_k", SHAPES)
@pytest.mark.parametrize("leaf_dtype", [torch.bfloat16, torch.float32])
def test_node_wire_every_config_every_M(mm, K, N, leaf_k, leaf_dtype):
    from isoexec.ops.mm.mm_sm90 import reference as R
    from isoexec.ops.mm.mm_sm90 import segments as seg

    m = K // leaf_k
    ms = (1, 64, 320) if K > 8192 else MS
    x, w = operands(max(ms), N, K, seed=K * 3 + N)
    leaves = ref_leaves(x, w, leaf_k)
    if leaf_dtype is torch.bfloat16:
        leaves = [t.to(torch.bfloat16).float() for t in leaves]
    node = R.tree(leaves)
    if m == 1 and leaf_dtype is torch.bfloat16:
        node = node.to(torch.bfloat16)
    for M in ms:
        # onchip folds walk consecutive leaves: whole 64-wide K blocks only
        names = [n for n in seg.configs(out_fp32=1, min_levels=1)
                 if (1 << seg._CONFIGS[n].levels) >= m and leaf_k % seg.SEGMENT_ALIGN == 0]
        # the split node plans: batched leaves + the combine kernel
        names += [mm.SPLIT_PREFIX + n for n in seg.configs(out_fp32=int(leaf_dtype is torch.float32),
                                                           exact_levels=0)]
        if m == 1:
            names = [None]
        for name in names:
            out = torch.full((M, N), float("nan"), dtype=node.dtype, device="cuda")
            mm.leaf_node(x[:M], w, leaf_k, out, leaf_dtype=leaf_dtype, config=name)
            assert torch.equal(_bits(out), _bits(node[:M])), (M, name)
        out = torch.empty((M, N), dtype=node.dtype, device="cuda")
        got = mm.leaf_node(x[:M], w, leaf_k, out, leaf_dtype=leaf_dtype)
        assert got is out  # written in place and handed back as given (pik passes its staging view)
        assert torch.equal(_bits(out), _bits(node[:M])), M


@pytest.mark.parametrize("K_full", [6144, 17408])
def test_rank_nodes_combine_to_the_tp1_node_at_every_tp(mm, K_full):
    """Row-parallel TP invariance: the subtree each rank produces, combined by the remaining tree levels,
    equals the TP 1 node over all G leaves -- bf16 and fp32 leaves, leaf wire and node wire."""
    from isoexec.ops.mm.mm_sm90 import reference as R

    N, M = 5120, 77
    leaf_k = K_full // G
    x, w = operands(M, N, K_full, seed=K_full)
    for bf16_leaves in (True, False):
        ld = torch.bfloat16 if bf16_leaves else torch.float32
        whole = torch.empty((M, N), dtype=torch.float32, device="cuda")
        mm.leaf_node(x, w, leaf_k, whole, leaf_dtype=ld)
        for tp in (1, 2, 4, 8):
            m = G // tp
            k = K_full // tp
            roots = []
            for r in range(tp):
                xs, ws = x[:, r * k:(r + 1) * k].contiguous(), w[:, r * k:(r + 1) * k].contiguous()
                if m == 1:
                    o = torch.empty((M, N), dtype=ld, device="cuda")
                    mm.leaf_node(xs, ws, leaf_k, o, leaf_dtype=ld)
                    roots.append(o.float())
                elif bf16_leaves and m == 2:
                    lv = torch.empty((m, M, N), dtype=torch.bfloat16, device="cuda")
                    mm.leaves(xs, ws, leaf_k, lv)  # the leaf wire: the reduce evaluates the whole tree
                    roots.extend(t.float() for t in lv)
                else:
                    o = torch.empty((M, N), dtype=torch.float32, device="cuda")
                    mm.leaf_node(xs, ws, leaf_k, o, leaf_dtype=ld)
                    roots.append(o)
            assert torch.equal(_bits(R.tree(roots)), _bits(whole)), (tp, bf16_leaves)


def test_leaves_equal_pik_leaf_kernels(mm):
    """The existing pik leaf contract: pik's own Triton leaf kernel produces the same leaves, so no new pik
    recipe is needed. (Before the pinned cuBLASLt leaf loop was retired, these leaves were also measured
    bit-equal to it at every width here -- 8 .. 2752 -- and every M; it was pik's admission reference.)"""
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik

    ensure_pik()

    for K, N, leaf_k in ((1536, 5120, 768), (4352, 5120, 2176), (2752, 512, 1376), (32, 256, 16)):
        for M in (1, 32, 320):
            x, w = operands(M, N, K, seed=M + K)
            m = K // leaf_k
            out = torch.empty((m, M, N), dtype=torch.bfloat16, device="cuda")
            mm.leaves(x, w, leaf_k, out)
            # pik's split schedule with one leaf per CTA writes the bf16 leaves themselves
            p = torch.empty((m, M, N), dtype=torch.bfloat16, device="cuda")
            from isoexec.ops.collectives.pik.codegen import split_tree_kernel
            from isoexec.ops.collectives.pik.fastlaunch import launch
            import triton

            grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), m)  # noqa: E731
            launch(split_tree_kernel(1, True, True), grid, x, w, p, M, N, leaf_k, x.stride(0), x.stride(1),
                   w.stride(0), w.stride(1), p.stride(0), p.stride(1), p.stride(2), NUM_LOCAL_LEAVES=m)
            assert torch.equal(_bits(out), _bits(p)), ("pik triton leaf", K, M)


def test_leaf_mode_refuses_what_it_cannot_express(mm):
    x, w = operands(4, 64, 1536, seed=1)
    with pytest.raises(mm.UnsupportedShape):
        mm.leaves(x, w, 100, torch.empty((15, 4, 64), dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(mm.UnsupportedShape):  # 3 leaves is not a balanced tree
        mm.leaf_node(x, w, 512, torch.empty((4, 64), device="cuda"))
    with pytest.raises(ValueError):
        mm.leaves(x, w, 768, torch.empty((2, 4, 64), dtype=torch.float32, device="cuda"))
