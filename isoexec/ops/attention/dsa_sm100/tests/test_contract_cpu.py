"""CPU contract of the three DSA recipes: no CUDA, no kernel.

What is checked here is the part of the numerical program that does not need a device: the declared
constants, the top-k key map is monotone and total with the declared NaN / tie / pad rules, the
radix-select algorithm class returns exactly the reference selection (single- and multi-CTA
histogram split), the group tree is the same object for one worker and for G workers and is not the
flat chain, and the fp32 emulations agree with the float64 references within tolerance.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from isoexec.ops.attention.dsa_sm100 import reference as ref
from isoexec.ops.attention.dsa_sm100 import spec

def test_cpu_declared_constants():
    assert (spec.H_KV, spec.D_V, spec.PAGE, spec.TOPK, spec.H_IDX, spec.D_IDX) == (1, 512, 64, 2048, 64, 128)
    assert (spec.SCORE_DTYPE, spec.TOPK_PAD, spec.NEXT_N, spec.MAX_CTX, spec.TILE_SLOTS) == ("bfloat16", -1, 1, 131072, 64)


def test_cpu_geometry_and_tables():
    assert spec.D_QK == 512 and spec.D_QK_NOPE + spec.D_QK_ROPE == spec.D_QK
    assert spec.KV_BYTES_PER_TOKEN == 528 == spec.D_QK + spec.D_QK // spec.KV_QUANT_TILE
    assert set(spec.H_Q_VALUES) == {64, 128}
    assert spec.LIST_GROUPS[(64, 2048)] >= 1 and spec.LIST_GROUPS[(128, 2048)] >= 1
    for (hq, k), g in spec.LIST_GROUPS.items():
        assert k % (g * spec.TILE_SLOTS) == 0, "a group must hold whole tiles"
    assert spec.SCORE_PAD_BITS == 0xFF80
    assert spec.MAX_INIT_VAL == -1e33 and spec.MAX_INIT_VAL_SM == -1e30
    assert set(spec.IMPL_IDS) == {"attention.dsa.indexer", "attention.dsa.topk", "attention.dsa.sparse"}
    from isoexec.core.registry_build import build_registry

    reg = build_registry()
    for op, impl in spec.IMPL_IDS.items():
        assert impl in reg.get_op(op).impls


# ------------------------------------------------------------------------------------------------
# top-k: key map, order, NaN, ties, pads
# ------------------------------------------------------------------------------------------------

def _all_bf16():
    return np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)


def test_cpu_key_map_bf16_monotone_total():
    bits = _all_bf16()
    vals = ref.bf16_bits_to_f32(bits)
    keys = ref.key_bf16(bits)
    nan = np.isnan(vals)
    assert (keys[nan] == 0).all(), "NaN_POLICY: every NaN maps to the lowest key"
    real = ~nan
    k_real, v_real, b_real = keys[real], vals[real], bits[real]
    assert (k_real >= 1).all(), "every non-NaN key is above every NaN"
    assert len(np.unique(k_real)) == len(k_real), "total: injective over non-NaN patterns"
    order = np.argsort(k_real, kind="stable")
    v_sorted = v_real[order]
    assert (np.diff(v_sorted.astype(np.float64)) >= 0).all(), "monotone: key order is value order"
    # -0 sits immediately below +0, and -inf is the lowest non-NaN key
    k_m0 = ref.key_bf16(np.array([0x8000], np.uint16))[0]
    k_p0 = ref.key_bf16(np.array([0x0000], np.uint16))[0]
    assert k_p0 == k_m0 + 1
    k_ninf = ref.key_bf16(np.array([0xFF80], np.uint16))[0]
    assert k_ninf == k_real.min()
    k_pinf = ref.key_bf16(np.array([0x7F80], np.uint16))[0]
    assert k_pinf == k_real.max()


def test_cpu_key_map_f32_monotone():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.standard_normal(20000).astype(np.float32),
                        np.array([0.0, -0.0, np.inf, -np.inf, 1e-45, -1e-45, 3.4e38, -3.4e38],
                                 np.float32)])
    k = ref.key_f32(x)
    o = np.argsort(k, kind="stable")
    assert (np.diff(x[o].astype(np.float64)) >= 0).all()
    assert (ref.key_f32(np.array([np.nan, -np.nan], np.float32)) == 0).all()


def _bits(vals):
    return ref.f32_to_bf16_bits(np.asarray(vals, np.float32))


def test_cpu_topk_ties_lowest_index_and_ascending_output():
    k = 4
    s = _bits([[1, 3, 3, 3, 3, 3, 0, 3]])
    out = ref.topk_ref(s, np.array([0]), np.array([8]), k)
    assert out.tolist() == [[1, 2, 3, 4]], "tie class at the threshold: lowest indices win"
    assert (np.diff(out[0]) > 0).all()


def test_cpu_topk_nan_below_neg_inf_and_pad():
    k = 3
    s = _bits([[np.nan, -np.inf, np.nan, 2.0, np.nan]])
    out = ref.topk_ref(s, np.array([0]), np.array([5]), k)
    assert out.tolist() == [[0, 1, 3]], "ascending output: {-inf@1, 2.0@3} then the lowest NaN @0"
    # rows shorter than K: every key selected, -1 pads trailing
    out = ref.topk_ref(_bits([[5.0, 1.0]]), np.array([0]), np.array([2]), 4)
    assert out.tolist() == [[0, 1, -1, -1]]
    # empty window
    out = ref.topk_ref(_bits([[5.0, 1.0]]), np.array([1]), np.array([1]), 2)
    assert out.tolist() == [[-1, -1]]
    # window respected: keys outside [ks, ke) are never candidates (even +inf)
    out = ref.topk_ref(_bits([[np.inf, 1.0, 2.0, np.inf]]), np.array([1]), np.array([3]), 1)
    assert out.tolist() == [[2]]
    assert spec.TOPK_PAD < 0


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
@pytest.mark.parametrize("ctas", [1, 3])
def test_cpu_radix_select_equals_reference(dtype, ctas):
    rng = np.random.default_rng(1 + ctas)
    rows, width, k = 12, 700, 64
    if dtype == "bf16":
        v = rng.standard_normal((rows, width)).astype(np.float32)
        v[0] = 1.0                                     # all equal
        v[1, :200] = 7.0                               # 200 equal maxima, K = 64 of them
        v[2, ::3] = np.nan
        v[3, :] = -np.inf
        v[4, 10:] = np.nan
        s = ref.f32_to_bf16_bits(v)
    else:
        s = rng.standard_normal((rows, width)).astype(np.float32)
        s[0] = 1.0
        s[1, :200] = 7.0
        s[2, ::3] = np.nan
        s[5, :] = -0.0
        s[5, ::2] = 0.0
    ks = rng.integers(0, 50, rows)
    ke = np.minimum(width, ks + rng.integers(1, width, rows))
    ke[6] = ks[6] + k                                  # exactly K
    ke[7] = ks[7] + 1                                  # one key
    ke[8] = ks[8]                                      # empty
    a = ref.topk_ref(s, ks, ke, k, dtype)
    b = ref.topk_radix_emul(s, ks, ke, k, dtype, ctas=ctas)
    assert np.array_equal(a, b)


# ------------------------------------------------------------------------------------------------
# the group tree
# ------------------------------------------------------------------------------------------------

def _sym(n, one_worker):
    leaves = list(range(n))
    comb = lambda a, b: (a, b)                             # noqa: E731
    f = ref.tree_fold_stack if one_worker else ref.tree_fold
    return f(leaves, combine=comb)


def _flat_chain(n):
    node = 0
    for i in range(1, n):
        node = (node, i)
    return node


@pytest.mark.parametrize("n", list(range(1, 65)))
def test_cpu_tree_one_worker_equals_g_workers(n):
    t_many = _sym(n, one_worker=False)
    t_one = _sym(n, one_worker=True)
    assert t_many == t_one == ref.tree_shape(n)
    if n >= 4:                                   # at n <= 3 the two shapes coincide
        assert t_many != _flat_chain(n), "the group tree is not the flat chain"


def test_cpu_flat_chain_refused_by_plan():
    from isoexec.ops.attention.dsa_sm100 import plan
    with pytest.raises(ValueError, match="flat chain"):
        plan.check_fold(groups=4, fold="chain")
    plan.check_fold(groups=4, fold="tree")
    plan.check_fold(groups=1, fold="chain")         # G = 1: the chain IS the tree
    assert plan.groups_for(64, 2048) == spec.LIST_GROUPS[(64, 2048)]
    with pytest.raises(KeyError):
        plan.groups_for(64, 1000)


# ------------------------------------------------------------------------------------------------
# emulations vs float64
# ------------------------------------------------------------------------------------------------

def _bf16(x):
    return ref.bf16_round(np.asarray(x, np.float32))


def test_cpu_sparse_leaf_emul_close_to_fp64():
    rng = np.random.default_rng(3)
    H, P = 64, 256
    q = _bf16(rng.standard_normal((H, 512)) * 0.3)
    rows = _bf16(rng.standard_normal((P, 512)) * 0.3)
    valid = rng.random(P) > 0.2
    rows[~valid] = np.nan                              # pad rows hold garbage: must not leak
    scale = 1 / math.sqrt(512)
    for g in (1, 2, 4):
        out_bits, lse, _ = ref.sparse_attn_emul(q, rows, valid, scale, groups=g)
        out = ref.bf16_bits_to_f32(out_bits)
        o64, l64 = ref.sparse_attn_fp64(q, rows, valid, scale)
        assert np.isfinite(out).all()
        assert np.abs(out - o64).max() < 2e-2
        assert np.abs(lse - l64).max() < 1e-4
    # one worker == G workers under CpuPrims too (same tree, same merges)
    a = ref.sparse_attn_emul(q, rows, valid, scale, groups=4)
    b = ref.sparse_attn_emul(q, rows, valid, scale, groups=4, one_worker=True)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_cpu_sparse_empty_list_declared_output():
    H, P = 64, 128
    q = _bf16(np.ones((H, 512)) * 0.1)
    rows = np.full((P, 512), np.nan, np.float32)
    valid = np.zeros(P, bool)
    out_bits, lse, st = ref.sparse_attn_emul(q, rows, valid, 1 / math.sqrt(512), groups=2)
    assert (out_bits == 0).all(), "all-pad row: out = bf16(0 / 1) = +0"
    assert (st.L == 0).all() and (st.M == np.float32(-1e30)).all()
    assert np.allclose(lse, -1e30 / math.log2(math.e), rtol=1e-6)


def test_cpu_indexer_emul_close_to_fp64():
    rng = np.random.default_rng(4)
    T, H, S = 3, 64, 200
    q = rng.standard_normal((T, H, 128)).astype(np.float32)
    k = rng.standard_normal((S, 128)).astype(np.float32)

    def quant(x):
        xv = x.reshape(*x.shape[:-1], 4, 32)
        amax = np.maximum(np.abs(xv).max(-1), 1e-4)
        e = np.ceil(np.log2(amax / 448.0)).astype(np.int64)
        s = np.exp2(e.astype(np.float64))
        import torch
        qf = torch.from_numpy((xv / s[..., None]).astype(np.float32)).to(torch.float8_e4m3fn)
        qb = qf.view(torch.uint8).numpy().reshape(x.shape)
        eb = (e + 127).astype(np.int64)
        words = eb[..., 0] | (eb[..., 1] << 8) | (eb[..., 2] << 16) | (eb[..., 3] << 24)
        return qb, words.astype(np.int64)

    qb, qs = quant(q)
    kb, ksf = quant(k)
    w = ref.f32_to_bf16_bits(rng.standard_normal((T, H)).astype(np.float32) * 0.2)
    ks = np.array([0, 10, 50])
    ke = np.array([200, 11, 50])
    e = ref.indexer_emul(qb, qs, kb, ksf, w, ks, ke)
    f = ref.indexer_fp64(qb, qs, kb, ksf, w, ks, ke)
    ev = ref.bf16_bits_to_f32(e)
    inwin = np.isfinite(f)
    assert (e[~inwin] == spec.SCORE_PAD_BITS).all()
    rel = np.abs(ev[inwin] - f[inwin]) / np.maximum(np.abs(f[inwin]), 1e-3)
    assert rel.max() < 1e-2                            # one bf16 rounding dominates
    assert inwin.sum() == 200 + 1 + 0
