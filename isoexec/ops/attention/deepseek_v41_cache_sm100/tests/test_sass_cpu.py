"""SASS gate (no GPU needed): no atomic or reduction instruction of any kind in any kernel of this
package's extension -- forward row kernels, cache writes, candidate masks, and every backward kernel.
The reused dsa_sm100 sink/backward kernels carry their own scans (dsa_sm100/tests).
"""
import re

import pytest


@pytest.fixture(scope="module")
def sass():
    from isoexec.ops._testing import kernel_sass
    from isoexec.ops.attention.deepseek_v41_cache_sm100._extension import extension

    return kernel_sass(extension)


def test_every_kernel_is_scanned_cpu(sass):
    names = " ".join(sass)
    for k in ("row_kernel", "pool_norm_kernel", "block_max_kernel", "candidate_mask_kernel", "build_lists_kernel",
              "scatter_rows_kernel", "row_bwd_kernel", "segment_partial_kernel", "segment_fold_kernel",
              "pool_norm_bwd_kernel", "decode_prep_kernel", "rope_tail_inplace_kernel", "permute_rows_kernel", "leaf_tree_kernel", "paged_quant_write_kernel",
              "pool_varlen_kernel", "state_update_kernel", "paged_lists_kernel", "comp_post_kernel",
              "qkv_post_kernel", "qkv_post_split_kernel", "paged_index_write128_kernel"):
        assert k in names, k


def test_no_atomics_anywhere_cpu(sass):
    # candidate_mask builds its bitmap words by search over the ascending selection (no shared OR), so not even
    # an order-free integer atomic or a warp reduction remains
    for name, lines in sass.items():
        bad = [ln for ln in lines if re.search(r"\b(ATOM|ATOMS|ATOMG|RED|REDUX)\b|\bATOM\.|\bATOMS\.|\bATOMG\.|\bRED\.|\bREDUX\.", ln)]
        assert not bad, (name, bad[:3])


# FFMA appears only inside the correctly rounded div.rn / sqrt.rn subroutines (Newton steps); the
# declared scalar graph is proven bit for bit by tests/test_rows_gpu.py against the CPU emulation.


def test_reused_kernels_have_no_float_atomics_cpu():
    """Every other extension the composition launches (forward and backward): integer/shared
    bookkeeping atomics and mbarrier arrives only -- no floating-point ATOM/RED anywhere."""
    from isoexec.ops.attention.dsa_sm100._extension import extension as dsa
    from isoexec.ops.mm.mm_sm100._extension import extension as mm
    from isoexec.ops.moe.moe_sm100._extension import extension as moe
    # extension() is lru_cached, and other packages' "importing builds nothing" tests read that
    # cache: restore each cache to its prior (empty) state so this test does not leak a build.
    from isoexec.ops._testing import kernel_sass
    # extension() is lru_cached, and other packages' "importing builds nothing" tests read that
    # cache: restore each cache to its prior (empty) state so this test does not leak a build.
    cached = [fn for fn in (moe, mm, dsa) if fn.cache_info().currsize == 0]
    try:
        scans = [kernel_sass(moe, family="moe", cutlass=True), kernel_sass(mm, family="mm", cutlass=True)]
        scans += [kernel_sass(lambda s=s: dsa(s), family="dsa", cutlass=True) for s in ("attn", "bwd", "indexer", "topk")]
        pat = re.compile(r"\b(ATOM|ATOMS|ATOMG|RED)\.[A-Z0-9._]*(F16|F32|F64|BF16|FADD|FMIN|FMAX)")
        for funcs in scans:
            for name, lines in funcs.items():
                bad = [ln for ln in lines if pat.search(ln)]
                assert not bad, (name, bad[:3])
    finally:
        for fn in cached:
            fn.cache_clear()
