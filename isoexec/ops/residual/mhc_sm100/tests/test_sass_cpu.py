"""SASS gate (needs only the built extension, no GPU): no atomic or reduction
instruction of any kind (ATOM / ATOMS / ATOMG / RED) in ANY kernel -- forward, merge or backward -- hence in particular
no float atomics. (The FFMA instructions that exist are the declared exact-product forms: sum-of-squares x*x and bf16 x bf16
accumulations, bitwise equal to separate multiply+add; the GPU tests compare against that torch program.)"""

from __future__ import annotations

import re

import pytest


@pytest.fixture(scope="module")
def sass():
    from isoexec.ops._testing import kernel_sass
    from isoexec.ops.residual.mhc_sm100._extension import extension

    return kernel_sass(extension)


def test_expected_kernels_present_cpu(sass):
    names = " ".join(sass)
    for k in ("proj_kernel", "tail_kernel", "collapse_kernel", "merge_fwd_kernel", "merge_bwd_kernel", "bwd_dcolx",
              "bwd_tail", "bwd_main", "bwd_reduce_fn", "bwd_rows_partial", "bwd_rows_final"):
        assert k in names, k


def test_no_atomics_anywhere_cpu(sass):
    for name, lines in sass.items():
        bad = [ln for ln in lines if re.search(r"\b(ATOM|ATOMS|ATOMG|RED)\b|\bATOMG?\.|\bRED\.", ln)]
        assert not bad, (name, bad[:3])
