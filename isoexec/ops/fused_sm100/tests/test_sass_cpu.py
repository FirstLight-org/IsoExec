"""SASS gate (needs the built extension, no GPU): no atomic or RED instruction of any
kind in any fused-glue kernel."""

from __future__ import annotations

import re

import pytest


@pytest.fixture(scope="module")
def sass():
    from isoexec.ops._testing import kernel_sass
    from isoexec.ops.fused_sm100._extension import extension

    return kernel_sass(extension)


def test_no_atomics_anywhere_cpu(sass):
    for needle in ("scatter_rows_kernel", "embed_expand_kernel", "argmax_partial_kernel", "argmax_final_kernel", "mul_rows_kernel"):
        assert any(needle in k for k in sass), needle
    for name, lines in sass.items():
        bad = [ln for ln in lines if re.search(r"\b(ATOM|ATOMS|ATOMG|RED)\b|\bATOMG?\.|\bRED\.", ln)]
        assert not bad, (name, bad[:3])
