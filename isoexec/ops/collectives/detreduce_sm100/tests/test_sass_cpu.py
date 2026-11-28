"""SASS gate on the built extension (no GPU needed).

Zero floating-point atomics or reductions in every kernel. Integer atomics are allowed and enumerated:
the per-call CTA completion counter of ``finish`` (global ``atomicAdd`` on an int) and ``moe_prep``'s
shared-memory integer group counts (GLM ``ep_prep``'s plan); neither carries a floating-point value.
"""
from __future__ import annotations

import re

import pytest

ATOMIC = re.compile(r"\b(ATOM|ATOMS|ATOMG|RED|REDG|REDUX)\b[.\w]*")
FLOATY = re.compile(r"\.(F16|F16x2|BF16|BF16x2|F32|F64|FADD|FMIN|FMAX)\b|FTZ|\.RN\b")


@pytest.fixture(scope="module")
def sass():
    from isoexec.ops._testing import kernel_sass
    from isoexec.ops.collectives.detreduce_sm100._extension import extension

    return kernel_sass(extension)


EXPECTED = ("tree_fused_kernel", "tree_allreduce_kernel", "terms_combine_kernel", "moe_prep_kernel")


def test_every_kernel_is_present_cpu(sass):
    for name in EXPECTED:
        assert any(name in k for k in sass), name


def test_no_float_atomic_or_reduction_cpu(sass):
    for name, lines in sass.items():
        bad = [ln for ln in lines if ATOMIC.search(ln) and FLOATY.search(ATOMIC.search(ln).group(0))]
        assert not bad, (name, bad[:3])


def test_integer_atomics_are_the_declared_ones_cpu(sass):
    for name, lines in sass.items():
        ops = {ATOMIC.search(ln).group(0) for ln in lines if ATOMIC.search(ln)}
        if "moe_prep_kernel" in name:
            assert all(o.startswith("ATOMS") for o in ops), (name, ops)
        else:
            # finish(): one global integer atomicAdd per CTA (CTA completion count)
            assert all(o.startswith(("ATOMG", "ATOM", "RED")) for o in ops), (name, ops)
            assert len([ln for ln in lines if ATOMIC.search(ln)]) <= 2, (name, ops)
