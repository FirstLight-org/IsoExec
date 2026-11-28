"""SASS gates on the built extension (no GPU needed; builds the extension if it is not cached; skipped without
nvcc / cuobjdump).

No kernel of this package -- forward, decode, backward or layer stage -- contains an atomic or reduction
instruction of any kind (ATOM, ATOMS, ATOMG, RED): every output element has one writer and every sum is a
fixed-order chain, tree or mma chain.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ATOMIC = re.compile(r"\b(ATOM|ATOMS|ATOMG|RED)(\.|\s)")


def _sass():
    from isoexec.ops.kda.glm_sm100._extension import extension
    from isoexec.ops.kda.tests._fixtures import require_cuda_toolchain

    require_cuda_toolchain()

    so = Path(extension().__file__)
    out = subprocess.run(["cuobjdump", "-sass", str(so)], capture_output=True, text=True, check=True).stdout
    funcs, cur = {}, None
    for line in out.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
        elif cur:
            funcs[cur].append(line)
    return funcs


@pytest.fixture(scope="module")
def sass():
    return _sass()


EXPECTED = ("plan_kernel", "intra_kernel", "state_kernel", "decode_prep_kernel", "decode_state_kernel", "decode_row_kernel", "decode_state_rec_kernel", "record_state_kernel", "bwd_state_kernel", "bwd_inter_kernel",
            "bwd_wy_kernel", "bwd_qkg_kernel", "gate_fwd_kernel", "gate_bwd_kernel", "beta_fwd_kernel",
            "beta_bwd_kernel", "gnorm_fwd_kernel", "gnorm_bwd_kernel", "colfold_kernel", "leaf_tree_kernel")


def test_every_kernel_is_present_cpu(sass):
    for name in EXPECTED:
        assert any(name in k for k in sass), name


def test_no_kernel_has_any_atomic_cpu(sass):
    for name, lines in sass.items():
        bad = [ln for ln in lines if ATOMIC.search(ln)]
        assert not bad, (name, bad[:3])
