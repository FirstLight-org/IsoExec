"""SASS gate: no float (or any) atomic/reduction instruction in any kernel of the recurrent recipe (no GPU needed;
builds the extension if it is not cached; skipped without nvcc / cuobjdump)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ATOMIC = re.compile(r"\b(ATOM|ATOMS|ATOMG|RED)(\.|\s)")


@pytest.fixture(scope="module")
def sass():
    from isoexec.ops.kda.glm_rec_sm100._extension import extension
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


def test_kernels_are_present_cpu(sass):
    assert any("kda_rec" in k for k in sass) and any("gate_kernel" in k for k in sass) and any("beta_kernel" in k for k in sass)


def test_no_kernel_has_any_atomic_cpu(sass):
    assert len(sass) >= 20
    for name, lines in sass.items():
        bad = [ln for ln in lines if ATOMIC.search(ln)]
        assert not bad, (name, bad[:3])
