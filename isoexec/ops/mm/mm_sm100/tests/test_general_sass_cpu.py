"""SASS gate on the mm_sm100/general extension (no GPU needed; builds outside the lease if not cached),
the same rule as checkpoint_tc_sm100's: no float atomic and no global atomic or reduction (ATOMG / RED /
generic ATOM) in any kernel -- the K-segment merge is the separate combine kernel in segment order, never
an atomic. The only atomics allowed are the compiler's tcgen05.alloc bookkeeping: integer bitwise
ATOMS.OR / AND / XOR on its shared allocation mask, never on data. SYNCS.ARRIVE.*.RED is an mbarrier
arrival (the 2-CTA remote arrive), not a memory reduction."""
import re
import subprocess
from pathlib import Path

import pytest

_ALLOWED = re.compile(r"^ATOMS\.(OR|AND|XOR)$")


def _opcode(line):
    m = re.search(r"\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_.]+)", line)
    return m.group(2) if m else ""


@pytest.fixture(scope="module")
def sass():
    from isoexec.ops.mm.mm_sm100.general._extension import extension

    out = subprocess.run(["cuobjdump", "-sass", str(Path(extension().__file__))], capture_output=True, text=True,
                         check=True).stdout
    funcs, cur = {}, None
    for line in out.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
        elif cur:
            funcs[cur].append(line)
    return funcs


def test_every_kernel_is_atomic_free_cpu(sass):
    gemm = [k for k in sass if "gemm_kernel" in k]
    combine = [k for k in sass if "combine_kernel" in k]
    # cta_group::1 {256,128,64} x 4 layouts + 32 x 2 K-major-B layouts; cta_group::2 {256,128} x 4 layouts
    assert len(gemm) == 22, sorted(gemm)
    # S = 2..16 x {bf16, fp32} root
    assert len(combine) == 30, sorted(combine)
    for name in gemm + combine:
        ops_ = [_opcode(ln) for ln in sass[name]]
        atomics = [o for o in ops_ if re.match(r"^(ATOM|ATOMS|ATOMG|RED)(\.|$)", o)]
        bad = [o for o in atomics if not _ALLOWED.match(o)]
        assert not bad, (name, bad[:3])


def test_gemm_kernels_issue_tensor_core_mma_cpu(sass):
    for name, lines in sass.items():
        if "gemm_kernel" in name:
            assert re.search(r"\bUTC\w*MMA", "\n".join(lines)), name
