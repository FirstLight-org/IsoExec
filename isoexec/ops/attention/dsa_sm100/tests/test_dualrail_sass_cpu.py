"""SASS gate of the dual-rail leaf (vendored FlashMLA-derived code): no floating-point or global atomics or
reductions; integer shared-memory atomics only in the TMEM allocator; no split-KV machinery compiled in."""
import os
import re
import shutil
import subprocess
from pathlib import Path

_ATOMIC = re.compile(r"\b(ATOM[A-Z0-9.]*|RED(?!UX)[A-Z0-9.]*|UBLKRED[A-Z0-9.]*|UTMAREDG[A-Z0-9.]*|UTCATOMSWS[A-Z0-9.]*)")


def _sass():
    from isoexec.ops.attention.dsa_sm100._extension import extension
    _skip_unless_buildable()
    so = Path(extension("dualrail").__file__)
    out = subprocess.run(["cuobjdump", "-sass", str(so)], capture_output=True, text=True, check=True).stdout
    funcs, cur = {}, None
    for line in out.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
        elif cur and "/*" in line:
            body = line.split("*/", 1)[1] if "*/" in line else line
            if _ATOMIC.search(body):
                funcs[cur].append(re.sub(r"^@!?U?P[T0-9]+\s+", "", body.strip()).split()[0])
    return funcs


def test_dualrail_atomics_only_tmem_allocation():
    funcs = _sass()
    leaves = [k for k in funcs if "dualrail_kernel" in k]
    assert len(leaves) == 3, list(funcs)                     # bf16, fp8, fp8 + fp8 extra
    for k in leaves:
        ops = set(funcs[k])
        assert ops <= {"UTCATOMSWS.FIND_AND_SET.ALIGN", "ATOMS.OR", "ATOMS.AND", "UTCATOMSWS.AND"}, (k, ops)
        assert any(o.startswith("UTCATOMSWS") for o in ops), (k, ops)


def test_split_plan_atomics_only_tmem_allocation():
    """The split execution plan (split.cuh): qk_kernel (fp8, fp8 + fp8 extra) and pv_kernel (NC 2 / 4 / 8, both cache forms) -- the published S
    and V are copied, never accumulated: no floating-point or global atomics or reductions either."""
    funcs = _sass()
    qk = [k for k in funcs if "qk_kernel" in k]
    pv = [k for k in funcs if "pv_kernel" in k]
    assert len(qk) == 2 and len(pv) == 6, list(funcs)   # pv: NC 2 / 4 / 8 x the fp8 and fp8 + fp8-extra tensor-map types
    for k in qk + pv:
        ops = set(funcs[k])
        assert ops <= {"UTCATOMSWS.FIND_AND_SET.ALIGN", "ATOMS.OR", "ATOMS.AND", "UTCATOMSWS.AND"}, (k, ops)


def test_no_split_kv_machinery():
    src = Path(__file__).resolve().parents[1] / "csrc" / "dualrail"
    text = "".join(p.read_text() for p in src.rglob("*") if p.suffix in {".cu", ".cuh", ".h"} and "kerutils" not in p.parts)
    for word in ("DecodingSchedMeta", "num_splits", "o_accum", "lse_accum", "tile_scheduler", "num_sm_parts"):
        assert word not in text, word


def _skip_unless_buildable():
    import pytest

    from isoexec.ops import build_cache

    if not os.environ.get("TORCH_EXTENSIONS_DIR"):
        pytest.skip("SASS scan: set TORCH_EXTENSIONS_DIR to an explicit extension build directory")
    if shutil.which("cuobjdump") is None:
        pytest.skip("SASS scan: cuobjdump (CUDA toolkit) is not on PATH")
    if build_cache.cutlass_include("dsa") is None:
        pytest.skip("SASS scan: the pinned CUTLASS headers are not installed (make setup)")
