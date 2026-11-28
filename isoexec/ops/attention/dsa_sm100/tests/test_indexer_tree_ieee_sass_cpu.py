"""SASS gate (no GPU needed) of the indexer tree's IEEE top levels (``tree_ieee_levels``).

The indexer TU is built ``--ftz=true``: every C-level float add/mul/max is FTZ (``FADD.FTZ`` /
``FMUL.FTZ`` / packed ``.FTZ`` forms). The canonical index-score tree completes its top levels with
inline ``add.rn.f32`` (no ``.ftz``) so that it reproduces the cross-rank ``leaf_tree`` completion (built
``--ftz=false``) bit for bit, subnormals included. This checks the compiler kept those adds IEEE: the
indexer kernel carries plain (non-FTZ) ``FADD`` instructions next to its FTZ arithmetic.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path


def _kernel_ops():
    from isoexec.ops.attention.dsa_sm100._extension import extension
    _skip_unless_buildable()
    so = Path(extension("indexer").__file__)
    out = subprocess.run(["cuobjdump", "-sass", str(so)], capture_output=True, text=True, check=True).stdout
    ops, cur = [], None
    for line in out.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1)
            continue
        if cur and "dsa_indexer_kernel" in cur and "/*" in line:
            body = line.split("*/", 1)[1] if "*/" in line else line
            body = re.sub(r"^\s*@!?U?P[T0-9]+\s+", "", body.strip())
            if body:
                ops.append(body.split()[0])
    return ops


def test_indexer_tree_top_levels_are_ieee_adds():
    ops = _kernel_ops()
    assert ops, "dsa_indexer_kernel not found"
    fadd = [o for o in ops if o.startswith("FADD")]
    assert any(".FTZ" in o for o in fadd), sorted(set(fadd))           # the kernel's own FTZ arithmetic
    assert any(".FTZ" not in o for o in fadd), sorted(set(fadd))       # ieee_add: add.rn.f32 survived


def _skip_unless_buildable():
    import pytest

    from isoexec.ops import build_cache

    if not os.environ.get("TORCH_EXTENSIONS_DIR"):
        pytest.skip("SASS scan: set TORCH_EXTENSIONS_DIR to an explicit extension build directory")
    if shutil.which("cuobjdump") is None:
        pytest.skip("SASS scan: cuobjdump (CUDA toolkit) is not on PATH")
    if build_cache.cutlass_include("dsa") is None:
        pytest.skip("SASS scan: the pinned CUTLASS headers are not installed (make setup)")
