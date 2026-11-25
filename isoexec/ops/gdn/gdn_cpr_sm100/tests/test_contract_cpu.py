"""Declarations only: no CUDA, no build. What this package claims and which arch it claims it for."""

from __future__ import annotations

import inspect

from isoexec.ops.gdn import gdn_cpr_sm100
from isoexec.ops.gdn.gdn_cpr_sm100 import spec


def test_arch_is_sm100():
    assert spec.ARCH == "sm100"


def test_impls_are_sm100_only_and_cover_the_owed_ops():
    if not spec.IMPLS:
        return   # the interim state: a row is a place to put a proof, never a proof
    ops = {s.op for s in spec.IMPLS}
    assert ops == {"gdn.core", "gdn.l2norm", "gdn.gating"}
    for s in spec.IMPLS:
        assert s.supported_archs == frozenset({"sm100"}), s
        assert s.impl_id == "sm100_cpr_v1", s
        assert s.version == 1, s


def test_impl_id_is_not_the_sm90_spelling():
    """The id IS the recipe. The sm100 kernel is a different kernel under a different id, even though the
    arithmetic it evaluates is the same program (see README.md, "Numerical contract")."""
    for s in spec.IMPLS:
        assert s.impl_id != "cuda_cpr_v1"


#: Optional execution-plan controls are gated by the bit-identity tests.
_EXTRA_KWARGS = {"decode": {"plan"}, "prefill": {"plan"}}


def test_public_surface_matches_the_sm90_package():
    """Same API, so a caller swaps packages by arch and nothing else: every sm90 parameter is accepted,
    every required parameter is the same one, and any addition is optional and declared above."""
    from isoexec.ops.gdn import gdn_cpr_sm90

    assert set(gdn_cpr_sm100.__all__) <= set(gdn_cpr_sm90.__all__)
    from isoexec.ops.gdn.gdn_cpr_sm100 import ops as new
    from isoexec.ops.gdn.gdn_cpr_sm90 import ops as old

    for name in ("prefill", "decode", "trainer_forward"):
        a = inspect.signature(getattr(old, name)).parameters
        b = inspect.signature(getattr(new, name)).parameters
        extra = _EXTRA_KWARGS.get(name, set())
        assert set(b) - extra <= set(a), (name, set(b) - extra - set(a))
        assert all(b[n].default is not inspect.Parameter.empty for n in extra), name
        assert {n for n, p in a.items() if p.default is inspect.Parameter.empty} == \
               {n for n, p in b.items() if p.default is inspect.Parameter.empty}, name


def test_chunk_is_sixty_four_and_geometry_is_128():
    from isoexec.ops.gdn.gdn_cpr_sm100 import ops

    assert ops.C == 64 and ops.D == 128 and ops.NPOS == 8


def test_no_import_builds_a_kernel():
    """Importing declarations must never compile: the extension is built on the first CALL."""
    import subprocess
    import sys

    code = ("import isoexec.ops.gdn.gdn_cpr_sm100 as pkg;"
            "import isoexec.ops.gdn.gdn_cpr_sm100.spec;"
            "import isoexec.ops.gdn.gdn_cpr_sm100.ops as ops;"
            "from isoexec.ops.gdn.gdn_cpr_sm100 import _extension;"
            "assert _extension.extension.cache_info().currsize == 0;"
            "print('clean')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and "clean" in out.stdout, (out.stdout[-2000:], out.stderr[-2000:])
