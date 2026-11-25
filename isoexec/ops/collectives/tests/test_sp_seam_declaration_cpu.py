"""Check sequence-parallel declarations against their gather/scatter factories.
The live byte movements and adjoints are tested separately with Gloo ranks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from isoexec.core.registry import ANY, Registry
from isoexec.ops.collectives import spec

_ROOT = Path(__file__).resolve().parents[4]

#: (op id, impl id, the factory in ``tp_bytes`` that builds it, the name that factory stamps).
SEAMS = (
    ("collectives.sp_gather", "seq_all_gather_replicated_fwd_slice_bwd", "make_seq_all_gather_replicated"),
    ("collectives.sp_scatter", "seq_scatter_fwd_all_gather_bwd", "make_seq_scatter"),
)


def test_the_family_declares_the_two_sequence_parallel_seams_and_nothing_else_new():
    """The whole family's op set, stated here so a stray entry cannot arrive quietly.

    Both seams are declared, each with exactly one impl, under an id that names its forward and its
    adjoint the way ``identity_fwd_tree_sum_bwd`` and ``seq_all_gather_fwd_tree_rs_bwd`` do.
    """
    reg = Registry.from_impls(spec.IMPLS)
    assert set(reg.ops) == {
        "collectives.row_parallel",
        "collectives.tp_copy",
        "collectives.all_gather",
        "collectives.sp_gather",
        "collectives.sp_scatter",
        "collectives.cp_gather",
        "collectives.cp_heads",
        # Contiguous context parallelism's recurrent-state / conv-tail hand-off (ops/gdn/cp.py).
        "collectives.cp_state",
    }
    for op_id, impl_id, _factory in SEAMS:
        assert set(reg.ops[op_id].impls) == {impl_id}


def test_the_seams_declare_the_degree_and_no_leaf_plan():
    """Declare only TP degree: both directions move bytes without a fixed-tree reduction."""
    reg = Registry.from_impls(spec.IMPLS)
    for op_id, impl_id, _factory in SEAMS:
        impl = reg.ops[op_id].impls[impl_id]
        assert impl.inputs == {"tp": ANY}
    tree_gather = reg.ops["collectives.tp_copy"].impls["seq_all_gather_fwd_tree_rs_bwd"]
    assert set(tree_gather.inputs) == {"tp", "leaves", "leaf_dtype"}


def test_the_seams_claim_one_arch_no_twin_class_and_one_version():
    """Require SM90/SM100 evidence, no cross-implementation twin claim and a shared gather/scatter recipe version."""
    reg = Registry.from_impls(spec.IMPLS)
    for op_id, impl_id, _factory in SEAMS:
        impl = reg.ops[op_id].impls[impl_id]
        assert impl.supported_archs == frozenset({"sm90", "sm100"})
        assert not impl.twin_class
        assert impl.version is spec.SP_SEAM_VERSION == 1


def test_each_impl_id_names_the_factory_that_builds_it():
    """The drift guard: an impl id is a claim about a callable, so the callable must be there.

    ``nn/_seams.py`` reads these ids off the declaration and the version with them, so an id here
    that no factory answers to would reach a peer's digest as a per-op DIFFERENCE rather than a
    refusal. The factory names and the seam names are checked the way the component reads them.
    """
    from isoexec.ops.collectives import tp_bytes

    for _op_id, impl_id, factory in SEAMS:
        assert callable(getattr(tp_bytes, factory))
        assert factory in tp_bytes.__all__
        # `seq_all_gather_replicated_fwd_slice_bwd` -> `seq_all_gather_replicated`, the forward the
        # factory builds and the name it stamps on the callable (`..._tp{degree}`).
        forward = impl_id.split("_fwd_")[0]
        assert factory == f"make_{forward}"


def test_reading_the_declaration_loads_no_kernel():
    """``spec.py`` stays import-light: op metadata is read on hosts that never build a collective.

    A fresh process, because the check is about what the import PULLS IN and a warm interpreter has
    already answered the question the wrong way.
    """
    code = (
        "import sys; import isoexec.ops.collectives.spec as s;"
        "siblings=[m for m in sys.modules if m.startswith('isoexec.ops.collectives.') "
        "and m != 'isoexec.ops.collectives.spec'];"
        "assert not siblings, siblings;"
        "assert not [m for m in sys.modules if m.startswith('isoexec_')];"
        "import torch; assert not torch.cuda.is_initialized();"
        "assert s.SP_SEAM_VERSION == 1"
    )
    subprocess.run([sys.executable, "-B", "-c", code], check=True, cwd=str(_ROOT))
