"""Check the vendored batch-invariant MM copy itself: its provenance record and its recorded hashes.

The functions in ``ops/mm/mm_batch_invariant.py`` were copied from the vLLM 0.22 revision recorded
in their provenance and qualified for determinism when they were vendored. Their local source must
still hash to that record. They are not compared with whatever vLLM is installed.
"""

from __future__ import annotations

import importlib

from isoexec.ops.tests._vendor_copy import (
    assert_copy_matches_recorded_digests,
    assert_provenance_record_is_complete,
)

V = importlib.import_module("isoexec.ops.mm.mm_batch_invariant")

#: Module from which the IsoExec copy was taken.
SOURCE_MODULE = "vllm.model_executor.layers.batch_invariant"


def test_provenance_record_is_complete():
    assert_provenance_record_is_complete(
        V,
        upstream_module=SOURCE_MODULE,
        keys=("project", "module", "version", "commit", "wheel", "url", "vendored_on", "symbols"),
    )


def test_isoexec_copy_matches_recorded_vllm_022_source_hashes():
    assert_copy_matches_recorded_digests(
        V, requalify="mm_sm90/tests/test_mm_sm90_gpu.py and moe/tests/shared_expert_biteq_gpu.py"
    )


if __name__ == "__main__":  # pragma: no cover
    import sys

    from isoexec.ops.tests._vendor_copy import run_as_script

    sys.exit(run_as_script(globals()))
