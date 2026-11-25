"""Check the vendored RoPE copy: provenance record, recorded digests and an empty substitution table.
The copy was qualified when it was vendored; it is not re-diffed against an installed Megatron."""

from __future__ import annotations

import importlib

from isoexec.ops.tests._vendor_copy import (
    assert_copy_matches_recorded_digests,
    assert_provenance_record_is_complete,
)

V = importlib.import_module("isoexec.ops.attention.rope_sm90.rope_reference")

#: Upstream module the copy came from. Recorded here as well as in the vendored file so a rename on
#: either side is a failure rather than a silent skip.
UPSTREAM_MODULE = "megatron.core.models.common.embeddings.rope_utils"


def test_provenance_record_is_complete():
    assert_provenance_record_is_complete(
        V,
        upstream_module=UPSTREAM_MODULE,
        keys=("project", "module", "version", "commit", "url", "vendored_on", "symbols"),
    )


def test_vendored_copy_matches_its_own_recorded_digests():
    assert_copy_matches_recorded_digests(
        V, requalify="the rope_sm90 GPU tests (test_forward_gpu.py, test_backward_gpu.py, test_dweight_determinism_gpu.py)"
    )


def test_the_copy_declares_no_substitution_and_has_none():
    """The verbatim claim, asserted rather than assumed.

    ``ops/mm`` and ``ops/gdn`` both needed substitution rules (import redirects, a re-expressed
    constant). This copy needs none, which is a stronger property and therefore worth pinning: if
    somebody later has to substitute something, the rule must be recorded before the code changes.
    """
    assert V.provenance()["substitutions"] == [], (
        "rope_reference.py claims to be verbatim; a substitution appeared without being recorded. "
        "Record it in _VENDORED_FROM['substitutions'] with its justification, and pin whatever "
        "upstream text it re-expresses (see ops/attention/attention_gate.py's jit_fuser rule)."
    )


if __name__ == "__main__":  # pragma: no cover
    import sys

    from isoexec.ops.tests._vendor_copy import run_as_script

    sys.exit(run_as_script(globals()))
