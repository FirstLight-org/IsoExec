"""Run every MLA conformance relation against mla_sm90.
ISOEXEC_MLA_SM90_SEG selects the segment span and ISOEXEC_MLA_SM90_HEADS the head-count sweep."""

import os

import pytest
import torch

from isoexec.ops.attention.mla_sm90.tests._harness import cuda_ok, load_backend
from isoexec.ops.attention.mla_sm90.tests._relations import RELATIONS, Config, run_relation

pytestmark = pytest.mark.skipif(not cuda_ok(), reason="requires an SM90 (H100) device")

SEG = int(os.environ.get("ISOEXEC_MLA_SM90_SEG", 512))
HEADS = [int(h) for h in os.environ.get("ISOEXEC_MLA_SM90_HEADS", "5,10,20").split(",")]


@pytest.fixture(scope="module")
def sm90():
    backend, why = load_backend("mla_sm90", SEG)
    if backend is None:
        pytest.skip(f"mla_sm90 forward not importable: {why}")
    return backend


_IDENTITY = {}


def _assert_outcome(o):
    ident = o.witness.get("kernel_identity")
    if ident is not None:
        first = _IDENTITY.setdefault("first", ident)
        assert ident == first, f"kernel sources changed during the session: {first} -> {ident}; rerun"
    if o.status == "n/a":
        pytest.skip("; ".join(o.notes + o.skipped) or "not applicable")
    lines = [f"{o.relation} [{o.backend}] -> {o.status}"]
    lines += [f"  {d.name}: shape={d.shape} {d.detail}" for d in o.failures[:20]]
    lines += [f"  note: {n.splitlines()[0]}" for n in o.notes[:10]]
    lines += [f"  metrics: {o.metrics.get('worst')}"] if o.metrics else []
    assert o.status == "pass", "\n".join(lines)


@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("relation", list(RELATIONS))
def test_relation(sm90, relation, heads):
    torch.backends.cuda.matmul.allow_tf32 = False
    _assert_outcome(run_relation(relation, sm90, Config(heads=heads, seg=SEG)))
