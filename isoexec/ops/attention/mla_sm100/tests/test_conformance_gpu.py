"""Run every MLA conformance relation against mla_sm100 on a Blackwell device."""

import os

import pytest
import torch

from isoexec.ops.attention.mla_sm90.tests._relations import RELATIONS, Config, run_relation


def _sm100_ok() -> bool:
    try:
        from isoexec.core.arch import live_device_arch

        return torch.cuda.is_available() and live_device_arch() == "sm100"
    except Exception:  # pragma: no cover
        return False


pytestmark = pytest.mark.skipif(not _sm100_ok(), reason="requires an SM100 (B200) device")

SEG = int(os.environ.get("ISOEXEC_MLA_SM100_SEG", 512))
HEADS = [int(h) for h in os.environ.get("ISOEXEC_MLA_SM100_HEADS", "5,10,20").split(",")]

from isoexec.ops.attention.mla_sm100.tests._relations_sm100 import RELATIONS_SM100, run_relation_sm100

# The trainer-op relations need the autograd wrapper, which stays on the sm90 FA3-basis path.  The
# sm90 file's ``cuda_graph`` hard-imports the sm90 kernel and reports n/a for this backend; this
# package states the same claim itself as ``cuda_graph_sm100``.
FWD_RELATIONS = [r for r in RELATIONS if not r.startswith("op_")] + list(RELATIONS_SM100)


def _run(relation, backend, cfg):
    return (run_relation_sm100 if relation in RELATIONS_SM100 else run_relation)(relation, backend, cfg)


@pytest.fixture(scope="module")
def sm100():
    from isoexec.ops.attention.mla_sm100.tests._backend import SM100Backend

    try:
        return SM100Backend(SEG)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"mla_sm100 forward not importable/buildable: {exc}")


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
@pytest.mark.parametrize("relation", FWD_RELATIONS)
def test_relation(sm100, relation, heads):
    torch.backends.cuda.matmul.allow_tf32 = False
    _assert_outcome(_run(relation, sm100, Config(heads=heads, seg=SEG)))
