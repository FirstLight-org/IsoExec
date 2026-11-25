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

from isoexec.ops.attention.mla_sm100 import fwd as _fwd, spec as _spec

#: The spans the relations run at, both outputs of the schedule policy: ``declared`` is the recipe's
#: (spec.py's declarations: two segments of 5120 keys at the longest context), ``batch1`` the same rule at
#: a smallest batch of 1 (32 segments of 320 keys), where the relations' lengths give rows of many
#: segments. ``ISOEXEC_MLA_SM100_SEG`` replaces both with one explicit span (a sensitivity run).
SEGS = {"declared": _fwd.segment_span(),
        "batch1": _fwd.seg_tokens_for(_spec.MODEL_HEADS, _spec.LONGEST_CONTEXT, min_batch=1)}
if os.environ.get("ISOEXEC_MLA_SM100_SEG"):
    SEGS = {"env": int(os.environ["ISOEXEC_MLA_SM100_SEG"])}
HEADS = [int(h) for h in os.environ.get("ISOEXEC_MLA_SM100_HEADS", "5,10,20").split(",")]

from isoexec.ops.attention.mla_sm100.tests._relations_sm100 import RELATIONS_SM100, run_relation_sm100

# The trainer-op relations need the autograd wrapper, which stays on the sm90 FA3-basis path.  The
# sm90 file's ``cuda_graph`` hard-imports the sm90 kernel and reports n/a for this backend; this
# package states the same claim itself as ``cuda_graph_sm100``.
FWD_RELATIONS = [r for r in RELATIONS if not r.startswith("op_")] + list(RELATIONS_SM100)


def _run(relation, backend, cfg):
    return (run_relation_sm100 if relation in RELATIONS_SM100 else run_relation)(relation, backend, cfg)


@pytest.fixture(scope="module", params=list(SEGS))
def sm100(request):
    from isoexec.ops.attention.mla_sm100.tests._backend import SM100Backend

    try:
        return SM100Backend(SEGS[request.param])
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
    seg = sm100.seg_tokens
    if seg == SEGS.get("declared") and len(SEGS) > 1:
        # at the declared span the relations' S-relative fixtures are ten times longer; they run at the model's
        # head count there, and at every head count under ``batch1``
        if relation == "saturated_decode_cut":
            # 512 requests of 8 S + 65 positions, each with every query row: about 145 GB of operands at S = 5120
            # and 5 heads before the kernel's buffers; it does not fit a B200. The cut rule runs under ``batch1``.
            pytest.skip(f"declared span {seg}: the fixture of saturated_decode_cut exceeds the device memory")
        if heads != _spec.MODEL_HEADS and _spec.MODEL_HEADS in HEADS:
            pytest.skip(f"declared span {seg}: run at the model's {_spec.MODEL_HEADS} heads; {heads} heads run under batch1")
    # the relations' fixtures take long_ctx above their own S-relative lengths (2 S + 7, 3 S + 100): 8213 at the
    # spans up to 2048 this module used to run, 4 S + 21 above
    _assert_outcome(_run(relation, sm100, Config(heads=heads, seg=seg, long_ctx=max(8192 + 21, 4 * seg + 21))))
