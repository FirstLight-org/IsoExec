"""Counterexamples with equal total counts must still fail exact online coverage."""

from dataclasses import replace
from itertools import product

import pytest

from isoexec.checks import Refusal
from isoexec.checks.prescore import ComparedShard, PrescorePlan, PrescoreReport, RowIdentity, verify_prescore


def example(shape=(2, 2, 2, 2), calls=2):
    owners = tuple(tuple((RowIdentity(c * shape[0] + dp, 0, 3, (10, 11)),) for dp in range(shape[0]))
                   for c in range(calls))
    rows = tuple(tuple(row for owner in call for row in owner) for call in owners)
    plan = PrescorePlan("batch-one", 1, 8, rows)
    reports = []
    for call in range(calls):
        for coord in product(*(range(n) for n in shape)):
            dp, pp, cp, tp = coord
            shards = ()
            # CP1 has no selected rows in this workload but must participate.
            if pp == shape[1] - 1 and cp == 0:
                start, stop = tp * 8 // shape[3], (tp + 1) * 8 // shape[3]
                shards = tuple(ComparedShard(row, start, stop, stop - start) for row in owners[call][dp])
            reports.append(PrescoreReport(plan.id, plan.step, call, coord, shape, owners[call][dp], shards, True))
    return plan, reports


@pytest.mark.parametrize("shape", [(1, 1, 1, 1), (1, 1, 1, 2), (2, 1, 1, 1), (1, 2, 1, 1),
                                   (1, 1, 2, 1), (1, 2, 1, 2), (2, 1, 1, 2), (2, 2, 2, 2)])
def test_complete_mesh_with_multiple_calls_and_empty_cp_owner(shape):
    plan, reports = example(shape)
    result = verify_prescore(plan, reports[::-1])
    assert result["rows"] == 2 * shape[0]
    assert result["compared_elements"] == result["rows"] * 8
    assert result["participant_completions"] == len(reports)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "wrong_batch", "wrong_step",
                                     "wrong_call", "incomplete", "error", "wrong_input", "duplicate_input", "unknown_rank",
                                     "gap", "overlap", "short_compare", "dropped_row", "wrong_prefix",
                                     "wrong_token", "wrong_dp", "pp_claims_rows", "split_cp"])
def test_invalid_reports_are_never_success(damage):
    plan, reports = example()
    index = next(i for i, r in enumerate(reports) if r.shards)
    report = reports[index]
    shard = report.shards[0]
    if damage == "missing":
        reports.pop()
    elif damage == "duplicate":
        reports[-1] = reports[-2]  # Same aggregate participant count.
    elif damage == "wrong_batch":
        reports[index] = replace(report, plan_id="other")
    elif damage == "wrong_step":
        reports[index] = replace(report, step=2)
    elif damage == "wrong_call":
        reports[index] = replace(report, call=99)
    elif damage == "incomplete":
        reports[index] = replace(report, completed=False)
    elif damage == "error":
        reports[index] = replace(report, errors=("full_distribution.bits",))
    elif damage == "wrong_input":
        reports[index] = replace(report, input_rows=())
    elif damage == "duplicate_input":
        reports[index] = replace(report, input_rows=report.input_rows * 2)
    elif damage == "unknown_rank":
        reports[index] = replace(report, coordinate=(9, 9, 9, 9))
    elif damage in ("gap", "overlap", "short_compare"):
        changed = {"gap": dict(start=1, checked_elements=3), "overlap": dict(stop=5, checked_elements=5),
                   "short_compare": dict(checked_elements=3)}[damage]
        reports[index] = replace(report, shards=(replace(shard, **changed),))
    elif damage == "dropped_row":
        reports[index] = replace(report, shards=())
    elif damage in ("wrong_prefix", "wrong_token", "wrong_dp"):
        changed = {"wrong_prefix": dict(prefix=(11, 10)), "wrong_token": dict(token=4), "wrong_dp": dict(sample=99)}[damage]
        reports[index] = replace(report, shards=(replace(shard, row=replace(shard.row, **changed)),))
    elif damage == "pp_claims_rows":
        reports[0] = replace(reports[0], shards=report.shards)
    else:
        peer = next(i for i, r in enumerate(reports) if r.call == report.call and r.coordinate == (0, 1, 1, 0))
        reports[peer] = replace(reports[peer], shards=report.shards)
        reports[index] = replace(report, shards=())  # Full vocab covered, but the two TP ranks disagree on CP owner.
    with pytest.raises(Refusal, match="prescore.coverage"):
        verify_prescore(plan, reports)


def test_identical_values_do_not_allow_duplicate_row_occurrences():
    plan, reports = example((1, 1, 1, 2))
    # Replace second-call comparisons by first-call rows; total elements unchanged.
    for i, report in enumerate(reports):
        if report.call == 1:
            reports[i] = replace(report, shards=tuple(replace(s, row=plan.calls[0][0]) for s in report.shards))
    with pytest.raises(Refusal):
        verify_prescore(plan, reports)


def test_empty_expected_batch_refuses():
    plan, _ = example()
    with pytest.raises(ValueError):
        replace(plan, calls=((),))
    with pytest.raises(ValueError):
        replace(plan, calls=(plan.calls[0], ()))


def test_dp_owner_union_and_disjointness_are_exact_not_counts():
    plan, reports = example((2, 1, 1, 1))
    first = plan.calls[0][0]
    for i, report in enumerate(reports):
        if report.call == 0 and report.coordinate[0] == 1:
            replacement = first
            reports[i] = replace(report, input_rows=(replacement,),
                                 shards=tuple(replace(s, row=replacement) for s in report.shards))
    with pytest.raises(Refusal, match="multiple DP owners"):
        verify_prescore(plan, reports)
