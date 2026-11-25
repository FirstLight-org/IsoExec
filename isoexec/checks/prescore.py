"""Framework-free exact coverage for an online, already-compared prescore batch."""

from dataclasses import dataclass
from itertools import product

from . import Refusal


@dataclass(frozen=True, order=True)
class RowIdentity:
    sample: int
    response: int
    token: int
    prefix: tuple[int, ...]

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in (self.sample, self.response, self.token)):
            raise ValueError("row identity requires nonnegative integer coordinates")
        if not self.prefix or any(type(t) is not int or t < 0 for t in self.prefix):
            raise ValueError("row identity requires the exact nonempty predictor prefix")


@dataclass(frozen=True)
class PrescorePlan:
    id: str
    step: int
    vocab: int
    # One native forward RPC per entry; rows are global across DP owners. The
    # live reports establish DP ownership and the attested mesh shape.
    calls: tuple[tuple[RowIdentity, ...], ...]

    def __post_init__(self):
        if not self.id:
            raise ValueError("prescore requires a fresh identity")
        if type(self.step) is not int or self.step < 0 or type(self.vocab) is not int or self.vocab < 1:
            raise ValueError("invalid prescore step/vocabulary")
        if not self.calls:
            raise ValueError("prescore requires at least one native forward call")
        if any(not call for call in self.calls):
            raise ValueError("every native forward call must own at least one expected row")
        rows = self.rows
        if not rows or len(rows) != len(set(rows)):
            raise ValueError("expected rows must be nonempty and uniquely assigned across calls/DP")
        if any(row.token >= self.vocab for row in rows):
            raise ValueError("sampled token is outside the declared vocabulary")

    @property
    def rows(self):
        return tuple(row for call in self.calls for row in call)


@dataclass(frozen=True)
class ComparedShard:
    row: RowIdentity
    start: int
    stop: int
    checked_elements: int


@dataclass(frozen=True)
class PrescoreReport:
    plan_id: str
    step: int
    call: int
    coordinate: tuple[int, int, int, int]
    shape: tuple[int, int, int, int]
    # Derived from incoming tensors before the native schedule, not from output.
    input_rows: tuple[RowIdentity, ...]
    # Produced only after the real comparator successfully checked the slice.
    shards: tuple[ComparedShard, ...]
    completed: bool
    errors: tuple[str, ...] = ()


def _validate_participants(plan, reports, calls=None):
    """Return the attested mesh and participant/input ownership, or refuse."""
    def refuse(reason):
        raise Refusal("prescore.coverage: " + reason)

    if not reports:
        refuse("no trainer participant reported")
    shapes = {report.shape for report in reports if isinstance(report, PrescoreReport)}
    if len(shapes) != 1:
        refuse(f"trainer ranks disagree on the resolved DP/PP/CP/TP shape: {sorted(shapes)}")
    shape = shapes.pop()
    if len(shape) != 4 or any(type(n) is not int or n < 1 for n in shape):
        refuse(f"invalid resolved DP/PP/CP/TP shape {shape}")
    participants = tuple(product(*(range(n) for n in shape)))
    calls = set(range(len(plan.calls))) if calls is None else set(calls)
    if not calls or any(type(call) is not int or not 0 <= call < len(plan.calls) for call in calls):
        refuse(f"invalid native forward call selection {sorted(calls)}")
    expected = {(call, coord) for call in calls for coord in participants}
    seen, inputs = set(), {}
    for report in reports:
        if not isinstance(report, PrescoreReport):
            refuse("invalid report type")
        if (report.plan_id, report.step) != (plan.id, plan.step):
            refuse("report belongs to a different batch or step")
        member = (report.call, report.coordinate)
        if member not in expected or member in seen:
            refuse(f"unknown or duplicate participant {member}")
        seen.add(member)
        if len(report.input_rows) != len(set(report.input_rows)):
            refuse(f"participant {member} received duplicate input rows")
        dp_key = (report.call, report.coordinate[0])
        prior = inputs.setdefault(dp_key, set(report.input_rows))
        if prior != set(report.input_rows):
            refuse(f"participants disagree on DP owner {dp_key}'s input rows")
    if seen != expected:
        refuse(f"missing completed participants: {sorted(expected - seen)}")
    for call in sorted(calls):
        expected_rows = plan.calls[call]
        owners = [inputs[(call, dp)] for dp in range(shape[0])]
        if any(owners[a] & owners[b] for a in range(len(owners)) for b in range(a + 1, len(owners))):
            refuse(f"call {call} assigned one logical row to multiple DP owners")
        if set().union(*owners) != set(expected_rows):
            refuse(f"call {call} DP owners do not tile the complete expected row set")
    return shape, expected


def verify_preflight(plan: PrescorePlan, reports: list[PrescoreReport], call: int) -> dict:
    """Coordinate every rank before one native schedule can enter collectives."""
    shape, _ = _validate_participants(plan, reports, {call})
    return {"status": "PASS", "participants": len(reports), "shape": shape}


def verify_prescore(plan: PrescorePlan, reports: list[PrescoreReport]) -> dict:
    """Fail closed on any missing, duplicate, foreign or incomplete participant."""
    def refuse(reason):
        raise Refusal("prescore.coverage: " + reason)

    shape, expected = _validate_participants(plan, reports)
    slices = {}
    for report in reports:
        member = (report.call, report.coordinate)
        if report.completed is not True or report.errors:
            refuse(f"participant {member} did not complete successfully: {report.errors}")
        dp, pp, cp, tp = report.coordinate
        if pp != shape[1] - 1 and report.shards:
            refuse(f"non-last pipeline stage {member} claimed logprob comparisons")
        input_rows = set(report.input_rows)
        for shard in report.shards:
            if not isinstance(shard, ComparedShard) or shard.row not in input_rows:
                refuse(f"participant {member} compared a foreign row/prefix/token")
            if (any(type(n) is not int for n in (shard.start, shard.stop, shard.checked_elements))
                    or not 0 <= shard.start < shard.stop <= plan.vocab
                    or shard.checked_elements != shard.stop - shard.start):
                refuse(f"participant {member} did not compare its complete vocabulary slice")
            slices.setdefault(shard.row, []).append((shard.start, shard.stop, cp, tp))
    if set(slices) != set(plan.rows):
        refuse("compared rows do not cover the complete expected batch")
    elements = 0
    for row, parts in slices.items():
        parts.sort()
        if len(parts) != shape[3] or {p[3] for p in parts} != set(range(shape[3])):
            refuse(f"row {row.sample}/{row.response} has missing or duplicate TP owners")
        if len({p[2] for p in parts}) != 1:
            refuse(f"row {row.sample}/{row.response} is split across different CP owners")
        cursor = 0
        for start, stop, _, _ in parts:
            if start != cursor:
                refuse(f"row {row.sample}/{row.response} has overlapping slices or a vocabulary gap")
            elements += stop - start
            cursor = stop
        if cursor != plan.vocab:
            refuse(f"row {row.sample}/{row.response} does not reach the end of the vocabulary")
    return {
        "status": "PASS", "plan_id": plan.id, "global_step": plan.step,
        "rows": len(plan.rows), "vocab_size": plan.vocab, "compared_elements": elements,
        "participant_completions": len(expected), "forward_calls": len(plan.calls),
        "shape_dp_pp_cp_tp": list(shape),
    }
