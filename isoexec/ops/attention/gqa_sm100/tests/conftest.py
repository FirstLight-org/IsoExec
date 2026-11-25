"""The GPU batteries run under two declarations of the smallest batch.

``declared`` is spec.py's (128: one segment of ``LONGEST_CONTEXT`` = 10240 keys). ``batch1`` declares a smallest
batch of 1, at which the schedule policy gives 16 segments of 640 keys, so the relations that need rows
of several segments (the tree, the fold run, the decode leaf's segment split, the combine) are exercised
at the lengths these modules use. Both are outputs of ``ops.schedule_for``; neither is a literal.
"""

from __future__ import annotations

import pytest

DECLARATIONS = {"declared": None, "batch1": 1}


@pytest.fixture(params=list(DECLARATIONS))
def declaration(request, monkeypatch):
    from isoexec.ops.attention.gqa_sm100 import ops, spec

    batch = DECLARATIONS[request.param]
    if batch is not None:
        monkeypatch.setattr(spec, "SMALLEST_BATCH", batch)
        ops.bind_schedule()          # the policy runs when the schedule is bound, never on a call
    yield request.param
    monkeypatch.undo()
    ops.bind_schedule()
