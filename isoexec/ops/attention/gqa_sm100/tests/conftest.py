"""The GPU batteries run under two declarations of the optional target batch.

``default`` declares none (the deployment default): the schedule policy gives one segment of
``LONGEST_CONTEXT`` = 10240 keys. ``batch1`` declares a target batch of 1, at which the policy gives 16
segments of 640 keys, so the relations that need rows of several segments (the tree, the fold run, the decode
leaf's segment split, the combine) are exercised at the lengths these modules use. Both are outputs of
``ops.schedule_for``; neither is a literal. Under each, every batch size these modules run gives the same bits.
"""

from __future__ import annotations

import pytest

#: label -> target batch (``None``: no target batch declared)
DECLARATIONS = {"default": None, "batch1": 1}


@pytest.fixture(params=list(DECLARATIONS))
def declaration(request, monkeypatch):
    from isoexec.ops.attention.gqa_sm100 import ops, spec

    monkeypatch.setattr(spec, "TARGET_BATCH", DECLARATIONS[request.param])
    ops.bind_schedule()          # the policy runs when the schedule is bound, never on a call
    yield request.param
    monkeypatch.undo()
    ops.bind_schedule()
