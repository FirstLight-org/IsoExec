"""CPU contract of ``TreeReducer.holds_execution_scope``: true exactly inside the calling thread's scope."""
from types import SimpleNamespace as NS


def test_tree_reducer_reports_its_scope_owner():
    from isoexec.ops.collectives.detreduce_sm100.ops import TreeReducer
    from isoexec.ops.collectives.detreduce_sm100.phased import ExecutionOwnership
    red = object.__new__(TreeReducer)
    red.world = 2
    red._phased = NS(ownership=ExecutionOwnership(lambda: "s", lambda: False, lambda: None))
    assert not red.holds_execution_scope()
    with red._phased.ownership.scope():
        assert red.holds_execution_scope()
    assert not red.holds_execution_scope()
