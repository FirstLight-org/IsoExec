"""CPU contract checks of detreduce_sm100 (no GPU, no process group)."""
from __future__ import annotations

import threading

import pytest

from isoexec.ops.collectives.detreduce_sm100 import ops, phased


def test_workspace_memory_is_independent_of_world_cpu():
    # the formula takes no world argument: every box slot is written by exactly one rank
    assert ops.tree_bytes() == 2 * ops.WORKSPACE_CAP * 4 + ops.WORKSPACE_CAP * 4 + ops.SYNC_BYTES
    assert ops.tree_bytes(1 << 20) < ops.tree_bytes(4 << 20)


def test_transport_id_and_the_one_transport_cpu():
    assert ops.TREE_TRANSPORT == "symm_tree_allreduce_v1"
    assert not hasattr(ops, "tree_allreduce") and not hasattr(ops, "terms_combine")
    import inspect
    assert "transport_mode" not in inspect.signature(ops.TreeReducer).parameters


def test_kernel_constants_agree_with_the_sync_layout_cpu():
    from pathlib import Path
    import re

    cu = (Path(ops.__file__).parent / "csrc" / "detreduce.cu").read_text()
    for name, value in (("kMaxCta", ops.MAX_CTA), ("kMaxWorld", ops.MAX_WORLD), ("kTagGrid", ops.TAG_GRID)):
        assert int(re.search(rf"constexpr int {name} = (\d+);", cu).group(1)) == value, name


def _ownership(stream="s0", capturing=False):
    state = {"stream": stream, "capturing": capturing, "syncs": 0}
    own = phased.ExecutionOwnership(lambda: state["stream"], lambda: state["capturing"],
                                    lambda: state.__setitem__("syncs", state["syncs"] + 1))
    return own, state


def test_execution_outside_the_owning_scope_refuses_cpu():
    own, _ = _ownership()
    with pytest.raises(RuntimeError, match="owning runtime"):
        own.validate()
    with own.scope():
        own.validate()


def test_a_changed_runtime_stream_refuses_cpu():
    own, state = _ownership()
    with own.scope():
        own.validate()
    state["stream"] = "s1"
    with pytest.raises(RuntimeError, match="stream changed"):
        with own.scope():
            pass


def test_concurrent_submission_refuses_cpu():
    own, _ = _ownership()
    errors = []
    entered, release = threading.Event(), threading.Event()

    def hold():
        with own.scope():
            entered.set()
            release.wait(5)

    t = threading.Thread(target=hold)
    t.start()
    entered.wait(5)
    try:
        with own.scope():
            pass
    except RuntimeError as e:
        errors.append(str(e))
    release.set()
    t.join()
    assert errors and "concurrent" in errors[0]


def test_capture_requires_entering_the_scope_first_cpu():
    own, state = _ownership(capturing=True)
    with pytest.raises(RuntimeError, match="before CUDA capture"):
        with own.scope():
            pass
