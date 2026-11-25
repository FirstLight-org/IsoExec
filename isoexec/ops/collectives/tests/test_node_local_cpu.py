"""Node-locality admission: a node-local mechanism refuses, by name, a group whose ranks span hosts, and the
tree reducer checks it before any symmetric-memory call."""
import pytest

from isoexec.ops.collectives import node_local


@pytest.fixture(autouse=True)
def _clear():
    node_local._CACHE.clear()
    node_local._ADMISSION.clear()
    yield
    node_local._CACHE.clear()
    node_local._ADMISSION.clear()


def _fake_world(monkeypatch, hosts):
    import torch
    import torch.distributed as dist

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    monkeypatch.setattr(dist, "get_world_size", lambda group=None: len(hosts))

    def gather(out, obj, group=None):
        # host strings for group_hosts, (host, uuid, ordinal, allow) rows for symmetric_admission
        out[:] = hosts if isinstance(obj, str) else [(h, f"GPU-{i}", i, False) for i, h in enumerate(hosts)]
    monkeypatch.setattr(dist, "all_gather_object", gather)


def test_one_host_admits(monkeypatch):
    _fake_world(monkeypatch, ["a/1", "a/1"])
    group = object()
    assert node_local.one_host(group)
    node_local.require_one_host(group, "mech")


def test_two_hosts_refused_by_name(monkeypatch):
    _fake_world(monkeypatch, ["a/1", "b/2", "a/1", "b/2"])
    with pytest.raises(node_local.NodeLocalityError, match=r"mech X is node-local.*2 hosts.*'a/1': \[0, 2\]"):
        node_local.require_one_host(object(), "mech X")


def test_same_hostname_other_boot_is_another_host(monkeypatch):
    _fake_world(monkeypatch, ["a/1", "a/2"])
    assert not node_local.one_host(object())


def test_tree_reducer_checks_before_symmetric_memory(monkeypatch):
    import torch
    import torch.distributed as dist

    from isoexec.ops.collectives.detreduce_sm100 import ops

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    _fake_world(monkeypatch, ["a/1", "b/2"])
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)

    def no_symm(*a, **k):
        raise AssertionError("symmetric memory touched before the host check")
    monkeypatch.setattr(ops, "TreeWorkspace", no_symm)
    reducer = ops.TreeReducer(object())
    with pytest.raises(node_local.NodeLocalityError, match="detreduce_sm100 TreeReducer"):
        reducer.prepare("cpu")
