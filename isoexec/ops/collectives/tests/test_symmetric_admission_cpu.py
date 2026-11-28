"""Symmetric-memory peer admission (``node_local.symmetric_admission``): one gathered (host, device UUID, ordinal)
row per rank, a pure decision every rank derives alike, a named reason; the tree reducer and the MegaMoE workspace
refuse by name before any symmetric-memory call; a real two-process gloo world takes the same verdict on every rank."""
import os
import tempfile

import pytest
import torch

from isoexec.ops.collectives import node_local
from isoexec.ops.collectives.node_local import PeerAdmission, classify_peers


@pytest.fixture(autouse=True)
def _clear():
    node_local._CACHE.clear()
    node_local._ADMISSION.clear()
    yield
    node_local._CACHE.clear()
    node_local._ADMISSION.clear()


H = "node-a/boot-1"


@pytest.mark.parametrize("rows, symmetric, reason", [
    ([(H, "GPU-0", 0, False)], True, "single_rank"),
    ([(H, "GPU-0", 0, False), (H, "GPU-1", 1, False)], True, "one_host_distinct_gpus"),          # real ranks
    ([(H, "GPU-1", 1, False), (H, "GPU-1", 1, False)], False, "shared_device"),                  # oversubscribed
    ([(H, "GPU-0", 0, False), (H, "GPU-0", 0, False), (H, "GPU-1", 1, False), (H, "GPU-1", 1, False)],
     False, "shared_device"),                                                                     # 4 on 2 GPUs
    ([(H, "GPU-0", 0, False), ("node-b/boot-2", "GPU-1", 0, False)], False, "multi_host"),       # fake nodes
    ([(H, "GPU-0", 0, False), ("node-a/boot-2", "GPU-1", 1, False)], False, "multi_host"),       # same name, rebooted
    ([(H, "GPU-0", 0, False), (H, "GPU-1", 0, False)], False, "overlapping_ordinals"),           # Ray: cuda:0 each
    ([(H, "GPU-0", 0, True), (H, "GPU-1", 0, True)], True, "one_host_distinct_gpus"),            # overlap allowed
    ([(H, "GPU-0", 0, True), (H, "GPU-1", 0, False)], False, "overlapping_ordinals"),            # allowed on one only
    ([(H, "", -1, False), (H, "", -1, False)], False, "no_cuda_device"),
])
def test_classify(rows, symmetric, reason):
    got = classify_peers(rows)
    assert isinstance(got, PeerAdmission) and (got.symmetric, got.reason) == (symmetric, reason)
    assert got.detail


def test_hosts_are_checked_before_devices():
    # two hosts whose ranks also collide on a UUID string: the host reason wins (it is the one that matters)
    rows = [(H, "GPU-0", 0, False), ("node-b/boot-2", "GPU-0", 0, False)]
    assert classify_peers(rows).reason == "multi_host"


def test_shared_device_names_ranks_and_device():
    got = classify_peers([(H, "GPU-x", 0, False), (H, "GPU-y", 1, False), (H, "GPU-x", 0, False)])
    assert got.reason == "shared_device" and "ranks [0, 2] share one GPU (GPU-x)" in got.detail


def _fake_world(monkeypatch, rows):
    import torch.distributed as dist
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: len(rows))
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)

    def gather(out, obj, group=None):
        calls.append(obj)
        out[:] = rows
    monkeypatch.setattr(dist, "all_gather_object", gather)
    return calls


def test_admission_is_one_cached_collective_per_group(monkeypatch):
    calls = _fake_world(monkeypatch, [(H, "GPU-1", 1, False), (H, "GPU-1", 1, False)])
    group = object()
    a = node_local.symmetric_admission(group)
    b = node_local.symmetric_admission(group)
    assert a == b and a.reason == "shared_device" and len(calls) == 1
    node_local.symmetric_admission(object())       # another group object: its own gather
    assert len(calls) == 2


def test_admission_refuses_inside_capture(monkeypatch):
    _fake_world(monkeypatch, [(H, "GPU-0", 0, False), (H, "GPU-1", 1, False)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        node_local.symmetric_admission(object())


def test_require_names_mechanism_and_reason(monkeypatch):
    _fake_world(monkeypatch, [(H, "GPU-1", 1, False), (H, "GPU-1", 1, False)])
    with pytest.raises(node_local.SymmetricPeersError, match=r"mech Y .*shared_device: ranks \[0, 1\] share one GPU"):
        node_local.require_symmetric_peers(object(), "mech Y")
    assert issubclass(node_local.SymmetricPeersError, node_local.NodeLocalityError)


def test_require_across_hosts_keeps_the_host_error_without_a_second_gather(monkeypatch):
    calls = _fake_world(monkeypatch, [(H, "GPU-0", 0, False), ("node-b/boot-2", "GPU-0", 0, False)])
    with pytest.raises(node_local.NodeLocalityError, match="mech Z is node-local.*spans 2 hosts") as info:
        node_local.require_symmetric_peers(object(), "mech Z")
    assert type(info.value) is node_local.NodeLocalityError and len(calls) == 1


def test_tree_reducer_refuses_a_shared_device_before_symmetric_memory(monkeypatch):
    from isoexec.ops.collectives.detreduce_sm100 import ops
    _fake_world(monkeypatch, [(H, "GPU-1", 1, False), (H, "GPU-1", 1, False)])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def no_symm(*a, **k):
        raise AssertionError("symmetric memory touched before the peer check")
    monkeypatch.setattr(ops, "TreeWorkspace", no_symm)
    with pytest.raises(node_local.SymmetricPeersError, match="detreduce_sm100 TreeReducer.*shared_device"):
        ops.TreeReducer(object()).prepare("cpu")


def test_megamoe_workspace_refuses_a_shared_device_before_symmetric_memory(monkeypatch):
    from types import SimpleNamespace as NS

    from isoexec.ops.moe.deepseek_v41_sm100 import mega
    _fake_world(monkeypatch, [(H, "GPU-1", 1, False), (H, "GPU-1", 1, False)])

    def no_symm(*a, **k):
        raise AssertionError("symmetric memory touched before the peer check")
    monkeypatch.setattr(mega, "_dg", lambda: NS(SymmBuffer=no_symm))
    with pytest.raises(node_local.SymmetricPeersError, match="BF16 MegaMoE.*shared_device"):
        mega.Workspace(object(), "cpu", 256, 8, 6, 4096, 2048)


# --------------------------------------------------------------------- real two-process world (gloo, CPU)
def _body(rank, world, out, init_method):
    import torch.distributed as dist

    from isoexec.models.deepseek_v4_1_flash import providers
    from isoexec.ops.collectives import node_local as nl
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world)
    res = {}
    scenarios = {
        "shared_device": lambda r: ("node-a/b", "GPU-7", 7, False),
        "distinct": lambda r: ("node-a/b", f"GPU-{r}", r, False),
        "multi_host": lambda r: (f"node-{r}/b", f"GPU-{r}", 0, False),
    }
    for name, row in scenarios.items():
        nl._ADMISSION.clear()
        nl.peer_row = lambda device=None, row=row: row(rank)
        prepared = []

        class Tree:
            group = dist.group.WORLD

            def prepare(self, device):
                prepared.append(device)

        class Gather:
            world = 2
        providers.TRANSPORT_RECEIPTS["tp_reducer"].clear()
        red = providers.ActorTPReducer(Tree(), Gather())
        red.prepare("cpu")
        red.prepare("cpu")                      # a second row of the group: already decided, no new collective
        res[name] = (red.tree_local, red.prepared, len(prepared),
                     list(providers.TRANSPORT_RECEIPTS["tp_reducer"].values()))
    dist.destroy_process_group()
    out[rank] = res


def test_two_ranks_agree_on_the_verdict_and_the_receipt():
    from isoexec.ops.collectives.tests._gloo_harness import run_world
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        res = run_world(_body, world=2, rendezvous=os.path.join(d, "rdv"))
    assert res[0] == res[1]
    assert res[0]["shared_device"] == (False, False, 0, ["all_gather:shared_device"])
    assert res[0]["multi_host"] == (False, False, 0, ["all_gather:multi_host"])
    assert res[0]["distinct"] == (True, True, 1, ["tree"])
