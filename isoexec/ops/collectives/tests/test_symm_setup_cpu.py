"""Every symmetric-memory rendezvous goes through ``collectives/symm_setup.rendezvous`` (one process-wide setup
counter; a group whose ranks are at different counts is refused by name on every rank instead of hanging).
CPU: torch.distributed and symmetric memory are stand-ins."""
import ast
import types
from pathlib import Path

import pytest
import torch

from isoexec.ops.collectives import symm_setup

ROOT = Path(__file__).resolve().parents[3]          # .../isoexec
#: The counted primitive symm_setup delegates to (pik's process-wide counter): the one other direct caller.
COUNTER = "ops/collectives/pik/allreduce.py"
#: Sites not yet on the entry point, each with its owner. A new direct rendezvous anywhere else fails the scan.
PENDING: dict = {}


def _direct_rendezvous_sites():
    hits = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if "/tests/" in f"/{rel}" or rel == "ops/collectives/symm_setup.py":
            continue
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "rendezvous"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id in ("symm_mem", "_symmetric_memory")):
                hits.append((rel, node.lineno))
    return hits


def test_no_symmetric_rendezvous_bypasses_the_entry_point():
    sites = _direct_rendezvous_sites()
    assert sites, "the scan found no rendezvous at all: it is not looking at the right tree"
    stray = sorted({rel for rel, _ in sites} - set(PENDING) - {COUNTER})
    assert not stray, f"direct symm_mem.rendezvous outside collectives/symm_setup: {stray}"
    assert not any("detreduce_sm100" in rel or "node_local" in rel for rel, _ in sites)
    assert PENDING == {}
    # exactly one counted primitive: pik's rendezvous, the only function in it that calls torch's
    assert [line for rel, line in sites if rel == COUNTER] and symm_setup.counter_owner() == "pik"


class FakeDist:
    """One rank of a group whose peers report ``peer_counts`` / ``peer_bytes`` (None: the same as this rank)."""

    def __init__(self, monkeypatch, ranks=(0, 1), rank=0, peer_counts=None, peer_bytes=None):
        import torch.distributed as dist
        from torch.distributed import _symmetric_memory as symm_mem
        self.calls, self.gathers = [], 0
        group = types.SimpleNamespace(name="g")
        self.group = group
        monkeypatch.setattr(dist, "get_process_group_ranks", lambda g: list(ranks))
        monkeypatch.setattr(dist, "get_world_size", lambda g=None: len(ranks))
        monkeypatch.setattr(dist, "get_rank", lambda g=None: rank)

        def gather(out, mine, group=None):
            self.gathers += 1
            count, nbytes = (int(v) for v in mine)
            counts = peer_counts if peer_counts is not None else [count] * len(ranks)
            sizes = peer_bytes if peer_bytes is not None else [nbytes] * len(ranks)
            out.copy_(torch.tensor([[c, b] for c, b in zip(counts, sizes)]).view(out.shape))
        monkeypatch.setattr(dist, "all_gather_into_tensor", gather)

        def rendezvous(t, g):
            import inspect
            caller = inspect.stack()[1]
            self.calls.append(Path(caller.filename).name)
            return types.SimpleNamespace(buffer_ptrs=[t.data_ptr()] * len(ranks), barrier=lambda: None)
        monkeypatch.setattr(symm_mem, "rendezvous", rendezvous)
        monkeypatch.setattr(symm_mem, "empty", lambda *shape, dtype, device: torch.empty(*shape, dtype=dtype))
        monkeypatch.setattr(symm_setup, "SHARE_PIK_COUNTER", False)
        monkeypatch.setattr(symm_setup, "_RENDEZVOUS", [0])
        monkeypatch.setattr(symm_setup, "_SKEWED", set())


def test_the_tree_workspace_sets_up_through_the_entry_point_only(monkeypatch, tmp_path):
    from isoexec.ops.collectives.detreduce_sm100 import ops
    fake = FakeDist(monkeypatch)
    monkeypatch.setattr(ops, "extension", lambda: types.SimpleNamespace(max_cta=lambda: 4, max_world=lambda: 16,
                                                                       tag_grid=lambda: 8))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setenv("ISOEXEC_SYMM_SETUP_LOG", str(tmp_path))
    ops.TreeWorkspace(fake.group, "cpu", cap=64)
    assert fake.calls == ["symm_setup.py"] * 3 and fake.gathers == 3      # flags, lbox, lybox; each counted
    assert symm_setup.setup_count() == 3
    log = "".join(p.read_text() for p in tmp_path.iterdir()).splitlines()
    assert [line.split()[3] for line in log] == ["site=detreduce_sm100.flags", "site=detreduce_sm100.lbox",
                                                 "site=detreduce_sm100.lybox"]
    assert [line.split()[5] for line in log] == ["count=0", "count=1", "count=2"]


def test_a_skewed_group_is_refused_on_every_rank_without_a_rendezvous(monkeypatch):
    fake = FakeDist(monkeypatch, peer_counts=[0, 1])
    with pytest.raises(symm_setup.SymmetricSetupSkew, match=r"\[0, 1\]"):
        symm_setup.rendezvous(torch.zeros(4), fake.group, site="t")
    assert fake.calls == [] and symm_setup.setup_count() == 0 and symm_setup.skewed(fake.group)
    gathers = fake.gathers
    with pytest.raises(symm_setup.SymmetricSetupSkew, match="already"):       # for good, without another gather
        symm_setup.rendezvous(torch.zeros(4), fake.group, site="t")
    assert fake.gathers == gathers and fake.calls == []
    # remembered by the rank tuple: a re-created group object with the same ranks stays refused
    with pytest.raises(symm_setup.SymmetricSetupSkew, match="already"):
        symm_setup.rendezvous(torch.zeros(4), types.SimpleNamespace(name="again"), site="t")


def test_buffers_of_different_sizes_are_refused_on_every_rank(monkeypatch):
    fake = FakeDist(monkeypatch, peer_bytes=[16, 32])
    with pytest.raises(symm_setup.SymmetricSetupSkew, match=r"buffers of \[16, 32\] bytes"):
        symm_setup.rendezvous(torch.zeros(4), fake.group, site="t")
    assert fake.calls == [] and symm_setup.skewed(fake.group)


def test_a_setup_inside_graph_capture_is_refused_before_any_collective(monkeypatch):
    fake = FakeDist(monkeypatch)
    t = types.SimpleNamespace(is_cuda=True, numel=lambda: 4, element_size=lambda: 4)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(symm_setup.SymmetricSetupInCapture):
        symm_setup.rendezvous(t, fake.group, site="t")
    assert fake.gathers == 0 and fake.calls == []


def test_the_tp_reducer_keeps_the_all_gather_form_on_a_skew():
    from isoexec.models.deepseek_v4_1_flash.providers import ActorTPReducer

    class Tree:
        def prepare(self, device):
            raise symm_setup.SymmetricSetupSkew("skew")
    reducer = ActorTPReducer(Tree(), types.SimpleNamespace(world=2)).prepare("cpu")
    assert reducer.tree_local is False and reducer.prepared is False and reducer.transport() == "all_gather"


def test_pik_counted_rendezvous_is_shared_when_present(monkeypatch):
    fake = FakeDist(monkeypatch)
    monkeypatch.setattr(symm_setup, "SHARE_PIK_COUNTER", True)          # sizes are still agreed here (one gather)
    import importlib

    from isoexec.ops.collectives.pik_bootstrap import ensure_pik
    ensure_pik()
    ar = importlib.import_module("pik.allreduce")
    import sys
    assert sys.modules.get("isoexec.ops.collectives.pik.allreduce", ar) is ar     # one module, one counter

    class Skew(RuntimeError):
        pass
    seen = []

    def pik_rendezvous(t, g):
        seen.append(g)
        ar._RENDEZVOUS[0] += 1
        if len(seen) == 2:
            raise Skew("counts [1, 2]")
        return "handle"
    monkeypatch.setattr(ar, "rendezvous", pik_rendezvous, raising=False)
    monkeypatch.setattr(ar, "_RENDEZVOUS", [7], raising=False)
    monkeypatch.setattr(ar, "SymmetricSetupSkew", Skew, raising=False)
    assert symm_setup.counter_owner() == "pik" and symm_setup.setup_count() == 7
    assert symm_setup.rendezvous(torch.zeros(4), fake.group, site="t") == "handle"
    assert symm_setup.setup_count() == 8 and symm_setup._RENDEZVOUS == [0]      # ONE counter: pik's
    with pytest.raises(symm_setup.SymmetricSetupSkew):
        symm_setup.rendezvous(torch.zeros(4), fake.group, site="t")
    assert symm_setup.skewed(fake.group)
