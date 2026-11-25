"""``moe.symm_rendezvous``: every symmetric-memory rendezvous of the MoE ops and the vendored DeepGEMM goes through it
(source scan); it is a stateless delegate to ``collectives.symm_setup`` (one counter: pik's), and through it a skewed
group is refused on every rank (two real gloo processes on CPU)."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from isoexec.ops.moe import symm_rendezvous as sr

OPS = Path(__file__).resolve().parents[2]
ROOTS = (OPS / "moe", OPS / "_vendor" / "deep_gemm")
ENTRY = OPS / "moe" / "symm_rendezvous.py"
CPP = {".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp"}


def _dotted(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


def _bypasses(source: str, filename: str = "<src>") -> list[str]:
    """Every reference to a rendezvous that is not ``symm_rendezvous.rendezvous``: an attribute ``.rendezvous`` on
    any other base (``symm_mem.rendezvous``, ``torch.distributed._symmetric_memory.rendezvous``,
    ``_SymmetricMemory.rendezvous``), ``from <not symm_rendezvous> import rendezvous``, ``getattr(x, "rendezvous")``."""
    out = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Attribute) and node.attr == "rendezvous":
            if not _dotted(node.value).endswith("symm_rendezvous"):
                out.append(f"{filename}:{node.lineno} {_dotted(node) or '<expr>.rendezvous'}")
        elif isinstance(node, ast.ImportFrom) and any(a.name == "rendezvous" for a in node.names):
            if not (node.module or "").endswith("symm_rendezvous"):
                out.append(f"{filename}:{node.lineno} from {node.module} import rendezvous")
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
              and len(node.args) > 1 and isinstance(node.args[1], ast.Constant) and node.args[1].value == "rendezvous"):
            out.append(f"{filename}:{node.lineno} getattr(..., 'rendezvous')")
    return out


def _library_files():
    for root in ROOTS:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "tests" in path.relative_to(root).parts or "__pycache__" in path.parts:
                continue
            if path.suffix == ".py" or path.suffix in CPP:
                yield path


def test_scanner_catches_every_bypass_form():
    bad = """
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._symmetric_memory import rendezvous
from torch._C._distributed_c10d import _SymmetricMemory
a = symm_mem.rendezvous(buf, group=g)
b = torch.distributed._symmetric_memory.rendezvous(buf, g)
c = _SymmetricMemory.rendezvous(buf, "name")
d = getattr(symm_mem, "rendezvous")(buf, g)
f = symm_mem.rendezvous
"""
    assert len(_bypasses(bad)) == 6
    good = """
from isoexec.ops.moe import symm_rendezvous
from isoexec.ops.moe.symm_rendezvous import rendezvous as _r
h = symm_rendezvous.rendezvous(buf, g)
k = isoexec.ops.moe.symm_rendezvous.rendezvous(buf, g)
"""
    assert _bypasses(good) == []


def test_no_symmetric_memory_rendezvous_bypasses_the_entry_point():
    offenders, routed = [], set()
    for path in _library_files():
        text = path.read_text(errors="replace")
        rel = path.relative_to(OPS).as_posix()
        if path.suffix in CPP:
            if "symmetric_memory::rendezvous" in text or "SymmetricMemory::rendezvous" in text:
                offenders.append(f"{rel}: C++ rendezvous")
            continue
        if path == ENTRY:
            continue
        offenders += _bypasses(text, rel)
        if "symm_rendezvous" in text and "rendezvous(" in text:
            routed.add(rel)
    assert not offenders, "symmetric-memory rendezvous outside moe.symm_rendezvous: " + "; ".join(offenders)
    # the scan is not vacuous: the known sites route through the entry point
    assert {"moe/deepseek_v41_sm100/tp_exchange.py", "moe/deepgemm_sm100/mega_ep.py",
            "_vendor/deep_gemm/mega/__init__.py"} <= routed, routed


def test_the_entry_point_is_a_stateless_delegate():
    """No direct rendezvous and no counter here: every setup is ``symm_setup.rendezvous`` (pik's counter)."""
    from isoexec.ops.collectives import symm_setup
    tree = ast.parse(ENTRY.read_text())
    direct = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "rendezvous"
              and _dotted(n.value) in ("symm_mem", "torch.distributed._symmetric_memory", "_symmetric_memory")]
    assert direct == []
    assert not hasattr(sr, "_SETUPS") and not hasattr(sr, "_SKEWED")
    assert sr.RendezvousSkewError is symm_setup.SymmetricSetupSkew
    assert symm_setup.counter_owner() == "pik"


class _Group:
    pass


def test_refused_inside_capture(monkeypatch):
    from isoexec.ops.collectives import symm_setup
    calls = []
    monkeypatch.setattr(symm_setup, "rendezvous", lambda *a, **k: calls.append(a) or "handle")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="inside CUDA graph capture"):
        sr.rendezvous(torch.zeros(4), _Group())
    assert calls == []


def test_megamoe_pool_key_is_the_rank_tuple(monkeypatch):
    from isoexec.ops.moe.deepseek_v41_sm100 import mega
    ranks = {}
    monkeypatch.setattr(sr, "group_ranks", lambda g: ranks[g])
    tp, ep, ep_again = _Group(), _Group(), _Group()
    ranks.update({tp: (0, 1), ep: (2, 3), ep_again: (2, 3)})
    key = lambda g: mega.Workspace.pool_key(g, "cuda:0", 8, 2, 256, 128, 64)
    assert key(tp) != key(ep)                    # two groups of one size never share a buffer
    assert key(ep) == key(ep_again)              # the same peers: one buffer, one rendezvous
    assert key(tp)[0] == (0, 1) and mega.Workspace.pool_key(None, "cuda:0", 8, 2, 256, 128, 64)[0] == "local"
    assert key(tp) != mega.Workspace.pool_key(tp, "cuda:0", 8, 2, 256, 128, 32)     # rows per rank is part of it


# --------------------------------------------------------------------------------------- two real CPU processes

def _two_ranks(tmp_path, scenario):
    child = Path(__file__).with_name("_symm_rendezvous_child.py")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GLOO_SOCKET_IFNAME=os.environ.get("GLOO_SOCKET_IFNAME", "lo"))
    init = tmp_path / "store"
    procs = [subprocess.Popen([sys.executable, str(child), str(r), "2", str(init), str(tmp_path / f"r{r}.json"),
                               scenario], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for r in range(2)]
    logs = []
    try:
        for p in procs:
            logs.append(p.communicate(timeout=120)[0])
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    for r, p in enumerate(procs):
        assert p.returncode == 0, (r, logs[r][-3000:])
    return [json.loads((tmp_path / f"r{r}.json").read_text()) for r in range(2)]


def test_two_processes_a_lone_setup_is_refused_on_every_rank(tmp_path):
    r0, r1 = _two_ranks(tmp_path, "lone")
    s0, s1 = ({name: (verdict, n) for name, verdict, n in r["steps"]} for r in (r0, r1))
    for s in (s0, s1):
        assert s["world"][0] == s["pair"][0] == s["world again"][0] == "ok"
        assert s["own singleton"][0] == s["world after symmetric singletons"][0] == "ok"
        assert s["world after lone"][0] == "skew" and s["pair after lone"][0] == "skew"
    assert s0["lone singleton"] == ("ok", 6) and "lone singleton" not in s1
    assert s0["world after lone"][1] == 6 and s1["world after lone"][1] == 5
    assert r0["skewed"] and r1["skewed"] and r0["done"] and r1["done"]


def test_two_processes_different_sizes_are_refused_on_every_rank(tmp_path):
    for r in _two_ranks(tmp_path, "size"):
        assert r["steps"] == [["sizes differ", "skew", 0]] and r["skewed"] and r["done"], r
