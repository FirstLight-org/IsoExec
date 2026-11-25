"""Check P2P fallback counters, warnings and transport hooks, and the engine transport's arch policy and flag.
CPU stubs exercise dispatch and observability; GPU tests cover transport arithmetic."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
import types

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_COLLECTIVES_DIR = os.path.dirname(_HERE)


def _load_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ``pik_bootstrap`` top-level imports are importlib/logging/pathlib/sys/threading, so it loads
# standalone -- and ``ensure_pik`` is the ONLY supported way to get the vendored package (it must
# be registered as top-level ``pik`` or the codegen'd kernels cannot resolve their own imports).
_BOOT = _load_by_path("_ix_pik_bootstrap", os.path.join(_COLLECTIVES_DIR, "pik_bootstrap.py"))
_BOOT.ensure_pik()
import pik.allreduce as AR  # type: ignore  # noqa: E402


def _reset_transport_counters():
    # inlined: the public module keeps the state but not this test hook
    for k in AR._TRANSPORT_COUNTS:
        AR._TRANSPORT_COUNTS[k] = 0
    AR._FIRST_FALLBACK = {}
    AR._LAST_STATE = "none"


# ``pik_tp_invariant`` relative-imports its sibling, so it needs A package -- a shell whose
# ``__path__`` is ops/collectives. Its own top level is logging/os/torch + that sibling; every
# megatron import in it is function-local, which is what lets this stay a CPU test.
_PKG = "_ix_collectives_pkg"
_shell = types.ModuleType(_PKG)
_shell.__path__ = [_COLLECTIVES_DIR]
_shell.__spec__ = importlib.machinery.ModuleSpec(_PKG, loader=None, is_package=True)
_shell.__spec__.submodule_search_locations = list(_shell.__path__)
sys.modules[_PKG] = _shell
_load_by_path(f"{_PKG}.pik_bootstrap", os.path.join(_COLLECTIVES_DIR, "pik_bootstrap.py"))
PTI = _load_by_path(f"{_PKG}.pik_tp_invariant", os.path.join(_COLLECTIVES_DIR, "pik_tp_invariant.py"))


def test_ensure_pik_hot_path_does_not_rescan_sys_modules(monkeypatch):
    """Once package identity is established, the MoE hot path must be O(1)."""
    canonical = sys.modules["pik"]
    assert sys.modules[_BOOT._DOTTED_NAME] is canonical

    def fail_rescan(*_args, **_kwargs):
        raise AssertionError("steady-state ensure_pik rescanned sys.modules")

    monkeypatch.setattr(_BOOT, "_alias_loaded_children", fail_rescan)
    for _ in range(1024):
        assert _BOOT.ensure_pik() is canonical


class _FakeDist:
    """The two calls ``tree_all_reduce`` makes before it picks a transport. Nothing else in the
    dispatch touches torch.distributed, and both transports are stubbed below."""

    WORLD = 4

    @staticmethod
    def is_initialized():
        return True

    @classmethod
    def get_world_size(cls, group=None):
        return cls.WORLD


@pytest.fixture()
def ar(monkeypatch):
    """A clean per-test view of the transport bookkeeping, with both transports stubbed.

    The stubs are the point: a CPU box has neither symmetric memory nor a triton tree kernel, and
    what is under test is which one gets CALLED and what gets recorded about it.
    """
    monkeypatch.setattr(AR, "dist", _FakeDist)
    monkeypatch.setattr(AR, "p2p_available", lambda group=None: True)
    monkeypatch.setattr(AR, "_tree_all_reduce_nccl", lambda partial, group, out, root_dtype: partial + 1)
    monkeypatch.setattr(AR, "_TRANSPORT_HOOKS", [])
    _reset_transport_counters()
    yield AR
    _reset_transport_counters()


def _x():
    return torch.ones(4, 8, dtype=torch.float32)


# The stubs are NAMED because the name is part of what is under test: the hook hands the install
# fingerprint the function that ACTUALLY moved the bytes, read off the module global at call time.
# In production those are ``pik.allreduce._tree_all_reduce_p2p`` and ``pik.allreduce._tree_reduce``.
def _stub_p2p(partial, group, out, rd):
    return partial * 2


def _stub_p2p_broken(partial, group, out, rd):
    raise RuntimeError("symm_mem rendezvous: allocations from overlapping devices")


# 1-2: the fallback is counted, warned about ONCE, and names the exception
def test_p2p_success_counts_and_stays_quiet(ar, capsys):
    ar._tree_all_reduce_p2p = _stub_p2p
    out = ar.tree_all_reduce(_x())
    assert torch.equal(out, _x() * 2), "the p2p stub's result must be the one returned"
    c = ar.transport_counts()
    assert (c["p2p_calls"], c["nccl_calls"], c["p2p_fallbacks"]) == (1, 0, 0), c
    assert c["state"] == "p2p" and c["first_fallback"] is None
    assert "WARNING" not in capsys.readouterr().out


def test_fallback_warns_once_names_the_exception_and_keeps_counting(ar, capsys):
    ar._tree_all_reduce_p2p = _stub_p2p_broken

    out = ar.tree_all_reduce(_x())
    assert torch.equal(out, _x() + 1), "the NCCL transport's result must be returned on fallback"
    first = capsys.readouterr().out
    # The exception is IN the message -- "it fell back" without "why" is not actionable.
    assert first.count("symmetric-memory P2P all-reduce FAILED") == 1, first
    assert "allocations from overlapping devices" in first
    assert "RuntimeError" in first

    # ...and every subsequent call is silent, but still COUNTED. This is the half that decays:
    # a warn-once with no counter degrades into "we warned about something, once, some time ago".
    for _ in range(50):
        ar.tree_all_reduce(_x())
    assert capsys.readouterr().out == "", "the fallback warning must fire exactly once per process"

    c = ar.transport_counts()
    assert (c["p2p_calls"], c["nccl_calls"], c["p2p_fallbacks"]) == (0, 51, 51), c
    assert c["state"] == "p2p_fallback", "a degraded process must not report itself as plain nccl"
    assert c["first_fallback"]["shape"] == (4, 8)
    assert c["first_fallback"]["world"] == 4
    assert "RuntimeError" in c["first_fallback"]["exception"]


def test_explicit_nccl_backend_counts_as_nccl_not_as_a_fallback(ar):
    ar._tree_all_reduce_p2p = _stub_p2p
    ar.tree_all_reduce(_x(), backend="nccl")
    c = ar.transport_counts()
    assert (c["p2p_calls"], c["nccl_calls"], c["p2p_fallbacks"]) == (0, 1, 0), c
    assert c["state"] == "nccl"


def test_world_1_records_no_transport(ar, monkeypatch):
    """TP=1 does no reduction at all, so it must report ``none`` rather than a transport it never
    chose -- otherwise the fingerprint would report the absence of a collective as a transport."""
    monkeypatch.setattr(_FakeDist, "WORLD", 1)
    ar.tree_all_reduce(_x())
    assert ar.transport_counts()["state"] == "none"


# 3-4: the hook, which is what carries the transport into the install fingerprint
def test_hook_fires_on_first_resolution_and_on_degradation(ar):
    seen = []
    ar.register_transport_hook(lambda state, fn: seen.append((state, getattr(fn, "__name__", fn))))

    ar._tree_all_reduce_p2p = _stub_p2p
    ar.tree_all_reduce(_x())
    ar.tree_all_reduce(_x())
    assert seen == [("p2p", "_stub_p2p")], "one hook call per STATE, not per reduce"

    ar._tree_all_reduce_p2p = _stub_p2p_broken
    ar.tree_all_reduce(_x())
    ar.tree_all_reduce(_x())
    # The degraded state names the NCCL transport's arithmetic (the real, unstubbed `_tree_reduce`).
    assert seen == [("p2p", "_stub_p2p"), ("p2p_fallback", "_tree_reduce")], seen


def test_late_registration_still_sees_the_resolved_transport(ar):
    """An install site cannot order itself against the first forward, so a hook registered after
    the fact must still fire -- otherwise the fingerprint silently records nothing."""
    ar._tree_all_reduce_p2p = _stub_p2p
    ar.tree_all_reduce(_x())
    seen = []
    ar.register_transport_hook(lambda state, fn: seen.append(state))
    assert seen == ["p2p"]


def test_a_broken_hook_cannot_break_the_reduce(ar):
    ar.register_transport_hook(lambda state, fn: 1 / 0)
    ar._tree_all_reduce_p2p = _stub_p2p
    assert torch.equal(ar.tree_all_reduce(_x()), _x() * 2)
    assert ar.transport_counts()["p2p_calls"] == 1


# 5: pik_tp_invariant wires the hook to the F1 fingerprint
def test_install_transport_hook_announces_the_resolved_transport(ar, capsys):
    """The hook used to also RECORD the resolved transport into the install fingerprint
    (`_record_transport_fingerprint`, deleted with the ledger in M6.3). What survives is the fact a
    reader needs and the one a check consumes: the banner says which transport resolved, and
    `pik_transport_status` -- the engagement report -- carries the counts."""
    PTI._install_transport_hook("ENGINE")

    ar._tree_all_reduce_p2p = _stub_p2p
    ar.tree_all_reduce(_x())

    assert PTI.pik_transport_status()["state"] == "p2p"
    banner = capsys.readouterr().out
    assert "[ISOEXEC-ENGINE] pik tree all-reduce TRANSPORT RESOLVED: p2p" in banner
    assert "_stub_p2p" in banner and "p2p=1 nccl=0 fallbacks=0" in banner


def test_transport_status_reports_the_counts(ar):
    ar._tree_all_reduce_p2p = _stub_p2p
    ar.tree_all_reduce(_x())
    st = PTI.pik_transport_status()
    assert st["state"] == "p2p" and st["p2p_calls"] == 1, st


# 6: the unified-model engine host makes ONE transport decision from the arch policy and ONE flag, and logs what ran
def _fresh_select(monkeypatch, arch):
    from isoexec.ops import _dispatch

    monkeypatch.setattr(_dispatch, "ARCH", arch)
    # the hook-once latch is per process; each test wants its own registration
    monkeypatch.setattr(PTI.select_engine_transport, "_hooked", False, raising=False)
    reasons = []

    def disable(reason):
        reasons.append(reason)
        monkeypatch.setattr(ar_mod(), "p2p_available", lambda g=None: False)

    monkeypatch.setattr(PTI, "_disable_pik_p2p", disable)
    return reasons


def ar_mod():
    return AR


def test_select_engine_transport_keeps_p2p_by_default_and_logs_the_resolution(ar, monkeypatch, capsys):
    monkeypatch.delenv("ISOEXEC_PIK_P2P", raising=False)
    _fresh_select(monkeypatch, "sm90")
    assert PTI.select_engine_transport("ENGINE") == "p2p"
    ar._tree_all_reduce_p2p = _stub_p2p
    ar.tree_all_reduce(_x())
    out = capsys.readouterr().out
    assert "pik P2P symmetric-memory all-reduce KEPT" in out
    assert "[ISOEXEC-ENGINE] pik tree all-reduce TRANSPORT RESOLVED: p2p" in out and "_stub_p2p" in out
    assert PTI.pik_transport_status()["state"] == "p2p"


def test_select_engine_transport_env_zero_forces_nccl(ar, monkeypatch, capsys):
    monkeypatch.setenv("ISOEXEC_PIK_P2P", "0")
    reasons = _fresh_select(monkeypatch, "sm90")
    assert PTI.select_engine_transport("ENGINE") == "nccl"
    assert ar.p2p_available() is False and reasons == ["ISOEXEC_PIK_P2P=0"]
    ar._tree_all_reduce_p2p = _stub_p2p
    ar.tree_all_reduce(_x())
    out = capsys.readouterr().out
    assert "TRANSPORT RESOLVED: nccl" in out
    assert PTI.pik_transport_status()["state"] == "nccl"


def test_sm100_engine_keeps_p2p(ar, monkeypatch, capsys):
    """No architecture is NCCL-only: the B200 TP4 hang was group-size keying, not the transport."""
    monkeypatch.delenv("ISOEXEC_PIK_P2P", raising=False)
    _fresh_select(monkeypatch, "sm100")
    assert PTI.p2p_arch_refusal("sm100") is None
    assert PTI.select_engine_transport("ENGINE") == "p2p"
    assert "pik P2P symmetric-memory all-reduce KEPT" in capsys.readouterr().out


def _trainer_selection_state(monkeypatch):
    import pik.lane as LANE  # type: ignore

    monkeypatch.setattr(PTI.select_trainer_transport, "_selected", None, raising=False)
    monkeypatch.setattr(AR, "_PRIVATE_OUTPUTS", False)
    monkeypatch.setattr(LANE, "_POLICY", None)
    monkeypatch.setattr(LANE, "_OVERLAP_DEFAULT", False)
    monkeypatch.setattr(PTI, "p2p_arch_refusal", lambda arch=None: None)
    return LANE


def test_select_trainer_transport_physical_keeps_p2p_and_privatizes_outputs(ar, monkeypatch, capsys):
    """Default trainer selection: P2P admitted per group by physical identity, never a symmetric output."""
    LANE = _trainer_selection_state(monkeypatch)
    monkeypatch.delenv("ISOEXEC_PIK_TRAINER_P2P", raising=False)
    original = AR.p2p_available
    assert PTI.select_trainer_transport("TRAINER") == "p2p_physical"
    assert AR.p2p_available is original  # not disabled: the per-group probe decides
    assert LANE.peer_policy() == "physical"
    assert AR._PRIVATE_OUTPUTS is True
    assert LANE._OVERLAP_DEFAULT is False  # comm-lane overlap is opt-in
    assert LANE._WANTED is True
    assert PTI.select_trainer_transport("TRAINER") == "p2p_physical"  # once
    assert "admitted PER GROUP by physical device identity" in capsys.readouterr().out


def test_trainer_p2p_off_votes_no_in_the_probe_instead_of_skipping_it(ar, monkeypatch):
    """ISOEXEC_PIK_TRAINER_P2P=0 on one rank must not make that rank skip the group's probe collective."""
    LANE = _trainer_selection_state(monkeypatch)
    monkeypatch.setattr(LANE, "_WANTED", True)
    monkeypatch.setenv("ISOEXEC_PIK_TRAINER_P2P", "0")
    original = AR.p2p_available
    assert PTI.select_trainer_transport("TRAINER") == "p2p_physical"
    assert AR.p2p_available is original and LANE.peer_policy() == "physical" and LANE._WANTED is False


def test_select_trainer_transport_disables_both_p2p_references_once(ar, monkeypatch, capsys):
    _trainer_selection_state(monkeypatch)
    monkeypatch.setattr(PTI, "p2p_arch_refusal", lambda arch=None: "arch policy: test NCCL-only")
    linear = types.ModuleType("pik.linear")
    linear.p2p_available = lambda group=None: True
    monkeypatch.setitem(sys.modules, "pik.linear", linear)
    monkeypatch.setattr(sys.modules["pik"], "linear", linear, raising=False)
    monkeypatch.setattr(AR, "p2p_available", lambda group=None: True)

    assert PTI.select_trainer_transport("TRAINER") == "nccl"
    assert AR.p2p_available() is False
    assert linear.p2p_available() is False
    assert len(ar._TRANSPORT_HOOKS) == 1

    assert PTI.select_trainer_transport("TRAINER") == "nccl"
    assert len(ar._TRANSPORT_HOOKS) == 1

    ar._tree_all_reduce_p2p = _stub_p2p
    ar.tree_all_reduce(_x())
    out = capsys.readouterr().out
    assert "disabled pik P2P symmetric-memory all-reduce -> NCCL transport" in out
    assert "[ISOEXEC-TRAINER] pik tree all-reduce TRANSPORT RESOLVED: nccl" in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
