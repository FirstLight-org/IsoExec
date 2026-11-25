"""Peer admission by physical device identity and the comm-lane schedule helpers (pik/lane.py, owner.py).

CPU only: the classification of gathered (host, UUID, ordinal) rows, the overlap switch, and the pure
pipeline-bound function every rank evaluates identically. The GPU batteries cover the transports."""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, *[".."] * 4))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from isoexec.ops.collectives.pik_bootstrap import ensure_pik  # noqa: E402

ensure_pik()
import pik.lane as LANE  # type: ignore  # noqa: E402
import pik.pipeline as PIPE  # type: ignore  # noqa: E402

U = [bytes([i]) * 16 for i in range(8)]


def test_ray_style_ranks_distinct_gpus_colliding_ordinals_are_admitted_with_relaxation():
    admit, relax, why = LANE.classify([(7, U[r], 0) for r in range(4)])
    assert admit and relax and "colliding device ordinals" in why


def test_torchrun_style_ranks_need_no_relaxation():
    admit, relax, _ = LANE.classify([(7, U[r], r) for r in range(8)])
    assert admit and not relax


def test_shared_physical_gpu_is_refused():
    admit, relax, why = LANE.classify([(7, U[0], 0), (7, U[0], 1)])
    assert not admit and not relax and "share one physical GPU" in why


def test_multi_host_group_is_refused():
    admit, _, why = LANE.classify([(7, U[0], 0), (9, U[1], 0)])
    assert not admit and "hosts" in why


def test_overlap_switch_default_and_env(monkeypatch):
    monkeypatch.setattr(LANE, "_OVERLAP_DEFAULT", False)
    monkeypatch.delenv("ISOEXEC_PIK_OVERLAP", raising=False)
    assert LANE.overlap_enabled() is False
    LANE.set_overlap_default(True)
    assert LANE.overlap_enabled() is True
    monkeypatch.setenv("ISOEXEC_PIK_OVERLAP", "0")
    assert LANE.overlap_enabled() is False
    monkeypatch.setenv("ISOEXEC_PIK_OVERLAP", "1")
    monkeypatch.setattr(LANE, "_OVERLAP_DEFAULT", False)
    assert LANE.overlap_enabled() is True


@pytest.mark.parametrize("M", [1, 255, 256, 2047, 2048, 2049, 5791, 11583, 16384])
@pytest.mark.parametrize("chunks", [1, 2, 3, 4, 8])
def test_pipeline_bounds_cover_rows_exactly_once_on_aligned_cuts(M, chunks):
    b = PIPE.pipeline_bounds(M, chunks, 2048)
    assert b[0][0] == 0 and b[-1][1] == M
    for (a0, a1), (b0, _) in zip(b, b[1:]):
        assert a1 == b0 and a1 % 128 == 0
    assert all(r1 > r0 for r0, r1 in b)
    if M < 2048 or chunks == 1:
        assert b == [(0, M)]
    else:
        assert len(b) <= chunks


def test_pipeline_bounds_is_a_pure_function():
    assert PIPE.pipeline_bounds(11583, 4, 2048) == PIPE.pipeline_bounds(11583, 4, 2048)


def test_physical_policy_routes_p2p_available_through_the_probe(monkeypatch):
    import pik.allreduce as AR  # type: ignore

    seen = []
    monkeypatch.setattr(AR.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(AR.dist, "get_world_size", lambda group=None: 4)
    monkeypatch.setattr(LANE, "physical_p2p", lambda group: seen.append(group) or False)
    monkeypatch.setattr(LANE, "_POLICY", None)
    assert AR.p2p_available("g") is True and seen == []  # historical rule (engine)
    monkeypatch.setattr(LANE, "_POLICY", "physical")
    assert AR.p2p_available("g") is False and seen == ["g"]


def test_derived_depth_follows_the_comm_compute_ratio():
    # fc2 of a 5120-hidden model: deeper with more rows, never above the cap, 1 below a wave per chunk
    f = lambda M, W, planes, elt, link="p2p", sms=132: LANE.plan_chunks(M, 5120, 17408 // W, W, planes, elt, 2,
                                                                      link, sms)
    for W, planes, elt in ((2, 1, 4), (4, 2, 2), (8, 1, 2)):
        seq = [f(M, W, planes, elt) for M in (256, 1024, 2048, 5800, 11600, 65536)]
        assert seq == sorted(seq) and seq[0] == 1 and max(seq) <= LANE.MAX_CHUNKS
        assert 3 <= f(5800, W, planes, elt) <= 6
    assert LANE.plan_chunks(4096, 5120, 4352, 1, 2, 2, 2, "p2p", 132) == 1  # no TP, no pipeline
    # a slower link moves the balance point (GEMM-bound min) but never deepens past the wave cap
    assert f(65536, 4, 2, 2, "nccl_internode") <= LANE.MAX_CHUNKS


def test_size_crossover_moves_large_p2p_reduces_to_the_nccl_owner_path(monkeypatch):
    import pik.pipeline as PIPE  # type: ignore
    from pik.owner import _State  # type: ignore

    st = _State(True, "p2p", 1 << 18, 32, p2p=object())
    st.twoshot = 3
    assert PIPE.pick_state(st, 1 << 30) is st  # no agreed cap (engine policy): no crossover
    st.p2p_max_bytes = 32 << 20  # what owner.state records from the group's probe
    assert PIPE.pick_state(st, 32 << 20) is st
    big = PIPE.pick_state(st, (32 << 20) + 1)
    assert big.transport == "nccl" and big.p2p is None and big.twoshot == 3
    assert PIPE.pick_state(st, 1 << 30) is big  # one cached shadow
    # the threshold is the state's agreed value, never this process's environment
    monkeypatch.setenv("ISOEXEC_PIK_TRAINER_P2P_MAX_MB", "0")
    assert PIPE.pick_state(st, 1 << 30) is big


def test_wanted_and_mig_columns_refuse_the_whole_group():
    admit, _, why = LANE.classify([(7, U[r], 0) for r in range(4)], wanted=[True, False, True, True])
    assert not admit and "[1]" in why
    admit, _, why = LANE.classify([(7, U[r], r) for r in range(2)], mig=[False, True])
    assert not admit and "MIG" in why


def test_host_identity_includes_the_boot_id(monkeypatch):
    import builtins
    import io

    real_open = builtins.open
    boot = {"id": b"aaaa"}

    def fake_open(path, *a, **k):
        if path == "/proc/sys/kernel/random/boot_id":
            return io.BytesIO(boot["id"])
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(LANE.socket, "gethostname", lambda: "same-name")
    a = LANE.host_id()
    boot["id"] = b"bbbb"
    assert LANE.host_id() != a  # same hostname on another node is another host


# ------------------------------------------------------------------- split environments (gloo, 2 ranks)
def _probe_rank(rank, world, env, rendezvous_ok=True):
    """One rank of the physical probe with this rank's own environment; CPU stand-ins for CUDA identity and the
    symmetric-memory rendezvous. Returns (verdict, agreed cap, calls into the rendezvous)."""
    import torch
    import torch.distributed as dist

    from isoexec.ops.collectives.pik_bootstrap import ensure_pik

    ensure_pik()
    import pik.lane as lane  # type: ignore

    for k, v in env[rank].items():
        os.environ[k] = v
    lane.set_peer_policy("physical", wanted=os.environ.get("ISOEXEC_PIK_TRAINER_P2P", "1") != "0")
    lane.device_uuid = lambda device=None: bytes([rank + 1]) * 16
    lane.is_mig = lambda device=None: False
    calls = []

    def rendezvous(group, device):
        calls.append(rank)
        dist.barrier(group=group)  # the real one is a collective: every admitted rank must reach it
        return rendezvous_ok

    lane._rendezvous_works = rendezvous
    got = lane._probe(dist.group.WORLD, torch.device("cpu"))
    # The next collective must pair with the peer's next collective (no rank left a probe collective behind).
    t = torch.tensor([rank + 1])
    dist.all_reduce(t)
    assert int(t) == 3
    return got, lane.p2p_max_bytes(dist.group.WORLD), calls


def test_split_p2p_max_mb_is_agreed_as_the_group_minimum():
    from isoexec.ops._testing import run_gloo

    out = run_gloo(_probe_rank, 2, [{"ISOEXEC_PIK_TRAINER_P2P_MAX_MB": "32"},
                                    {"ISOEXEC_PIK_TRAINER_P2P_MAX_MB": "64"}])
    assert out[0][:2] == out[1][:2] == (True, 32 << 20) and out[0][2] == [0] and out[1][2] == [1]
    out = run_gloo(_probe_rank, 2, [{"ISOEXEC_PIK_TRAINER_P2P_MAX_MB": "0"},
                                    {"ISOEXEC_PIK_TRAINER_P2P_MAX_MB": "48"}])
    assert out[0][1] == out[1][1] == 48 << 20  # 0 = no limit on that rank; the bounded rank still binds


def test_split_trainer_p2p_flag_refuses_on_every_rank_without_a_hang():
    from isoexec.ops._testing import run_gloo

    out = run_gloo(_probe_rank, 2, [{"ISOEXEC_PIK_TRAINER_P2P": "0"}, {"ISOEXEC_PIK_TRAINER_P2P": "1"}])
    assert out[0][0] is False and out[1][0] is False
    assert out[0][2] == [] and out[1][2] == []  # decided from the gathered rows: nobody entered a rendezvous


def test_a_rendezvous_that_fails_on_one_rank_refuses_the_group():
    from isoexec.ops._testing import run_gloo

    out = run_gloo(_probe_rank, 2, [{}, {}], False)
    assert out[0][0] is False and out[1][0] is False


def _overlap_rank(rank, world, env, sms):
    import torch
    import torch.distributed as dist

    from isoexec.ops.collectives.pik_bootstrap import ensure_pik

    ensure_pik()
    import pik.lane as lane  # type: ignore

    for k, v in env[rank].items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    lane._sm_count = lambda device: sms[rank]
    lane._capturing = lambda: False
    return lane.overlap_agreed(dist.group.WORLD, torch.device("cpu"))


@pytest.mark.parametrize("env,expected", [
    ([{"ISOEXEC_PIK_OVERLAP": None}, {"ISOEXEC_PIK_OVERLAP": None}], False),  # default: off
    ([{"ISOEXEC_PIK_OVERLAP": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
      {"ISOEXEC_PIK_OVERLAP": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"}], True),
    ([{"ISOEXEC_PIK_OVERLAP": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "1"},
      {"ISOEXEC_PIK_OVERLAP": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "1"}], False),  # one queue: no lane
    ([{"ISOEXEC_PIK_OVERLAP": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
      {"ISOEXEC_PIK_OVERLAP": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "1"}], False),  # split: off for both
    ([{"ISOEXEC_PIK_OVERLAP": "1"}, {"ISOEXEC_PIK_OVERLAP": "0"}], False),
])
def test_overlap_is_opt_in_needs_queues_and_is_group_agreed(env, expected):
    from isoexec.ops._testing import run_gloo

    out = run_gloo(_overlap_rank, 2, env, [132, 132])
    assert out == {0: expected, 1: expected}


def test_serial_tp_copy_backward_is_tree_all_reduce(monkeypatch):
    """Without a lane link the adjoint is pik's tree_all_reduce, so ISOEXEC_PIK_OWNER_ROUTE gates it as before."""
    import types

    import torch

    import isoexec.ops.collectives.pik.allreduce as AR_by_path
    from isoexec.ops.collectives import tp_bytes as TB

    seen = []
    monkeypatch.setattr(AR_by_path, "tree_all_reduce", lambda g, group=None: seen.append(group) or g * 2)
    ctx = types.SimpleNamespace(group="tp", link=None)
    g = torch.ones(3)
    out, *_ = TB._TpCopy.backward(ctx, g)
    assert seen == ["tp"] and torch.equal(out, g * 2)


def test_admitted_mode_needs_an_admitted_p2p_class(monkeypatch):
    import torch

    from pik import owner as OWNER  # type: ignore
    from pik.owner import _State  # type: ignore

    monkeypatch.setattr(OWNER.dist, "get_world_size", lambda group=None: 4)
    monkeypatch.setattr(OWNER, "_ADMIT", {})
    st = _State(True, "p2p", 1 << 18, 32, p2p=object())
    st.twoshot = 2
    flat = torch.zeros(1, 4096, dtype=torch.bfloat16)
    assert OWNER.admitted_mode(flat, "g", torch.bfloat16, st) is None  # never admitted: not lane-safe
    mode = OWNER.branch(4096, 4, 1, 2, st.oneshot_bytes)
    OWNER._ADMIT[OWNER._class_key(st, 4, 1, flat.dtype, torch.bfloat16, mode, OWNER.vec_for(4096))] = True
    assert OWNER.admitted_mode(flat, "g", torch.bfloat16, st) == mode
    nccl = _State(True, "nccl", 1 << 18, 32)
    assert OWNER.admitted_mode(flat, "g", torch.bfloat16, nccl) is None  # NCCL never goes on the lane
