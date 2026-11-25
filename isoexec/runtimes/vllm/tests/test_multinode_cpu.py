"""Multi-node engine launcher: placement rule, NCCL log parsing and the admission comparison (CPU, no vLLM engine)."""
from __future__ import annotations

import copy
import json

import pytest

from isoexec.runtimes.vllm import launch_receipt as lr
from isoexec.runtimes.vllm import multinode as mn


def test_recipe_placement_puts_one_pp_stage_on_each_host():
    rows = mn.placement(tensor_parallel_size=8, pipeline_parallel_size=2, nnodes=2)
    assert [r["node"] for r in rows] == [0] * 8 + [1] * 8
    assert [r["pp_rank"] for r in rows] == [0] * 8 + [1] * 8
    # vLLM's layout: TP innermost, PP groups pair rank r with r + TP
    assert rows[3]["tp_ranks"] == list(range(8)) and rows[11]["tp_ranks"] == list(range(8, 16))
    assert rows[3]["pp_ranks"] == [3, 11] and rows[11]["pp_ranks"] == [3, 11]
    assert all(len({rows[q]["node"] for q in r["tp_ranks"]}) == 1 for r in rows)


@pytest.mark.parametrize("tp,pp,nnodes", [(8, 2, 4), (4, 2, 8), (2, 3, 4)])
def test_layouts_that_split_a_tp_group_or_the_world_are_refused(tp, pp, nnodes):
    with pytest.raises(lr.LaunchRefusal, match=r"launcher\.(tp_spans_hosts|nnodes)"):
        mn.placement(tensor_parallel_size=tp, pipeline_parallel_size=pp, nnodes=nnodes)


def test_placement_matches_vllm_group_construction():
    torch = pytest.importorskip("torch")
    tp, pp = 4, 2
    ranks = torch.arange(tp * pp).reshape(-1, 1, pp, 1, tp)       # vLLM initialize_model_parallel's all_ranks
    tp_groups = [list(map(int, g)) for g in ranks.view(-1, tp)]
    pp_groups = [list(map(int, g)) for g in ranks.transpose(2, 4).reshape(-1, pp)]
    rows = mn.placement(tensor_parallel_size=tp, pipeline_parallel_size=pp, nnodes=pp)
    assert sorted({tuple(r["tp_ranks"]) for r in rows}) == sorted(map(tuple, tp_groups))
    assert sorted({tuple(r["pp_ranks"]) for r in rows}) == sorted(map(tuple, pp_groups))


LOG = """\
fake-node-0:11:11 [0] NCCL INFO Using network IB
fake-node-0:11:11 [0] NCCL INFO [Rank 0] ncclCommInitRankConfig comm 0xaaa rank 0 nranks 2 cudaDev 0 nvmlDev 1 busId 1b000 commId 0x1111 - Init START
fake-node-0:11:11 [0] NCCL INFO comm 0xaaa rank 0 nRanks 2 nNodes 2 localRanks 1 localRank 0 MNNVL 0
fake-node-0:11:11 [0] NCCL INFO Channel 00/0 : 1[1] -> 0[0] [receive] via NET/IB/0
fake-node-0:11:11 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] [send] via NET/IB/0
fake-node-0:11:11 [0] NCCL INFO ncclCommInitRankConfig comm 0xaaa rank 0 nranks 2 cudaDev 0 nvmlDev 1 busId 1b000 commId 0x1111 localSize 1 used 1 bytes - Init COMPLETE
fake-node-0:11:12 [0] NCCL INFO [Rank 0] ncclCommInitRankConfig comm 0xbbb rank 0 nranks 2 cudaDev 0 nvmlDev 1 busId 1b000 commId 0x2222 - Init START
fake-node-0:11:12 [0] NCCL INFO comm 0xbbb rank 0 nRanks 2 nNodes 2 localRanks 1 localRank 0 MNNVL 0
fake-node-0:11:12 [0] NCCL INFO Channel 00/1 : 0[0] -> 1[1] [send] via NET/IB/0/Shared
"""


def test_nccl_log_parser_assigns_channels_to_their_communicator():
    comms = {c["comm_id"]: c for c in lr.parse_nccl_log(LOG.splitlines())}
    assert comms["0x1111"]["nnodes"] == 2 and comms["0x1111"]["transports"] == {"NET/IB": 2}
    assert comms["0x2222"]["transports"] == {"NET/IB": 1}
    assert comms["0x1111"]["host"] == "fake-node-0"


def _launch(tp=2, pp=2, nnodes=2, **over):
    launch = {"launch_id": "L", "placement": mn.placement(tensor_parallel_size=tp, pipeline_parallel_size=pp,
                                                          nnodes=nnodes),
              "env": {"ISOEXEC_PIK": "1", "NCCL_DEBUG": "INFO"}, "node_env": {}, "require_prebuilt": False,
              "transports": {"required": True, "cross_host": ["NET/IB"], "intra_host": "not_net"},
              "expect_hostnames": None}
    launch.update(over)
    return launch


def _records(launch, *, tp_kind="P2P/CUMEM", pp_kind="NET/IB"):
    places = launch["placement"]
    recs = []
    for p in places:
        host = f"h{p['node']}"
        comms = []
        if len(p["tp_ranks"]) > 1:
            comms.append({"comm_id": f"tp{p['tp_ranks'][0]}", "rank": p["tp_rank"], "nranks": len(p["tp_ranks"]),
                          "nnodes": 1, "transports": {tp_kind: 2}})
        if len(p["pp_ranks"]) > 1:
            comms.append({"comm_id": f"pp{p['pp_ranks'][0]}", "rank": p["pp_rank"], "nranks": len(p["pp_ranks"]),
                          "nnodes": len({places[q]["node"] for q in p["pp_ranks"]}), "transports": {pp_kind: 2}})
        recs.append({"rank": p["rank"], "launch_id": "L", "tp_ranks": p["tp_ranks"], "pp_ranks": p["pp_ranks"],
                     "host": {"hostname": host, "boot_id": f"b{p['node']}"}, "device_uuid": f"GPU-{p['rank']}",
                     "software": {"commit": "c", "sources": {"a.py": "1"}, "sources_digest": "s",
                                  "binaries": {"x.so": "2"}, "binaries_digest": "b"},
                     "versions": {"torch": "t"}, "program_identity": {"identity": {"p": 1}},
                     "vllm_config": {"tensor_parallel_size": 2}, "prebuilt": {"module": False, "only": None,
                                                                              "dir_set": False, "manifest_digest": None},
                     "env": {"ISOEXEC_PIK": "1", "NCCL_DEBUG": "INFO"}, "nccl_registration": {},
                     "nccl": {"state": "parsed", "comms": comms}})
    return recs


def test_agreeing_workers_are_admitted_with_the_pp_edge_as_the_only_cross_host_comm():
    launch = _launch()
    table = lr.verify(_records(launch), launch)
    roles = {c["comm_id"]: c["role"] for c in table["comms"]}
    assert roles == {"tp0": "tp", "tp2": "tp", "pp0": "pp", "pp1": "pp"}
    assert table["cross_host_edges"] == [(0, 2), (1, 3)]


def _refused(records, launch, reason):
    with pytest.raises(lr.LaunchRefusal, match=rf"^launcher\.{reason}:"):
        lr.verify(records, launch)


def test_every_disagreement_is_refused_by_name():
    launch = _launch()
    base = _records(launch)

    def edit(fn):
        recs = copy.deepcopy(base)
        fn(recs)
        return recs

    _refused(edit(lambda r: r[3]["software"].update(sources={"a.py": "9"}, sources_digest="t")), launch, "sources")
    _refused(edit(lambda r: r[3]["software"].update(commit="d")), launch, "commit")
    _refused(edit(lambda r: r[1]["software"].update(binaries={"x.so": "9"}, binaries_digest="c")), launch, "binaries")
    _refused(edit(lambda r: r[2].update(program_identity={"identity": {"p": 2}})), launch, "program_identity")
    _refused(edit(lambda r: [x.update(program_identity=None) for x in r]), launch, "program_identity_missing")
    _refused(edit(lambda r: r[2]["vllm_config"].update(tensor_parallel_size=4)), launch, "vllm_config")
    _refused(edit(lambda r: r[2]["env"].update(ISOEXEC_PIK="0")), launch, "environment")
    _refused(edit(lambda r: r[2]["env"].update(NCCL_IB_TIMEOUT="20")), launch, "nccl_settings")
    _refused(edit(lambda r: r[2].update(nccl_live={"NCCL_GRAPH_REGISTER": "1"})), launch, "nccl_settings")
    _refused(edit(lambda r: r[1]["host"].update(hostname="h1", boot_id="b1")), launch, "node_spans_hosts")
    _refused(edit(lambda r: [x["host"].update(hostname="h0", boot_id="b0") for x in r]), launch, "nodes_share_host")
    _refused(edit(lambda r: r[1].update(device_uuid="GPU-0")), launch, "shared_device")
    _refused(edit(lambda r: r[0].update(launch_id="other")), launch, "launch_id")
    _refused(edit(lambda r: r[0].update(pp_ranks=[0, 3])), launch, "groups")
    _refused(edit(lambda r: r[0]["nccl"].update(state="not_logged")), launch, "transport_unknown")
    _refused(_records(launch, pp_kind="NET/Socket"), launch, "pp_transport")
    _refused(_records(launch, tp_kind="NET/IB"), launch, "tp_transport")
    _refused(edit(lambda r: [c.update(nnodes=1) for x in r for c in x["nccl"]["comms"] if c["comm_id"] == "pp0"]),
             launch, "nccl_host_view")
    _refused(edit(lambda r: [x["nccl"].update(comms=[c for c in x["nccl"]["comms"] if c["comm_id"] != "pp1"])
                             for x in r]), launch, "pp_transport_unobserved")

    def world_comm(recs):
        for x in recs:
            x["nccl"]["comms"].append({"comm_id": "world", "rank": x["rank"], "nranks": 4, "nnodes": 2,
                                       "transports": {"NET/IB": 1}})
    _refused(edit(world_comm), launch, "unexpected_cross_host_comm")
    _refused(base, _launch(expect_hostnames=["h1", "h0"]), "hostname")


def test_declared_per_host_keys_may_differ_and_nothing_else():
    launch = _launch(node_env={"1": {"NCCL_IB_HCA": "=mlx5_2:1"}})
    recs = _records(launch)
    for r in recs:
        r["env"]["NCCL_IB_HCA"] = "=mlx5_2:1" if r["rank"] >= 2 else "=mlx5_0:1"
    with pytest.raises(lr.LaunchRefusal, match="launcher.environment"):
        lr.verify(recs, launch)          # node 0 has no declared value: its key must match the resolved env (absent)
    launch = _launch(node_env={"0": {"NCCL_IB_HCA": "=mlx5_0:1"}, "1": {"NCCL_IB_HCA": "=mlx5_2:1"}})
    lr.verify(recs, launch)


def test_prebuilt_only_launch_needs_the_mode_and_the_resolved_manifest():
    launch = _launch(require_prebuilt=True, prebuilt_manifest_digest="m1")
    recs = _records(launch)
    _refused(recs, launch, "prebuilt_mode_unavailable")
    for r in recs:
        r["prebuilt"] = {"module": True, "only": "1", "dir_set": True, "manifest_digest": "m2"}
    _refused(recs, launch, "prebuilt_manifest")
    for r in recs:
        r["prebuilt"]["manifest_digest"] = "m1"
    lr.verify(recs, launch)
    recs[3]["prebuilt"] = dict(recs[3]["prebuilt"], only=None)
    _refused(recs, launch, "prebuilt")


def test_resolved_launch_is_digest_sealed(tmp_path):
    launch = _launch(format=mn.LAUNCH_FORMAT)
    launch["digest"] = mn.launch_digest(launch)
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(launch))
    assert mn.load(path)["launch_id"] == "L"
    launch["env"]["ISOEXEC_PIK"] = "0"
    path.write_text(json.dumps(launch))
    with pytest.raises(lr.LaunchRefusal, match="launcher.digest"):
        mn.load(path)


def test_node_environment_is_the_resolved_one_plus_declared_keys_and_the_marker():
    launch = _launch(node_env={"1": {"CUDA_VISIBLE_DEVICES": "6"}})
    env0, env1 = mn.node_environment(launch, 0), mn.node_environment(launch, 1)
    assert env0 == {**launch["env"], mn.LAUNCH_ID_ENV: "L", mn.LAUNCH_NODE_ENV: "0"}
    assert env1 == {**launch["env"], "CUDA_VISIBLE_DEVICES": "6", mn.LAUNCH_ID_ENV: "L", mn.LAUNCH_NODE_ENV: "1"}


def test_kill_removes_only_the_named_launch_and_node():
    import subprocess
    import sys
    import time

    def start(launch_id, node):
        env = {"PATH": "/usr/bin:/bin", mn.LAUNCH_ID_ENV: launch_id, mn.LAUNCH_NODE_ENV: str(node)}
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], env=env)

    procs = {("A", 0): start("A", 0), ("A", 1): start("A", 1), ("B", 0): start("B", 0)}
    procs[("A", 1)].send_signal(__import__("signal").SIGSTOP)        # a stopped (hung) process is killed too
    time.sleep(0.5)
    try:
        out = mn.kill_launch("A", node=1, grace=5)
        assert out["found"] == [procs[("A", 1)].pid] and out["left"] == []
        assert procs[("A", 0)].poll() is None and procs[("B", 0)].poll() is None
        out = mn.kill_launch("A", grace=5)
        assert out["found"] == [procs[("A", 0)].pid] and procs[("B", 0)].poll() is None
    finally:
        for p in procs.values():
            p.kill()
            p.wait()


def test_search_paths_extended_in_process_are_not_a_launch_difference():
    launch = _launch(env={"ISOEXEC_PIK": "1", "NCCL_DEBUG": "INFO", "LD_LIBRARY_PATH": "/a/lib64:/b:"})
    recs = _records(launch)
    seen = {"ISOEXEC_PIK": "1", "NCCL_DEBUG": "INFO", "LD_LIBRARY_PATH": "/a/lib64:/a/lib64:/a/lib64:/b:"}
    for r in recs:
        r["env"] = lr.environment_view(launch["env"], seen)
    lr.verify(recs, launch)
    recs[2]["env"] = lr.environment_view(launch["env"], dict(seen, LD_LIBRARY_PATH="/c:/b:"))
    with pytest.raises(lr.LaunchRefusal, match="launcher.environment"):
        lr.verify(recs, launch)
    recs[2]["env"] = lr.environment_view(launch["env"], dict(seen, LD_LIBRARY_PATH="/tvm:/a/lib64:/b:"))
    lr.verify(recs, launch)                     # an entry a library added in-process: reported, not compared
    assert lr.environment_added(launch["env"], {"LD_LIBRARY_PATH": "/tvm:/a/lib64:/b:"}) == {"LD_LIBRARY_PATH": ["/tvm"]}
    recs[2]["env"] = lr.environment_view(launch["env"], dict(seen, LD_LIBRARY_PATH="/b:/a/lib64:"))
    with pytest.raises(lr.LaunchRefusal, match="launcher.environment"):
        lr.verify(recs, launch)                 # the launch's entries reordered
