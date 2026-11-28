"""What every worker of a multi-node engine reports before the first token, and the comparison that admits it.

``collect`` runs in each engine worker (vLLM ``collective_rpc``; the CPU dry run calls it directly) and returns a plain
record: who the worker is (global rank, its TP / PP group ranks, host = hostname + boot id, device), what it runs
(IsoExec sources and extension binaries actually loaded, the forward program identity of its stage, the vLLM config
hash, the prebuilt-cache manifest digest and mode), the environment it resolved (every key of the launch's resolved
environment plus every NCCL key it sees) and, from its own NCCL debug log, every NCCL communicator it holds with the
number of nodes NCCL counted and the transport of each channel. ``verify`` compares the records of all workers with
the launch's placement and refuses by name (``LaunchRefusal``, a ``checks.Refusal``) on any disagreement.

The transport rule it enforces is the placement rule of ``multinode``: a communicator whose members are one TP group
lies on ONE host (NCCL counts one node) and, unless the launch relaxes it, uses an intra-node transport (P2P / NVLS /
SHM, not the network); a communicator that crosses hosts is a pipeline edge (its members belong to one PP group) and
uses the network the launch names (``NET/IB`` on the target cluster). Any other communicator that crosses hosts is
refused: at TP x PP with one stage per host the PP edge is the only cross-host NCCL traffic.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from isoexec.checks import Refusal

#: Keys every worker reports from its environment besides the launch's resolved environment (NCCL settings).
NCCL_PREFIXES = ("NCCL_", "TORCH_NCCL_")
#: Keys that legitimately differ per process and are never compared (the launch's ``node_env`` covers per-host keys).
PER_PROCESS_KEYS = frozenset({"CUDA_VISIBLE_DEVICES", "LOCAL_RANK", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"})
NCCL_DEBUG_FILE = "NCCL_DEBUG_FILE"
#: Search-path lists that imported libraries extend in-process (OpenCV's cv2 prepends its lib64 to LD_LIBRARY_PATH on
#: every import; TileLang prepends its TVM path to PYTHONPATH during graph capture). The launch's entries must all be
#: there, in the launch's order; entries a library added in-process are reported, not compared.
PATH_LISTS = frozenset({"LD_LIBRARY_PATH", "PATH", "PYTHONPATH", "CPATH", "LIBRARY_PATH"})


class LaunchRefusal(Refusal):
    """A multi-node engine launch whose workers disagree; the message starts with ``launcher.<reason>``."""


def refuse(reason: str, detail: str) -> None:
    raise LaunchRefusal(f"launcher.{reason}: {detail}")


# --------------------------------------------------------------------------------------------- worker side


def _sha(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def host_identity() -> dict:
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = "?"
    return {"hostname": socket.gethostname(), "boot_id": boot}


def loaded_sources(env: Mapping[str, str]) -> dict:
    """sha256 of every loaded IsoExec source, keyed by its path relative to the package root (checkout paths may
    differ across hosts; the bytes may not), plus every loaded shared object of IsoExec or a torch-extension cache
    (keyed by file name), plus the git commit of the tree when it is a checkout."""
    import isoexec

    root = Path(isoexec.__file__).resolve().parent
    files: dict[str, str] = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if not path or name.split(".")[0] != "isoexec" or not os.path.exists(path):
            continue
        p = Path(path).resolve()
        try:
            files[str(p.relative_to(root))] = _sha(p)
        except ValueError:
            files[f"outside:{p}"] = _sha(p)
    binaries: dict[str, str] = {}
    try:
        maps = Path("/proc/self/maps").read_text().splitlines()
    except OSError:
        maps = []
    # IsoExec's own extension binaries: under the package, the torch-extension cache or the prebuilt store; keyed by
    # the path below that root (the roots may sit at different paths on different hosts).
    roots = [("isoexec", root)] + [(name, Path(p).resolve()) for name in ("TORCH_EXTENSIONS_DIR", "ISOEXEC_PREBUILT_DIR")
                                   if (p := env.get(name))]
    for line in maps:
        parts = line.split()
        if len(parts) < 6 or ".so" not in Path(parts[5]).name:
            continue
        lib = Path(parts[5])
        for label, base in roots:
            try:
                key = f"{label}:{lib.relative_to(base)}"
            except ValueError:
                continue
            if key not in binaries:
                binaries[key] = _sha(lib)
            break
    commit = None
    head = root.parent / ".git"
    if head.exists():
        try:
            import subprocess

            commit = subprocess.run(["git", "-C", str(root.parent), "rev-parse", "HEAD"], capture_output=True,
                                    text=True, timeout=20).stdout.strip() or None
        except Exception:  # noqa: BLE001 -- the commit is reported, the bytes are compared
            commit = None
    return {"commit": commit, "sources": files, "sources_digest": _digest(files), "binaries": binaries,
            "binaries_digest": _digest(binaries)}


def prebuilt_state(env: Mapping[str, str]) -> dict:
    """The prebuilt-cache mode of this process: whether the store module exists in this tree, the mode switch, and
    the digest recorded in ``$ISOEXEC_PREBUILT_DIR/MANIFEST.json`` (read as a file: no import of the store)."""
    from importlib.util import find_spec

    root = env.get("ISOEXEC_PREBUILT_DIR")
    manifest = None
    if root:
        try:
            manifest = json.loads((Path(root) / "MANIFEST.json").read_text()).get("digest")
        except (OSError, ValueError):
            manifest = None
    return {"module": find_spec("isoexec.ops.prebuilt") is not None, "only": env.get("ISOEXEC_PREBUILT_ONLY"),
            "dir_set": bool(root), "manifest_digest": manifest}


_CHANNEL = re.compile(r"Channel \d+/\d+ : .*\[(?:send|receive)\] via (\S+)")
_START = re.compile(r"comm (0x[0-9a-f]+) rank (\d+) nranks (\d+) .*commId (0x[0-9a-f]+)")
_NODES = re.compile(r"comm (0x[0-9a-f]+) rank (\d+) nRanks (\d+) nNodes (\d+) localRanks (\d+)")
_PREFIX = re.compile(r"^([^:\s]+):(\d+):(\d+) ")


def parse_nccl_log(lines: Iterable[str]) -> list[dict]:
    """NCCL communicators and their channel transports from one process's ``NCCL_DEBUG=INFO`` log.

    A communicator is announced by its ``Init START`` line (comm pointer, rank, nranks, commId) and its topology line
    (``nRanks .. nNodes ..``); channel lines (``Channel cc/x : a[d] -> b[d] [send|receive] via T``) belong to the last
    communicator whose initialization the same thread started (NCCL connects a communicator's rings and its later
    point-to-point channels on the thread that drives it). The transport kind is the text after ``via`` without the
    device index (``NET/IB/0/Shared`` -> ``NET/IB``)."""
    comms: dict[str, dict] = {}
    by_ptr: dict[str, str] = {}
    current: dict[str, str] = {}
    for line in lines:
        prefix = _PREFIX.match(line)
        tid = prefix.group(3) if prefix else "?"
        m = _START.search(line)
        if m and "Init START" in line:
            ptr, rank, nranks, comm_id = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
            comms[comm_id] = {"comm_id": comm_id, "rank": rank, "nranks": nranks, "nnodes": None,
                              "transports": {}, "host": prefix.group(1) if prefix else None}
            by_ptr[ptr] = comm_id
            current[tid] = comm_id
            continue
        m = _NODES.search(line)
        if m and m.group(1) in by_ptr:
            comms[by_ptr[m.group(1)]]["nnodes"] = int(m.group(4))
            continue
        m = _CHANNEL.search(line)
        if m and tid in current:
            kind = "/".join(m.group(1).split("/")[:2])
            t = comms[current[tid]]["transports"]
            t[kind] = t.get(kind, 0) + 1
    return list(comms.values())


def nccl_log_path(env: Mapping[str, str], pid: int | None = None, host: str | None = None) -> str | None:
    pattern = env.get(NCCL_DEBUG_FILE)
    if not pattern:
        return None
    return pattern.replace("%h", host or socket.gethostname()).replace("%p", str(pid or os.getpid()))


def nccl_communicators(env: Mapping[str, str]) -> dict:
    path = nccl_log_path(env)
    if path is None or env.get("NCCL_DEBUG", "").upper() not in ("INFO", "TRACE"):
        return {"log": path, "state": "not_logged", "comms": []}
    try:
        text = Path(path).read_text(errors="replace").splitlines()
    except OSError as e:
        return {"log": path, "state": f"unreadable: {e}", "comms": []}
    return {"log": path, "state": "parsed", "comms": parse_nccl_log(text)}


def env_value(key: str, value: str | None, resolved: str | None = None) -> str | None:
    """The compared form of an environment value. A search path keeps its entries in first-seen order without
    repeats and, when the launch resolved the key, only the launch's own entries (what imported libraries add
    in-process is not part of the launch)."""
    if value is None or key not in PATH_LISTS:
        return value
    entries = list(dict.fromkeys(value.split(":")))
    if resolved is not None:
        own = set(resolved.split(":"))
        entries = [e for e in entries if e in own]
    return ":".join(entries)


def environment_view(resolved_env: Mapping[str, str], env: Mapping[str, str]) -> dict:
    """This process's value of every resolved key and of every NCCL key it sees (absent keys as None), in compared
    form (``env_value``)."""
    keys = {k for k in resolved_env if k not in PER_PROCESS_KEYS}
    keys |= {k for k in env if k.startswith(NCCL_PREFIXES)}
    return {k: env_value(k, env.get(k), resolved_env.get(k)) for k in sorted(keys)}


def environment_added(resolved_env: Mapping[str, str], env: Mapping[str, str]) -> dict:
    """Search-path entries this process carries beyond the launch's (added in-process; reported only)."""
    out = {}
    for k in sorted(PATH_LISTS & set(resolved_env)):
        extra = [e for e in dict.fromkeys((env.get(k) or "").split(":")) if e not in set(resolved_env[k].split(":"))]
        if extra:
            out[k] = extra
    return out


def _vllm_config_view(vllm_config) -> dict:
    if vllm_config is None:
        return {}
    pc, mc, sc, cc = (vllm_config.parallel_config, vllm_config.model_config, vllm_config.scheduler_config,
                      vllm_config.cache_config)
    comp = vllm_config.compilation_config
    view = {"tensor_parallel_size": pc.tensor_parallel_size, "pipeline_parallel_size": pc.pipeline_parallel_size,
            "nnodes": pc.nnodes, "enable_expert_parallel": pc.enable_expert_parallel,
            "max_model_len": mc.max_model_len, "dtype": str(mc.dtype), "seed": mc.seed,
            "max_num_batched_tokens": sc.max_num_batched_tokens, "max_num_seqs": sc.max_num_seqs,
            "enable_chunked_prefill": sc.enable_chunked_prefill, "block_size": cc.block_size,
            "cache_dtype": str(cc.cache_dtype), "num_gpu_blocks": cc.num_gpu_blocks,
            "cudagraph_mode": str(comp.cudagraph_mode), "cudagraph_capture_sizes": list(comp.cudagraph_capture_sizes or []),
            "enforce_eager": mc.enforce_eager}
    try:
        view["compute_hash"] = vllm_config.compute_hash()
    except Exception as e:  # noqa: BLE001 -- reported; the explicit fields above are compared regardless
        view["compute_hash"] = f"unavailable: {type(e).__name__}"
    return view


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    return value if value is None or isinstance(value, (bool, int, float, str)) else str(value)


def _program_identity(model) -> Any:
    seen = set()
    while model is not None and id(model) not in seen:
        seen.add(id(model))
        fn = getattr(model, "forward_program_identity", None)
        if callable(fn):
            return fn()
        model = getattr(model, "model", None) or getattr(model, "module", None)
    return None


def collect(resolved_env: Mapping[str, str], *, rank: int, tp_ranks: list[int], pp_ranks: list[int],
            model=None, vllm_config=None, device_uuid: str | None = None) -> dict:
    """This worker's launch record (plain data)."""
    env = dict(os.environ)
    record = {"rank": rank, "pid": os.getpid(), "tp_ranks": list(tp_ranks), "pp_ranks": list(pp_ranks),
              "host": host_identity(), "device_uuid": device_uuid, "launch_id": env.get("ISOEXEC_LAUNCH_ID"),
              "software": loaded_sources(env), "prebuilt": prebuilt_state(env),
              "env": environment_view(resolved_env, env), "env_added": environment_added(resolved_env, env),
              "nccl_live": {k: v for k, v in sorted(env.items()) if k.startswith(NCCL_PREFIXES)},
              "vllm_config": _vllm_config_view(vllm_config),
              "program_identity": _program_identity(model), "nccl": nccl_communicators(env)}
    try:
        import torch

        record["versions"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version.split()[0],
                              "nccl": ".".join(map(str, torch.cuda.nccl.version())) if torch.cuda.is_available()
                              else None}
    except Exception as e:  # noqa: BLE001
        record["versions"] = {"error": repr(e)}
    try:
        import vllm

        record["versions"]["vllm"] = vllm.__version__
    except Exception:  # noqa: BLE001
        record["versions"]["vllm"] = None
    # transport choices a model package recorded at placement (e.g. V4.1's TP reducer / EP exchange: symmetric memory
    # or the NCCL form, and why); reported with the receipt, not compared (the packages agree them per group)
    choices = {}
    for name, module in list(sys.modules.items()):
        table = getattr(module, "TRANSPORT_RECEIPTS", None) if name.startswith("isoexec.") else None
        if isinstance(table, Mapping):
            choices[name] = _plain(table)
    record["transport_choices"] = choices
    try:
        from isoexec.runtimes.vllm.vllm_plugin import NCCL_REGISTRATION_RECEIPT

        record["nccl_registration"] = dict(NCCL_REGISTRATION_RECEIPT)
    except Exception:  # noqa: BLE001
        record["nccl_registration"] = None
    return record


def warm_groups() -> None:
    """Create the communicators the first token would create, so their transport is known before it: one tiny
    point-to-point exchange on every PP edge of this worker (the same torch process group and device as vLLM's
    ``isend_tensor_dict``) and one all-reduce on the TP device group (IsoExec's own TP collectives use it)."""
    import torch
    import torch.distributed as dist
    from vllm.distributed import get_pp_group, get_tp_group

    device = torch.device("cuda", torch.cuda.current_device())
    tp = get_tp_group()
    if tp.world_size > 1:
        t = torch.ones(1, device=device)
        dist.all_reduce(t, group=tp.device_group)
    pp = get_pp_group()
    if pp.world_size > 1:
        buf = torch.zeros(1, device=device)
        ops = []
        if not pp.is_last_rank:
            ops.append(dist.P2POp(dist.isend, buf, pp.ranks[pp.rank_in_group + 1], group=pp.device_group))
        if not pp.is_first_rank:
            ops.append(dist.P2POp(dist.irecv, torch.empty_like(buf), pp.ranks[pp.rank_in_group - 1],
                                  group=pp.device_group))
        for work in dist.batch_isend_irecv(ops):
            work.wait()
    torch.cuda.synchronize()


def worker_receipt(worker, resolved_env: Mapping[str, str], warm: bool = True) -> dict:
    """``LLM.collective_rpc`` entry: warm this worker's groups, then return its record."""
    import torch
    from vllm.distributed import get_pp_group, get_tp_group, get_world_group

    if warm:
        warm_groups()
    runner = getattr(worker, "model_runner", None)
    model = getattr(runner, "model", None) if runner is not None else None
    vllm_config = getattr(worker, "vllm_config", None)
    uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    return collect(resolved_env, rank=get_world_group().rank, tp_ranks=list(get_tp_group().ranks),
                   pp_ranks=list(get_pp_group().ranks), model=model, vllm_config=vllm_config, device_uuid=uuid)


# --------------------------------------------------------------------------------------------- comparison


def _host_key(record: Mapping) -> str:
    h = record["host"]
    return f"{h['hostname']}/{h['boot_id']}"


def _first_diff(records: list[Mapping], field) -> tuple | None:
    ref = field(records[0])
    for r in records[1:]:
        if field(r) != ref:
            return records[0]["rank"], r["rank"], ref, field(r)
    return None


def _diff_maps(a: Mapping, b: Mapping, limit: int = 6) -> list[str]:
    out = []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            out.append(f"{k}: {a.get(k, '<absent>')!r} vs {b.get(k, '<absent>')!r}")
            if len(out) >= limit:
                break
    return out


def verify(records: list[Mapping], launch: Mapping) -> dict:
    """Compare every worker's record with the launch and with each other; return the transport table on success.

    ``launch`` is the resolved launch (``multinode.resolve``): ``placement`` (rank -> node / tp / pp), ``env``,
    ``node_env``, ``transports`` policy, ``require_prebuilt``. Raises ``LaunchRefusal`` naming the first reason."""
    records = sorted(records, key=lambda r: r["rank"])
    placement = {int(p["rank"]): p for p in launch["placement"]}
    if [r["rank"] for r in records] != sorted(placement):
        refuse("ranks", f"workers reported ranks {[r['rank'] for r in records]}, the launch places "
                        f"{sorted(placement)}")
    if any(r.get("launch_id") != launch["launch_id"] for r in records):
        bad = [(r["rank"], r.get("launch_id")) for r in records if r.get("launch_id") != launch["launch_id"]]
        refuse("launch_id", f"workers of another launch (rank, launch id): {bad}; expected {launch['launch_id']}")

    # placement: hosts are nodes, TP groups inside one host, each node holds whole PP stages
    hosts_of_node: dict[int, set] = {}
    for r in records:
        hosts_of_node.setdefault(placement[r["rank"]]["node"], set()).add(_host_key(r))
    for node, hosts in sorted(hosts_of_node.items()):
        if len(hosts) != 1:
            refuse("node_spans_hosts", f"node {node} workers report hosts {sorted(hosts)}")
    node_hosts = {n: next(iter(h)) for n, h in hosts_of_node.items()}
    if len(set(node_hosts.values())) != len(node_hosts):
        refuse("nodes_share_host", f"distinct nodes report the same host: {node_hosts}")
    expected_names = launch.get("expect_hostnames")
    if expected_names:
        for node, key in node_hosts.items():
            if key.split("/")[0] != expected_names[node]:
                refuse("hostname", f"node {node} runs on {key.split('/')[0]!r}, the launch names {expected_names[node]!r}")
    for r in records:
        p = placement[r["rank"]]
        if list(r["tp_ranks"]) != list(p["tp_ranks"]) or list(r["pp_ranks"]) != list(p["pp_ranks"]):
            refuse("groups", f"rank {r['rank']}: vLLM built TP {r['tp_ranks']} / PP {r['pp_ranks']}, the placement "
                             f"rule gives TP {p['tp_ranks']} / PP {p['pp_ranks']}")
    by_rank = {r["rank"]: r for r in records}
    for r in records:
        tp_hosts = {_host_key(by_rank[q]) for q in r["tp_ranks"]}
        if len(tp_hosts) != 1:
            refuse("tp_group_spans_hosts", f"TP group {r['tp_ranks']} spans hosts {sorted(tp_hosts)}")
    uuids = [(r["device_uuid"], _host_key(r)) for r in records if r.get("device_uuid")]
    if len(set(uuids)) != len(uuids):
        refuse("shared_device", f"two workers report the same device: {uuids}")

    # software and program
    # the commit is compared when every worker runs a checkout; the loaded source bytes always are
    checks = [("versions", lambda r: r["versions"])]
    if all(r["software"]["commit"] for r in records):
        checks.insert(0, ("commit", lambda r: r["software"]["commit"]))
    for name, field in checks:
        d = _first_diff(records, field)
        if d:
            refuse(name, f"rank {d[0]} {d[2]!r} vs rank {d[1]} {d[3]!r}")
    d = _first_diff(records, lambda r: r["software"]["sources_digest"])
    if d:
        a, b = by_rank[d[0]]["software"]["sources"], by_rank[d[1]]["software"]["sources"]
        common = set(a) & set(b)
        changed = [k for k in sorted(common) if a[k] != b[k]]
        if changed:   # a module only some ranks imported is not a difference of the program; a changed byte is
            refuse("sources", f"IsoExec source bytes differ between rank {d[0]} and rank {d[1]}: {changed[:6]}")
    d = _first_diff(records, lambda r: r["software"]["binaries_digest"])
    if d:
        a, b = by_rank[d[0]]["software"]["binaries"], by_rank[d[1]]["software"]["binaries"]
        changed = [k for k in sorted(set(a) & set(b)) if a[k] != b[k]]
        if changed:
            refuse("binaries", f"extension binaries differ between rank {d[0]} and rank {d[1]}: {changed[:6]}")
    d = _first_diff(records, lambda r: r["program_identity"])
    if d:
        refuse("program_identity", f"rank {d[0]} {d[2]!r} vs rank {d[1]} {d[3]!r}")
    if launch.get("require_program_identity", True) and any(r["program_identity"] is None for r in records):
        refuse("program_identity_missing", f"ranks without a forward program identity: "
                                           f"{[r['rank'] for r in records if r['program_identity'] is None]}")
    d = _first_diff(records, lambda r: r["vllm_config"])
    if d:
        refuse("vllm_config", f"rank {d[0]} vs rank {d[1]}: {_diff_maps(d[2], d[3])}")

    # prebuilt cache
    d = _first_diff(records, lambda r: r["prebuilt"])
    if d:
        refuse("prebuilt", f"rank {d[0]} {d[2]!r} vs rank {d[1]} {d[3]!r}")
    if launch.get("require_prebuilt"):
        p = records[0]["prebuilt"]
        if not p["module"]:
            refuse("prebuilt_mode_unavailable", "the launch requires ISOEXEC_PREBUILT_ONLY=1 but this IsoExec tree has "
                                                "no isoexec.ops.prebuilt (prebuilt-only cache mode not merged)")
        if p["only"] != "1" or not p["dir_set"] or not p["manifest_digest"]:
            refuse("prebuilt_not_active", f"prebuilt-only required, worker state {p}")
        want = launch.get("prebuilt_manifest_digest")
        if want and p["manifest_digest"] != want:
            refuse("prebuilt_manifest", f"workers load manifest {p['manifest_digest']}, the launch resolved {want}")

    # environment: every worker == resolved env + its node's declared differences; NCCL keys agree everywhere
    resolved_env = launch["env"]
    for r in records:
        node = placement[r["rank"]]["node"]
        want = dict(resolved_env)
        want.update(launch.get("node_env", {}).get(str(node), {}))
        want = {k: env_value(k, v, v) for k, v in want.items()}
        bad = [f"{k}: worker {r['env'].get(k)!r} vs launch {v!r}" for k, v in sorted(want.items())
               if k in r["env"] and r["env"].get(k) != v]
        if bad:
            refuse("environment", f"rank {r['rank']} (node {node}) differs from the resolved launch environment: "
                                  + "; ".join(bad[:6]))
    node_keys = {k for env in launch.get("node_env", {}).values() for k in env}
    for r in records:       # a key declared for some hosts: a host without a declaration keeps the resolved value
        node_env = launch.get("node_env", {}).get(str(placement[r["rank"]]["node"]), {})
        for k in sorted(node_keys - PER_PROCESS_KEYS):
            want = node_env.get(k, resolved_env.get(k))
            want = env_value(k, want, want)
            if k in r["env"] and r["env"][k] != want:
                refuse("environment", f"rank {r['rank']}: {k}={r['env'][k]!r}, its host declares {want!r}")

    def shared_env(r):
        return {k: v for k, v in r["env"].items() if k not in node_keys}

    d = _first_diff(records, shared_env)
    if d:
        refuse("nccl_settings", f"rank {d[0]} vs rank {d[1]}: {_diff_maps(d[2], d[3])}")
    # the NCCL settings in effect (after in-process defaults such as the plugin's NCCL_GRAPH_REGISTER / NVLS) agree too
    d = _first_diff(records, lambda r: {k: v for k, v in r.get("nccl_live", {}).items() if k not in node_keys})
    if d:
        refuse("nccl_settings", f"in effect, rank {d[0]} vs rank {d[1]}: {_diff_maps(d[2], d[3])}")
    d = _first_diff(records, lambda r: r.get("nccl_registration"))
    if d:
        refuse("nccl_settings", f"NCCL graph registration: rank {d[0]} {d[2]!r} vs rank {d[1]} {d[3]!r}")

    return transport_table(records, launch, placement)


def transport_table(records: list[Mapping], launch: Mapping, placement: Mapping[int, Mapping]) -> dict:
    """Join the communicators every worker parsed by commId and apply the placement's transport rule."""
    policy = launch.get("transports") or {}
    if not policy.get("required", True):
        return {"state": "not_required", "comms": []}
    unparsed = [(r["rank"], r["nccl"]["state"]) for r in records if r["nccl"]["state"] != "parsed"]
    if unparsed:
        refuse("transport_unknown", f"no NCCL debug log to read the selected transports from: {unparsed} (the launch "
                                    "sets NCCL_DEBUG=INFO and NCCL_DEBUG_FILE for this receipt)")
    by_rank = {r["rank"]: r for r in records}
    comms: dict[str, dict] = {}
    for r in records:
        for c in r["nccl"]["comms"]:
            entry = comms.setdefault(c["comm_id"], {"comm_id": c["comm_id"], "nranks": c["nranks"], "members": {},
                                                    "nnodes": set(), "transports": {}})
            entry["members"][c["rank"]] = r["rank"]
            entry["nnodes"].add(c["nnodes"])
            for k, n in c["transports"].items():
                entry["transports"][k] = entry["transports"].get(k, 0) + n
    tp_groups = {tuple(p["tp_ranks"]) for p in placement.values()}
    pp_groups = [set(p["pp_ranks"]) for p in placement.values()]
    cross_ok = tuple(policy.get("cross_host", ("NET/IB",)))
    intra = policy.get("intra_host", "not_net")
    table, seen_pp_edges = [], set()
    for cid, c in sorted(comms.items()):
        ranks = tuple(c["members"][i] for i in sorted(c["members"]))
        if len(ranks) != c["nranks"]:
            refuse("transport_partial", f"communicator {cid} has {c['nranks']} ranks but only workers {list(ranks)} "
                                        "logged it")
        hosts = {_host_key(by_rank[q]) for q in ranks}
        nnodes = c["nnodes"]
        if len(nnodes) != 1 or None in nnodes or next(iter(nnodes)) != len(hosts):
            refuse("nccl_host_view", f"communicator {cid} over ranks {list(ranks)} spans {len(hosts)} hosts by the "
                                     f"workers' host ids, NCCL counted nNodes {sorted(map(str, nnodes))}")
        kinds = sorted(c["transports"])
        if len(ranks) == 1:
            role = "single"
        elif tuple(sorted(ranks)) in {tuple(sorted(g)) for g in tp_groups}:
            role = "tp"
            if len(hosts) != 1:
                refuse("tp_group_spans_hosts", f"TP communicator {cid} over {list(ranks)} spans {sorted(hosts)}")
            if intra == "not_net" and any(k.startswith("NET/") for k in kinds):
                refuse("tp_transport", f"TP communicator {cid} over {list(ranks)} uses {kinds}: an intra-host TP "
                                       "group must not take the network")
        elif len(hosts) > 1 and any(set(ranks) <= g for g in pp_groups):
            role = "pp"
            if not kinds or any(not k.startswith(cross_ok) for k in kinds):
                refuse("pp_transport", f"PP communicator {cid} over {list(ranks)} uses {kinds or 'no channel'}; the "
                                       f"launch requires {list(cross_ok)}")
            seen_pp_edges.add(tuple(sorted(ranks)))
        elif len(hosts) > 1:
            refuse("unexpected_cross_host_comm", f"communicator {cid} over ranks {list(ranks)} crosses hosts "
                                                 f"{sorted(hosts)} and is not a pipeline edge (transports {kinds})")
        else:
            role = "intra_other"
            if intra == "not_net" and any(k.startswith("NET/") for k in kinds):
                refuse("intra_transport", f"intra-host communicator {cid} over {list(ranks)} uses {kinds}")
        table.append({"comm_id": cid, "ranks": list(ranks), "hosts": len(hosts), "nnodes": next(iter(nnodes)),
                      "role": role, "transports": c["transports"]})
    edges = {tuple(sorted({p["pp_ranks"][i], p["pp_ranks"][i + 1]})) for p in placement.values()
             for i in range(len(p["pp_ranks"]) - 1)
             if placement[p["pp_ranks"][i]]["node"] != placement[p["pp_ranks"][i + 1]]["node"]}
    covered = {e for e in edges if any(set(e) <= set(s) for s in seen_pp_edges)}
    if edges - covered:
        refuse("pp_transport_unobserved", f"no NCCL communicator was logged for the cross-host PP edges "
                                          f"{sorted(edges - covered)}")
    return {"state": "verified", "comms": table, "cross_host_edges": sorted(edges)}


__all__ = ["LaunchRefusal", "collect", "parse_nccl_log", "transport_table", "verify", "warm_groups",
           "worker_receipt"]
