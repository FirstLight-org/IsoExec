"""One vLLM engine over several hosts: TP inside a host, PP stages across hosts (the V4.1 recipe's TP8 x PP2 per node
pair), launched through vLLM's own multi-node multiprocessing executor with ONE resolved environment on every host.

Mature references: vLLM 0.33's multi-node ``mp`` backend (``config/parallel.py`` ``nnodes`` / ``node_rank`` /
``master_addr`` / ``master_port``; ``v1/executor/multiproc_executor.py`` global rank = ``node_rank x local_world_size
+ local_rank``; ``entrypoints/cli/serve.py`` ``run_headless``: a non-leader node runs only a headless
``MultiprocExecutor``), and IsoExec's rule that every actor of a job runs one resolved environment
(``runtimes/environment.resolved_environment``, as ``integrations/skyrl/bootstrap.worker_environment`` and
``config.engine_runtime_env`` carry it into Ray actors).

Placement rule. vLLM orders ranks TP-innermost (rank = pp_rank x TP + tp_rank) and the mp backend gives host
``node_rank`` the contiguous ranks ``[node_rank x L, (node_rank + 1) x L)`` with ``L = world / nnodes``. With ``L`` a
multiple of TP every TP group lies inside one host, and with ``nnodes == PP`` (TP8 x PP2 on two hosts: ``L = 8``) host
``i`` holds exactly PP stage ``i``. Then the only cross-host NCCL traffic is the PP edge (``{hidden_states, pre_mix}``
at the encoder/decoder cut, sent by each TP rank to its peer on the next host), and every TP reduction, the step
guard's digest, the vocab gather of the logits, the Engram / wkv gathers and the expert exchange stay inside a host,
where IsoExec's node-local symmetric-memory paths are admitted. A layout whose TP group would span hosts is refused.

Workflow (the same files on every host; see ``__main__`` for the CLI):
  1. ``resolve(spec)`` once: run ``register_native`` and ``resolved_environment`` for the spec and write the
     resolved launch (engine kwargs, environment, placement, digest) as JSON.
  2. ``run_node(launch, i)`` on host ``i``: apply exactly that environment (+ the spec's declared per-host keys),
     node 0 builds the engine (``build_engine``), every other host runs vLLM's headless executor.
  3. On node 0, before the first token, ``admit(llm, launch)``: every worker reports ``launch_receipt.collect``
     through ``collective_rpc`` and ``launch_receipt.verify`` refuses by name on any disagreement (commit / sources /
     binaries, program identity, vLLM config, prebuilt manifest, environment and NCCL settings, and the transport NCCL
     actually selected per communicator).
  4. Teardown: node 0 shuts its engine down; a headless host watches the leader's rendezvous port and shuts its
     executor down when the leader is gone; ``kill_launch`` removes every process of the launch on a host (marker
     ``ISOEXEC_LAUNCH_ID`` in its environment), which is how a supervisor kills a hung host.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import sys
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .launch_receipt import LaunchRefusal, refuse, verify, worker_receipt

#: Every process of one launch carries this in its environment (spawned workers inherit it).
LAUNCH_ID_ENV = "ISOEXEC_LAUNCH_ID"
#: ... and the node it belongs to (several fake nodes share one machine in tests; a host's abort kills only its node).
LAUNCH_NODE_ENV = "ISOEXEC_LAUNCH_NODE"
#: Format version of the resolved launch file.
LAUNCH_FORMAT = 1
#: Keys of the resolving shell the launch carries besides ``resolved_environment`` (cache roots and build identity the
#: extension and JIT keys depend on; ``runtimes/environment`` does not forward them, COLD-START.md 1d).
CARRIED = ("PYTHONPATH", "CUTE_DSL_CACHE_DIR", "VLLM_CACHE_ROOT", "TORCH_CUDA_ARCH_LIST", "CXX", "PYTORCH_CUDA_ALLOC_CONF")
#: Environment the launch needs for its own receipt: NCCL's per-process debug log (transport actually selected) and
#: callables through ``collective_rpc``.
RECEIPT_ENV = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET", "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}


def placement(*, tensor_parallel_size: int, pipeline_parallel_size: int, nnodes: int) -> list[dict]:
    """rank -> node, local rank, TP / PP rank and group ranks, as vLLM's mp backend builds them; refuses layouts whose
    TP group would span hosts or whose hosts would split a PP stage."""
    tp, pp = int(tensor_parallel_size), int(pipeline_parallel_size)
    world = tp * pp
    if nnodes < 1 or world % nnodes:
        refuse("nnodes", f"{nnodes} hosts do not divide the world of TP {tp} x PP {pp} = {world}")
    local = world // nnodes
    if local % tp:
        refuse("tp_spans_hosts", f"{local} ranks per host is not a multiple of TP {tp}: a TP group would span hosts "
                                 "(TP stays inside a host; only PP crosses hosts)")
    out = []
    for rank in range(world):
        pp_rank, tp_rank = divmod(rank, tp)
        out.append({"rank": rank, "node": rank // local, "local_rank": rank % local, "tp_rank": tp_rank,
                    "pp_rank": pp_rank, "tp_ranks": [pp_rank * tp + i for i in range(tp)],
                    "pp_ranks": [s * tp + tp_rank for s in range(pp)]})
    return out


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def launch_digest(launch: Mapping) -> str:
    body = {k: v for k, v in launch.items() if k != "digest"}
    return hashlib.sha256(_canonical(body).encode()).hexdigest()


def _raw_config(checkpoint: str) -> dict:
    """The checkpoint's config.json (a renamed subset checkpoint folds in its isoexec_subset.json)."""
    path = Path(checkpoint)
    root = path.parent if path.is_file() else path
    raw = json.loads((root / "config.json").read_text())
    subset = root / "isoexec_subset.json"
    if "isoexec_subset" not in raw and subset.exists():
        raw["isoexec_subset"] = {"layer_sources": json.loads(subset.read_text())["layer_sources"]}
    return raw


def resolve(spec: Mapping[str, Any]) -> dict:
    """The resolved launch: engine kwargs after ``register_native``, ONE environment for every host, placement.

    ``spec`` keys: ``hosts`` (one entry per node: its name, also the expected hostname when ``expect_hostnames``),
    ``master_addr``, ``master_port``, ``tensor_parallel_size``, ``pipeline_parallel_size``, ``model`` (``package``,
    ``checkpoint``, ``cudagraph_mode``, optional ``model_runner``), ``engine_kwargs`` (vLLM kwargs given to
    ``register_native``), optional ``env`` (extra keys for every host), ``node_env`` ({node: {key: value}}: the only
    keys allowed to differ per host, e.g. a per-host HCA list), ``run_dir`` (same path on every host: NCCL logs and
    receipts), ``require_prebuilt``, ``transports`` ({``required``, ``cross_host``: allowed kinds for the PP edge,
    ``intra_host``: ``not_net`` | ``any``}), ``launch_id`` (default: a fresh one).

    Runs ``apply_trainer_env`` and ``register_native`` in THIS process (they export the engine's process
    environment), so call it in a process of its own (the CLI ``resolve`` command does)."""
    from isoexec.compiler.source import ModelSource
    from isoexec.runtimes.environment import resolved_environment
    from isoexec.runtimes.trainer_installs import apply_trainer_env
    from isoexec.runtimes.vllm.registration import register_native

    hosts = list(spec["hosts"])
    tp, pp = int(spec["tensor_parallel_size"]), int(spec["pipeline_parallel_size"])
    places = placement(tensor_parallel_size=tp, pipeline_parallel_size=pp, nnodes=len(hosts))
    model = dict(spec["model"])
    checkpoint, mode = model["checkpoint"], model.get("cudagraph_mode", "NONE")
    runner = model.get("model_runner", "v2")
    if runner != "v2":
        refuse("model_runner", "IsoExec requires vLLM Model Runner V2")
    before = dict(os.environ)
    apply_trainer_env(checkpoint)
    engine = dict(spec.get("engine_kwargs") or {})
    for name in ("nnodes", "node_rank", "master_addr", "master_port", "tensor_parallel_size",
                 "pipeline_parallel_size", "distributed_executor_backend"):
        if name in engine:
            refuse("engine_kwargs", f"{name} is set by the launch, not by engine_kwargs")
    engine.update(tensor_parallel_size=tp, pipeline_parallel_size=pp, distributed_executor_backend="mp")
    kwargs = register_native(ModelSource(model["package"]), _raw_config(checkpoint), checkpoint=checkpoint,
                             cudagraph_mode=mode, engine_kwargs=engine,
                             score_temperature=float(model.get("score_temperature", 1.0)),
                             enable_chunked_prefill=bool(model.get("enable_chunked_prefill", False)))
    env = resolved_environment(checkpoint)
    # What the driver-side preparation exported (register_native / apply_trainer_env), as a launcher process would
    # carry it into the engine processes it spawns.
    env.update({k: v for k, v in os.environ.items() if before.get(k) != v and k not in env})
    # As SkyRL's worker environment does (integrations/skyrl/bootstrap.worker_environment): every ISOEXEC* key of the
    # resolving shell; and every NCCL / Python path / build-identity setting (hosts must agree on them).
    env.update({k: v for k, v in before.items() if (k.startswith(("ISOEXEC", "NCCL_", "TORCH_NCCL_")) or k in CARRIED)
                and k not in env})
    env.update({k: str(v) for k, v in (spec.get("env") or {}).items()})
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    run_dir = str(Path(spec["run_dir"]).resolve())
    policy = {"required": True, "cross_host": ["NET/IB"], "intra_host": "not_net"}
    policy.update(spec.get("transports") or {})
    if policy["required"]:
        env.update(RECEIPT_ENV)
        env["NCCL_DEBUG_FILE"] = f"{run_dir}/nccl/nccl.%h.%p.log"
    if spec.get("require_prebuilt"):
        env["ISOEXEC_PREBUILT_ONLY"] = "1"
        if not env.get("ISOEXEC_PREBUILT_DIR"):
            refuse("prebuilt_dir_missing", "require_prebuilt is set but ISOEXEC_PREBUILT_DIR is not")
    manifest = None
    if env.get("ISOEXEC_PREBUILT_DIR"):
        try:
            manifest = json.loads((Path(env["ISOEXEC_PREBUILT_DIR"]) / "MANIFEST.json").read_text()).get("digest")
        except (OSError, ValueError):
            manifest = None
        if spec.get("require_prebuilt") and not manifest:
            refuse("prebuilt_manifest_missing", f"no MANIFEST.json digest under {env['ISOEXEC_PREBUILT_DIR']}")
    launch_id = spec.get("launch_id") or uuid.uuid4().hex
    env.pop(LAUNCH_ID_ENV, None)
    launch = {"format": LAUNCH_FORMAT, "launch_id": launch_id, "hosts": hosts,
              "expect_hostnames": hosts if spec.get("expect_hostnames") else None,
              "master_addr": spec["master_addr"], "master_port": int(spec["master_port"]),
              "tensor_parallel_size": tp, "pipeline_parallel_size": pp, "nnodes": len(hosts),
              "placement": places, "engine_kwargs": kwargs, "env": dict(sorted(env.items())),
              "node_env": {str(k): {kk: str(vv) for kk, vv in v.items()} for k, v in (spec.get("node_env") or {}).items()},
              "run_dir": run_dir, "require_prebuilt": bool(spec.get("require_prebuilt")),
              "prebuilt_manifest_digest": manifest, "transports": policy,
              "require_program_identity": bool(spec.get("require_program_identity", True)),
              "teardown": {"leader_probe_s": float(spec.get("leader_probe_s", 5.0)),
                           "leader_lost_after": int(spec.get("leader_lost_after", 3)),
                           "leader_wait_s": float(spec.get("leader_wait_s", 1800.0))}}
    launch["digest"] = launch_digest(launch)
    return launch


def load(path: str | os.PathLike) -> dict:
    launch = json.loads(Path(path).read_text())
    if launch.get("format") != LAUNCH_FORMAT:
        refuse("format", f"{path}: launch format {launch.get('format')!r}, this IsoExec reads {LAUNCH_FORMAT}")
    if launch_digest(launch) != launch.get("digest"):
        refuse("digest", f"{path}: the resolved launch was edited after resolve (digest mismatch)")
    return launch


def node_environment(launch: Mapping, node_rank: int) -> dict:
    """The exact environment of host ``node_rank``: the resolved environment, the host's declared keys, the marker."""
    env = dict(launch["env"])
    env.update(launch.get("node_env", {}).get(str(node_rank), {}))
    env[LAUNCH_ID_ENV] = launch["launch_id"]
    env[LAUNCH_NODE_ENV] = str(node_rank)
    return env


def apply_node_environment(launch: Mapping, node_rank: int) -> dict:
    env = node_environment(launch, node_rank)
    os.environ.update(env)
    (Path(launch["run_dir"]) / "nccl").mkdir(parents=True, exist_ok=True)
    return env


def engine_kwargs(launch: Mapping, node_rank: int) -> dict:
    kw = dict(launch["engine_kwargs"])
    kw.update(nnodes=launch["nnodes"], node_rank=int(node_rank), master_addr=launch["master_addr"],
              master_port=int(launch["master_port"]))
    return kw


def _check_local_devices(launch: Mapping) -> None:
    import torch

    local = launch["tensor_parallel_size"] * launch["pipeline_parallel_size"] // launch["nnodes"]
    have = torch.cuda.device_count()
    if have < local:
        refuse("local_devices", f"this host sees {have} CUDA devices (CUDA_VISIBLE_DEVICES="
                                f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}); the launch places {local} ranks per host")


def _write_node_record(launch: Mapping, node_rank: int, state: str, **extra) -> None:
    d = Path(launch["run_dir"]) / "nodes"
    d.mkdir(parents=True, exist_ok=True)
    rec = {"node": node_rank, "state": state, "pid": os.getpid(), "pgid": os.getpgid(0), "time": time.time(),
           "hostname": socket.gethostname(), "launch_id": launch["launch_id"], **extra}
    tmp = d / f".node{node_rank}.{os.getpid()}.json"
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    os.replace(tmp, d / f"node{node_rank}.json")


def build_engine(launch: Mapping):
    """Node 0: ``vllm.LLM`` over every host (returns after vLLM's start-up; call ``admit`` before generating)."""
    from vllm import LLM

    _check_local_devices(launch)
    _write_node_record(launch, 0, "starting")
    llm = LLM(**engine_kwargs(launch, 0))
    _write_node_record(launch, 0, "engine_up")
    return llm


def admit(llm, launch: Mapping, *, timeout: float = 600.0) -> dict:
    """Before the first token: every worker's receipt, compared; ``LaunchRefusal`` on any disagreement. Writes
    ``<run_dir>/receipts.json`` (records + verdict) in either case."""
    records = llm.collective_rpc(worker_receipt, timeout=timeout, args=(launch["env"],))
    out = Path(launch["run_dir"]) / "receipts.json"
    try:
        table = verify(records, launch)
    except LaunchRefusal as e:
        out.write_text(json.dumps({"verdict": str(e), "records": records}, indent=1, default=str))
        raise
    out.write_text(json.dumps({"verdict": "admitted", "transports": table, "records": records}, indent=1, default=str))
    _write_node_record(launch, 0, "admitted")
    return table


def _leader_reachable(addr: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((addr, port), timeout=timeout):
            return True
    except OSError:
        return False


def run_headless(launch: Mapping, node_rank: int) -> int:
    """Host ``node_rank`` > 0: vLLM's headless multiprocessing executor (its workers join the leader's engine) and a
    leader watch: once the leader's rendezvous port has answered, losing it for ``leader_lost_after`` probes shuts this
    host's executor down (the leader finished or died); never answering within ``leader_wait_s`` is a failure."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    if node_rank < 1:
        raise ValueError("run_headless is for node_rank > 0; node 0 builds the engine")
    _check_local_devices(launch)
    _write_node_record(launch, node_rank, "starting")
    args = engine_args(engine_kwargs(launch, node_rank))
    config = args.create_engine_config(usage_context=UsageContext.LLM_CLASS, headless=True)
    executor = MultiprocExecutor(config, monitor_workers=False)
    _write_node_record(launch, node_rank, "workers_up")
    state = {"reason": None}

    def lost(reason):
        state["reason"] = reason
        _write_node_record(launch, node_rank, reason)
        executor.shutdown()

    _watch_leader(launch, lost)

    def on_signal(signum, frame):
        state["reason"] = f"signal {signum}"
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        executor.start_worker_monitor(inline=True)
    finally:
        executor.shutdown()
        _write_node_record(launch, node_rank, "exited", reason=state["reason"])
    return 0 if state["reason"] == "leader_gone" else 1


def engine_args(kwargs: Mapping[str, Any]):
    """``EngineArgs`` exactly as ``vllm.LLM`` builds them from the same kwargs (so a headless host's config equals the
    leader's: the receipt compares the two)."""
    import inspect

    from vllm import LLM
    from vllm.config import (AttentionConfig, CompilationConfig, ProfilerConfig, StructuredOutputsConfig,
                             is_init_field)
    from vllm.config.compilation import CompilationMode
    from vllm.engine.arg_utils import EngineArgs

    kw = dict(kwargs)
    # LLM's own defaults where they differ from EngineArgs' (e.g. gpu_memory_utilization), as LLM passes them.
    fields = set(inspect.signature(EngineArgs).parameters)
    for name, p in inspect.signature(LLM.__init__).parameters.items():
        if name in fields and name not in kw and p.default is not inspect.Parameter.empty:
            kw[name] = p.default
    kw.setdefault("disable_log_stats", True)
    kw["hf_overrides"] = kw.get("hf_overrides") or {}

    def make(value, cls):
        if value is None:
            return cls()
        if isinstance(value, dict):
            return cls(**{k: v for k, v in value.items() if is_init_field(cls, k)})
        return value

    comp = kw.pop("compilation_config", None)
    kw["compilation_config"] = (CompilationConfig(mode=CompilationMode(comp)) if isinstance(comp, int)
                                else make(comp, CompilationConfig))
    kw["structured_outputs_config"] = make(kw.pop("structured_outputs_config", None), StructuredOutputsConfig)
    kw["profiler_config"] = make(kw.pop("profiler_config", None), ProfilerConfig)
    kw["attention_config"] = make(kw.pop("attention_config", None), AttentionConfig)
    kw["offload_params"] = kw.get("offload_params") or set()
    return EngineArgs(**kw)


def _marked(launch_id: str, node: int | None = None) -> list[int]:
    me, uid, out = os.getpid(), os.getuid(), []
    needles = [f"{LAUNCH_ID_ENV}={launch_id}\0".encode()]
    if node is not None:
        needles.append(f"{LAUNCH_NODE_ENV}={node}\0".encode())
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            if os.stat(f"/proc/{d}").st_uid == uid:
                env = b"\0" + Path(f"/proc/{d}/environ").read_bytes()
                if all(b"\0" + n in env for n in needles):
                    out.append(int(d))
        except OSError:
            pass
    return out


def kill_launch(launch_id: str, *, node: int | None = None, grace: float = 10.0) -> dict:
    """Every process of this uid on THIS host whose environment carries the launch marker (and, with ``node``, that
    node's marker): TERM, then KILL after ``grace`` (a stopped or hung process is killed too). Only marked processes
    are signalled."""
    found = _marked(launch_id, node)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        live = _marked(launch_id, node)
        if not live:
            break
        for pid in live:
            try:
                os.kill(pid, sig)
                if sig == signal.SIGTERM:
                    os.kill(pid, signal.SIGCONT)   # a stopped process must run to act on TERM
            except ProcessLookupError:
                pass
        end = time.time() + grace
        while time.time() < end and _marked(launch_id, node):
            time.sleep(0.2)
    return {"host": socket.gethostname(), "node": node, "found": found, "left": _marked(launch_id, node)}


def _watch_leader(launch: Mapping, on_lost) -> threading.Thread:
    td = launch["teardown"]

    def watch():
        seen, misses, start = False, 0, time.monotonic()
        while True:
            time.sleep(td["leader_probe_s"])
            if _leader_reachable(launch["master_addr"], int(launch["master_port"])):
                seen, misses = True, 0
                continue
            misses += 1
            if (seen and misses >= td["leader_lost_after"]) or (not seen and time.monotonic() - start > td["leader_wait_s"]):
                on_lost("leader_gone" if seen else "leader_never_seen")
                return

    thread = threading.Thread(target=watch, daemon=True, name="isoexec-leader-watch")
    thread.start()
    return thread


# ------------------------------------------------------------------------------------------ CPU dry run
# The launch's own code paths without vLLM or a GPU: real processes per host (one per placed rank), a gloo world over
# the launch's rendezvous, the placement's TP / PP groups, every worker's receipt, the comparison on rank 0, and the
# same teardown (leader watch on the other hosts). Not checked here: program identity (needs the model), the vLLM
# config, and NCCL transports (no NCCL); the dry run says so in its verdict.


def _dry_worker(launch: dict, node_rank: int, local_rank: int, timeout_s: float) -> None:
    import datetime

    import torch.distributed as dist

    local = launch["tensor_parallel_size"] * launch["pipeline_parallel_size"] // launch["nnodes"]
    rank = node_rank * local + local_rank
    world = launch["tensor_parallel_size"] * launch["pipeline_parallel_size"]
    dist.init_process_group("gloo", init_method=f"tcp://{launch['master_addr']}:{launch['master_port']}", rank=rank,
                            world_size=world, timeout=datetime.timedelta(seconds=timeout_s))
    places = {p["rank"]: p for p in launch["placement"]}
    groups = {}
    for key in ("tp_ranks", "pp_ranks"):
        for ranks in sorted({tuple(p[key]) for p in places.values()}):
            groups[(key, ranks)] = dist.new_group(list(ranks))
    mine = places[rank]
    record = __import__("isoexec.runtimes.vllm.launch_receipt", fromlist=["collect"]).collect(
        launch["env"], rank=rank, tp_ranks=mine["tp_ranks"], pp_ranks=mine["pp_ranks"])
    # every group really exchanges once (the placement's groups exist and reach each other)
    for key in ("tp_ranks", "pp_ranks"):
        peers = [None] * len(mine[key])
        dist.all_gather_object(peers, record["host"], group=groups[(key, tuple(mine[key]))])
        record.setdefault("group_hosts", {})[key] = peers
    records = [None] * world
    dist.all_gather_object(records, record)
    verdict = None
    if rank == 0:
        relaxed = dict(launch, require_program_identity=False, transports={"required": False})
        try:
            table = verify(records, relaxed)
            verdict = {"verdict": "admitted", "transports": table}
        except LaunchRefusal as e:
            verdict = {"verdict": str(e)}
        verdict["not_checked"] = ["program_identity (no model)", "vllm_config (no engine)", "NCCL transports (gloo)"]
        verdict["records"] = records
        out = Path(launch["run_dir"]) / "dryrun.json"
        out.write_text(json.dumps(verdict, indent=1, default=str))
    box = [verdict["verdict"] if verdict else None]
    dist.broadcast_object_list(box, src=0)
    hold = float(os.environ.get("ISOEXEC_LAUNCH_DRY_HOLD", "0") or 0)   # keeps the world alive for teardown tests
    if hold:
        time.sleep(hold)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if box[0] == "admitted" else 3)


def run_dry(launch: dict, node_rank: int, *, timeout_s: float = 120.0) -> int:
    import multiprocessing

    local = launch["tensor_parallel_size"] * launch["pipeline_parallel_size"] // launch["nnodes"]
    _write_node_record(launch, node_rank, "starting", mode="dry-run")
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_dry_worker, args=(launch, node_rank, i, timeout_s), name=f"dry-{node_rank}-{i}")
             for i in range(local)]
    for p in procs:
        p.start()
    state = {"reason": None}

    def lost(reason):
        state["reason"] = reason
        _write_node_record(launch, node_rank, reason, mode="dry-run")
        for p in procs:
            if p.is_alive():
                p.terminate()

    if node_rank > 0:
        _watch_leader(launch, lost)
    for p in procs:
        p.join()
    codes = [p.exitcode for p in procs]
    _write_node_record(launch, node_rank, "exited", mode="dry-run", exitcodes=codes, reason=state["reason"])
    if state["reason"]:          # this host's workers were stopped by the leader watch: a teardown, not a result
        return 0 if state["reason"] == "leader_gone" else 4
    return 0 if all(c == 0 for c in codes) else (3 if 3 in codes else 1)


def abort_node(launch: Mapping, node_rank: int, reason: str, code: int = 5) -> None:
    """Leave after an engine failure without waiting on it: a worker whose peer host stopped sits in a NCCL
    collective and vLLM's shutdown (and the interpreter's exit) then waits forever (seen on the fake hosts: the
    leader logged EngineDeadError and never exited). Kill every process of the launch on THIS host, record why, exit."""
    try:
        killed = kill_launch(launch["launch_id"], node=node_rank, grace=5.0)
    except Exception as e:  # noqa: BLE001
        killed = {"error": repr(e)}
    try:
        _write_node_record(launch, node_rank, "aborted", reason=reason[:2000], killed=killed)
    finally:
        print(f"[ISOEXEC-LAUNCH] node {node_rank} aborted: {reason[:500]}", file=sys.stderr, flush=True)
        os._exit(code)


def _smoke(llm, launch: Mapping, tokens: int) -> dict:
    from vllm import SamplingParams

    prompts = [{"prompt_token_ids": [1 + (7 * i + j) % 1000 for j in range(16 + i)]} for i in range(4)]
    out = llm.generate(prompts, [SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True, logprobs=1)] * 4,
                       use_tqdm=False)
    rows = [{"tokens": list(o.outputs[0].token_ids),
             "logprobs": [float(d[t].logprob) for t, d in zip(o.outputs[0].token_ids, o.outputs[0].logprobs)]}
            for o in out]
    (Path(launch["run_dir"]) / "smoke.json").write_text(json.dumps(rows))
    return {"requests": len(rows), "tokens": sum(len(r["tokens"]) for r in rows)}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m isoexec.runtimes.vllm.multinode",
                                 description="one vLLM engine over several hosts (TP in a host, PP across hosts)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("resolve", help="resolve a spec once into the launch every host runs")
    r.add_argument("--spec", required=True)
    r.add_argument("--out", required=True)
    n = sub.add_parser("node", help="run this host's part of a resolved launch")
    n.add_argument("--launch", required=True)
    n.add_argument("--node-rank", type=int, required=True)
    n.add_argument("--dry-run", action="store_true", help="CPU: gloo world, receipts and teardown without vLLM")
    n.add_argument("--smoke-tokens", type=int, default=0, help="node 0: generate this many tokens after admission")
    n.add_argument("--dry-timeout", type=float, default=120.0)
    k = sub.add_parser("kill", help="kill every process of a launch on THIS host")
    k.add_argument("--launch", default=None)
    k.add_argument("--launch-id", default=None)
    k.add_argument("--grace", type=float, default=10.0)
    k.add_argument("--node-rank", type=int, default=None, help="only that node's processes (default: every node's)")
    p = sub.add_parser("placement", help="print the placement of TP x PP over N hosts")
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--pp", type=int, required=True)
    p.add_argument("--nnodes", type=int, required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "resolve":
            launch = resolve(json.loads(Path(a.spec).read_text()))
            Path(a.out).write_text(json.dumps(launch, indent=1, default=str))
            print(json.dumps({"launch_id": launch["launch_id"], "digest": launch["digest"], "out": a.out}))
            return 0
        if a.cmd == "placement":
            for row in placement(tensor_parallel_size=a.tp, pipeline_parallel_size=a.pp, nnodes=a.nnodes):
                print(json.dumps(row))
            return 0
        if a.cmd == "kill":
            launch_id = a.launch_id or load(a.launch)["launch_id"]
            print(json.dumps(kill_launch(launch_id, node=a.node_rank, grace=a.grace)))
            return 0
        launch = load(a.launch)
        apply_node_environment(launch, a.node_rank)
        if a.dry_run:
            return run_dry(launch, a.node_rank, timeout_s=a.dry_timeout)
        if a.node_rank > 0:
            return run_headless(launch, a.node_rank)
        llm = build_engine(launch)
        try:
            table = admit(llm, launch)
            print(json.dumps({"admitted": True, "comms": len(table.get("comms", []))}), flush=True)
            if a.smoke_tokens:
                print(json.dumps({"smoke": _smoke(llm, launch, a.smoke_tokens)}), flush=True)
        except LaunchRefusal:
            llm.llm_engine.engine_core.shutdown()      # workers are healthy: the ordinary shutdown
            _write_node_record(launch, 0, "refused")
            raise
        except BaseException as e:  # noqa: BLE001 -- a failed engine: its workers may sit in a collective
            abort_node(launch, 0, repr(e))
        llm.llm_engine.engine_core.shutdown()
        _write_node_record(launch, 0, "exited")
        return 0
    except LaunchRefusal as e:
        print(f"REFUSED {e}", file=sys.stderr, flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["LAUNCH_ID_ENV", "LaunchRefusal", "admit", "apply_node_environment", "build_engine", "engine_args",
           "abort_node", "kill_launch", "load", "main", "node_environment", "placement", "resolve", "run_dry", "run_headless"]
