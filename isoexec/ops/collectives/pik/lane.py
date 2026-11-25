"""Where pik's collectives run: peer admission by PHYSICAL device identity, and one comm lane per group.

Two schedule-only facilities. Neither touches arithmetic: every reduce still evaluates the same fixed tree.

Peer admission (``physical_p2p``). Symmetric memory needs every rank of a group on one host, each on its
own GPU. Ray (verl, SkyRL) gives every trainer actor exactly one visible GPU, so every rank calls it
``cuda:0`` and ``symm_mem.rendezvous`` refuses the group as "overlapping devices" although the ranks sit on
distinct GPUs. The right identity is the physical one: the device UUID (and the host: hostname + kernel boot
id). The probe gathers ``(host, UUID, ordinal, wanted, MIG, P2P size cap)`` from every rank and decides from
the gathered rows only, so every rank takes the same branch before any rendezvous: a group whose ranks all
want P2P, share a host, are not MIG instances and have pairwise distinct UUIDs is tried -- with torch's
ordinal check relaxed only when the ordinals collide but the UUIDs do not -- and any other group is not
(NCCL moves the bytes there, same bits). The tried transport must then work on every rank (MIN vote). The
verdict and the group's P2P size cap (MIN over ranks) are cached per group.

Comm lane (``lane``). One side CUDA stream per (group, device). A collective submitted to the lane waits for
everything already enqueued on the caller's stream, runs on the lane, and hands back an event the consumer
waits on; the caller's stream meanwhile runs independent work (the next GEMM chunk, the weight-gradient
GEMM). Every symmetric-memory collective of a process either runs on the lane or first waits for the lane's
tail (``fence``), so two collectives never touch a symmetric buffer concurrently and the per-group order
of collectives is the program order on every rank.
"""

from __future__ import annotations

import logging
import os
import socket
import zlib

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

ENV_OVERLAP = "ISOEXEC_PIK_OVERLAP"
ENV_OVERLAP_MIN_M = "ISOEXEC_PIK_OVERLAP_MIN_M"
ENV_OVERLAP_CHUNKS = "ISOEXEC_PIK_OVERLAP_CHUNKS"
ENV_OVERLAP_SMS = "ISOEXEC_PIK_OVERLAP_COMM_SMS"
_OVERLAP_ALLOW = "TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip() or default)
    except ValueError:
        return default


def _group_key(group):
    try:
        return tuple(dist.get_process_group_ranks(group))
    except Exception:  # noqa: BLE001 -- default group, or an older torch
        return ("world", dist.get_world_size(group))


def _capturing() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------------------------- physical peers

#: None = the process keeps pik's historical rule (``p2p_available`` == "a group of >1 ranks"; the engine,
#: whose ranks see every GPU). "physical" = admit by (host, UUID) with a group-agreed probe (the trainer).
_POLICY: str | None = None
#: This rank's own vote in the physical probe (``ISOEXEC_PIK_TRAINER_P2P``). A rank that does not want P2P
#: still enters the probe and votes no, so no rank skips a collective its peers run.
_WANTED = True
_VERDICTS: dict = {}
_CAPS: dict = {}  # group key -> agreed per-rank P2P byte cap (MIN over ranks; 0 = none)
_COUNTS = {"probes": 0, "admitted": 0, "refused": 0, "ordinal_overlap_relaxed": 0}


def set_peer_policy(policy: str | None, wanted: bool = True) -> None:
    global _POLICY, _WANTED
    if policy not in (None, "physical"):
        raise ValueError(f"peer policy must be None or 'physical', got {policy!r}")
    _POLICY = policy
    _WANTED = bool(wanted)


def peer_policy() -> str | None:
    return _POLICY


def peer_counts() -> dict:
    return {**_COUNTS, "policy": _POLICY, "groups": {str(k): v for k, v in _VERDICTS.items()}}


def device_uuid(device=None) -> bytes:
    """The physical identity of ``device`` (16 bytes), independent of CUDA_VISIBLE_DEVICES remapping."""
    props = torch.cuda.get_device_properties(device if device is not None else torch.cuda.current_device())
    u = getattr(props, "uuid", None)
    if u is None:
        raise RuntimeError("torch.cuda device properties expose no uuid")
    raw = getattr(u, "bytes", None)
    if raw is None:  # older torch: the str() form "GPU-xxxxxxxx-..." or a bare hex string
        s = str(u).removeprefix("GPU-").replace("-", "")
        raw = bytes.fromhex(s)
    return bytes(raw)[:16].ljust(16, b"\0")


def is_mig(device=None) -> bool:
    """Whether ``device`` is a MIG instance (CUDA IPC, hence symmetric memory, is unsupported there)."""
    if "MIG-" in os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        return True
    try:
        props = torch.cuda.get_device_properties(device if device is not None else torch.cuda.current_device())
        return str(getattr(props, "uuid", "")).startswith("MIG-")
    except Exception:  # noqa: BLE001
        return False


def host_id() -> int:
    """This host's identity: hostname AND kernel boot id (containers on different nodes may share a hostname)."""
    try:
        with open("/proc/sys/kernel/random/boot_id", "rb") as f:
            boot = f.read().strip()
    except OSError:
        boot = b""
    return zlib.crc32(socket.gethostname().encode() + b"/" + boot) & 0x7FFFFFFF


def classify(rows: list[tuple[int, bytes, int]], wanted=None, mig=None) -> tuple[bool, bool, str]:
    """(admit, relax_ordinals, why) for every rank's ``(host_id, uuid, ordinal)`` and, optionally, every rank's
    P2P vote and MIG flag. Pure; rank-uniform."""
    if wanted is not None and not all(wanted):
        off = [r for r, w in enumerate(wanted) if not w]
        return False, False, f"ranks {off} do not want P2P (ISOEXEC_PIK_TRAINER_P2P=0)"
    if mig is not None and any(mig):
        return False, False, "MIG instances (no CUDA IPC)"
    hosts = {r[0] for r in rows}
    if len(hosts) != 1:
        return False, False, f"ranks span {len(hosts)} hosts (symmetric memory is intra-node)"
    uuids = [r[1] for r in rows]
    if len(set(uuids)) != len(uuids):
        return False, False, "two ranks share one physical GPU"
    ordinals = [r[2] for r in rows]
    relax = len(set(ordinals)) != len(ordinals)
    why = ("distinct GPUs behind colliding device ordinals (one visible GPU per rank, e.g. Ray actors)"
           if relax else "distinct GPUs, distinct ordinals")
    return True, relax, why


def _identity_row(device) -> torch.Tensor:
    """This rank's probe row: [host, ordinal, uuid x16, wanted | mig << 1, P2P byte cap]."""
    mine = torch.zeros(_ROW, dtype=torch.int64, device=device)
    mine[0] = host_id()
    if device.index is not None:
        mine[1] = int(device.index)
    elif device.type == "cuda":
        mine[1] = int(torch.cuda.current_device())
    mine[2:18] = torch.tensor(list(device_uuid(device)), dtype=torch.int64)
    mine[18] = int(_WANTED) | (int(is_mig(device)) << 1)
    mine[19] = _local_p2p_max_bytes()
    return mine


_ROW = 20


def _rendezvous_works(group, device) -> bool:
    """Map every peer's buffer and run one barrier through it (every rank of the group calls this)."""
    from torch.distributed import _symmetric_memory as symm_mem

    from .allreduce import rendezvous

    world = dist.get_world_size(group)
    t = symm_mem.empty(64, dtype=torch.int32, device=device)
    t.fill_(dist.get_rank(group))
    h = rendezvous(t, group)
    h.barrier()
    peer = (dist.get_rank(group) + 1) % world
    ok = int(h.get_buffer(peer, (64,), torch.int32)[0].item()) == peer
    h.barrier()
    return ok


def _probe(group, device) -> bool:
    world = dist.get_world_size(group)
    mine = _identity_row(device)
    allv = torch.empty((world, _ROW), dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(allv, mine, group=group)
    gathered = allv.cpu().tolist()
    rows = [(int(r[0]), bytes(int(b) for b in r[2:18]), int(r[1])) for r in gathered]
    # Everything below is decided from the gathered rows, identically on every rank, before any rendezvous.
    admit, relax, why = classify(rows, wanted=[bool(r[18] & 1) for r in gathered],
                                 mig=[bool(r[18] & 2) for r in gathered])
    bounded = [int(r[19]) for r in gathered if int(r[19]) > 0]  # 0 = no limit on that rank
    _CAPS[_group_key(group)] = min(bounded) if bounded else 0
    ok = admit
    err = ""
    if ok:
        if relax:
            # torch checks distinct ORDINALS; the UUIDs above prove distinct GPUs, which is what it means. It
            # stays set: every later rendezvous of this process (owner, pools, Lamport buffers) needs it, and
            # each of those groups is itself admitted by this probe first (p2p_available under this policy).
            os.environ[_OVERLAP_ALLOW] = "1"
            _COUNTS["ordinal_overlap_relaxed"] += 1
        try:  # the transport must also WORK
            ok = _rendezvous_works(group, device)
        except Exception as e:  # noqa: BLE001 -- refused below, the group runs NCCL
            ok, err = False, repr(e)
    t = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    agreed = bool(t.item())
    msg = (f"[ISOEXEC-PIK] P2P peers for group {_group_key(group)}: {'ADMITTED' if agreed else 'REFUSED'} "
           f"by physical identity ({why}{'; probe failed: ' + err if err else ''}"
           f"{'; a peer refused' if ok and not agreed else ''}). Refused groups move bytes over NCCL "
           "(same tree, same bits).")
    print(msg, flush=True)
    (logger.info if agreed else logger.warning)(msg)
    return agreed


def physical_p2p(group) -> bool:
    """Group-agreed: may this group use symmetric memory? First call per group is a collective."""
    if not dist.is_initialized() or dist.get_world_size(group) < 2 or not torch.cuda.is_available():
        return False
    key = _group_key(group)
    got = _VERDICTS.get(key)
    if got is not None:
        return got
    if _capturing():
        return False  # decided only eagerly; NCCL bytes meanwhile (same bits)
    _COUNTS["probes"] += 1
    try:
        got = _probe(group, torch.device("cuda", torch.cuda.current_device()))
    except Exception as e:  # noqa: BLE001 -- a probe that cannot even gather leaves NCCL in place
        logger.warning("[ISOEXEC-PIK] P2P peer probe raised %r; NCCL transport", e)
        got = False
    _VERDICTS[key] = got
    _COUNTS["admitted" if got else "refused"] += 1
    return got


# ------------------------------------------------------------------------------------------- comm lane


#: The process default when ``ISOEXEC_PIK_OVERLAP`` is unset: off on both sides (opt in with
#: ``ISOEXEC_PIK_OVERLAP=1``; it engages only with more than one hardware queue, see ``overlap_agreed``).
_OVERLAP_DEFAULT = False


def set_overlap_default(on: bool) -> None:
    global _OVERLAP_DEFAULT
    _OVERLAP_DEFAULT = bool(on)


def overlap_enabled() -> bool:
    v = os.environ.get(ENV_OVERLAP, "").strip().lower()
    if not v:
        return _OVERLAP_DEFAULT
    return v not in ("0", "false", "no", "off")


def overlap_min_m() -> int:
    """Explicit row floor for pipelining (``ISOEXEC_PIK_OVERLAP_MIN_M``); 0 = derived (``plan_chunks``)."""
    return max(0, _env_int(ENV_OVERLAP_MIN_M, 0))


def overlap_chunks() -> int:
    """``ISOEXEC_PIK_OVERLAP_CHUNKS``: 0 (default) = forward sites unpipelined; -1 = depth derived per site
    (``plan_chunks``); n >= 2 = n chunks. Default off: on 4xH100 at TP4 a pipelined trainer forward measured
    ~12% slower than unpipelined with 8 hardware queues, and with ``CUDA_DEVICE_MAX_CONNECTIONS=1`` (one
    hardware queue, the Megatron/verl setting) the streams cannot overlap at all."""
    return max(-1, _env_int(ENV_OVERLAP_CHUNKS, 0))


# ---- the derived pipeline depth. Schedule only: any chunking gives the same bits (pik_overlap_gpu --sim).
#: Sustained dense bf16 GEMM rate of the segment-tree leaf GEMM per SM (H100: 600-775 TF/s on 132 SMs).
GEMM_FLOPS_PER_SM = 5.0e12
#: Effective per-rank bandwidth of the owner reduce: NVLink symmetric memory / intra-node NCCL measured
#: 2.25 S bytes in 1.02-1.05 ms at S=118 MB (TP4, H100) = 260 GB/s; inter-node NCCL (400 Gb/s NIC) ~45 GB/s.
LINK_BYTES_PER_S = {"p2p": 260e9, "nccl": 260e9, "nccl_internode": 45e9}
#: Fixed cost each extra chunk adds (one more reduce launch + its two barriers + a GEMM tail), seconds.
CHUNK_OVERHEAD_S = 15e-6
MAX_CHUNKS = 8
_TILE_M, _TILE_N = 128, 256


def hw_queues() -> int:
    """Hardware work queues per device this process may use (``CUDA_DEVICE_MAX_CONNECTIONS``, CUDA default 8).

    At 1 (Megatron/verl set it), every stream feeds one queue and kernels of different streams never run
    concurrently: an nsys trace of the pipelined trainer (4xH100, recurrent GDN) showed 0 of 43,746 kernels
    starting before another stream's kernel ended. Pipelining then only adds launches, so the derived
    depth is 1 there; the early dgrad launch stays (it adds no kernel) and overlaps once queues > 1.
    """
    try:
        return max(1, int(os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "8")))
    except ValueError:
        return 8


def plan_chunks(M: int, N: int, k_local: int, world: int, planes: int, wire_elt: int, root_elt: int,
                link: str, sm_count: int) -> int:
    """Chunks for a pipelined row-parallel site from what sets its comm/compute ratio. Pure; rank-uniform.

    GEMM time g = 2 M N k_local / (sm_count * GEMM_FLOPS_PER_SM); reduce time c = owner bytes per rank /
    link bandwidth, bytes = (W-1)/W * M N (planes * wire_elt + root_elt) (the leaf/node wire of
    ``owner.wire`` read from peers plus the pushed roots). With n chunks the site costs about
    max(g, c) + min(g, c)/n + n * CHUNK_OVERHEAD_S, minimised at n = sqrt(min(g, c) / overhead). n is
    capped so each chunk's GEMM still fills one wave of 128x256 tiles on every SM (no partial-wave
    inefficiency) and by MAX_CHUNKS. An explicit ``ISOEXEC_PIK_OVERLAP_CHUNKS`` overrides it (``site_chunks``);
    ``ISOEXEC_PIK_OVERLAP_MIN_M`` is the row floor ``pipeline.pipeline_bounds`` applies after it.

    On Qwen3.8-27B (N=5120, fc2 K=17408) at M=5.8k this gives 5 at TP2 (fp32 node wire), 5 at TP4 (2 bf16
    leaves), 4 at TP8 (1 leaf); inter-node NCCL (c >> g) gives the GEMM-bound sqrt; small M gives 1.
    """
    if world < 2 or M <= 0:
        return 1
    g = 2.0 * M * N * k_local / (max(1, sm_count) * GEMM_FLOPS_PER_SM)
    c = (world - 1) / world * M * N * (planes * wire_elt + root_elt) / LINK_BYTES_PER_S.get(link, 45e9)
    n = int(round((min(g, c) / CHUNK_OVERHEAD_S) ** 0.5))
    rows_per_wave = _TILE_M * -(-max(1, sm_count) // max(1, -(-N // _TILE_N)))
    n = min(n, M // max(1, rows_per_wave), MAX_CHUNKS)
    return max(1, n)


ENV_P2P_MAX_MB = "ISOEXEC_PIK_TRAINER_P2P_MAX_MB"


def _local_p2p_max_bytes() -> int:
    return max(0, _env_int(ENV_P2P_MAX_MB, 32)) << 20


def p2p_max_bytes(group) -> int:
    """Per-rank wire bytes above which an admitted P2P group still moves a reduce over NCCL.

    Group-agreed: the physical probe gathers every rank's ``ISOEXEC_PIK_TRAINER_P2P_MAX_MB`` and the group
    uses the minimum (a rank with 0 = no limit counts as unbounded), so every rank picks the same transport
    for the same size. Only under the physical (trainer) policy; 0 = no limit. Measured on 4xH100
    (row-parallel owner reduce, N=5120, 2 bf16 leaf planes): P2P 0.119 vs NCCL 0.371 ms at 1024 tokens
    (21 MB/rank), but 1.019 vs 0.917 at 10240 and 2.011 vs 1.728 at 20480; P2P loses from ~4096 tokens
    (84 MB). Default 32 MiB sits between. Same bits either way (both classes are admitted against the
    legacy path).
    """
    if _POLICY != "physical":
        return 0
    return _CAPS.get(_group_key(group), 0)


def overlap_comm_sms() -> int:
    """SMs left free for the lane while a producer GEMM chunk overlaps a reduce (0 = no reservation)."""
    return max(0, _env_int(ENV_OVERLAP_SMS, 0))


_AGREED: dict = {}


def _sm_count(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def overlap_agreed(group, device=None) -> bool:
    """Group-agreed overlap switch. Pipelining changes how many collectives a site issues, so every rank of
    the group must take the same schedule: the first call per group (eager; a collective) gathers
    ``(on, chunks, min_m, SM count, hardware queues)`` from every rank and turns overlap on only if all are on
    and identical and there is more than one hardware queue. A split configuration is logged and runs
    unpipelined everywhere (same bits), never hangs."""
    if not dist.is_initialized() or dist.get_world_size(group) < 2:
        return False
    key = _group_key(group)
    got = _AGREED.get(key)
    if got is not None:
        return got
    if _capturing():
        return False
    dev = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    mine = torch.tensor([int(overlap_enabled()), overlap_chunks(), overlap_min_m(), _sm_count(dev), hw_queues()],
                        dtype=torch.int64, device=dev)
    world = dist.get_world_size(group)
    allv = torch.empty((world, mine.numel()), dtype=torch.int64, device=dev)
    dist.all_gather_into_tensor(allv, mine, group=group)
    rows = allv.cpu().tolist()
    same = all(r == rows[0] for r in rows)
    # With one hardware queue (CUDA_DEVICE_MAX_CONNECTIONS=1, the Megatron/verl setting) streams never run
    # concurrently: the lane would only add stream hops, so overlap stays off.
    got = same and bool(rows[0][0]) and rows[0][4] > 1
    if not same:
        msg = (f"[ISOEXEC-PIK] comm/compute overlap DISABLED for group {key}: ranks disagree on "
               f"({ENV_OVERLAP}, {ENV_OVERLAP_CHUNKS}, {ENV_OVERLAP_MIN_M}, SM count, CUDA_DEVICE_MAX_CONNECTIONS) "
               f"= {rows}; set them identically.")
        print(msg, flush=True)
        logger.warning(msg)
    elif rows[0][0] and not got:
        print(f"[ISOEXEC-PIK] comm/compute overlap requested but OFF for group {key}: one hardware queue "
              "(CUDA_DEVICE_MAX_CONNECTIONS=1), streams cannot overlap.", flush=True)
    elif got:
        print(f"[ISOEXEC-PIK] comm/compute overlap ON for group {key} (chunks={rows[0][1] or 'derived'}, "
              f"min_m={rows[0][2] or 'derived'}): pipelined row-parallel sites and async dgrad tree sums on a "
              "comm lane.",
              flush=True)
    _AGREED[key] = got
    return got


class Lane:
    """A side stream for one group's collectives plus the event of the last thing submitted to it."""

    def __init__(self, device):
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.tail: torch.cuda.Event | None = None

    def enter(self) -> torch.cuda.Stream:
        """Order the lane after everything already enqueued on the caller's stream; return the lane."""
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        return self.stream

    def leave(self) -> torch.cuda.Event:
        ev = torch.cuda.Event()
        ev.record(self.stream)
        self.tail = ev
        return ev


_LANES: dict = {}


def lane(group, device) -> Lane:
    key = (_group_key(group), str(device))
    ln = _LANES.get(key)
    if ln is None:
        ln = Lane(device)
        _LANES[key] = ln
    return ln


def fence(device=None) -> None:
    """Make the caller's stream wait for every lane's tail on ``device`` (cheap: event waits only)."""
    if not _LANES:
        return
    cur = torch.cuda.current_stream(device)
    for ln in _LANES.values():
        if ln.tail is not None and ln.stream != cur and (device is None or ln.device == cur.device):
            cur.wait_event(ln.tail)


class Pending:
    """A collective running on a lane: ``result`` is valid on the caller's stream after ``wait()``."""

    __slots__ = ("result", "event", "_done")

    def __init__(self, result: torch.Tensor, event: torch.cuda.Event):
        self.result = result
        self.event = event
        self._done = False

    def wait(self) -> torch.Tensor:
        if not self._done:
            cur = torch.cuda.current_stream(self.result.device)
            cur.wait_event(self.event)
            self.result.record_stream(cur)
            self._done = True
        return self.result


def submit(group, fn, *tensors: torch.Tensor) -> Pending:
    """Run ``fn(*tensors)`` on the group's lane after the caller's pending work; return a :class:`Pending`.

    ``tensors`` are produced on the caller's stream; they are kept alive for the lane (``record_stream``).
    """
    dev = tensors[0].device
    ln = lane(group, dev)
    s = ln.enter()
    with torch.cuda.stream(s):
        res = fn(*tensors)
    for t in tensors:
        t.record_stream(s)
    return Pending(res, ln.leave())


__all__ = [
    "Lane",
    "Pending",
    "classify",
    "device_uuid",
    "fence",
    "lane",
    "overlap_agreed",
    "overlap_chunks",
    "overlap_comm_sms",
    "overlap_enabled",
    "overlap_min_m",
    "host_id",
    "hw_queues",
    "is_mig",
    "p2p_max_bytes",
    "plan_chunks",
    "peer_counts",
    "peer_policy",
    "physical_p2p",
    "set_overlap_default",
    "set_peer_policy",
    "submit",
]
