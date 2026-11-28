"""Per-component cold-start timers of the engine and the trainer, emitted once per process.

Wall-clock seconds (``time.monotonic``) of each named start-up component, accumulated per process:

``process.*``   process start (kernel start time of this pid) to a milestone
``engine.*``    vLLM worker lifecycle: ``init_device`` (device + process groups / NCCL), ``load_model``,
                ``profile`` (``determine_available_memory``), ``kv_cache``, ``warmup`` (warm-ups + graph capture),
                ``graph_capture`` and ``graph_capture.<MODE>.<tokens>`` per bucket, ``first_step``
``trainer.*``   ``install_kernels``, ``build_meta``, ``capture``, ``define_stage``, ``model_init``,
                ``bind_state``, ``ddp_optimizer``, ``first_forward``, ``first_backward``
``model_init.*`` shared by both sides: ``build_meta``, ``capture``, ``allocate``, ``host_segment`` (node-shared
                host buffers: segment create / map), ``bind``
``weights.*``   checkpoint load shared by both sides: ``iterate`` (time inside the checkpoint iterator),
                ``device_copy`` (conversion + device writes), ``host_copy`` (writes into host-resident buffers,
                the Engram tables), ``host_finish`` (publish + ``cudaHostRegister``), ``finish_engine_load``
``symm.rendezvous``  symmetric-memory setups (count = setups)
``ext.store_hit`` / ``ext.build``  torch-extension loads from the prebuilt store / builds (count = extensions)
``dg_jit.compile``  DeepGEMM JIT compiles committed while this process ran (from the cache entries' file times)
``triton.compile`` / ``triton.hit``  Triton kernel compiles (seconds) and cache hits (count)

Phases nest (``weights.*`` inside ``engine.load_model``; ``ext.*`` and ``dg_jit.*`` cut across); every record keeps
``seconds`` and ``count``. ``emit(side)`` prints ONE line ``ISOEXEC_LOAD_TIMERS <json>`` (stable keys: ``side``,
``host``, ``pid``, ``rank``, ``timers`` {name: {"s", "n"}}) the first time it is called in a process (or at exit if
nothing called it) and, with ``ISOEXEC_LOAD_TIMERS_DIR`` set, writes the same record to
``<dir>/<side>-<host>-<pid>.json``. ``ISOEXEC_LOAD_TIMERS=0`` turns recording off. Cost: two clock reads per
phase and a dict update; no device synchronisation except where a phase says so (the first step / forward).

Mature reference: ``runtimes/vllm/nccl_channels.py`` ``_RECEIPTS`` (a module-level, process-local record that a
run reports), ``models/deepseek_v4_1_flash/providers.py`` ``TRANSPORT_RECEIPTS``.
"""
from __future__ import annotations

import atexit
import json
import os
import socket
import sys
import time
from contextlib import contextmanager

MARKER = "ISOEXEC_LOAD_TIMERS"

_TIMERS: dict[str, list[float]] = {}          # name -> [seconds, count]
_STATE = {"emitted": False, "side": None, "atexit": False}
_T0_MONO = time.monotonic()
_T0_WALL = time.time()


def enabled() -> bool:
    return os.environ.get("ISOEXEC_LOAD_TIMERS", "1") != "0"


def add(name: str, seconds: float, count: int = 1) -> None:
    """Accumulate ``seconds`` (and ``count`` occurrences) under ``name``."""
    if not enabled():
        return
    slot = _TIMERS.setdefault(name, [0.0, 0])
    slot[0] += float(seconds)
    slot[1] += int(count)


@contextmanager
def phase(name: str):
    """Time the body under ``name`` (accumulates if entered again)."""
    if not enabled():
        yield
        return
    start = time.monotonic()
    try:
        yield
    finally:
        add(name, time.monotonic() - start)


class Lap:
    """Consecutive phases without re-indenting a constructor: ``lap = Lap(); ...; lap("a"); ...; lap("b")`` records
    the time since the previous lap (or the construction) under each name."""

    def __init__(self) -> None:
        self.last = time.monotonic()

    def __call__(self, name: str) -> None:
        now = time.monotonic()
        add(name, now - self.last)
        self.last = now


def timed(name: str, *, before=None, after=None):
    """Decorator: time every call under ``name``; ``before()`` / ``after()`` run around it (milestones)."""
    import functools

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if before is not None:
                before()
            with phase(name):
                result = fn(*args, **kwargs)
            if after is not None:
                after()
            return result

        return wrapper

    return decorate


class _FirstCall:
    def __init__(self, fn, name: str, emit_side: str | None):
        self.fn, self.name, self.emit_side, self.done = fn, name, emit_side, False

    def __call__(self, *args, **kwargs):
        if self.done:
            return self.fn(*args, **kwargs)
        self.done = True
        start = time.monotonic()
        result = self.fn(*args, **kwargs)
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
        since = _STATE.get("first_forward_end")
        add(self.name, time.monotonic() - (since if since is not None else start))
        mark(self.name.split(".")[-1])
        if self.emit_side:
            emit(self.emit_side)
        return result


def first_call(fn, name: str, *, emit_side: str | None = None):
    """``fn`` whose FIRST call records ``name``: seconds from the ``process.first_forward`` milestone (else from the
    call) to the end of that call, closed by one device synchronisation; then ``emit(emit_side)``. Later calls
    pass straight through. Used on the trainer's gradient finaliser, which ends the first optimizer step's
    backward. A ``functools.partial`` stays a partial with the same arguments (callers read ``.keywords``)."""
    import functools

    if isinstance(fn, functools.partial):
        return functools.partial(_FirstCall(fn.func, name, emit_side), *fn.args, **fn.keywords)
    return functools.wraps(fn)(_FirstCall(fn, name, emit_side))


def process_start_wall() -> float:
    """Wall-clock time this process was started (``/proc/self/stat`` start ticks + boot time)."""
    try:
        with open("/proc/self/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        ticks = int(fields[19])                      # field 22 overall: starttime in clock ticks since boot
        with open("/proc/stat") as f:
            boot = next(int(line.split()[1]) for line in f if line.startswith("btime "))
        return boot + ticks / os.sysconf("SC_CLK_TCK")
    except Exception:  # noqa: BLE001 -- non-Linux: fall back to this module's import time
        return _T0_WALL


def mark(name: str) -> None:
    """Record ``process.<name>``: seconds from process start to now (set once)."""
    if not enabled() or f"process.{name}" in _TIMERS:
        return
    _TIMERS[f"process.{name}"] = [max(0.0, time.time() - process_start_wall()), 1]
    _STATE[f"{name}_end"] = time.monotonic()


def snapshot() -> dict[str, dict[str, float]]:
    return {name: {"s": round(s, 4), "n": n} for name, (s, n) in sorted(_TIMERS.items())}


def _rank() -> int | None:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:  # noqa: BLE001
        pass
    return None


def _dg_jit_compiles() -> None:
    """Fold DeepGEMM JIT compiles committed since this process started into ``dg_jit.compile``: per new cache
    entry, the time between its source and its cubin being written (``deep_jit`` backend: write kernel.cu, run
    nvcc, cubin). Entries another process on the same cache compiled meanwhile are included."""
    root = os.environ.get("ISOEXEC_DG_JIT_CACHE_DIR", "").split(":")[0]
    if not root or "dg_jit.compile" in _TIMERS:
        return
    start = process_start_wall()
    seconds, count = 0.0, 0
    try:
        with os.scandir(os.path.join(root, "cache")) as entries:
            for entry in entries:
                try:             # (not ``.committed``: a cache hit refreshes its mtime)
                    cubin = os.stat(os.path.join(entry.path, "kernel.cubin")).st_mtime
                    if cubin < start:
                        continue
                    src = os.stat(os.path.join(entry.path, "kernel.cu")).st_mtime
                except OSError:
                    continue
                seconds += max(0.0, cubin - src)
                count += 1
    except OSError:
        return
    _TIMERS["dg_jit.compile"] = [seconds, count]


def install_triton_listener() -> bool:
    """Count Triton compiles (seconds) and cache hits through ``triton.knobs.compilation.listener`` (chained)."""
    if not enabled() or "triton" not in sys.modules or _STATE.get("triton"):
        return False
    try:
        from triton import knobs
    except Exception:  # noqa: BLE001
        return False
    previous = knobs.compilation.listener

    def listener(*, src, metadata, metadata_group, times, cache_hit):
        if cache_hit:
            add("triton.hit", 0.0)
        else:
            add("triton.compile", float(getattr(times, "total", 0) or 0) / 1e6)   # CompileTimes: microseconds
        if previous is not None:
            previous(src=src, metadata=metadata, metadata_group=metadata_group, times=times, cache_hit=cache_hit)

    knobs.compilation.listener = listener
    _STATE["triton"] = True
    return True


def emit(side: str, **extra) -> dict | None:
    """Print the process's record once (``ISOEXEC_LOAD_TIMERS <json>``); later calls return None."""
    if not enabled() or _STATE["emitted"]:
        return None
    _STATE["emitted"] = True
    _dg_jit_compiles()
    mark("emit")
    record = {"side": side, "host": socket.gethostname(), "pid": os.getpid(), "rank": _rank(),
              "timers": snapshot(), **extra}
    line = f"{MARKER} {json.dumps(record, sort_keys=True)}"
    print(line, file=sys.stderr, flush=True)
    directory = os.environ.get("ISOEXEC_LOAD_TIMERS_DIR")
    if directory:
        try:
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, f"{side}-{record['host']}-{record['pid']}.json")
            with open(path + ".tmp", "w") as f:
                json.dump(record, f, indent=1, sort_keys=True)
            os.replace(path + ".tmp", path)
        except OSError as exc:
            print(f"{MARKER} write to {directory} failed: {exc}", file=sys.stderr, flush=True)
    return record


def emit_at_exit(side: str) -> None:
    """Emit at interpreter exit unless a milestone emitted first (a process that never reached it)."""
    _STATE["side"] = side
    if _STATE["atexit"] or not enabled():
        return
    _STATE["atexit"] = True
    atexit.register(lambda: emit(_STATE["side"] or side, at_exit=True))


__all__ = ["Lap", "MARKER", "add", "emit", "emit_at_exit", "enabled", "first_call", "install_triton_listener",
           "mark", "phase", "process_start_wall", "snapshot", "timed"]
