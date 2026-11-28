"""vLLM monkey-patches for bitwise-matched rollouts."""

from __future__ import annotations

import importlib.abc
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

ISOEXEC_VLLM_ENV = {
    "VLLM_BATCH_INVARIANT": "1",
    "VLLM_USE_AOT_COMPILE": "0",
    # The IsoExec engine runs on vLLM Model Runner V2: the sampler hooks, the GDN driver and the
    # breakable PIECEWISE graphs target it.
    "VLLM_USE_V2_MODEL_RUNNER": "1",
}

# Preserve per-communicator channel settings instead of inheriting process-wide overrides.
_VLLM_NCCL_TRANSPORT_OVERRIDES = (
    "NCCL_LAUNCH_MODE",
    "NCCL_COLLNET_ENABLE",
    "NCCL_NVLS_ENABLE",
    "NCCL_P2P_NET_DISABLE",
    "NCCL_PROTO",
    "NCCL_NTHREADS",
    "NCCL_SOCKET_NTHREADS",
)


def _without_nccl_overrides(original):
    from functools import wraps

    from .nccl_channels import clear_process_pins

    @wraps(original)
    def wrapped(*args, **kwargs):
        inherited = {name: os.environ.get(name) for name in _VLLM_NCCL_TRANSPORT_OVERRIDES}
        try:
            return original(*args, **kwargs)
        finally:
            for name, value in inherited.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            clear_process_pins()

    wrapped._isoexec_orig = original
    return wrapped


def apply_vllm_isoexec_env(env: dict | None = None) -> dict:
    from .nccl_channels import clear_process_pins

    target = os.environ if env is None else env
    to_set = dict(ISOEXEC_VLLM_ENV)
    clear_process_pins(target)
    for k, v in to_set.items():
        target[k] = v
    logger.info("[isoexec] set vLLM batch-invariant env: %s", to_set)
    return dict(to_set)


def neutralize_vllm_nccl_channel_pin() -> bool:
    """Remove vLLM's process pins and cap its original communicators individually."""
    from ...checks import Refusal
    from ...ops.collectives.nccl_identity import (
        engine_channel_cap,
        engine_channel_pin_neutralized,
    )
    from .nccl_channels import install_engine_channel_policy

    if not engine_channel_pin_neutralized():
        return False
    try:
        import vllm.model_executor.determinism.batch_invariant as _bi
    except ImportError as error:
        raise Refusal(
            "vllm.model_executor.determinism.batch_invariant: required NCCL environment boundary is unavailable"
        ) from error
    override = getattr(_bi, "override_envs_for_invariance", None)
    if not callable(override):
        raise Refusal(
            "vllm.model_executor.determinism.batch_invariant.override_envs_for_invariance: "
            "required NCCL environment boundary is not callable"
        )
    cap = engine_channel_cap()
    if cap not in ("8", "16"):
        raise Refusal(f"ISOEXEC_ENGINE_NCCL_MAX_NCHANNELS: certified ceilings are 8 and 16, got {cap!r}")
    install_engine_channel_policy(int(cap))
    if getattr(_bi, "_isoexec_channel_pin_neutralized", False):
        return True
    _bi.override_envs_for_invariance = _without_nccl_overrides(override)
    _bi._isoexec_channel_pin_neutralized = True
    logger.info("[isoexec] original engine NCCL communicators configured with bounds 1/%s", cap)
    return True


#: Model Runner V2 sampler routing counters, kept apart from the numerical-kernel engagement counters.
_V2_STATE = {"calls": 0, "full_rows": 0, "rowinv_owned": 0, "incumbent": 0, "partial": 0, "reported": 0}


def _v2_report() -> None:
    n = _V2_STATE["calls"]
    if n < 1 or (n & (n - 1)) != 0 or n == _V2_STATE["reported"]:
        return
    _V2_STATE["reported"] = n
    print(
        f"[ISOEXEC-SAMPLER-LOGPROBS-V2] pid={os.getpid()} vLLM Model Runner V2 compute_token_logprobs hook: "
        f"calls={n} full_rows={_V2_STATE['full_rows']} rowinv_owned={_V2_STATE['rowinv_owned']} "
        f"incumbent={_V2_STATE['incumbent']} partial={_V2_STATE['partial']}",
        flush=True,
    )


def _make_vllm_v2_token_logprobs(original, rowinv_full_logprobs, prepare_logits, temperature_tensor):
    """Build the Model Runner V2 ``compute_token_logprobs`` wrapper from injected, vLLM-free callables.

    V2 gathers every requested column (``logprobs=0``: the sampled token alone; ``logprobs=-1``:
    ``1 + vocab``) from one log-softmax per row. Every gather is routed through IsoExec's full-row
    producer, which the trainer's scorer also uses, so the sampled-token logprob vLLM returns is
    the trainer's value bit for bit; vLLM's own kernel reduces the row in a different order.
    """

    from functools import wraps

    @wraps(original, updated=())
    def wrapped(logits, token_ids):
        _V2_STATE["calls"] += 1
        if logits.ndim != 2 or token_ids.ndim != 2 or token_ids.shape[0] != logits.shape[0]:
            _V2_STATE["partial"] += 1
            _v2_report()
            return original(logits, token_ids)
        rows = rowinv_full_logprobs(
            logits,
            src_dtype=logits.dtype,
            reference=lambda: None,
            temperature=temperature_tensor(),
        )
        if rows is None:
            _V2_STATE["incumbent"] += 1
            _v2_report()
            return original(prepare_logits(logits), token_ids)
        if str(rows.dtype) != "torch.float32" or tuple(rows.shape) != tuple(logits.shape):
            raise RuntimeError(
                f"[isoexec] V2 full-row producer returned {tuple(rows.shape)} {rows.dtype} "
                f"for logits {tuple(logits.shape)}"
            )
        _V2_STATE["rowinv_owned"] += 1
        _V2_STATE["full_rows"] += int(logits.shape[0])
        _v2_report()
        return rows.gather(-1, token_ids.long())

    setattr(wrapped, "_isoexec_v2_full_row_hook", True)
    setattr(wrapped, "_isoexec_orig", original)
    return wrapped


def _install_vllm_v2_token_logprobs(logprob_module, rowinv_full_logprobs, prepare_logits, temperature_tensor) -> bool:
    """Replace the module global that ``compute_topk_scores`` resolves at call time, once."""
    raw = getattr(logprob_module, "compute_token_logprobs", None)
    if raw is None:
        return False
    if getattr(raw, "_isoexec_v2_full_row_hook", False):
        return True
    logprob_module.compute_token_logprobs = _make_vllm_v2_token_logprobs(
        raw, rowinv_full_logprobs, prepare_logits, temperature_tensor,
    )
    return True


def patch_vllm_v2_sampler_logprobs_rowinv() -> bool:
    """Install the fixed-recipe full-row producer on vLLM Model Runner V2's logprob gather."""
    try:
        from vllm.v1.worker.gpu.sample import logprob as v2_logprob
    except Exception as e:
        logger.warning("[isoexec] cannot hook vLLM Model Runner V2 compute_token_logprobs: %s", e)
        return False
    try:
        from isoexec.ops.logprobs import integration as _rowinv_module
    except ImportError as e:
        print(f"[ISOEXEC-ROWINV-LOGPROB] V2 hook REFUSED: logprobs provider import failed ({e!r})", flush=True)
        return False
    from .scoring_temperature import prepare_logits, temperature_tensor

    def _rowinv_full_logprobs(*args, **kwargs):
        return _rowinv_module.full_logprobs(*args, **kwargs)

    if not _install_vllm_v2_token_logprobs(v2_logprob, _rowinv_full_logprobs, prepare_logits, temperature_tensor):
        logger.warning("[isoexec] vLLM Model Runner V2 has no compute_token_logprobs; hook not installed")
        return False
    print(
        "[ISOEXEC-ROWINV-LOGPROB] vLLM Model Runner V2 compute_token_logprobs HOOKED (complete "
        "1+vocab gathers use the full-row leaf-tree producer). Judge engagement by served>0.",
        flush=True,
    )
    return True


#: Model Runner V2 CUDA graph census: captures and full-graph replays, counted on the host.
_V2_GRAPH_STATE = {"captures": 0, "replays": 0, "reported": 0, "piecewise_runs": 0, "piecewise_reported": 0}


def _v2_graph_report() -> None:
    n = _V2_GRAPH_STATE["replays"]
    if n < 1 or (n & (n - 1)) != 0 or n == _V2_GRAPH_STATE["reported"]:
        return
    _V2_GRAPH_STATE["reported"] = n
    print(f"[ISOEXEC-V2-CUDAGRAPH] pid={os.getpid()} captures={_V2_GRAPH_STATE['captures']} "
          f"replays={_V2_GRAPH_STATE['replays']}", flush=True)


def patch_vllm_v2_cudagraph_census() -> bool:
    """Count Model Runner V2 graph captures, full-graph replays and PIECEWISE runs (evidence only; no behavior change)."""
    try:
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
    except Exception as e:
        logger.warning("[isoexec] cannot count Model Runner V2 CUDA graphs: %s", e)
        return False
    if getattr(ModelCudaGraphManager, "_isoexec_census", False):
        return True
    capture, replay = ModelCudaGraphManager.capture, ModelCudaGraphManager.run_fullgraph

    def counted_capture(self, *args, **kwargs):
        result = capture(self, *args, **kwargs)
        _V2_GRAPH_STATE["captures"] += len(getattr(self, "graphs", {}) or {})
        print(f"[ISOEXEC-V2-CUDAGRAPH] pid={os.getpid()} captured graphs={len(getattr(self, 'graphs', {}) or {})}",
              flush=True)
        return result

    def counted_replay(self, *args, **kwargs):
        _V2_GRAPH_STATE["replays"] += 1
        _v2_graph_report()
        return replay(self, *args, **kwargs)

    run_pw_graph = ModelCudaGraphManager.run_pw_graph

    def counted_pw_graph(self, *args, **kwargs):
        # Breakable PIECEWISE runs (first sight of a descriptor captures, later ones replay).
        _V2_GRAPH_STATE["piecewise_runs"] += 1
        n = _V2_GRAPH_STATE["piecewise_runs"]
        if n & (n - 1) == 0 and n != _V2_GRAPH_STATE["piecewise_reported"]:
            _V2_GRAPH_STATE["piecewise_reported"] = n
            print(f"[ISOEXEC-V2-CUDAGRAPH] pid={os.getpid()} piecewise_runs={n}", flush=True)
        return run_pw_graph(self, *args, **kwargs)

    dispatch = ModelCudaGraphManager.dispatch

    def counted_dispatch(self, *args, **kwargs):
        # Every model step's execution mode (FULL replay, PIECEWISE, or NONE = eager), with the token counts
        # each mode ran at: evidence that mixed steps really run piecewise graphs and decode steps full ones.
        desc = dispatch(self, *args, **kwargs)
        mode = getattr(getattr(desc, "cg_mode", None), "name", str(getattr(desc, "cg_mode", "?")))
        modes = _V2_GRAPH_STATE.setdefault("modes", {})
        sizes = _V2_GRAPH_STATE.setdefault("sizes", {})
        modes[mode] = modes.get(mode, 0) + 1
        sizes.setdefault(mode, set()).add(int(getattr(desc, "num_tokens", -1)))
        total = sum(modes.values())
        if total & (total - 1) == 0:
            shown = {m: f"{c} at {len(sizes[m])} sizes (max {max(sizes[m])})" for m, c in sorted(modes.items())}
            print(f"[ISOEXEC-V2-CUDAGRAPH] pid={os.getpid()} dispatch steps={total} {shown}", flush=True)
        return desc

    setattr(ModelCudaGraphManager, "dispatch", counted_dispatch)
    setattr(ModelCudaGraphManager, "capture", counted_capture)
    setattr(ModelCudaGraphManager, "run_fullgraph", counted_replay)
    setattr(ModelCudaGraphManager, "run_pw_graph", counted_pw_graph)
    setattr(ModelCudaGraphManager, "_isoexec_census", True)
    return True


_V2_REQUEST_STATE = {"new_requests": 0, "reported": 0}


def patch_vllm_v2_request_census() -> bool:
    """Count real requests each Model Runner V2 worker admits, with its data-parallel rank (evidence only)."""
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    except Exception as e:
        logger.warning("[isoexec] cannot count Model Runner V2 requests: %s", e)
        return False
    if getattr(GPUModelRunner, "_isoexec_request_census", False):
        return True
    add_requests = GPUModelRunner.add_requests

    def counted(self, scheduler_output):
        result = add_requests(self, scheduler_output)
        added = len(scheduler_output.scheduled_new_reqs)
        if added:
            _V2_REQUEST_STATE["new_requests"] += added
            n = _V2_REQUEST_STATE["new_requests"]
            if n & (n - 1) == 0 or added > 1:
                parallel = self.vllm_config.parallel_config
                print(f"[ISOEXEC-V2-REQUESTS] pid={os.getpid()} dp_rank={getattr(parallel, 'data_parallel_rank', 0)} "
                      f"new_requests={n}", flush=True)
        return result

    setattr(GPUModelRunner, "add_requests", counted)
    setattr(GPUModelRunner, "_isoexec_request_census", True)
    return True


_FULL_ROW_STATE = {"calls": 0, "rows": 0, "fallback": 0, "reported": 0}


def _full_row_report() -> None:
    n = _FULL_ROW_STATE["rows"] + _FULL_ROW_STATE["fallback"]
    if n < 1 or (n & (n - 1)) != 0 or n == _FULL_ROW_STATE["reported"]:
        return
    _FULL_ROW_STATE["reported"] = n
    print(
        f"[ISOEXEC-FULL-ROWS] pid={os.getpid()} complete-vocabulary rows kept as float32 arrays: "
        f"rows={_FULL_ROW_STATE['rows']} fallback={_FULL_ROW_STATE['fallback']}",
        flush=True,
    )


def _make_full_vocabulary_logprobs_update(original, make_row):
    """Build the output-processor wrapper from injected, vLLM-free callables."""

    from functools import wraps

    @wraps(original, updated=())
    def wrapped(self, logprobs_lists):
        # Only a complete-vocabulary request without detokenization has a compact form. Every other
        # request, including one that asks for decoded tokens or flat containers, keeps vLLM's own.
        if self.num_logprobs != -1 or self.tokenizer is not None or type(self.logprobs) is not list:
            return original(self, logprobs_lists)
        _FULL_ROW_STATE["calls"] += 1
        token_ids_lst, logprobs_lst, ranks_lst, _ = logprobs_lists
        try:
            # Zip over the ranks too, as vLLM does, so both forms take the same number of positions.
            rows = [
                make_row(token_ids_np, logprobs_np)
                for _, logprobs_np, token_ids_np in zip(ranks_lst, logprobs_lst, token_ids_lst)
            ]
        except ValueError as e:
            # Not a sampled-first complete row: vLLM's containers keep whatever was sent, and the
            # packer refuses an incomplete row exactly as it did before this hook existed.
            _FULL_ROW_STATE["fallback"] += 1
            logger.warning("[isoexec] complete-vocabulary row kept in vLLM's own container: %s", e)
            _full_row_report()
            return original(self, logprobs_lists)
        for row in rows:
            # The sampled entry is what vLLM itself accumulates (its first column).
            self.cumulative_logprob += row.sampled_logprob
            self.logprobs.append(row)
        _FULL_ROW_STATE["rows"] += len(rows)
        _full_row_report()

    wrapped._isoexec_full_vocabulary_rows = True
    wrapped._isoexec_orig = original
    return wrapped


def _install_full_vocabulary_logprobs_update(processor_cls, make_row) -> bool:
    """Install the injected wrapper once on the output processor's sample-logprob update."""

    raw = processor_cls.__dict__.get("_update_sample_logprobs")
    if raw is None:
        return False
    if getattr(raw, "_isoexec_full_vocabulary_rows", False):
        return True
    processor_cls._update_sample_logprobs = _make_full_vocabulary_logprobs_update(raw, make_row)
    return True


def patch_vllm_full_vocabulary_logprobs() -> bool:
    """Keep ``logprobs=-1`` rows as float32 arrays in vLLM's output processor.

    The stock processor creates one ``Logprob`` object per vocabulary entry and keeps those
    objects until the response finishes. Compact rows retain the float32 values, and the
    packer writes the same token-ID-ordered payload without creating one object per entry.
    """
    try:
        from vllm.v1.engine.logprobs import LogprobsProcessor
    except Exception as e:
        logger.warning("[isoexec] cannot hook LogprobsProcessor._update_sample_logprobs: %s", e)
        return False
    from isoexec.integrations.full_distribution import full_vocabulary_row

    if not _install_full_vocabulary_logprobs_update(LogprobsProcessor, full_vocabulary_row):
        logger.warning("[isoexec] LogprobsProcessor has no _update_sample_logprobs; hook not installed")
        return False
    return True


def patch_vllm_cumem_sleep() -> bool:
    """Release cached pool blocks before vLLM sleep unmaps their memory."""
    try:
        import inspect

        import torch
        from vllm.device_allocator.cumem import CuMemAllocator
    except Exception as e:
        logger.warning("[isoexec] cannot patch vLLM cumem sleep: %s", e)
        return False
    if getattr(CuMemAllocator.sleep, "_isoexec_cumem_sleep", False):
        return True
    params = list(inspect.signature(CuMemAllocator.sleep).parameters)
    if params != ["self", "offload_tags"] or not hasattr(CuMemAllocator, "_python_free_callback"):
        print(
            f"[ISOEXEC-CUMEM-SLEEP] NOT INSTALLED: CuMemAllocator.sleep{tuple(params)} is not the (self, offload_tags) "
            "surface this patch wraps; stock vLLM sleep is kept",
            flush=True,
        )
        return False
    _orig_sleep = CuMemAllocator.sleep

    def sleep(self, offload_tags=None):
        import gc

        table = getattr(self, "pointer_to_data", None)
        before = len(table) if table is not None else -1
        gc.collect()
        torch.cuda.empty_cache()
        released = before - len(table) if table is not None else 0
        if released > 0:
            # The evidence line of the defect: each of these would have been a double unmap after the sleep.
            print(
                f"[ISOEXEC-CUMEM-SLEEP] pid={os.getpid()} released {released} cached pool allocation(s) BEFORE the "
                f"sleep unmap ({before} -> {len(table)} tracked)",
                flush=True,
            )
        return _orig_sleep(self, offload_tags)

    sleep._isoexec_cumem_sleep = True
    CuMemAllocator.sleep = sleep
    return True


def patch_vllm_kernel_warmup() -> bool:
    """Skip vLLM's CuTe-DSL MoE-router GEMM warmup for an IsoExec model; other models keep it.

    An IsoExec model routes with its own kernels and builds no vLLM ``GateLinear``, so this warmup
    has nothing to compile. On SM100 vLLM imports the warmup's CuTe-DSL module, and through it
    ``quack``, before it looks for a gate; ``quack`` needs a newer CuTe DSL than the pinned one, so
    the import alone fails engine start. Every other warmup checks for its own layers first.
    """
    try:
        from vllm.model_executor.warmup import kernel_warmup
    except Exception as e:
        logger.warning("[isoexec] cannot patch vLLM kernel warmup: %s", e)
        return False
    original = getattr(kernel_warmup, "_warmup_bf16x3_router_gemm", None)
    if original is None:
        return False
    if getattr(original, "_isoexec_skips_native", False):
        return True

    def _warmup_bf16x3_router_gemm(model, *args, **kwargs):
        from isoexec.runtimes.vllm.model import VllmModel

        if isinstance(model, VllmModel):
            return None
        return original(model, *args, **kwargs)

    _warmup_bf16x3_router_gemm._isoexec_skips_native = True  # type: ignore[attr-defined]
    kernel_warmup._warmup_bf16x3_router_gemm = _warmup_bf16x3_router_gemm
    return True


#: vLLM GPU worker lifecycle methods timed as ``engine.<label>`` (``core/load_timers``), in call order.
_WORKER_PHASES = (("init_device", "init_device"), ("load_model", "load_model"),
                  ("determine_available_memory", "profile"), ("initialize_from_config", "kv_cache"),
                  ("compile_or_warm_up_model", "warmup"))


_GPU_WORKER = "vllm.v1.worker.gpu_worker"


class _AfterImport:
    """A one-shot meta-path finder: run ``hook`` right after ``name`` is first imported (a process that never
    imports the GPU worker, such as the API server, pays nothing)."""

    def __init__(self, name, hook):
        self.name, self.hook = name, hook

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.name:
            return None
        import importlib.util

        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return spec
        loader, hook = spec.loader, self.hook

        class _Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return loader.create_module(spec)

            def exec_module(self, module):
                loader.exec_module(module)
                hook()

        spec.loader = _Loader()
        return spec


def patch_vllm_load_timers() -> bool:
    """Cold-start timers in every vLLM worker: one ``engine.<phase>`` record per lifecycle method, the graph capture
    per bucket (``engine.graph_capture.<MODE>.<tokens>``), and the first real step (``engine.first_step``: the first
    ``execute_model`` after warm-up through its ``sample_tokens``, closed by ONE device synchronisation), after which
    the worker emits its record once (``ISOEXEC_LOAD_TIMERS``). Host clock reads only, otherwise. Installed when the
    GPU worker module is imported (now, if it already is)."""
    from isoexec.core import load_timers

    if not load_timers.enabled():
        return False
    if _GPU_WORKER in sys.modules:
        return _install_load_timers()
    if not any(isinstance(f, _AfterImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _AfterImport(_GPU_WORKER, _install_load_timers))
    return True


def _install_load_timers() -> bool:
    try:
        from vllm.v1.worker import gpu_worker
        from vllm.v1.worker.gpu import cudagraph_utils
    except Exception as e:  # pragma: no cover - vLLM without the V1 GPU worker / Model Runner V2
        logger.warning("[isoexec] cannot install load timers: %s", e)
        return False
    from isoexec.core import load_timers

    worker = gpu_worker.Worker
    if getattr(worker, "_isoexec_load_timers", False):
        return True
    state = {"warm": False, "first": None, "done": False}

    def timed(name, label):
        original = getattr(worker, name)

        def method(self, *args, **kwargs):
            if label == "init_device":
                load_timers.mark("init_device")
                load_timers.emit_at_exit("engine")
            with load_timers.phase(f"engine.{label}"):
                result = original(self, *args, **kwargs)
            if label == "load_model":
                load_timers.install_triton_listener()
            if label == "warmup":
                state["warm"] = True
                load_timers.mark("ready")
            return result

        method.__wrapped__ = original
        setattr(worker, name, method)

    for name, label in _WORKER_PHASES:
        if hasattr(worker, name):
            timed(name, label)

    def finish_first_step():
        import torch

        torch.cuda.synchronize()
        load_timers.add("engine.first_step", time.monotonic() - state["first"])
        load_timers.mark("first_step")
        state["done"] = True
        load_timers.emit("engine")

    execute = worker.execute_model

    def execute_model(self, *args, **kwargs):
        if state["warm"] and state["first"] is None:
            state["first"] = time.monotonic()
        result = execute(self, *args, **kwargs)
        if state["first"] is not None and not state["done"] and result is not None:
            finish_first_step()        # the step returned its output here (no separate sample_tokens)
        return result

    execute_model.__wrapped__ = execute
    worker.execute_model = execute_model
    if hasattr(worker, "sample_tokens"):
        sample = worker.sample_tokens

        def sample_tokens(self, *args, **kwargs):
            result = sample(self, *args, **kwargs)
            if state["first"] is not None and not state["done"]:
                finish_first_step()
            return result

        sample_tokens.__wrapped__ = sample
        worker.sample_tokens = sample_tokens

    manager = cudagraph_utils.CudaGraphManager
    capture = manager.capture

    def capture_timed(self, create_forward_fn, *args, **kwargs):
        marks = []

        def create(desc, warmup):
            if warmup:
                marks.append((f"{desc.cg_mode.name}.{desc.num_tokens}", time.monotonic()))
            return create_forward_fn(desc, warmup=warmup)

        with load_timers.phase("engine.graph_capture"):
            result = capture(self, create, *args, **kwargs)
        end = time.monotonic()
        for (label, start), nxt in zip(marks, [m[1] for m in marks[1:]] + [end]):
            load_timers.add(f"engine.graph_capture.{label}", nxt - start)
        return result

    capture_timed.__wrapped__ = capture
    manager.capture = capture_timed
    worker._isoexec_load_timers = True
    return True


_FUSED_SAMPLER_ENV = "ISOEXEC_FUSED_SAMPLER"


def fused_sampler_enabled() -> bool:
    """Default on; ``0`` keeps vLLM's own sampler for every step (declared in ``core/flags``)."""
    return os.environ.get(_FUSED_SAMPLER_ENV, "1").strip().lower() not in ("", "0", "false", "no")


def _fused_step_refusal(*, spec_decode, logprobs_mode, top_k_top_p, per_request_seed, logprob_token_ids,
                        num_logprobs):
    """The first condition that keeps a step on vLLM's sampler, or ``None``.

    The decode sampler draws neither a truncated distribution nor a per-request RNG stream, scores
    the raw row only, and returns the sampled token plus at most the top-1.
    """
    if spec_decode:
        return "spec_decode"
    if logprobs_mode != "raw_logprobs":
        return "logprobs_mode"
    if top_k_top_p:
        return "top_k/top_p"
    if per_request_seed:
        return "per_request_seed"
    if logprob_token_ids:
        return "logprob_token_ids"
    if num_logprobs not in (None, 0, 1):
        return f"max_num_logprobs={num_logprobs}"
    return None


#: Decode sampler census: steps served, steps left to vLLM, the last refusal.
_FUSED_STATE = {"v2": {"served": 0, "fallback": 0, "last_fallback": None, "reported": 0}}


def _fused_census(runner: str, refusal: str | None) -> None:
    st = _FUSED_STATE[runner]
    if refusal is None:
        st["served"] += 1
    else:
        st["fallback"] += 1
        st["last_fallback"] = refusal
    n = st["served"] + st["fallback"]
    if n & (n - 1) != 0 or n == st["reported"]:
        return
    st["reported"] = n
    print(f"[ISOEXEC-FUSED-SAMPLER] pid={os.getpid()} runner={runner} served={st['served']} "
          f"fallback={st['fallback']} last_fallback={st['last_fallback']}", flush=True)


def _decode_sampler_provider(resolve_sampler):
    """``resolve_sampler()`` once, on the first step (the device is known by then), then its answer."""
    provider = []

    def get():
        if not provider:
            provider.append(resolve_sampler())
            logger.info("[isoexec] decode sampler %s", "engaged" if provider[0] is not None else "not available")
        return provider[0]

    return get


def _fused_draw(fs, x, temperature, all_greedy):
    """``(temperature, seed, offset)`` of one step: an all-greedy step draws no noise and keeps the RNG."""
    if all_greedy:
        return None, 0, 0
    seed, offset = fs.next_philox(x.device)
    return temperature, seed, offset


def _resolve_decode_sampler():
    """The selected logprobs package's decode sampler, or ``None`` on an arch without one."""
    from isoexec.ops import logprobs
    from isoexec.ops._dispatch import ArchSelectionError

    try:
        return logprobs.sampler
    except ArchSelectionError:
        return None


#: The Model Runner V2 Sampler methods that draw the token; a subclass overriding any of them
#: (the watermarking sampler) samples its own way and keeps its own ``__call__``.
_V2_SAMPLING_METHODS = ("apply_sampling_params", "sample", "_sample_random")


def _v2_sampler_override(cls, base) -> str | None:
    """The name of a ``base`` subclass that overrides how tokens are drawn, else ``None``."""
    if cls is base:
        return None
    for name in _V2_SAMPLING_METHODS:
        if getattr(cls, name, None) is not getattr(base, name):
            return cls.__name__
    return None


def _v2_processed_flags(sampler):
    """Per request slot: whether a logits processor rewrites the row (see ``_make_vllm_v2_add_request``)."""
    flags = sampler.__dict__.get("_isoexec_processed")
    if flags is None:
        # Slots admitted before the hook carry vLLM's wider flag: never under-reports a processor.
        flags = sampler.needs_logits_processing.copy()
        sampler._isoexec_processed = flags
    return flags


def _v2_min_only_lens(sampler):
    """Per request slot: the length below which only a ``min_tokens`` mask rewrites the row, else -1.

    -1 also covers every slot whose processing is something else (or more than the mask), and slots
    admitted before the hook.
    """
    import numpy as np

    lens = sampler.__dict__.get("_isoexec_min_only_len")
    if lens is None:
        lens = np.full(len(sampler.needs_logits_processing), -1, dtype=np.int64)
        sampler._isoexec_min_only_len = lens
    return lens


def _make_vllm_v2_add_request(original):
    """Wrap the V2 ``Sampler.add_request`` to keep the logits-processor part of its per-slot flag apart.

    V2's ``needs_logits_processing`` ORs the sampling states' answer (temperature, min_p, top-k,
    top-p) with every logits processor's (logit bias, allowed tokens, min_tokens, penalties, bad
    words, thinking budget, custom processors). The decode sampler applies temperature itself and
    refuses min_p and top-k/top-p on their own, so only the processors' part decides whether a slot
    is plain. The sampling states' answer is held out of the OR for the call and merged back after,
    so vLLM's flag is unchanged.

    A slot whose only processing is ``LogitBiasState``'s ``min_tokens`` mask (no allowed tokens, no
    logit bias, no other processor) also records the length the mask lasts to: vLLM masks the stop
    tokens only while the sampled position is below it, so after that the row is plain. SkyRL asks
    for ``min_tokens=1`` on every request.
    """
    from functools import wraps

    @wraps(original)
    def add_request(self, req_idx, sampling_params):
        states = self.sampling_states
        own = [False]
        answers: list[tuple[object, bool]] = []

        def sampling_states_part(add):
            def part(i, params):
                own[0] = bool(add(i, params))
                return False
            return part

        def recorded(proc, add):
            def answer(i, params):
                out = add(i, params)
                answers.append((proc, bool(out)))
                return out
            return answer

        shadows = [(states, sampling_states_part(states.add_request))]
        shadows += [(proc, recorded(proc, proc.add_request))
                    for proc in (self.thinking_budget_state, *self.logits_processors)]
        saved = [(obj, obj.__dict__.get("add_request")) for obj, _ in shadows]
        for obj, fn in shadows:
            obj.add_request = fn
        try:
            original(self, req_idx, sampling_params)
        finally:
            for obj, prior in saved:
                if prior is None:
                    del obj.add_request
                else:
                    obj.add_request = prior
        processed = bool(self.needs_logits_processing[req_idx])
        self.needs_logits_processing[req_idx] = processed or own[0]
        _v2_processed_flags(self)[req_idx] = processed
        active = [proc for proc, used in answers if used]
        min_only = (len(active) == 1 and type(active[0]).__name__ == "LogitBiasState"
                    and not sampling_params.allowed_token_ids and not sampling_params.logit_bias)
        _v2_min_only_lens(self)[req_idx] = int(active[0].min_lens.np[req_idx]) if min_only else -1

    return add_request


def _v2_fused_sampler_plan(sampler, logits, idx_mapping_np, supports, override=None, next_lens=None):
    """``(mode, detail)`` for one Model Runner V2 sampling step, host-side checks only (no device sync).

    ``("plain", k)``: sample and score ``logits`` directly, returning ``k`` (None, 0 or 1) logprob
    columns beyond the sampled token. ``(None, why)``: V2's own ``__call__`` serves the step. Any
    request slot a logits processor rewrites (logit bias, allowed tokens, min_tokens, penalties,
    bad words, thinking budget, custom) keeps V2's sampler: V2 applies them as dense passes over
    an fp32 copy that the plain path does not reproduce. A slot processed only by a ``min_tokens``
    mask counts while the sampled position (``next_lens``, per batch row: computed + scheduled
    tokens) is below its mask length, as vLLM's own kernel decides it.
    """
    import numpy as np

    if override is not None:
        return None, f"sampler_override:{override}"
    states = sampler.sampling_states
    dims = sampler.get_logprobs_dims(idx_mapping_np)
    refusal = _fused_step_refusal(
        # V2 builds the sampler with decode_query_len: 1 + the speculative steps.
        spec_decode=sampler.num_speculative_tokens != 1 or logits.shape[0] != len(idx_mapping_np),
        logprobs_mode=sampler.logprobs_mode,
        top_k_top_p=bool(np.any(states.top_k.np[idx_mapping_np] != states.vocab_size)
                         or np.any(states.top_p.np[idx_mapping_np] != 1.0)),
        per_request_seed=bool(np.any(states.seeds_set[idx_mapping_np])),
        logprob_token_ids=dims is not None and dims[1] > 0,
        num_logprobs=None if dims is None else dims[0],
    )
    if refusal is not None:
        return None, refusal
    if np.any(states.min_p.np[idx_mapping_np] != 0.0):
        return None, "min_p"
    if sampler.trace_replay_state is not None:
        return None, "trace_replay"
    if sampler.return_sampling_mask:
        return None, "sampling_mask"
    processed = _v2_processed_flags(sampler)[idx_mapping_np]
    if np.any(processed):
        min_len = _v2_min_only_lens(sampler)[idx_mapping_np]
        live = processed & (min_len < 0)
        if next_lens is None:
            live |= processed
        else:
            live |= processed & (np.asarray(next_lens) < min_len)
        if np.any(live):
            return None, "logits_processors"
    if not supports(logits):
        return None, "layout"
    import torch

    # V2 samples outside its CUDA graphs; a capturing stream would freeze one step's Philox offset.
    if logits.is_cuda and torch.cuda.is_current_stream_capturing():
        return None, "cuda_graph_capture"
    return "plain", None if dims is None else dims[0]


def _vllm_v2_sampler_output(sampler, logits, input_batch, token, lp):
    """V2's ``SamplerOutput`` for a served step, built exactly as ``Sampler.__call__`` builds it."""
    import torch
    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.metrics.logits import get_num_nans
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    num_nans = get_num_nans(logits) if sampler.compute_nans else None
    tensors = None
    if lp is not None:
        indices, logprobs, ranks = lp
        # V2's gather carries the sampled column as the int64 token tensor itself.
        tensors = LogprobsTensors(logprob_token_ids=indices.to(torch.int64), logprobs=logprobs,
                                  selected_token_ranks=ranks)
    num_sampled, num_rejected = get_num_sampled_and_rejected(
        input_batch.seq_lens.new_ones(input_batch.num_reqs),
        input_batch.seq_lens,
        input_batch.cu_num_logits,
        input_batch.idx_mapping,
        sampler.req_states.prefill_len.gpu,
    )
    return SamplerOutput(sampled_token_ids=token.view(-1, 1), logprobs_tensors=tensors, num_nans=num_nans,
                         num_sampled=num_sampled, num_rejected=num_rejected)


def _make_vllm_v2_fused_sampler_call(original, base, resolve_sampler, temperature_tensor):
    """Build the Model Runner V2 ``Sampler.__call__`` wrapper; ``base`` is V2's ``Sampler`` class.

    A served step draws with the decode sampler's kernels on the step's logits, per-row temperature
    (greedy rows 0), the default CUDA generator seed and the step offset, and scores the raw row.
    """
    from functools import wraps

    decode_sampler = _decode_sampler_provider(resolve_sampler)

    @wraps(original, updated=())
    def call(self, logits, input_batch):
        fs = decode_sampler()
        idx = input_batch.idx_mapping_np
        mode, detail = (None, "arch") if fs is None else _v2_fused_sampler_plan(
            self, logits, idx, fs.supports, _v2_sampler_override(type(self), base),
            next_lens=input_batch.num_computed_tokens_np + input_batch.num_scheduled_tokens,
        )
        _fused_census("v2", detail if mode is None else None)
        if mode is None:
            return original(self, logits, input_batch)

        import numpy as np
        from vllm.sampling_params import _SAMPLING_EPS

        states = self.sampling_states
        # V2 keeps the raw temperature; a request below vLLM's sampling epsilon is greedy, and the
        # kernel reads it so too.
        all_greedy = bool(np.all(states.temperature.np[idx] < _SAMPLING_EPS))
        temperature = None if all_greedy else states.temperature.gpu[input_batch.expanded_idx_mapping]
        temp, seed, offset = _fused_draw(fs, logits, temperature, all_greedy)
        token, lp = fs.decode_step(logits, temp, seed, offset, detail, temperature_tensor())
        return _vllm_v2_sampler_output(self, logits, input_batch, token, lp)

    call._isoexec_fused_sampler = True
    call._isoexec_orig = original
    return call


def patch_vllm_v2_fused_sampler() -> bool:
    """Serve plain Model Runner V2 decode steps with the arch's decode sampler.

    Hooks V2's ``Sampler.__call__``, the one entry ``GPUModelRunner.sample`` calls for steps
    without draft tokens (eagerly, after the model's CUDA graph). Plain steps draw a Gumbel-max
    token keyed by (default CUDA generator seed, step offset, row, column), not V2's per-request
    (seed, position) Gumbel stream, and score it with the pinned rowinv recipe, the trainer's
    kernels. Steps with top-k/top-p, min_p, explicit
    per-request seeds, speculative decoding, ``logprob_token_ids``, logprobs outside {0, 1}, a
    non-raw logprobs mode, any logits processor, trace replay or sampling masks, and samplers that
    override token drawing (watermarking), keep V2's ``__call__``. ``ISOEXEC_FUSED_SAMPLER=0``
    keeps it for every step.
    """
    if not fused_sampler_enabled():
        logger.info("[isoexec] V2 decode sampler disabled (%s=0)", _FUSED_SAMPLER_ENV)
        return False
    try:
        from vllm.v1.worker.gpu.sample.sampler import Sampler
    except Exception as e:
        logger.warning("[isoexec] cannot hook vLLM Model Runner V2 Sampler: %s", e)
        return False
    if getattr(Sampler.__call__, "_isoexec_fused_sampler", False):
        return True
    from .scoring_temperature import temperature_tensor

    Sampler.add_request = _make_vllm_v2_add_request(Sampler.add_request)
    Sampler.__call__ = _make_vllm_v2_fused_sampler_call(
        Sampler.__call__, Sampler, _resolve_decode_sampler, temperature_tensor,
    )
    logger.info("[isoexec] hooked vLLM Model Runner V2 Sampler.__call__ -> decode sampler for plain decode steps")
    return True


#: What the shard head's noise was verified against (tests/test_shard_sample_gpu.py): Triton's lowering of
#: libdevice logf (major.minor) and the source of vLLM's Gumbel-max path (the functions below, hashed). Any
#: other version declines the head at install, so a vLLM or Triton bump cannot silently change served bits.
SHARD_SAMPLER_TRITON = "3.8"
SHARD_SAMPLER_GUMBEL_FUNCTIONS = (
    "_murmur3_rotl32", "_murmur3_mix", "_murmur3_fmix32", "murmur3_hash32", "murmur3_uniform32",
    "_log1p_neg_stable", "gumbel_noised_argmax", "gumbel_block_argmax", "_gumbel_sample_kernel", "gumbel_sample",
)
SHARD_SAMPLER_GUMBEL_SHA256 = "cf04a7443ff342cf4877c720103552b9f5d7546636185c3e274cd2b5ae40f3c6"


def _gumbel_source_digest(gumbel_module) -> str:
    import hashlib
    import inspect

    h = hashlib.sha256()
    for name in SHARD_SAMPLER_GUMBEL_FUNCTIONS:
        fn = getattr(gumbel_module, name)
        src = getattr(fn, "src", None) or inspect.getsource(fn)  # triton.JITFunction keeps its source in .src
        h.update(name.encode())
        h.update(src.encode())
    return h.hexdigest()


def shard_sampler_version_refusal() -> str | None:
    """None when the installed vLLM/Triton are the ones the shard head's noise reproduces, else why not."""
    try:
        import triton
        from vllm.v1.worker.gpu.sample import gumbel
    except Exception as e:  # pragma: no cover - import surface moved
        return f"cannot import vLLM's gumbel sampler / Triton ({e!r})"
    tv = ".".join(str(triton.__version__).split(".")[:2])
    if tv != SHARD_SAMPLER_TRITON:
        return f"Triton {triton.__version__} (the noise reproduces Triton {SHARD_SAMPLER_TRITON}'s logf lowering)"
    try:
        digest = _gumbel_source_digest(gumbel)
    except Exception as e:
        return f"vLLM's gumbel sampler surface moved ({e!r})"
    if digest != SHARD_SAMPLER_GUMBEL_SHA256:
        return f"vLLM's gumbel sampler source changed (sha256 {digest[:12]}, verified {SHARD_SAMPLER_GUMBEL_SHA256[:12]})"
    return None


def vllm_all_nan_column(vocab: int, device) -> int | None:
    """The token vLLM's ``gumbel_sample`` returns for a row that is NaN in every column, or None.

    That row has no order-free argmax: Triton's block reduction picks a column by its combine tree and
    ``torch.argmax`` over the block maxima takes the first NaN block. The answer depends only on the
    vocabulary and the kernel (not on seeds, positions or temperature), so it is probed once: one greedy
    and one noisy row, which must agree and be in range.
    """
    import torch
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    logits = torch.full((2, int(vocab)), float("nan"), dtype=torch.float32, device=device)
    req = torch.arange(2, dtype=torch.int32, device=device)
    temp = torch.tensor([0.0, 1.0], dtype=torch.float32, device=device)
    seed = torch.tensor([0, 12345], dtype=torch.int64, device=device)
    pos = torch.tensor([0, 7], dtype=torch.int64, device=device)
    got = gumbel_sample(logits, req, temp, seed, pos, apply_temperature=False, is_drafting=False).tolist()
    if got[0] != got[1] or not 0 <= got[0] < vocab:
        return None
    return int(got[0])


def _shard_head_local_check(head, world: int) -> tuple[bool, int, str]:
    """This rank's verdict on the shard head: (ok, all-NaN token, reason). Builds the extension."""
    import torch

    try:
        from vllm.v1.worker.gpu import model_runner as v2_runner

        installed = bool(getattr(v2_runner.GPUModelRunner, "_isoexec_shard_sampler", False))
    except Exception:
        installed = False
    if not installed:
        return False, 0, "head not installed"
    try:
        from isoexec.ops.logprobs import supports

        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
            return False, 0, "not SM90"
        if head.padded_vocab != head.vocab or not supports(head.vocab, world):
            return False, 0, "head geometry (padding / vocab split)"
        from isoexec.ops import logprobs as _logprobs  # SM90-only export, reached after the capability check

        _logprobs.shard_sample_build()
        col = vllm_all_nan_column(head.vocab, torch.device("cuda", torch.cuda.current_device()))
        if col is None:
            return False, 0, "vLLM's all-NaN token is not one column"
        return True, col, ""
    except Exception as e:
        return False, 0, f"build/probe failed: {e!r}"


def agree_shard_head(ok: bool, nan_col: int, group) -> tuple[bool, int, str]:
    """MIN over the group of (ok, nan_col, -nan_col): every rank serves, or none does, with one token."""
    import torch
    import torch.distributed as dist

    device = (torch.device("cpu") if dist.get_backend(group) == "gloo"
              else torch.device("cuda", torch.cuda.current_device()))
    v = torch.tensor([int(ok), int(nan_col), -int(nan_col)], dtype=torch.int64, device=device)
    dist.all_reduce(v, op=dist.ReduceOp.MIN, group=group)
    agreed, lo, hi = bool(int(v[0])), int(v[1]), -int(v[2])
    if not agreed:
        return False, 0, "declined on a TP rank"
    if lo != hi:
        return False, 0, f"TP ranks probed different all-NaN tokens ({lo} vs {hi})"
    return True, lo, ""


def prepare_v2_shard_head(model) -> int | None:
    """Engine init: build the shard head's extension and probe its all-NaN token, agreed over the TP group.

    Runs on every rank of the final PP stage at model construction (before any step), so a slow build cannot
    land between a step's collectives, and a failure on one rank declines the head on every rank (a step's
    eligibility must be rank-uniform: the shard path and the gathered path issue different collectives).
    Once the head is installed in this process, the agreement runs whatever this rank's own verdict is (build
    or probe failures are per rank). Sets and returns ``model.isoexec_shard_nan_col``
    (None = the head is declined for the run).
    """
    model.isoexec_shard_nan_col = None
    head = getattr(model, "head", None)
    if head is None:
        return None
    group = getattr(model, "_tp_group", None)
    world = 1 if group is None else group.size()
    ok, nan_col, why = _shard_head_local_check(head, world)
    if why == "head not installed":
        # Installation is decided identically on every rank (one flag, one vLLM/Triton install, one hook): a
        # process that did not install the head never serves it, and its peers did not install it either.
        return None
    if world > 1:
        ok, nan_col, why_group = agree_shard_head(ok, nan_col, group)
        why = why or why_group
    if ok:
        model.isoexec_shard_nan_col = nan_col
        print(f"[ISOEXEC-SHARD-SAMPLER] pid={os.getpid()} head ready (extension built at init; all-NaN row -> "
              f"token {nan_col})", flush=True)
    else:
        print(f"[ISOEXEC-SHARD-SAMPLER] pid={os.getpid()} head DECLINED for this run: {why}", flush=True)
    return model.isoexec_shard_nan_col


#: Vocab-parallel sampling head census: steps served shard-local vs declined (host counters only).
_SHARD_STATE = {"served": 0, "declined": 0, "reason": "", "reported": 0}


def _shard_report() -> None:
    n = _SHARD_STATE["served"] + _SHARD_STATE["declined"]
    if n < 1 or (n & (n - 1)) != 0 or n == _SHARD_STATE["reported"]:
        return
    _SHARD_STATE["reported"] = n
    print(f"[ISOEXEC-SHARD-SAMPLER] pid={os.getpid()} served={_SHARD_STATE['served']} "
          f"declined={_SHARD_STATE['declined']} last_decline={_SHARD_STATE['reason']!r}", flush=True)


def _shard_decline(reason: str):
    _SHARD_STATE["declined"] += 1
    _SHARD_STATE["reason"] = reason
    _shard_report()
    return None


def _v2_shard_sample(runner, hidden_states, input_batch, grammar_output, *, sampler_cls, sampler_output_cls,
                     logprobs_tensors_cls, get_num_sampled_and_rejected, temperature_tensor):
    """Model Runner V2 ``sample`` on this rank's vocab shard, or None to run vLLM's gathered path.

    Serves exactly the steps whose gathered path is: logits -> (no processor) -> ``gumbel_sample(...,
    apply_temperature=False)`` -> optional sampled-token logprob (the hooked leaf8 full-row producer) and
    rank. Bitwise equal outputs (logprobs_sm90.ops.shard_sample; tests/test_shard_sample_gpu.py)."""
    import numpy as np

    if grammar_output is not None or getattr(runner, "batch_sharder", None) is not None:
        return _shard_decline("structured output / batch-sharded sampling")
    if input_batch.num_reqs == 0 or input_batch.num_draft_tokens != 0:
        return _shard_decline("no requests / speculative step")
    sampler = runner.sampler
    if type(sampler) is not sampler_cls:
        return _shard_decline(f"sampler {type(sampler).__name__}")
    if (sampler.compute_nans or sampler.trace_replay_state is not None or sampler.return_sampling_mask
            or sampler.use_fp64_gumbel):
        return _shard_decline("nan census / trace replay / sampling mask / fp64 gumbel")
    idx_np = input_batch.idx_mapping_np
    if np.any(sampler.needs_logits_processing[idx_np]):
        return _shard_decline("logits processing (temperature != 0/1, min_p, top-k/top-p, penalties, ...)")
    states = sampler.sampling_states
    # A greedy row may carry top-k/top-p without needing processing; vLLM still masks the batch then.
    if np.any(states.top_k.np[idx_np] != states.vocab_size) or np.any(states.top_p.np[idx_np] != 1.0):
        return _shard_decline("top-k/top-p masking")
    dims = sampler.get_logprobs_dims(idx_np)
    if dims is not None and (tuple(dims) != (0, 0) or sampler.logprobs_mode not in ("raw_logprobs",
                                                                                    "processed_logprobs")):
        return _shard_decline(f"logprobs dims {dims} / mode {sampler.logprobs_mode}")
    if input_batch.logits_indices.shape[0] != input_batch.num_reqs:
        return _shard_decline("expanded logits")
    shard_logits = getattr(runner.model, "isoexec_shard_logits", None)
    if shard_logits is None:
        return _shard_decline("model has no shard head")
    nan_col = getattr(runner.model, "isoexec_shard_nan_col", None)
    if nan_col is None:
        return _shard_decline("shard head not prepared at init (see the init line)")
    local = shard_logits(hidden_states[input_batch.logits_indices])
    if local is None:
        return _shard_decline("head geometry (padding / vocab split)")
    from isoexec.ops import logprobs as _logprobs

    import torch

    req = input_batch.expanded_idx_mapping
    if req.dtype != torch.int32:
        req = req.to(torch.int32)
    sampled, logprob, rank = _logprobs.shard_sample(
        local, group=runner.model._tp_group, req_of_row=req, req_temperature=states.temperature.gpu,
        req_seed=states.seeds.gpu, row_pos=input_batch.positions[input_batch.logits_indices], nan_col=nan_col,
        temperature=temperature_tensor(), ranks=dims is not None)
    logprobs_tensors = None
    if dims is not None:
        logprobs_tensors = logprobs_tensors_cls(logprob_token_ids=sampled.unsqueeze(-1),
                                                logprobs=logprob.unsqueeze(-1), selected_token_ranks=rank)
    num_sampled, num_rejected = get_num_sampled_and_rejected(
        input_batch.seq_lens.new_ones(input_batch.num_reqs), input_batch.seq_lens, input_batch.cu_num_logits,
        input_batch.idx_mapping, sampler.req_states.prefill_len.gpu)
    out = sampler_output_cls(sampled_token_ids=sampled.view(-1, 1), logprobs_tensors=logprobs_tensors,
                             num_nans=None, num_sampled=num_sampled, num_rejected=num_rejected)
    _SHARD_STATE["served"] += 1
    _shard_report()
    return out, out.num_sampled, out.num_rejected


def patch_vllm_v2_shard_sampler() -> bool:
    """Route eligible Model Runner V2 sampling steps through the vocab-parallel head (ISOEXEC_SHARD_SAMPLER)."""
    if os.environ.get("ISOEXEC_SHARD_SAMPLER", "1").lower() in ("", "0", "false", "no"):
        print("[ISOEXEC-SHARD-SAMPLER] OFF (ISOEXEC_SHARD_SAMPLER): every step gathers the vocabulary", flush=True)
        return False
    try:
        from vllm.v1.outputs import LogprobsTensors
        from vllm.v1.worker.gpu import model_runner as v2_runner
        from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
        from vllm.v1.worker.gpu.sample import logprob as v2_logprob
        from vllm.v1.worker.gpu.sample.output import SamplerOutput
        from vllm.v1.worker.gpu.sample.sampler import Sampler
    except Exception as e:
        logger.warning("[isoexec] cannot install the V2 vocab-parallel sampling head: %s", e)
        return False
    refusal = shard_sampler_version_refusal()
    if refusal is not None:
        print(f"[ISOEXEC-SHARD-SAMPLER] NOT INSTALLED: {refusal}; every step gathers the vocabulary", flush=True)
        return False
    if not getattr(getattr(v2_logprob, "compute_token_logprobs", None), "_isoexec_v2_full_row_hook", False):
        # Declined steps would then score with vLLM's own logprob kernel and served steps with leaf8.
        print("[ISOEXEC-SHARD-SAMPLER] NOT INSTALLED: the V2 leaf8 logprob hook is not active; every step gathers "
              "the vocabulary", flush=True)
        return False
    import torch

    runner_cls = v2_runner.GPUModelRunner
    if getattr(runner_cls, "_isoexec_shard_sampler", False):
        return True
    original = runner_cls.sample
    from .scoring_temperature import temperature_tensor

    def sample(self, hidden_states, input_batch, grammar_output):
        if torch.cuda.get_device_capability(hidden_states.device) == (9, 0):
            got = _v2_shard_sample(self, hidden_states, input_batch, grammar_output, sampler_cls=Sampler,
                                   sampler_output_cls=SamplerOutput, logprobs_tensors_cls=LogprobsTensors,
                                   get_num_sampled_and_rejected=get_num_sampled_and_rejected,
                                   temperature_tensor=temperature_tensor)
            if got is not None:
                return got
        else:
            _shard_decline("not SM90")
        return original(self, hidden_states, input_batch, grammar_output)

    runner_cls.sample = sample
    runner_cls._isoexec_shard_sampler = True
    print("[ISOEXEC-SHARD-SAMPLER] Model Runner V2 vocab-parallel sampling head INSTALLED", flush=True)
    return True


def _local_kv_cache_tensors(config):
    """``config`` with the KV cache tensors of layers this worker does not hold removed.

    vLLM 0.33 projects the global KV cache groups onto a pipeline stage (``_project_kv_cache_groups_to_worker``)
    and keeps a group none of whose layers is on the stage with its full ``UniformTypeKVCacheSpecs`` (models with
    packed per-layer specs, e.g. DeepSeek V4.1), so the per-worker config lists tensors of other stages' layers and
    the worker's ``allocate_kv_cache`` finds no group for them (``StopIteration``). Such a tensor names only
    layers of no group of this worker; it is dropped. Every other tensor is kept unchanged (offsets and strides
    are absolute), so a config without such tensors -- every single-stage config, every model without packed
    group specs -- is returned as it was."""
    held = {name for group in config.kv_cache_groups for name in group.layer_names}
    kept = [t for t in config.kv_cache_tensors if any(name in held for name in t.layers)]
    if len(kept) == len(config.kv_cache_tensors):
        return config
    mixed = [t for t in kept if not all(name in held for name in t.layers)]
    if mixed:
        raise ValueError(f"a KV cache tensor mixes held and foreign layers: {mixed[0].layers}")
    config.kv_cache_tensors = kept
    return config


def patch_vllm_pp_kv_cache_tensors() -> bool:
    """Wrap the engine core's ``get_kv_cache_configs`` with ``_local_kv_cache_tensors`` per worker (idempotent)."""
    try:
        from vllm.v1.engine import core
    except Exception:  # pragma: no cover - vLLM absent
        return False
    original = getattr(core, "get_kv_cache_configs", None)
    if original is None:
        return False
    if getattr(original, "_isoexec_orig", None) is not None:
        return True
    from functools import wraps

    @wraps(original)
    def wrapped(*args, **kwargs):
        return [_local_kv_cache_tensors(config) for config in original(*args, **kwargs)]

    wrapped._isoexec_orig = original
    core.get_kv_cache_configs = wrapped
    return True
