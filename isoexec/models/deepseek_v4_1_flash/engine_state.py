"""The per-step host state of a vLLM V2 engine serving this package (``engine_model_state_cls``).

vLLM's ``DeepseekV41ModelState`` gathers the Engram token-history window (``lookback_token_ids``) every step,
graph steps included. This subclass adds what the rows need from the host:

* at load: the tokenizer's compressed token map for the Engram hash, and the device views of the host-resident
  Engram tables (one host copy per node, mapped by the runtime; row lookups on the device, inside captured
  graphs);
* before every captured decode step (vLLM replays it without calling the model): the derived-weight check
  (fail closed when a weight write bypassed the update commit: a version compare over the entries the model
  resolved once, ``VllmModel.derived_weight_check``), the score capacity of the step's bucket and the replay's
  program receipt (``engine.RECEIPTS``: bucket and the weight version the graph reads);
* two sampled sweeps of the same contract, round robin: every ``CONTENT_INTERVAL`` captured steps one
  derived copy re-derived on the device and compared byte for byte (read back at a later step without waiting,
  failing closed; it sees source writes no version counter records), and every captured step
  ``ADDRESS_SAMPLE`` of the parameters the graphs read in place checked for a replacement or a moved storage;
* every ``consistency_check_interval`` steps: a fingerprint of the step inputs compared across the TP ranks
  on the device, read back at the next check (deferred, no wait in the step) and failing closed.
"""
from __future__ import annotations

import weakref

import numpy as np
import torch

#: ``additional_config.isoexec`` key: steps between cross-rank step-input checks (0 disables them).
CHECK_INTERVAL_KEY = "consistency_check_interval"
DEFAULT_CHECK_INTERVAL = 64
#: Graph-read parameters whose address one step checks (round robin over the rows).
ADDRESS_SAMPLE = 32
#: Captured steps between two sampled re-derivations of a derived weight copy.
CONTENT_INTERVAL = 8
#: Steps between two cross-rank digests of the token ids, positions and slots (``StepGuard``).
DIGEST_INTERVAL = 1

_CLASS: list = []


def _rows(model):
    from .block import V41Layer
    return [m for m in model.modules() if isinstance(m, V41Layer)]


def _embed(model):
    from .embed import DeepSeekV41Embed
    return next((m for m in model.modules() if isinstance(m, DeepSeekV41Embed)), None)


def _geometry(model):
    """The served geometry from any component this rank holds: the embedding (first pipeline stage) or a row
    (every stage). A later pipeline stage holds no embedding; its decoder plan still needs the replay window
    (taking it from the embedding alone gave window 0 there: PP2 logits differed from PP1, defect 13d)."""
    holder = _embed(model) or next(iter(_rows(model)), None)
    return None if holder is None else holder.cfg


def bind_tokenizer(model, vllm_config) -> None:
    """Bind the compressed token map of the served tokenizer to the Engram hash."""
    embed = _embed(model)
    if embed is None or not embed.hashing:
        return
    from transformers import AutoTokenizer

    from .embed import compressed_token_map
    mc = vllm_config.model_config
    tokenizer = AutoTokenizer.from_pretrained(mc.tokenizer or mc.model, trust_remote_code=mc.trust_remote_code)
    embed.bind_token_map(compressed_token_map(tokenizer))


def register_host_table(table, device) -> tuple:
    """Device views of a table this process holds alone (a trainer's host copy): each buffer registered with
    CUDA (``cudaHostRegister``, mapped) and viewed from the device. Fails closed.

    Registered in ``host_tables.REGISTER_CHUNK`` pieces and unregistered piece by piece
    (``release_host_table``; a finalizer does it at interpreter exit at the latest). A registration left to the
    CUDA context teardown is dropped in one driver critical section at process exit: for the ~92 GiB tables of
    the released checkpoint that stalls every other driver client beyond 10 s (``nvidia-smi``)."""
    from cuda.bindings import runtime as rt
    from isoexec.runtimes.vllm.host_tables import _unregister, register_pieces

    release_host_table(table)
    chunks: list[int] = []
    table.__dict__["_host_registration"] = chunks
    weakref.finalize(table, _unregister, chunks)
    views = []
    for host in (table.weight, table.scale):
        if not host.is_contiguous() or host.data_ptr() % 4096:
            _unregister(chunks)
            raise RuntimeError("host Engram tables must be page-aligned dense host memory")
        nbytes = host.numel() * host.element_size()
        if nbytes == 0:
            _unregister(chunks)
            raise RuntimeError("an empty host Engram table")
        # on failure every piece registered so far (both buffers) is dropped before the error propagates
        pointer = register_pieces(host.data_ptr(), nbytes, rt.cudaHostRegisterMapped | rt.cudaHostRegisterPortable,
                                  chunks, "an Engram table")

        class _Interface:
            __cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (int(pointer), False),
                                        "version": 3, "strides": None}
        flat = torch.as_tensor(_Interface(), device=device)
        views.append(flat.view(host.dtype).view(host.shape))
    return views[0], views[1]


def release_host_table(table) -> None:
    """Drop the CUDA registration ``register_host_table`` made for ``table`` (idempotent), piece by piece, and
    its device views. The host bytes stay."""
    from isoexec.runtimes.vllm.host_tables import _unregister

    chunks = table.__dict__.pop("_host_registration", None)
    if chunks is not None:
        if getattr(table, "device_view", None) is not None:
            table.device_view = None
        _unregister(chunks)


def release_host_tables(model) -> int:
    """Release, in order and piece by piece, every CUDA registration of a host Engram table this process holds
    before it exits: a table it registered itself (``register_host_table``) and this rank's registration of a
    node-shared copy (``host_tables.NodeSharedHostBuffers.release``; the shared pages stay mapped until the
    process exits, the segment itself was unlinked at load). Returns the number of tables released."""
    released, shared = 0, []
    for row in _rows(model):
        table = None if row.engram is None else row.engram.embed
        if table is None:
            continue
        if "_host_registration" in table.__dict__:
            release_host_table(table)
            released += 1
        for buffers in table.__dict__.get("_host_shared", {}).values():
            if buffers.registered is not None and not any(buffers is b for b in shared):
                shared.append(buffers)
        if table.__dict__.get("_host_shared") and getattr(table, "device_view", None) is not None:
            table.device_view = None
            released += 1
    for buffers in shared:
        buffers.release()
    return released


def map_host_tables(model, device) -> int:
    """Give every host-resident Engram table (``EngramTable.HOST_RESIDENT_BUFFERS``) its device view. On the
    engine the runtime keeps one host copy per node, registered with CUDA on every rank
    (``runtimes/vllm/host_tables``), and exposes each buffer's device mapping (``_host_device_views``); a
    table without them (a trainer's own host copy) is registered here (``register_host_table``). A fixed row
    sample looked up through the view must equal the lookup of the host rows. Fails closed. Returns the number
    of tables mapped."""
    from isoexec.ops.embed.engram_sm100.ops import fp8_rowwise_lookup

    mapped = 0
    for row in _rows(model):
        table = None if row.engram is None else row.engram.embed
        if table is None or table.weight.is_cuda or getattr(table, "device_view", None) is not None:
            continue
        given = table.__dict__.get("_host_device_views", {})
        if "weight" in given and "scale" in given:
            views = (given["weight"], given["scale"])
        elif "_host_shared" in table.__dict__:
            raise RuntimeError("a node-shared host Engram table has no device mapping (the runtime's load did "
                               "not finish)")
        else:
            views = register_host_table(table, device)
        rows = table.weight.shape[0]
        g = torch.Generator().manual_seed(0x5EED ^ rows)
        sample = torch.cat([torch.tensor([0, rows - 1], dtype=torch.int64),
                            torch.randint(0, rows, (4094,), generator=g, dtype=torch.int64)])
        host_rows = (table.weight.index_select(0, sample).to(device), table.scale.index_select(0, sample).to(device))
        local = torch.arange(sample.numel(), device=device).view(1, 1, -1)
        want = fp8_rowwise_lookup(*host_rows, local, validate=True)
        got = fp8_rowwise_lookup(views[0], views[1], sample.to(device).view(1, 1, -1), validate=True)
        if not torch.equal(got.view(torch.uint8), want.view(torch.uint8)):
            raise RuntimeError("Engram lookup through the device mapping differs from the host rows")
        table.device_view = views
        mapped += 1
    return mapped


class DeferredFlag:
    """A device verdict copied to the host without waiting and read at a later step (fail closed)."""

    def __init__(self, what: str) -> None:
        self.what, self.pending, self._host = what, None, None

    def put(self, flag_dev, label) -> None:
        if self._host is None:
            self._host = torch.empty(1, dtype=torch.int32, pin_memory=True)
        flag = self._host
        flag.copy_(flag_dev.reshape(1).to(torch.int32), non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.pending = (flag, event, label)

    def resolve(self, *, wait: bool = False) -> bool:
        """Raise on a set verdict; return whether nothing is pending any more."""
        if self.pending is None:
            return True
        flag, event, label = self.pending
        if not wait and not event.query():
            return False
        event.synchronize()
        self.pending = None
        if int(flag[0]):
            raise RuntimeError(f"DeepSeek V4.1 {self.what}: {label}")
        return True


class StepGuard:
    """Per-step device checks of the step inputs, read back one step later without waiting (fail closed):

    * block ids: every entry of the step's block-table rows lies below ``blocks``, the largest page count of
      the engine's cache layers (one multi-tensor max over the tables; vLLM's KV manager owns them, and this
      catches an id that would address outside every cache allocation), and every ``LOW_INTERVAL`` steps
      none is negative;
    * cross-rank agreement (TP > 1, every ``interval`` steps): one int64 digest of the token ids, positions and
      slot mappings, all-gathered over the TP group, equal on every rank.

    One small packet per step goes to the host; it is read at the next step if it has arrived. At TP = 1 a step whose
    previous packet is still in flight waits for none and issues none; at TP > 1 the step waits for it, so the digest
    all-gather is issued at the same steps on every rank (a rank-local skip mis-pairs the TP collectives)."""

    LOW_INTERVAL = 64

    def __init__(self, interval: int, group, limit: int | None = None) -> None:
        self.interval, self.group, self.limit = interval, group, limit
        self.count, self.pending, self._host, self._weights = 0, None, None, None
        self._world = 1

    def resolve(self, *, wait: bool = False) -> bool:
        if self.pending is None:
            return True
        event, tables, low, digest = self.pending
        if not wait and not event.query():
            return False
        event.synchronize()
        self.pending = None
        packet = self._host[:tables + (1 if low else 0) + (self._world if digest else 0)].tolist()
        high = max(packet[:tables])
        if self.limit is not None and high >= self.limit:
            raise RuntimeError(f"DeepSeek V4.1 step block tables hold an id outside [0, {self.limit}): max {high}")
        if low and packet[tables] < 0:
            raise RuntimeError(f"DeepSeek V4.1 step block tables hold a negative id ({packet[tables]})")
        if digest and len(set(packet[-self._world:])) > 1:
            raise RuntimeError("DeepSeek V4.1 engine ranks disagree on the step inputs (token ids, positions, "
                               f"slots): digests {packet[-self._world:]}")
        return True

    def _weighted(self, n: int, device):
        w = self._weights
        if w is None or w.numel() < n:
            w = self._weights = torch.arange(1, max(n, 4096) + 1, dtype=torch.int64, device=device)
        return w[:n]

    def step(self, input_batch, block_tables, slot_mappings) -> None:
        # The cross-rank digest is a TP COLLECTIVE: whether a rank issues it must not depend on a rank-local fact.
        # "Is my previous packet still in flight" (event.query()) is rank-local timing, so at TP > 1 the previous
        # packet is waited for (it was recorded at the previous step's input preparation, so this waits for that
        # point of the device stream, not for the previous forward); every rank then counts and issues the same
        # steps. At TP = 1 nothing is collective and the in-flight skip stays (defect 12: one rank skipped the
        # digest all-gather while the other issued it, so NCCL paired it with the next logits all-gather).
        if not self.resolve(wait=self.group is not None and self.interval > 0):
            return
        R, n = input_batch.num_reqs, input_batch.num_tokens
        tables = [t[:R] for t in block_tables if R and t.numel()]
        if not tables:
            return
        self.count += 1
        parts = list(torch._foreach_max(tables))            # one multi-tensor launch
        low = self.count % self.LOW_INTERVAL == 0
        if low:
            parts.append(torch.stack([t.min() for t in tables]).min())
        digest = bool(self.group is not None and self.interval and self.count % self.interval == 0)
        if digest:
            import torch.distributed as dist
            v = torch.cat([input_batch.input_ids[:n].to(torch.int64), input_batch.positions[:n].to(torch.int64),
                           slot_mappings.reshape(-1).to(torch.int64)])
            mine = (v * self._weighted(v.numel(), v.device)).sum().view(1)
            self._world = dist.get_world_size(self.group)
            gathered = torch.empty(self._world, dtype=torch.int64, device=mine.device)
            dist.all_gather_into_tensor(gathered, mine, group=self.group)
            parts += list(gathered.unbind())
        packet = torch.stack(parts) if len(parts) > 1 else parts[0].view(1)
        if self._host is None or self._host.numel() < packet.numel():
            self._host = torch.empty(max(packet.numel(), 64), dtype=torch.int64, pin_memory=True)
        self._host[:packet.numel()].copy_(packet, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.pending = (event, len(tables), low, digest)


def cache_page_limit(vllm_config, attn_groups) -> int | None:
    """The largest leading (page) dimension of the step's cache layers: no block-table id reaches it."""
    context = vllm_config.compilation_config.static_forward_context
    pages = []
    for groups in attn_groups:
        for group in groups:
            for name in group.layer_names:
                cache = getattr(context.get(name), "kv_cache", None)
                if isinstance(cache, (list, tuple)):
                    cache = cache[0] if cache else None
                if isinstance(cache, torch.Tensor) and cache.dim() and cache.numel():
                    pages.append(int(cache.shape[0]))
    return max(pages) if pages else None


class StepChecks:
    """Sampled, deferred cross-rank comparison of the step inputs (token ids, positions, slots, block
    tables): every ``interval`` steps each rank all-gathers a fingerprint over the TP group on the device;
    the verdict is copied to the host without waiting and read at the next check, which fails closed."""

    def __init__(self, interval: int, device) -> None:
        self.interval, self.device, self.count = interval, device, 0
        self.pending = None
        self.group = None
        if interval:
            from vllm.distributed import get_tp_group
            group = get_tp_group()
            self.group = group.device_group if group.world_size > 1 else None

    def resolve(self, *, wait: bool = False) -> None:
        if self.pending is None:
            return
        flag, event = self.pending
        if not wait and not event.query():
            return
        event.synchronize()
        self.pending = None
        if int(flag[0]):
            raise RuntimeError("DeepSeek V4.1 engine ranks disagree on the step inputs (sampled check)")

    def step(self, input_batch, block_tables, slot_mappings) -> None:
        if self.group is None:
            return
        self.count += 1
        if self.count % self.interval:
            return
        self.resolve(wait=True)
        import torch.distributed as dist
        n = input_batch.num_tokens

        def fold(t):
            t = t.reshape(-1).to(torch.int64)
            weights = torch.arange(1, t.numel() + 1, dtype=torch.int64, device=t.device)
            return (t * weights).sum().view(1)
        parts = [fold(input_batch.input_ids[:n]), fold(input_batch.positions[:n]), fold(slot_mappings),
                 *(fold(t[:input_batch.num_reqs]) for t in block_tables)]
        mine = torch.cat(parts)
        world = dist.get_world_size(self.group)
        gathered = torch.empty((world, mine.numel()), dtype=torch.int64, device=mine.device)
        dist.all_gather_into_tensor(gathered, mine, group=self.group)
        flag_dev = (gathered != gathered[:1]).any().view(1).to(torch.int32)
        flag = torch.empty(1, dtype=torch.int32, pin_memory=True)
        flag.copy_(flag_dev, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.pending = (flag, event)


def _ced_mode(model):
    """The CED prefill program of the served model (the IsoExec definition carries it on every block)."""
    modes = {getattr(m, "ced_prefill", None) for m in _rows(model)} - {None}
    if len(modes) != 1:
        raise ValueError(f"the served V4.1 model declares no single CED prefill program: {sorted(modes)}")
    return modes.pop()


def _dummy(input_batch) -> bool:
    from .engine_ced import is_dummy_batch
    return is_dummy_batch(input_batch)


def _pw_core() -> bool:
    from .piecewise import mode
    return mode() == "core"


def model_state_class():
    """``DeepSeekV41EngineState``: vLLM's ``DeepseekV41ModelState`` with this package's host steps."""
    if _CLASS:
        return _CLASS[0]
    from vllm.config import CUDAGraphMode
    from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState

    from .engine import graph_context_capacity, record_replay

    class DeepSeekV41EngineState(DeepseekV41ModelState):
        #: Tests and cost measurements only: False skips the per-step input guard (``StepGuard``).
        step_guard = True

        def __init__(self, vllm_config, model, encoder_cache, device):
            cfg = _geometry(model)
            model.token_lookback_depth = (cfg.engram_max_ngram_size - 1
                                          if cfg is not None and cfg.engram_layer_ids else 0)
            super().__init__(vllm_config, model, encoder_cache, device)
            self._model = model
            bind_tokenizer(model, vllm_config)
            self.engram_device_tables = map_host_tables(model, device)
            settings = (vllm_config.additional_config or {}).get("isoexec") or {}
            interval = settings.get(CHECK_INTERVAL_KEY, DEFAULT_CHECK_INTERVAL)
            if type(interval) is not int or interval < 0:
                raise ValueError(f"{CHECK_INTERVAL_KEY} must be a nonnegative integer")
            self._checks = StepChecks(interval, device)
            self._guard = None                  # StepGuard, built at the first step (the caches are allocated)
            self._content = DeferredFlag("derived weight differs from its sources without the update commit "
                                         "(sampled re-derivation)")
            rows = _rows(model)
            self.ced_prefill = getattr(model, "ced_prefill", None) or _ced_mode(model)
            self.ced_window = cfg.replay_window if cfg is not None else 0
            self._max_len = vllm_config.model_config.max_model_len
            self._budget = rows[0].engine_state.score_budget if rows and rows[0].engine_state is not None else None
            self.graph_steps = 0

        def add_request(self, req_index, new_req_data):
            from .engine_ced import refuse_request
            params = new_req_data.sampling_params
            refuse_request(self.ced_prefill,
                           prompt_logprobs=params is not None and params.prompt_logprobs is not None,
                           prompt_logprob_token_ids=params is not None
                           and getattr(params, "prompt_logprob_token_ids", None) is not None,
                           return_routed_experts=bool(getattr(getattr(self.vllm_config, "aux_output_config", None),
                                                              "enable_return_routed_experts", False)),
                           req_id=str(getattr(new_req_data, "req_id", "")))
            super().add_request(req_index, new_req_data)

        def prepare_inputs(self, input_batch, req_states):
            """The step's decoder rows (``engine_ced.plan_from_runner``: host arrays only), published for the
            decoder components with the model inputs."""
            from .engine_ced import plan_from_runner, publish_plan
            model_inputs = super().prepare_inputs(input_batch, req_states)
            publish_plan(plan_from_runner(input_batch, req_states, window=self.ced_window, mode=self.ced_prefill))
            return model_inputs

        def prepare_dummy_inputs(self, num_reqs, num_tokens):
            """Graph warm-up and capture build their batches here, not in ``prepare_inputs``: the dummy plan
            (every row a decoder row, segment start 0), split as ``InputBatch.make_dummy`` splits the tokens."""
            from .engine_ced import dummy_plan, dummy_split, publish_plan
            publish_plan(dummy_plan(dummy_split(num_reqs, num_tokens)))
            return super().prepare_dummy_inputs(num_reqs, num_tokens)

        def prepare_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups,
                         kv_cache_config, for_capture=False, *args, **kwargs):
            # captured graphs read the weights in place: FULL decode graphs and, under the ``core`` break structure,
            # the PIECEWISE segments (``piecewise``); both get the guards and the replay receipt
            replay = not for_capture and (cudagraph_mode == CUDAGraphMode.FULL
                                          or (cudagraph_mode == CUDAGraphMode.PIECEWISE and _pw_core()
                                              and not _dummy(input_batch)))      # vLLM passes no for_capture there
            if replay:
                check = self._model.derived_weight_check()
                stale = check.stale()
                if stale:
                    raise RuntimeError("DeepSeek V4.1 weights changed without the update commit; captured "
                                       f"derived weights are stale: {stale[:4]}")
                moved = check.moved_sample(ADDRESS_SAMPLE)
                if moved:
                    raise RuntimeError("DeepSeek V4.1 parameters the captured graphs read in place moved without "
                                       f"the update commit: {moved[:4]}")
                if self._content.resolve() and self.graph_steps % CONTENT_INTERVAL == 0:
                    probe = check.probe_content()
                    if probe is not None:
                        self._content.put(probe[1], probe[0])
                reqs = input_batch.num_reqs
                longest = 0 if cudagraph_mode != CUDAGraphMode.FULL else int(np.max(input_batch.num_computed_tokens_np[:reqs]
                                     + input_batch.num_scheduled_tokens[:reqs])) if reqs else 0
                capacity = graph_context_capacity(input_batch.num_tokens_after_padding, self._max_len, self._budget)
                if longest > capacity:
                    raise RuntimeError(f"a captured decode step of {input_batch.num_tokens_after_padding} rows "
                                       f"scores {capacity} context positions; a request reached {longest}")
                self.graph_steps += 1
                T = input_batch.num_tokens_after_padding
                record_replay(T if cudagraph_mode == CUDAGraphMode.FULL else f"pw{T}")
            if not for_capture:
                if self._guard is None:
                    from vllm.distributed import get_tp_group
                    tp = get_tp_group()
                    self._guard = StepGuard(DIGEST_INTERVAL, tp.device_group if tp.world_size > 1 else None,
                                            cache_page_limit(self.vllm_config, attn_groups))
                if self.step_guard:
                    self._guard.step(input_batch, block_tables, slot_mappings)
                self._checks.resolve()
                self._checks.step(input_batch, block_tables, slot_mappings)
            return super().prepare_attn(input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups,
                                        kv_cache_config, for_capture, *args, **kwargs)

    _CLASS.append(DeepSeekV41EngineState)
    return DeepSeekV41EngineState


__all__ = ["ADDRESS_SAMPLE", "CHECK_INTERVAL_KEY", "CONTENT_INTERVAL", "DIGEST_INTERVAL", "DeferredFlag",
           "StepChecks", "StepGuard", "cache_page_limit", "bind_tokenizer", "map_host_tables", "model_state_class",
           "register_host_table", "release_host_table", "release_host_tables"]
