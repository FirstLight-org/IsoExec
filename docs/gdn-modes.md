# GDN modes

Gated DeltaNet layers (Qwen3.5/3.6) carry a recurrent state per request. IsoExec supports two ways to
compute that recurrence and hold its state. Both are bitwise: in either mode the trainer and the engine
run one kernel recipe, so rollout logprobs equal the trainer's, and both modes are batch and parallelism
invariant. The modes differ in speed and in what owns the engine's state memory. Trainer and engine read
the same flags, so they always run the same mode.

| | CPR | Recurrent |
| --- | --- | --- |
| Select with | `ISOEXEC_GDN_KERNEL=cpr` (the trainer environment's default) | `ISOEXEC_GDN_KERNEL=recurrent` and `ISOEXEC_GDN_NATIVE_STATE=1` |
| Core package on sm90 (`gdn.core` impl) | `ops/gdn/gdn_cpr_sm90` (`cuda_cpr_v1`) | `ops/gdn/gdn_recurrent_sm90` (`fla_native_recurrent_v1`), also when the flag is unset |
| Core package on sm100 | `ops/gdn/gdn_cpr_sm100` (`sm100_cpr_v1`) | `ops/gdn/gdn_recurrent_sm100` (`fla_native_recurrent_v1`) |
| Trainer forward | chunk-parallel CUDA kernel | vLLM's fused recurrent Triton kernel, one token after another |
| Trainer backward | FLA's chunked backward | FLA's chunked backward (through an eager rebuild of l2norm and gating) |
| Engine state | IsoExec's private pool, sized by `max_num_seqs` and kept outside vLLM's KV cache; `ISOEXEC_GDN_CPR_MIN_PAGES=1` shrinks vLLM's mamba pages to slot ids | vLLM's own mamba cache pages (fp32), allocated, evicted and slept by vLLM |
| Around training | the pool is discarded with the engine's KV cache by CPR sleep (`ISOEXEC_GDN_CPR_SLEEP=1`) | nothing beyond vLLM's own sleep |
| Chunked prefill | with `ISOEXEC_GDN_CHUNKED_PREFILL=1` | always available |
| vLLM runner and CUDA graphs | Model Runner V2: `FULL_AND_PIECEWISE` (default), `FULL_DECODE_ONLY` or `NONE` | Model Runner V2: `FULL_AND_PIECEWISE` (default), `FULL_DECODE_ONLY` or `NONE` |

## Choosing

CPR's trainer forward is chunk-parallel, where the recurrent forward scans one token after another, so
CPR is the cheaper trainer forward: at 10k tokens one Qwen3.6-27B GDN layer takes about 0.8 ms forward
under CPR on B200 and about 15 ms under the recurrent core. Use CPR unless you need something only the
recurrent mode offers.

The recurrent mode keeps nothing outside vLLM's KV cache, so its state memory comes out of the KV budget
and vLLM sizes it for you. The CPR pool is allocated beside the KV cache at engine build, one row per
`max_num_seqs`, and so is extra memory to budget for (see below). Choose the recurrent mode when that
memory is not available.

## How the engine runs a CPR pool

vLLM gives every request a mamba block id, and the CPR pool maps each id to one of its rows. The
engine's request lifecycle owns that map: a row is claimed when a request first prefills and is returned
when vLLM retires the request (finished, aborted or preempted). On Model Runner V2 the release happens
where V2 retires requests, at the head of every step, including steps that schedule no tokens; a
preempted request that comes back is prefilled again from token zero into a fresh row. Warmup and CUDA
graph capture run dummy batches on vLLM's null block, which resolve to the pool's null row and never
claim a row. The pool never evicts a row a live request owns, so `max_num_seqs` live requests always fit.

A step's prefill rows are claimed once for every GDN layer before the forward, so the layers make no
host-to-device copies of their own for them.

Under `FULL_AND_PIECEWISE` the whole GDN core of each layer (conv, state read and write, the
decode/prefill split) is an eager break between captured graph segments, so a mixed prefill/decode step
replays the captured segments and runs the core on that step's own metadata. Pure decode steps replay a
FULL graph.

With chunked prefill, a continuation chunk resumes from the request's row, which holds the GDN state,
the conv window and the open chunk exactly as a decode at that position would. The attention core's
segment boundaries are absolute, so a prompt split across steps, or prefilled beside decodes, gives the
trainer's rows.

With prefix caching, a request whose prompt shares a prefix with an earlier one resumes from a boundary
checkpoint of exactly that prefix (`runtimes/vllm/cpr_apc.py`). CPR's state is not in vLLM's mamba blocks,
so the private pool checkpoints itself: at a multiple of 64 a record's open chunk is empty and its state is
the encoded boundary state plus the conv window and position. At the start of each step every worker
copies that checkpoint for each request whose previous chunk ended on the checkpoint grid (lcm of 64 and
vLLM's mamba block size, which the GDN layer makes a multiple of 64), keyed by a digest of the request's
first tokens, and lists the key in a per-worker index under `/dev/shm`. The scheduler keeps vLLM's
prefix-cache hit of `h` tokens only if `h` is a multiple of 64 and every worker of the engine holds the
key of the request's first `h` tokens; otherwise it takes vLLM's next shorter hit, down to none, and the
request recomputes. The admitted request then takes a fresh row with the checkpoint installed and continues
as a chunked-prefill continuation. A checkpoint is removed from the index before its memory is released
(16 steps later), so an admitted hit always finds it; a reset of vLLM's prefix cache (verl resets at every
wake) starts a new epoch, and every sleep drops the store. The store is bounded by
`ISOEXEC_GDN_CPR_APC_MB` (default 1024 MiB per GPU, outside `gpu_memory_utilization`); a checkpoint is
`GDN layers x (local value heads x 67584 B + local conv channels x 3 x 2 B)`, 38 MiB for Qwen3.8-27B at TP4.
Prompts given as embeddings, multimodal prompts and LoRA requests are never served from it.

Sleep keeps vLLM's own behaviour for both levels (its offload tags, backend state and weights) with the
pool hidden from it: the pool is never copied to the host, and the engine starts from an empty pool
after a wake.

## Running CPR

The trainer environment's defaults select CPR. For the engine, export (trainer and engine processes
inherit them):

```sh
export ISOEXEC_GDN_KERNEL=cpr ISOEXEC_GDN_NATIVE_KERNELS=1 \
  ISOEXEC_GDN_CHUNKED_PREFILL=1 ISOEXEC_GDN_CPR_MIN_PAGES=1 ISOEXEC_GDN_CPR_SLEEP=1
```

`ISOEXEC_GDN_CHUNKED_PREFILL=1` is needed only with chunked prefill on; prefix caching also needs
`ISOEXEC_GDN_CPR_APC=shm` (the boundary store). `register_native` and the verl integration set both when the
caller turns the features on (verl's defaults are on; the verl integration forwards the flags to every Ray
worker), and refuse a flag the caller set to `0`. With SkyRL the engines run Model Runner V2 with
`FULL_AND_PIECEWISE` graphs unless
`generator.inference_engine.engine_init_kwargs.compilation_config.cudagraph_mode` selects
`FULL_DECODE_ONLY` (mixed batches eager) or `NONE`. Chunked prefill is an explicit
opt-in, `generator.inference_engine.engine_init_kwargs.enable_chunked_prefill=true` (SkyRL's own
top-level `enable_chunked_prefill` is not read). For example (SkyRL overrides):

```text
generator.inference_engine.engine_init_kwargs.compilation_config={"mode": 0, "cudagraph_mode": "FULL_AND_PIECEWISE"}
generator.inference_engine.engine_init_kwargs.enable_chunked_prefill=true
generator.inference_engine.max_num_batched_tokens=8192
```

### Pool memory

Each pool row holds, per GDN layer, one record per local value head (the fp32 state, the fp32 boundary
state and the open chunk's operands; `ops.record_layout()["bytes"]` of the CPR package, 168704 bytes on
sm100, 137984 bytes on sm90, whose record (format 3) stores the fp32 state exactly as its bf16 image -- what
decode reads -- plus the low 16 bits and one tie bit per element)
and the bf16 conv window. Per GPU the pool is about

```text
(max_num_seqs + 1) x GDN layers x (local value heads x record bytes + local conv channels x 3 x 2 B)
```

For Qwen3.6-27B on B200 (48 GDN layers, 48 value heads, 10240 conv channels) that is 0.37 GiB per
sequence at TP1 and 0.18 GiB per sequence at TP2, so `max_num_seqs=256` at TP2 takes 47 GiB per GPU. It
fits beside the 25 GiB of TP2 weights and a 48 GiB KV cache on a B200 (about 120 GiB of its 179 GiB).
With `ISOEXEC_GDN_CPR_MIN_PAGES=1` the KV budget goes almost entirely to attention, since the mamba pages
shrink to slot ids. For Qwen3.8-27B on H100 at TP4 (48 GDN layers, 12 local value heads, 2560 local conv
channels) the sm90 pool is 76.5 MiB per sequence, so `max_num_seqs=256` takes about 19.2 GiB per GPU; the pool
is allocated before vLLM profiles memory, so it comes out of the `gpu_memory_utilization` budget.

### Decode split

The sm90 decode kernel splits each value head over 1, 2, 4 or 8 CTAs; every split gives the same bits. When a
pool binds, the engine measures each split at a grid of batch sizes on the device and uses the fastest
(`ISOEXEC_GDN_CPR_DECODE_TUNE=1`, the default; `0` keeps a shape heuristic). It is side-local: ranks and
processes may pick differently.

### Async scheduling

On Model Runner V2 the CPR pool is exact under async scheduling: the lifecycle driver reads only the
scheduler's host output for the step it prepares (request ids, block-table slot ids, scheduled and computed
token counts, finished and preempted ids, which V2 retires in `finish_requests` first), and every pool
mutation is enqueued on the compute stream after the previous step's kernels. The verl integration turns
`async_scheduling` on for CPR on V2.

## Running the recurrent mode

Export the two flags for the whole job (trainer and engine processes inherit them):

```sh
export ISOEXEC_GDN_KERNEL=recurrent ISOEXEC_GDN_NATIVE_STATE=1
```

Runner selection and chunked prefill work as for CPR, with no extra flag: the recurrent composition
resumes a split prompt exactly from vLLM's state pages. The vLLM mamba cache must be fp32, which
`register_native` sets.

## Context parallelism

Under trainer context parallelism GDN takes the form of the model's CP layout. In the `contiguous`
layout (sm90) rank `r` starts the native recurrent core from the fp32 state and conv tail rank `r-1`
ended with, so it needs `ISOEXEC_GDN_KERNEL=recurrent` (a CPR hand-off is not implemented and is refused
by name). In the `zigzag` layout (CP=2) GDN re-cuts the token cut by heads (`collectives.cp_heads`): one
all-to-all gives every CP rank the whole packed sequence for `1/cp` of this TP rank's heads, the selected
core runs unchanged, and a second all-to-all returns the rank's own tokens. See
[Trainer context parallelism](context-parallelism.md), which also covers attention, MoE and context
parallelism together with expert parallelism.

## Refused combinations

The selected package checks the flags when the engine is built and names every conflict:

- CPR with `ISOEXEC_GDN_NATIVE_STATE=1`: the CPR pool is private; state in vLLM's pages is the recurrent
  mode's.
- CPR with chunked prefill but without `ISOEXEC_GDN_CHUNKED_PREFILL=1`; CPR with prefix caching but without
  `ISOEXEC_GDN_CPR_APC` (and `ISOEXEC_GDN_CHUNKED_PREFILL=1`); `ISOEXEC_GDN_CPR_APC` on the sm100 CPR package
  (no boundary checkpoints there yet); CPR prefix caching with a KV connector.
- Recurrent without `ISOEXEC_GDN_NATIVE_STATE=1`: the recurrent core keeps its state in vLLM's pages only.
- Recurrent with `ISOEXEC_GDN_NATIVE_KERNELS=0`, `ISOEXEC_GDN_CPR_MIN_PAGES=1` or `ISOEXEC_GDN_CPR_APC`.
