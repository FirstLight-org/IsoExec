# DeepSeek V4.1 Flash text

Use `isoexec.models.deepseek_v4_1_flash` as the model source package with the released
`deepseek-ai/DeepSeek-V4.1-Flash` checkpoint (or a reduced-depth checkpoint, below). The HF
`architectures` entry selects `DeepseekV41ForCausalLM`; importing the package registers the
`deepseek_v41` config type with `AutoConfig`.

## The program: a Causal Encoder-Decoder

DeepSeek V4.1 Flash is the tech report's Causal Encoder-Decoder (CED, Fig. 3): a causal encoder of `L/2`
blocks on every token, then a decoder of `L/2` blocks whose global (compressed sparse) KV is projected from
the encoder output, not computed from the decoder's own hidden states. The definition is written that way:

```text
embed (DeepSeekV41Embed)          token gather + four-stream expand + one-hot pre; Engram n-gram hashes
encoder.blocks.{i} (EncoderBlock) layers [0, b): SWA (0, 1), CSA2 FULL ratio 2 (2, 8, 14) + REUSE; Engram (1, 14)
  -> EncoderOutput                the handoff: streams X_b [T, 4, H] bf16 + the last encoder FFN's pre [T, 4] fp32
decoder.entry (DecoderEntry)      every token: layer b's attention boundary, window KV, ratio-1 compressor
                                  (the decoder global KV) and index keys
decoder.blocks.{j} (DecoderBlock) layers [b, L) on the decoder segments' rows: FULL (query side, candidate
                                  pool), REUSE, REINDEX (masked to the pool); windows truncated at the segment start
norm (DeepSeekV41HeadStage)       last deferred expand, final collapse, final RMSNorm
head (LMHead, logits fp32)        FP32 head root (vocab-parallel under TP)
```

`config.ced_geometry` derives the split (`b = max(kv_source_layer_ids)`, the candidate source; production
`b = 20`) and validates every CED invariant (`validate_ced`: a config that is not an encoder-decoder is
refused by name, `CedError.code`); `CedGeometry.encoder` / `.decoder` hold one `LayerKind` per layer in the
report's terms. Checkpoint names stay `layers.{i}.*`; `weight.stack_name` / `checkpoint_name` map them to the
module paths above (the boundary layer's attention boundary and projection, `weight.ENTRY_PARTS`, belong to
`decoder.entry`).

Two prefill programs are declared (`config.CED_PREFILL_MODES`, part of `program_identity()`, which engine and
trainer compare):

- `bounded_replay` (release): for a turn whose prompt ends at `P` and whose generated tokens end at `E`, the
  decoder runs on the segment `[max(0, P - 128), E)` (Decoder SWA Bounded Replay, report p.20); every decoder
  layer after the boundary truncates its window at the segment start; the boundary layer's window keys are
  token-wise in the handoff and are not truncated. The rule depends on the prompt length only, never on
  scheduler chunks, graph mode or logprob requests. The trainer scores per decoder segment, the segments taken
  from the rollout's turns (`program.DecoderSegments.replay`).
- `full_decoder`: the decoder on every token (the official reference code's program); reference parity only.

State classes (`program.ced_states`; record widths from the cache package's `spec`): encoder window KV per
layer and the three ratio-2 global KV / index-key caches (request lifetime; the pending ratio-2 group crosses
calls); one decoder global KV and index-key cache plus the boundary layer's window keys (`decoder.entry`,
request lifetime, every token); the window KV of decoder layers after the boundary (decoder-segment lifetime:
rebuilt by every replay, never prefix-cached); the segment start; the Engram token history. Top-k, the
candidate pool, mHC carries and routing never cross calls. Positions are absolute and no state assumes one
rank holds a whole sequence (context parallelism is admitted later; `admission.CONTEXT_PARALLEL` until then).

Arithmetic: BF16 weights and activations; FP8 cache records as `port/CACHE-SPEC` (window and compressed KV as
528-byte records, FP8 index keys) with the trainer applying the same quantize/dequantize; index queries are
FP8 as vLLM's engine quantizes them, through the same op on the trainer. The recipe gives the indexer no
gradient (SkyRL's Megatron V4.1 selects top-k under `no_grad` on detached `qr` / `x`, and freezes the indexer
without an indexer loss), so no straight-through estimator is needed for the query quantizer. The MoE rounding
program is the one named `config.MOE_RECIPE` (the official reference's). Vision and DSpark are excluded by
declaration (`CedGeometry.excluded`).

`reference.py` is the CPU statement of the program: the flat reference-code shape, the CED program over packed
segments (the trainer's form) and an incremental engine (chunked prefill, decode, caches), on row-invariant
CPU arithmetic. Its tests show bitwise: `full_decoder` == the flat program; `bounded_replay` == the flat
program with the decoder windows floored at the segment start; trainer per-segment scoring == the engine's
generation (single and multi-turn, packed); every chunk split of a prompt gives the same decoder rows, state
and logits.

On SM100, the trainer runs the decoder entry projection over packed tokens. The engine writes the
same projections into paged KV and index caches. Both execute decoder attention on the rows selected
by bounded replay. These entry points are in `isoexec/ops/attention/deepseek_v41_cache_sm100/ced.py`.
The [six-layer SkyRL run](../../../docs/v41-six-layer-skyrl.md) exercises both paths in one GRPO step.

Kernels (all SM100, `isoexec/ops`): `fused_sm100.embed_expand`; `embed/engram_sm100` (hash, FP8 row
lookup, gated residual); `residual/deepseek_mhc_sm100` (coefficients, fused boundary rows, expand,
collapse); `norms/norms_sm100`; `attention/deepseek_v41_cache_sm100` (window + compressed sparse KV,
compressor, indexer, candidate filter, top-k, dual-rail learned-sink attention, grouped `wo_a`, the fixed
8-leaf `wo_b` tree, BF16-actor backward); `moe/deepseek_v41_sm100` (sqrt-softplus router, clamped-SwiGLU
grouped experts, fixed-order EP exchange with backward, and the same forward on canonical BF16 MegaMoE,
`mega.py`, over vendored DeepGEMM); `collectives/detreduce_sm100` (the TP tree reducer); BF16 projections
on vendored DeepGEMM where a weight's published `mm_sm100` segment count is 1 (`mm/deepgemm_sm100`; under
`mm_sm100` segment policy v4 that is every weight, since S is a function of K only), else on the
batch-invariant `mm` chain with `mm_sm100` on top. Every alternative evaluates the same per-output program, so the choice
between them never moves a bit.

Each block takes and returns explicit carry tensors: the streams `x`, the shifted `pre`, and the deferred
FFN expand `(out, post, comb)` that the next boundary fuses with its collapse and RMSNorm. The cross-row
attention state (a compressed-KV source's `kv`, index keys, `topk`, the candidate blocks) is returned by its
producer and passed to its consumers by the model's forward, so every dependency between rows is in the
captured graph. Decoder rows are the pipeline layers.

Seams (`providers.PROVIDERS[recipe]`), resolved once per block at placement:

- `linear`: one provider per projection kind (`attention`, `indexer`, `engram`): DeepGEMM's BF16 GEMM for
  the weights it serves, the `mm` chain otherwise;
- `moe_forward`: the expert execution per side and EP degree. The trainer runs the grouped BF16 experts
  (custom backward under autograd) and, under EP, the fixed-order exchange with backward. The engine runs
  canonical BF16 MegaMoE: at EP = 1 directly (its workspace is created and warmed on the first eager call),
  at EP > 1 once `prepare_expert_workspace()` has run (collective, after the weights load, outside capture;
  the engine's load hook `prepare_engine_execution()` runs it):
  replicated rows feed their canonical home slice and come back through the fused NVLink gather, with no
  host sync. Without the workspace (or under autograd) the engine runs the trainer's forward, and graph
  buckets switch to static-capacity EP routes (`prepare_static_ep(max_home_rows)` once at load,
  `use_static_ep(enabled)` per forward, no collective);
- the TP reducer: `wo_b`'s FP32 subtree roots on the symmetric-memory tree reducer whenever autograd does
  not track them (rollout, scoring, the trainer's forward), the rank-ordered all-gather otherwise. Placement
  prepares it (collective). The tree runs inside its execution scope: on the engine, `VllmModel` holds it
  around vLLM's graph captures (the capture lifecycle), replays and eager forwards
  (`engine_execution_scopes()`); elsewhere each row enters it around its own reductions in eager steps. A
  capture by a runtime that holds no scope records the all-gather form (the same bits, slower);
- derived weights (BF16/FP32 copies, concatenations, `wo_b` leaves) live in the kernels' per-holder caches;
  `refresh_derived_weights()` re-derives them into their existing storage after a weight update, and
  `stale_derived_weights()` names copies whose sources changed since.

## Parallelism

Every block and `decoder.entry` is placement-bound: engine and trainer bind it with the side's full
`Placement`; `admission.admit` is the one function that admits or refuses a layout (`Refusal.code`). Pipeline
stages may start only where the residual carry alone crosses (`CedGeometry.admissible_pipeline_cuts()`:
production 1, 2, 8, 14, 20 -- the CED handoff is the recipe's PP2 cut -- B14 1, 2, 4, 6, 8); no cut inside the
decoder, whose every layer reads the decoder global KV. A binding declares its stage starts
(`declare_pipeline_stages`) before placement.
TP (1, 2, 4 or 8) splits attention heads (`wq_b` rows, `attn_sink`) and output groups (`wo_a` rows,
`wo_b` columns); `wo_b`'s FP32 subtree roots are gathered in rank order and completed by the same leaf
tree, so every TP degree gives the same bits. The FP32 head computes this rank's contiguous vocab shard
(`linear(h, w[rows], root_fp32=True)`) and the engine gathers the shards in rank order; placement refuses a
TP degree whose shard `mm_sm100` does not serve (under policy v4 a shard has the replicated head's segment
count). The indexer and the other attention tensors are replicated
(their gradients are per-rank partials, `sum_over_tp`). EP holds whole experts in contiguous ranges under
their global `experts.{e}` names (other ranks' experts are remote names); the shared expert runs at home.
EP runs over the TP ranks (replicated tokens, one home per row) or over token-disjoint ranks at TP = 1.

## Checkpoint

`weight.py` maps the checkpoint names to the CED module paths. Each FP8 (E4M3, 32 x 32 UE8M0 blocks) or FP4 (packed E2M1,
1 x 32 UE8M0 blocks) projection is a many-to-one converter of its codes and scales into the BF16 master;
the Engram FP8 rows and scales load byte for byte as checkpoint buffers. The vision tower (and the routers'
`bias_vl`), the DSpark ("MTP") rows and their heads are ignored (`ignored_checkpoint_patterns`). A checkpoint
without a `quantization_config` (the B14 BF16 debug checkpoint) loads its BF16 tensors as stored. Conversion is
load-only.

A reduced-depth checkpoint states its rows in the root config: `"isoexec_subset": {"layer_sources":
[checkpoint layer of each row], "layout": "renamed" | "physical"}` (it must still be a CED whose boundary row
is production layer 20), with the reduced geometry (layers, compression roles, routed experts,
expert channels) in `text_config`. Its rows' checkpoint layers are renamed, the leading routed experts,
expert channels and router rows are kept exactly as stored, and the Engram hash keeps the released layout.

## vLLM engine

The decoder rows of each step come from `engine_ced.plan_step` (positions at or after the request's segment
start, from `req_states.prompt_len`; published by the model state's `prepare_inputs`): encoder and
`decoder.entry` on every scheduled token, the decoder on those rows. Its module doc states the request
lifecycle, the chunk-independence invariant and the capture plan. The trainer side (`training_ced`) takes each
trained sequence's turns from the rollout. The binding below serves the encoder blocks' composition.

Served through the general vLLM path (`register_native`, vLLM 0.33 V2 runner). The model class declares
its engine settings (`engine_settings()`: `kv_cache_dtype="fp8_ds_mla"`, `block_size=128`, chunked
prefill, no global attention backend) and its per-step host state (`engine_model_state_cls()`: vLLM's
`DeepseekV41ModelState`, which gathers the Engram token-history window, plus the checks below).

- Cache memory and per-step metadata are vLLM's own V4.1 ones. `bind_engine_state` registers, under the
  row's `.attn` prefix, the layers native vLLM registers: `swa_cache` (`DeepseekV4SWACache`, 32-token
  blocks of FP8 records), the compressed-KV source (`CompressedKVSource`, native's `MLAAttentionSpec`, FP4
  records), `indexer.k_cache` (`DeepseekV4IndexerCache`) and `compressor.state_cache`
  (`CompressorStateCache`), with native's specs and metadata backends (FlashMLA sparse family), so vLLM
  groups, allocates and shares the pages as for its own model; consumer rows read their source's pages.
- `engine.EngineStep` adapts vLLM's metadata to the kernels, address math only: window rows and main lists
  from vLLM's sliding-window slots, the compressed rail as 64-record pages of a block, index keys as pages
  of `128 / ratio` entries, the compressor state in the request's ring block. Records are the kernels'
  own formats; `tests/test_engine_cache_cpu.py` checks byte-exclusive addressing on vLLM's allocator.
- In the SkyRL experiment, uniform decode steps use full CUDA graphs. Mixed batches use vLLM's
  breakable piecewise graphs with `ISOEXEC_V41_PIECEWISE_BREAKS=core`. The model declares these
  boundaries in `piecewise.py`. A captured decode step scores at most
  `decode_graph_score_element_budget / tokens` context positions.
- After the checkpoint load each row packs its experts, prepares the MegaMoE workspace (EP > 1) and records
  the address of every parameter the kernels read in place (DeepGEMM projections and MegaMoE read the BF16
  masters with no derived copy). Weight updates write in place (`VllmModel.finish_weight_update` refuses an
  update that moved a parameter, re-derives every cached weight copy into its storage and clears the cache
  pages), so captured graphs replay the new weights without recapture; before each captured step the model
  state refuses copies whose sources changed without the update commit. That check walks no module tree:
  `VllmModel.derived_weight_check()` resolves the placement-bound rows once after the load, again after
  every update commit and after any component placement, and each step compares the recorded source
  versions of every derived copy (`DerivedWeightGuard`). Two sampled sweeps cover what a version compare
  cannot see, round robin and failing closed: every `CONTENT_INTERVAL` (8) captured steps one copy is
  re-derived on the device and compared byte for byte (read back at a later step, no wait), and every
  captured step `ADDRESS_SAMPLE` (32) of the parameters the graphs read in place are checked for a
  replacement or a moved storage (`engine_state`).
- Program receipts (`engine.RECEIPTS`, per rank): every forward that runs Python records the TP reducer
  transport, the expert path and the EP routes its rows ran (captured buckets with the weight version at
  capture), and every graph replay records its bucket and the weight version it reads. Every `consistency_check_interval` steps the TP ranks compare
  a fingerprint of the step inputs on the device, read back at the next check (fail closed).
- Per-step input guard (`engine_state.StepGuard`), computed on the device and read back one step later
  without waiting (fail closed): every block id of the step's block-table rows lies inside the cache
  allocations (one multi-tensor max per step; no id is negative, every 64 steps), and (TP > 1, every `DIGEST_INTERVAL` (1) steps) one digest of the token ids, positions and slot
  mappings is equal on every TP rank. The block tables, slot mappings and page ownership are vLLM's own (its
  KV cache manager allocates, frees and maps the pages, and the rows read its per-step metadata through
  `EngineStep`, address math only): the engine keeps no host mirror of the tables, so it has no
  ownership, coverage or mirror-equality check of its own; the addressing is checked against vLLM's
  allocator by `tests/test_engine_cache_cpu.py`.
- The Engram tables stay in host memory and are read through a device mapping of the same bytes; the
  compressed token map comes from the served tokenizer. One copy serves a node: the lowest engine rank of
  each host loads the tables into a shared memory segment (`/dev/shm`, named after the checkpoint, buffer
  layout and stage, unlinked once every co-located rank has mapped it), the other ranks map it read-only, and
  every rank registers it with CUDA. The two released tables are about 196 GB of shared, page-locked host
  memory per node, whatever the number of ranks on it (a reduced-depth checkpoint keeps full-size tables);
  `/dev/shm` must hold them plus 1 GiB. A rank whose mapped segment has another identity fails closed. Each
  rank's registration is released at engine shutdown. The segment's name is removed before the weights
  load (and a segment left by a crashed creator is removed by the next one), so nothing persists after
  the engine exits: the kernel frees the memory when the last mapping goes, on a crash too.

## Megatron trainer

Built by the release factory (`runtimes.megatron.factory.build_model`) from the same package and
checkpoint; the trainer `Placement` binds the same components with their training paths (custom
backward of the sparse attention, mHC, grouped clamped-SwiGLU experts and the fixed-order EP exchange,
Engram residual, FP32 head root).

- Packed sequences: the trainer's packed row (`input_ids [N]`, boundaries in the `SequenceContext`)
  runs row-wise through the embedding, boundaries, Engram, MoE and head. The Engram hash restarts at
  each sequence; the attention runs on the right-padded `[sequences, longest]` view of the rows
  (`training.SequenceRows`, a gather each way) and the cross-row state stays in that view. The
  logits of a sequence are bit for bit the validated trainer's padded batch.
- Trainable: every BF16/FP32 parameter except the indexer and the router's selection biases
  (`training.trainable`); the Engram FP8 tables are frozen checkpoint buffers held in host memory
  and read through a device mapping, as on the engine (`bind_trainer_state`, after loading; each
  trainer rank registers its own host copy).
- Expert packing: the trainer's parameters stay in Megatron's DDP buffers; the grouped expert
  operands are stacked copies, their gradient returns as per-expert slices.
- Gradient reductions: under TP the replicated attention tensors (`sum_over_tp`) get one gradient hook
  each that sums the ranks' partials in rank order (`collectives.ordered_sum`), before DDP
  accumulates; routed experts are `allreduce=False` (expert data parallel), the router and the
  shared expert are completed by the EP exchange's backward. Megatron DDP reduces over DP.
- After an optimizer step `finish_native_optimizer_step` re-derives the components' cached weight
  copies in place (the optimizer writes through the DDP buffers).
- The updated policy streams to the engine through `IsoExecWeightSource` (full logical tensors under
  the engine's registered names, every global expert) into `VllmModel.start_weight_update` /
  `load_weights` / `finish_weight_update`.

The six-layer run used token-disjoint EP4 at TP1. Placement admission rejects sequence and context
parallelism for this model.

## Not yet in this package

- vLLM's `inputs_embeds` paths (`embed_input_ids`: multimodal inputs, prompt embeddings, speculative
  drafts): the embedding and the Engram hash run inside the forward on token ids; none of these paths is
  enabled for this model, and `embed_input_ids` refuses.
- Trainable Engram row masters. The FP8 tables are frozen.

CPU tests cover the CED geometry of the released, B14 and reduced configs and the named refusals of
non-CED layouts, the stacks, layer kinds, handoff ownership and state classes, admission, checkpoint key
coverage against the released tensor index and the B14 BF16 layout, the exact FP8/FP4 decode, loading a
reduced row, placement at TP 1/2 and EP 2, and the CPU reference program's equalities. They are not a GPU
numerical qualification.
