# Trainer context parallelism

IsoExec's Megatron trainer runs `context_parallel_size` > 1 on the native model, alone or together
with tensor, pipeline and expert parallelism. The forward is bitwise invariant to the cut: every row a
CP rank computes equals the CP=1 row, so rollout and trainer logprobs agree at any CP degree the trainer
admits. Input gradients of every row are CP=1's as well. Weight gradients are sums over tokens, and a
different token split sums them in a different order, so they are not bitwise across layouts.

The rollout engine does not run context parallelism for IsoExec models: vLLM PCP/DCP is refused before
any worker starts (`runtimes/vllm/validation.py`, `register_native`, verl's rollout validator). Trainer
CP=k against an engine without CP gives the same bits as CP=1.

## Two row layouts, one per model

A model binds exactly one CP row layout (`runtimes/context_parallel.select_cp_layout`), and the host
must shard its packed batches the same way:

| Layout | Degrees | Rank `r` holds | Where |
| --- | --- | --- | --- |
| `contiguous` | any | rows `[r*T/cp, (r+1)*T/cp)` of the packed stream; sequence boundaries fall anywhere | every attention core has a varlen query/key form (`trainer_core_qk`) and every GDN core a state hand-off (`gdn_core_stateful`): sm90 today |
| `zigzag` | 2 | Megatron's load-balanced cut: each sequence padded to a multiple of `2 * cp`, split into `2 * cp` chunks, rank `r` holds chunks `r` and `2 * cp - 1 - r` | any arch (the gqa_sm100 owner-tile core on sm100, the generic gathered core elsewhere) |

Both layouts hold the same number of rows per rank, so a host that cuts one way while the model binds
the other would run on the wrong tokens without a shape error. The layout is therefore declared, never
inferred:

- **verl** follows the model: `model.cp_layout` is `contiguous` on sm90 (`integrations/verl/thd.py` packs
  the rows with verl's own contiguous semantics) and `zigzag` otherwise.
- **SkyRL** packs and scores Megatron's zigzag rows itself, so it builds the model with
  `cp_layout="zigzag"` (CP=2, any arch). `select_cp_layout(required=...)` refuses a required layout the
  arch cannot bind.

## What each component does under CP

Each component kind states its context-parallel form (`Component.CONTEXT_PARALLEL`); placement refuses
CP>1 for a kind that states none (MLA's absorbed latent core today).

| Component | `contiguous` | `zigzag` |
| --- | --- | --- |
| Embedding, RMSNorm, dense MLP, LM head | per token, no movement | per token, no movement |
| Softmax attention | each rank receives the KV prefix of the sequences it continues (p2p, `ops/collectives/cp_attention.py`) and runs the varlen core on its own query rows | K/V gathered into the global packed order, each local chunk is one request of the CP=1 forward (`collectives.cp_gather`; on sm100 see the [gqa_sm100 README](../isoexec/ops/attention/gqa_sm100/README.md#3a-context-parallelism-trainer-head-dim-256)) |
| Gated DeltaNet | rank `r` starts from the fp32 recurrent state and conv tail rank `r-1` ended with (`ops/gdn/cp.py`, native recurrent core) | re-cut by heads: every rank runs the whole sequence for `1/cp` of this TP rank's heads (`collectives.cp_heads`, one all-to-all per operand group each way) |
| MoE (routing, experts, shared expert, combine) | per token | per token |

## Context and expert parallelism together

A mixture needs no CP movement of its own. Routing scores one row, the expert kernels compute each row
independently of the batch around it, and the combine folds one token's `k` terms over ascending global
expert ids in fp32. Expert-parallel dispatch (`ops/moe/expert_parallel.py`) sends a rank's own routes to
the owners of their experts and folds the returned terms over that rank's own tokens; it never reads a
token's sequence position. So the ranks of a CP group are expert peers exactly like DP replicas, in
either layout.

Megatron carves the expert mesh `EP x expert-DP` (ETP=1) from the same ranks as the dense
`TP x CP x DP` mesh, and IsoExec admits exactly Megatron's constraint: **EP divides TP x CP x DP**
(`runtimes/parallel.expert_data_parallel_degree`), plus a check that the live expert-DP group has the
implied size. EP need not be a multiple of TP: the dispatch cuts token slices over the TP group inside
each EP group and restores the TP replication afterwards, so TP4/EP2 or TP2/EP3 run with the EP=1 bits.
At CP2 EP2 on two GPUs each rank owns half of the experts and serves the tokens of both CP ranks; at
TP2 CP2 EP2 the two CP peers of a TP rank replicate each expert shard (expert DP 2), and Megatron's DDP
reduces those gradients over the expert-DP group.

Megatron's load-balancing loss (`moe_router_load_balancing_type=aux_loss`, SkyRL) follows the cut: the
mask selects the rank's zigzag rows, the expert counts and the token denominator are reduced over the CP
group (and the TP group under SP), the probabilities stay local, and each rank's objective is its share
of the global one. Every row's router gradient is CP=1's.

## Evidence

- `ops/collectives/tests/test_cp_contiguous_battery_gpu.py`, `cp_contiguous_nccl_gpu.py`: contiguous CP
  at CP=2/4/8 on the sm90 kernels (simulated and real NCCL p2p), forward bitwise vs CP=1.
- `ops/collectives/tests/context_parallel_seams_gpu.py`: the zigzag attention and GDN seams at CP=2 equal
  CP=1 bitwise, rows and gradients (sm100).
- `ops/moe/tests/moe_context_parallel_gpu.py`: the released MoE layer at CP2 EP1 and CP2 EP2 equals CP=1
  bitwise, rows and input gradients (two Gloo ranks on one sm90 or sm100 GPU).
- `ops/moe/tests/ep_matrix_gpu.py`: the EP mesh matrix, including CP x EP layouts.
- CPU: `test_cp_contiguous_cpu.py`, `test_cp_simulated_cpu.py`, `test_cp_dist_transport_cpu.py`,
  `runtimes/megatron/tests/test_native_cp_cpu.py` (layout selection and the host-declared layout),
  `integrations/verl/tests/test_contiguous_thd_cpu.py`, `nn/tests/test_context_parallel_refusal_cpu.py`
  (MoE binds the CP=1 record at CP=2, EP 1 and 2; GDN's head-wise region cuts the TP rank's heads),
  `integrations/skyrl/tests/test_aux_loss_context_parallel_cpu.py`,
  `runtimes/tests/test_expert_data_parallel_cpu.py`, `integrations/skyrl/tests/test_megatron_mesh_cpu.py`,
  `ops/moe/tests/test_ep_mesh_dispatch_cpu.py`, and the engine-CP refusal tests.
