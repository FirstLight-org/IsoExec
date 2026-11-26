# GLM KDA on SM100, recurrent-everywhere recipe (`sm100_kda_rec_v1`)

On this tree: the per-token reducer of `isoexec/ops/kda/schedule.py` (`TOKEN`, block 1, rule 1 of the schedule policy:
segments of one token, one continued chain). It is not the default. A deployment selects it explicitly with
`schedule.select(schedule.TOKEN)` and uses it for engine and trainer alike; its manifest differs from the chunkwise
reducer's (`schedule.manifest`), and each engine refuses the other's state pool. The vLLM / SGLang selection keys and
the adapter tests named below belong to the tree the package was written on and are not present here.

Companion of `kda/glm_sm100` (CPR recipe `sm100_kda_v1`, the default and frozen). Prefill, decode and the trainer forward are the SAME per-token
fp32 update; the state is only vLLM's regular KDA cache (fp32 `[slots, H, 128 (v), 128 (k)]` + the conv window). `csrc/recipe.cuh` is the contract;
design and plans: `/var/tmp/isoexec-kernel-parity-20261014/kda-recurrent/DESIGN.md`.

```python
from isoexec.ops.kda.glm_rec_sm100 import ops, engine
o, ht = ops.forward(q, k, v, g, beta, cu_seqlens, h0)               # trainer forward (packed [T,H,128]; h0/ht [N,H,128,128] v-first)
o = ops.prefill(q, k, v, g, beta, cu, state, slot_in, slot_out)     # engine prefill (slot_in < 0: fresh, +0 state)
o = ops.decode_step(q, k, v, g, beta, state, slots, null_slot=0)    # one token per sequence, in place; null lanes -> zeros, state untouched
o = ops.decode_step_fused(q, k, v, f, b, dt_bias, A_log, lb, state, slots, null_slot=0)   # gate + beta in the kernel (same bits)
```
Selection is explicit: vLLM `additional_config={"isoexec": {"kda_recipe": "sm100_kda_rec_v1"}}`; SGLang model override `isoexec_kda_recipe` (parallel/engine programs host only); default `sm100_kda_v1`.
State ownership: the engine's (vLLM MambaSpec via `kda_state_shape`; SGLang's native mamba/hybrid pool: `mamba2_layer_cache`, slots from `get_mamba_indices`). No Iso pool, record or reset kernel.

## Recipe (summary)
R128 = every dot product: 32 partials (4-element fma chains, channel c = 32m + 4s + e) -> Q_s = (P0+P1)+(P2+P3) -> adjacent-pair tree over s. Token:
k^ = k rsqrt(R128(k,k)+1e-6), q^ = q rsqrt(R128(q,q)+1e-6) 128^-1/2, a = ex2(g log2e); A = S a; kS = R128(A,k^); d = beta (v - kS);
S' = fma(d, k^, A); o = R128(S', q^) -> bf16. T tokens = this T times. 8/16/32 lanes per column and any columns per CTA are bit-identical mappings.

## Gates (tests/, all bitwise unless noted; 35 GPU/CPU tests)
split anywhere + decode == whole sequence (outputs and state bytes); row-alone / batch composition / order / size; one-token prefill == decode; head shards == full (TP);
null/negative lanes touch nothing; CUDA graph == eager; run-to-run; every launch shape and the packed fp32x2 path == scalar reference (incl. subnormals);
fused gate/beta == unfused == glm_sm100's ops; SASS: no ATOM/RED; engine layer: TP 1/2/4/8 invariant, continuous batching == whole sequence, graph == eager,
strided pages; adapter seams (fresh singleton slot, captured decode over changed slots/positions). Tolerance: float64 recurrence (o rel-L2 < 4e-3, state < 1e-5),
CPR recipe (o < 1e-2). Same bits under torch 2.13 (SkyRL venv) and 2.14 (.venv-sm100).

## Performance (B200, H = 32 heads = TP2; scripts and raw json in kda-recurrent/bench/)
Decode, L2 flushed, us/step: B=1/4/16/64/256/512/1024 rec 4.1/6.5/15.7/51/167/327/651; vLLM native fused_recurrent 4.0/7.0/16.2/52/190/372/740; FLA 8.5/10.7/20/54/188/381/768.
Prefill, ms (rec | FlashKDA (vLLM native GLM prefill) | Triton chunk_kda | FLA fused_recurrent), one sequence, H=32:
T=256 0.13|0.05|0.65|0.41; 2K 0.91|0.15|0.68|3.8; 8K 3.6|0.49|1.55|16.0; 32K 14.3|1.8|5.3|64.1 (H=64: 32K 14.7|2.1|9.5|66.2).
Batched: 64x256 tokens 2.5|0.55|2.8|6.1; 256x256 9.8|3.0|10.5|23.0; 16x2048 5.6|0.84|4.9|14.2. (Before the warp-specialized prefill kernel: 32K 33.6 ms.)
Decode kernel = the flat one-token kernel; prefill kernel = warp-specialized (below). Same arithmetic, bit-identical (tested).

## Prefill kernel (execution only, bits unchanged)
Sequential over tokens, parallel over heads, sequences and value columns, the fp32 state of a CTA's columns in registers. CTA = consumer warps
(column groups x 8 lanes) + 2 producer warps (token parity): producers cp.async each token's rows (prefetch distance 3), prepare k^, q^, a, beta ONCE per CTA
(not per column group) and publish them through a 4-stage shared-memory ring with named FULL/EMPTY barriers; consumers only run the update. Measured per
token (clock64, one head): 1.5k cycles -> 0.6k. Shapes by work: N*H >= 64 -> (8 lanes, 4 columns/lane group, 64 columns/CTA), else (8, 2, 32). Packed FFMA2 instructions
(bit-identical, `ops.PACKED`) and a reformulated output (o = dq + delta kq) were tried and are not faster.

## Backward (csrc/bwd.cu, `ops.forward_train` / `ops.backward` / `ops.core`; engine layer under autograd)
Deterministic, no atomics, may depend on T. `forward_train` is `forward` plus the fp32 state entering every 64th token (exec-only stores; o and state bitwise `forward`'s).
`backward`: CTA = (sequence, head), 512 threads (channel x 4 column groups of 32); per 64-token block, last to first: restore the checkpoint, re-run the block writing each
token's entering state to a per-CTA scratch (persistent grid), then walk the tokens in reverse with the adjoint state in registers (equations in DESIGN.md section 6);
every cross-thread sum is a fixed warp-shuffle tree + fixed-order shared-memory combine. Gates (tests/test_bwd_gpu.py, test_engine_rec_gpu.py): float64 gradients (dq/dk/dv/dbeta
rel-L2 1.6e-3 = bf16 rounding of the outputs, dg and dh0 < 1e-5), run-to-run and CTA-schedule bitwise, sequence-alone == in-batch, full layer gradients (input and every parameter)
vs the float64 equation across a checkpoint boundary, SASS without atomics. Speed: T=8192, H=32: 107 ms (13 us/token) -- correctness first; not tuned.

## Not built
Speculative decode / MTP slots, mamba prefix caching (refused by the adapters), distributed (TP>1) backward of the layer, backward speed tuning.
