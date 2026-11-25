"""The FLA gated-delta-rule kernels, vendored. `gdn.core`, `gdn.l2norm`, `gdn.gating`.

WHY THIS EXISTS (kernel-ownership.md 1, two tests in order):

1. Do these kernels DETERMINE a rounding schedule the IsoExec contract pins? Yes. The chunked
   delta-rule forward fixes the reduction order of the whole GDN core; `l2norm_fwd` fixes
   `gdn.l2norm`; `fused_sigmoid_gating_delta_rule_update` fixes `gdn.gating`. Every one of them
   appears in a composition entry with a pinned schedule.
2. Is the upstream a released, versioned artefact with an API-stability contract whose schedule we
   fix by the parameters we pass? No. `vllm.model_executor.layers.fla.ops` is an internal module
   of vLLM -- itself a port of flash-linear-attention -- with no stability promise, moved between
   releases, and reachable only by importing engine internals. Test 2 says: vendor it.

The failure mode this closes is not a crash. An upstream patch can change a reduction order and
NOTHING fails loudly: trainer and engine still agree with each other, while no longer agreeing
with the recorded gate evidence. Owning the file makes that change a diff we have to accept.

WHAT IS AND IS NOT HERE

Copied: chunk_delta_h, chunk_o, chunk_scaled_dot_kkt, cumsum, fused_recurrent,
fused_sigmoid_gating, index, l2norm, op, solve_tril, utils, wy_fast -- what `ops/gdn/` calls, plus
`chunk_o`, whose autotuner the GDN pin walks in both trees (`gdn_batch_invariant._FLA_KERNELS`).

NOT copied: `layernorm_guard`. It is the INVERTED direction -- `pin_gdn_rmsnorm_rows_per_block`
rebinds `calc_rows_per_block` on vLLM's OWN module so vLLM's `RMSNormGated` becomes M-invariant.
Vendoring can never remove that import; a copy of it would be dead code that pins nothing. Also
not copied: `kda`, `fused_gdn_prefill_post_conv` -- vLLM calls them, we never do.

DO NOT EDIT the copied files. They are `header + rewrite(upstream)` under the rules recorded in
PROVENANCE.json, and `_provenance.py` pins each file to its recorded vendored_sha256. Changing the numerics means
changing our advertised rounding schedule, which is a digest bump with a canonical diff (D8), not
an edit. Fixing a bug upstream fixed means re-vendoring at a new commit and re-running the gates.
"""
