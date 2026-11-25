# `tp_exchange.cu`: provenance

`gather_cols` (the pull kernel) is IsoExec-authored. `lamport_gather_cols` (the multicast push kernel) and
`lamport_probe` are derived from vLLM's MNNVL Lamport all-gather; the rest of this note is about them.

| Source | Revision | Licence |
| --- | --- | --- |
| vLLM `csrc/custom_all_gather_reduce_scatter.cuh` (`mnnvl_lamport_all_gather`, `store_multimem_lamport_payload`, `load_lamport_pack`, `lamport_cta_arrive`, `mnnvl_lamport_dirty_stage` / `_next_stage`), `csrc/custom_collective_common.cuh` (packed 16-byte vectors), host launch in `csrc/custom_all_gather_reduce_scatter.cu` (`CustomAllreduce::mnnvl_lamport_allgather`: 128 threads, PDL launch attribute) and the buffer set-up in `vllm/distributed/device_communicators/custom_all_reduce.py` (`_init_mnnvl_buffer`: torch symmetric memory, `handle.multicast_ptr`, 3 stages) | files as of commit `d79b52b18219ecbaba780f72ae0a4aa0546ac00f` (2026-10-12; the Lamport kernel last changed in `a047e2543da5`, "Fix MNNVL Lamport mailbox publication and cleanup", #53000), read in a tree at `v0.33.0rc0-910-g437c8dfe09`; sha256 of the .cuh `5100cad1...b100c3`, common header `42ea189b...ffadb9`, .cu `ab516059...037993` | Apache-2.0, `licenses/vllm-LICENSE.txt` (vLLM's `LICENSE`, sha256 `c71d239d...0ab4`) |

No vLLM file is copied verbatim; the kernel was rewritten from the one above, so the vLLM code is not imported or
linked at run time (nothing outside this directory and torch is needed to build it).

## Kept from upstream
NVLink SHARP multicast push (`multimem.st.relaxed.sys.global.v4.f32` through the multicast address of a torch
symmetric-memory region, one store per 16 bytes, every rank receiving in its own memory); the Lamport receive (spin on
the stored words, no flag round trip); three rotating stages with the previous call's stage cleaned during the current
call; device-resident stage state advanced once per call by the grid; PDL (`griddepcontrol.wait` before the first
access, `griddepcontrol.launch_dependents` after the publish; `cudaLaunchAttributeProgrammaticStreamSerialization`
at launch); 128 threads a block.

## Changes, each with the reason
1. **No payload sentinel** (byte-exactness). Upstream marks unwritten 32-bit words with `0x80000000` and rewrites
   payload words equal to it (`sanitize_lamport_payload`), changing bits. Here every 32-bit word on the wire is
   `(tag << 16) | unit`, `unit` one 16-bit payload unit, `tag` a per-call value that is never 0; cleaned words are 0.
   Every payload bit pattern travels unchanged; wire bytes are 2x the payload.
2. **Tags are normal f32 words** (byte-exactness): `multimem.st` has only floating vector forms; every tagged word has
   an exponent field in [1, 254], so no float path can treat it as NaN / subnormal. 65024 tags cycle.
3. **Column layout** (the exchange's contract): rank r's vector (row i, column vector c) goes to `out[i, r C + c]`
   (`NcclExchange`'s all-gather + reorder), not upstream's rank-major result.
4. **Grid-stride, capped grid, publish-all-then-wait** (no dependence on block scheduling order): upstream launches one
   thread per vector.
5. **Stage advance by the last-arriving block** (`atom.add.acq_rel.gpu`), not by block 0 spinning on a counter.
6. **Bounded waits** (never hang): `%globaltimer` bounds every spin; on expiry the first failing thread records
   (rank, stage, call, tag counter, tag, block, thread, source rank, vector, words seen) in a mapped host record and
   prints it, then the kernel traps.
7. **Own slot neither read nor cleaned** (the own copy comes from the input).
8. **Construction probe** (`lamport_probe`): every rank stores marks through the multicast address and every rank
   checks them in its unicast alias before the region is used; a mismatch is a named refusal.
9. Element type: a byte copy of 16-bit units (any dtype whose row is a multiple of 16 bytes), no arithmetic.

## Re-vendoring
Diff the upstream files listed above against the recorded commit; port protocol changes into `lamport_gather_cols`
keeping changes 1-8; update the commit, hashes and this list.
