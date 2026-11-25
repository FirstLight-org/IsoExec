# FlashAttention-4 SM100 backward: source provenance

This private package contains copied upstream source, not an independent IsoExec
implementation. It is FlashAttention-4's CuTe DSL backward for Blackwell (SM100)
and the QuACK helper modules it imports. `isoexec/ops/attention/gqa_sm100` runs it
as the backward of its own forward. No separately installed `flash_attn`,
`flash-attn-4` or `quack-kernels` package is imported.

| Source | Revision | Retained license |
| --- | --- | --- |
| [Dao-AILab FlashAttention](https://github.com/Dao-AILab/flash-attention) (`flash_attn/cute`, FlashAttention-4) | `0bbb25a3a5ad3c58c029b3d287d6c9af56a5cad5` | [BSD-3-Clause](licenses/flash_attention.txt), contributors in [AUTHORS](licenses/flash_attention_AUTHORS.txt) |
| [Dao-AILab QuACK](https://github.com/Dao-AILab/quack) (`quack/`) | `25b4b879b66cbab75ca127c5f5b1c3d84c84aadb` (tag `v0.5.0`) | [Apache-2.0](licenses/quack.txt) |

`upstream_manifest.json` maps every copied file to its source, upstream path,
original SHA-256 and local SHA-256. Modified entries say what changed, and an
import-rewritten file lists each rewritten line (`import_rewrites`), so
`isoexec/ops/tests/test_fa4_vendor_provenance_cpu.py` re-derives the upstream bytes
from the copy. The original copyright lines stay at the top of every file; a
modified file carries one `# Modified for IsoExec:` line directly after them. The
full license texts are in `licenses/`.

## What was copied

The transitive import closure of FA4's SM100 backward entry points, and nothing
else: `flash_bwd_sm100.py` (head dims up to 128, and 192/128), the dedicated
head-dim-256 two-CTA backward (`sm100_hd256_2cta_fmha_backward*.py`),
`flash_bwd_preprocess.py`, `flash_bwd_postprocess.py`, and the FA4 and QuACK
modules they import. Modules the closure imports but the SM100 varlen path never
traces (block sparsity, the SM80 helpers) are copied unmodified so the imports
resolve. FA4's forward, its other architectures and the rest of `interface.py`
are not copied.

## Modifications

* Every `flash_attn.cute.*` and `quack.*` import is rewritten to a relative import
  of this package. No statement other than imports changes in those files, except
  for the CuTe DSL 4.8 edits below.
* CuTe DSL 4.8 compatibility. `nvidia-cutlass-dsl` 4.8.0 removed APIs that were
  deprecated in 4.5.2, and this copy predates upstream's own replacements. Each
  removed name is replaced the way upstream later replaced it, and every edited
  line is listed in the manifest (`cute_dsl_rewrites`, `[upstream, local]` pairs):
  * `cute.core.ThrMma` -> `cute.ThrMma` and `cute.make_fragment` ->
    `cute.make_rmem_tensor` (FlashAttention
    `eaf806d78fa9e18d9aadc323a77fbabe2c539331`, "replace deprecated apis",
    #2602): `flash_bwd_sm100.py`, `flash_bwd_postprocess.py`, `softmax.py`,
    `utils.py`, `block_sparse_utils.py`.
  * `cute.core.ThrCopy` -> `cute.ThrCopy` (QuACK
    `3bc56bf228f9585051810b84a83e40afdfcb23ea`, "replace deprecated apis", #147):
    `quack/copy_utils.py`, `quack/layout_utils.py`.
  * `vector.extractelement` / `vector.insertelement` with a constant position ->
    `vector.extract` / `vector.insert` with `static_position` (QuACK
    `f7a9d860931f0806b1c364857bb5a05b99d8abf0`, "Fix compatibility issues with
    CuTe DSL 4.6.0+", #151): `quack/rounding.py`, `quack/utils.py`. The SM100
    backward does not trace these two functions; they are fixed so that no
    copied module calls an API 4.8 lacks.

  In 4.5.2 `cute.make_fragment` is a deprecated wrapper that returns
  `cute.make_rmem_tensor(...)`, and `cute.core.ThrMma` / `cute.core.ThrCopy` are
  deprecated subclasses of `cute.ThrMma` / `cute.ThrCopy` used here only as type
  annotations. The backward's dQ/dK/dV (and the forward output) are bit-identical
  with and without these edits under 4.5.2, and under 4.8.0 with the edits equal
  the 4.5.2 bits, at head dims 128 and 256 (see the pull request that made them).
* `interface_bwd_sm100.py` is extracted from `flash_attn/cute/interface.py`: the
  SM100 varlen branch of `_flash_attn_bwd`, with `_bwd_preprocess`,
  `_bwd_postprocess_convert` and their fake-tensor compile helpers. Other
  architectures, the forward, `score_mod`/`mask_mod`, block sparsity, softcap,
  `dlse`, `seqused` and the on-disk JIT cache are removed; compiled kernels are
  cached in process-local dicts; the 2-CTA choice is the `disable_2cta` argument
  instead of the `FA_DISABLE_2CTA` environment variable. Kernel selection, tile
  sizes, buffer layouts and the launch sequence are upstream's.
* Context-parallel tile selection (IsoExec, `gqa_sm100/context_parallel.py`). The head-dim-256 dQ and
  dK/dV kernels take an optional int32 tile mask (`q_tile_mask` per 256-row dQ tile, `kv_tile_mask` per
  128-key CTA-pair tile); a masked-off tile has no work, every warp role reads the same predicate, and a
  kept tile runs exactly the unmasked program. `flash_attn_varlen_bwd_sm100` takes the masks and optional
  precomputed statistics (`stats`), and `backward_stats_sm100` runs the preprocess alone. Every edited
  line is listed in the manifest (`isoexec_edits`, `[line, upstream, local]`).
* `__init__.py` and `quack/__init__.py` are authored for IsoExec. Upstream's
  `quack/__init__.py` imports unrelated kernels and installs process-wide CuTe DSL
  patches (`quack.dsl`); the copied modules need neither.

## Why QuACK v0.5.0

The copy was taken when the release pinned `nvidia-cutlass-dsl==4.5.2`. QuACK
`v0.5.0` is the newest QuACK release whose declared range
(`nvidia-cutlass-dsl>=4.5.2`) admits that pin, and FA4 at the commit above
declares `quack-kernels>=0.4.0`. Later QuACK revisions, including
`bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8` (`v0.6.5-4`), require
`nvidia-cutlass-dsl>=4.7` and import modules 4.5.2 does not have
(`cutlass._mlir_helpers`, `cutlass.experimental`). FA4 at this commit does not run
on 4.7.1 either (`cutlass.cute.core.ThrMma` is gone there).

The release now installs `nvidia-cutlass-dsl==4.8.0` (vLLM 0.33's pin; see
`environments/default.toml`). No upstream pair was re-vendored for it: the first
FlashAttention revision whose declared range admits 4.8.0 is
`0251105a2fb19d2957484b7f023cd8c115286ced` (`>=4.6.2`, 2026-08-15), 39
`flash_attn/cute` commits after the one above, which change the SM100 backward
(`flash_bwd_sm100.py` +/-428 lines, the head-dim-256 driver +/-103); on `main` the
dedicated head-dim-256 backward is deleted;
and the only QuACK releases that admit 4.8.0 are `v0.6.5` and later (`>=4.7`;
`v0.5.1`-`v0.6.4` pin 4.6.x). Re-vendoring would change the backward this package
was qualified with, so the copy stays at the revisions above with the CuTe DSL 4.8
edits listed under Modifications.

## Runtime dependencies

Not vendored, and required at run time on the trainer of an SM100 host:

* `nvidia-cutlass-dsl` 4.8.0 (CuTe DSL), the release's pin. The copy also still
  compiles under 4.5.2, where it was first qualified; there the files must be a
  consistent CUDA 13 set: `nvidia-cutlass-dsl-libs-cu13==4.5.2` and
  `nvidia-cutlass-dsl-libs-base==4.5.2` write 99 of the same paths with different
  contents (the compiler library, the generated MLIR dialect bindings, the
  TVM-FFI provider); a mixture of the two cannot compile this backward, and
  `gqa_sm100.ops.require_consistent_cute_dsl()` refuses one.
* `apache-tvm-ffi` (0.1.x): kernels are compiled with `--enable-tvm-ffi`.
* PyTorch and the CUDA driver.

Compiled kernels live in process memory; nothing is written into the source tree.

## Determinism

At head dim 128 the backward runs `FlashAttentionBackwardSm100` with
`deterministic=True`: dQ, and dK/dV when several query heads share a KV head,
accumulate through `cp.reduce.async.bulk add.f32` in a semaphore-ordered sequence.
At head dim 256 FA4 runs its dedicated two-CTA kernels, which have no
`deterministic` flag and no global float reduction: each dQ tile and each dK/dV
tile is owned by one CTA of a two-CTA cluster, which loops over the KV tiles (dQ)
or over the query tiles and the KV head's query heads (dK/dV) and stores once.
Both are bitwise repeatable per layout
(`isoexec/ops/attention/gqa_sm100/tests/test_ops_gpu.py`); neither is invariant to
how heads are split across tensor-parallel ranks.
