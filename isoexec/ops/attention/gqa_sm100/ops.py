"""SM100 GQA: the deterministic causal forward over a paged or packed cache, FA4's backward, and the
autograd core the trainer binds.

The forward's numerical program is stated in README.md. Two things in this module are part of the
op's identity and nothing else is:

* ``SEG_TOKENS`` -- the segment span, a table over ``(head_dim, max_context)`` read at the recipe's
  fixed ``spec.SEG_CONTEXT`` (:func:`segment_span`). A change to the table or to that context is a
  recipe version bump.
* ``KEY_TILE`` -- the KV tile width, fixed at 128.

Everything the call chooses beyond those (the leaf, the query-tile wave count, the item granularity,
the grid capacity, whether a row's tree is folded by one worker or published and merged by the
combine, the KV strides) is execution, and ``tests/test_relations_gpu.py`` and
``tests/test_decode_gpu.py`` force the alternatives and require identical bits.

The backward is FlashAttention-4's SM100 backward (Dao-AILab/flash-attention ``flash_attn/cute``,
BSD-3-Clause, Copyright (c) 2025 Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao; vendored at
``spec.FA4_COMMIT`` in ``isoexec/ops/_vendor/flash_attn_cute``, see PROVENANCE.md there), run on
this forward's bf16 output and LSE. At head dim 128 it runs ``FlashAttentionBackwardSm100`` with
``deterministic=True``: dQ (and dK/dV when several query heads share a KV head) accumulate through
``cp.reduce.async.bulk add.f32`` in a semaphore-ordered sequence, so the same inputs give the same
bits on every run. At head dim 256 FA4 runs its dedicated two-CTA kernels, which have no
deterministic flag because they need none: each dQ and dK/dV tile is owned by one CTA of a two-CTA
cluster, which loops over the KV tiles (dQ) or over the query tiles and the KV head's query heads
(dK/dV) and stores once, with no global float reduction. Neither is invariant to the head layout (the tensor-parallel split), and
neither is required to be.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import torch

from isoexec.ops.attention.gqa_sm100 import spec

#: The KV tile width. Numerical: every tile boundary inside a segment is a max/rescale point.
KEY_TILE = 128

#: The head dims this package compiles a leaf for. 128 runs one CTA per work item at one or two
#: query waves; 256 runs a CTA PAIR per item (each CTA owns 128 packed query rows and all 256 output
#: features). Both run the SAME numerical program -- the recipe is head-dim independent -- and the
#: call REFUSES any other head dim rather than reading past a tile.
HEAD_DIMS = (128, 256)

#: Granularity overrides for the relations (execution only): aligned runs of the TMEM stack's reach,
#: one segment per item, or aligned pairs. Each cuts a row's tree only at a node of it.
GRAN_AUTO, GRAN_SINGLE, GRAN_PAIRS = 0, 1, 2

#: Which leaf serves a head-dim-256 call (execution only). ``LEAF_AUTO`` takes the decode leaf
#: (``csrc/decode_d256.cu``) for a decode-shaped call -- ``max_seqlen_q * (h_q / h_kv) <= DECODE_ROWS``
#: and no granularity or run override -- and the general leaf otherwise; the other two force one.
LEAF_AUTO, LEAF_GENERAL, LEAF_DECODE = 0, 1, 2

#: Packed query rows the decode leaf holds per (request, KV head): decode, and speculative decode up
#: to ``DECODE_ROWS / ratio`` tokens per request.
DECODE_ROWS = 16

#: The segment span, per deployment-fixed ``(head_dim, max_context)``. Regret-chosen across decode and
#: prefill on the target shapes; README.md carries the measurements and the reasoning. The rule the
#: table follows is ``n_max = max_context / SEG = 16``, floored at 128 (one KV tile) -- sixteen
#: segments give decode sixteen-way split parallelism per (request, KV head) while keeping a prefill
#: row's tree shallow enough for one worker to fold.
SEG_TOKENS: dict[tuple[int, int], int] = {
    (128, 2048): 128,
    (128, 8192): 512,
    (128, 32768): 2048,
    (128, 131072): 8192,
    (256, 2048): 128,
    (256, 8192): 512,
    (256, 32768): 2048,
    (256, 131072): 8192,
}


def seg_for(head_dim: int, max_context: int) -> int:
    """The deployment-fixed segment span for this geometry, from ``SEG_TOKENS``.

    A context that is not a table key resolves to the smallest key at or above it, so a deployment
    declaring ``max_context = 6000`` gets the 8192 row rather than a span invented per call.
    """
    keys = sorted(c for (d, c) in SEG_TOKENS if d == head_dim)
    if not keys:
        raise ValueError(f"no SEG table row for head_dim={head_dim}")
    for c in keys:
        if max_context <= c:
            return SEG_TOKENS[(head_dim, c)]
    raise ValueError(
        f"max_context={max_context} exceeds the largest SEG table row for head_dim={head_dim} "
        f"({keys[-1]}); the table is part of the op's identity, so extend it deliberately"
    )


def segment_count(border: int, seg: int) -> int:
    """Segments of a row whose causal border is ``border`` keys, anchored at key 0."""
    return 0 if border <= 0 else (border + seg - 1) // seg


def gqa_sm100_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    *,
    softmax_scale: float,
    seg_tokens: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_table: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    granularity: int = GRAN_AUTO,
    waves: int = 0,
    max_run: int = 0,
    leaf: int = LEAF_AUTO,
    decode_split: int = 0,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(out bf16 [T, h_q, D], lse fp32 [T, h_q])`` for packed varlen queries.

    ``k_cache``/``v_cache`` are either a paged cache ``[blocks, page, h_kv, D]`` with ``block_table``
    ``[B, max_blocks]`` and ``page % 128 == 0``, or a packed cache ``[T_kv, h_kv, D]`` with
    ``cu_seqlens_k``. Both addressing modes hand the leaf the same logical keys in the same order, so
    page ids never reach the arithmetic.

    Query row ``i`` of request ``b`` sits at logical position ``seqused_k[b] - q_len_b + i`` and
    attends keys ``[0, position]``. ``seg_tokens`` comes from :func:`seg_for`; it is not a capacity
    knob and a caller that varies it has changed the op.

    ``waves`` is a head-dim-128 execution override and is refused at head dim 256, which runs one
    CTA pair per work item and has no wave count. ``leaf`` (``LEAF_*``) and ``decode_split`` (CTAs
    per (request, KV head) of the decode leaf; 0 chooses) are execution overrides too. ``out``
    (contiguous bf16 with ``q``'s element count) is written in place and returned.
    """
    from isoexec.ops.attention.gqa_sm100._extension import extension

    if seg_tokens <= 0 or seg_tokens % KEY_TILE:
        raise ValueError(
            f"seg_tokens must be a positive multiple of {KEY_TILE} (got {seg_tokens}); it is the "
            "span the canonical tree merges over, not a capacity knob"
        )
    return extension().forward(
        q, k_cache, v_cache, cu_seqlens_q, seqused_k, block_table, cu_seqlens_k,
        float(softmax_scale), int(seg_tokens), int(max_seqlen_q), int(max_seqlen_k),
        int(granularity), int(waves), int(max_run), int(leaf), int(decode_split), out,
    )


def default_scale(head_dim: int) -> float:
    """The scale the trainer and the engine both pass: ``1 / sqrt(head_dim)``."""
    return 1.0 / math.sqrt(head_dim)


def segment_span(head_dim: int) -> int:
    """The recipe's segment span at ``head_dim``: the ``SEG_TOKENS`` row at ``spec.SEG_CONTEXT``."""
    return seg_for(head_dim, spec.SEG_CONTEXT)


@lru_cache(None)
def _is_sm100(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (10, 0)


def _require_sm100(t: torch.Tensor, where: str) -> None:
    index = t.device.index if t.device.index is not None else torch.cuda.current_device()
    if t.device.type != "cuda" or not _is_sm100(index):
        raise RuntimeError(f"{where}: gqa_sm100 runs on SM100 CUDA devices only (got {t.device})")


def _check(q, k, v, where):
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise RuntimeError(f"{where}: bf16 Q/K/V only (got {q.dtype}, {k.dtype}, {v.dtype})")
    d = q.shape[-1]
    if d not in HEAD_DIMS or k.shape[-1] != d or v.shape[-1] != d:
        raise RuntimeError(f"{where}: Q/K/V head dims {q.shape[-1]}/{k.shape[-1]}/{v.shape[-1]}; this package "
                           f"compiles equal head dims {HEAD_DIMS} only")
    hq, hk = q.shape[-2], k.shape[-2]
    if hq % hk:
        raise RuntimeError(f"{where}: {hq} query heads over {hk} kv heads; the ratio must be a whole number")
    _require_sm100(q, where)


#: Ratios that do not divide the 128-row tile but that the decode leaf serves in one launch.
DECODE_ONLY_RATIOS = (3, 6)


def _launch_ratio(ratio: int, decode: bool = False) -> int:
    """The head ratio one launch serves: ``ratio`` when it divides the 128-row tile (at most 16), or on a
    decode-shaped call one of :data:`DECODE_ONLY_RATIOS`; else its largest power-of-two divisor, launched
    once per group of that many heads (:func:`_by_head_groups`)."""
    if (128 % ratio == 0 and ratio <= 16) or (decode and ratio in DECODE_ONLY_RATIOS):
        return ratio
    return min(ratio & -ratio, 16)


def _decode_shaped(q, hk: int, max_seqlen_q: int) -> bool:
    """The call the decode leaf takes under ``LEAF_AUTO`` (host.cu's rule): head dim 256, packed rows <= 16."""
    return q.shape[-1] == 256 and int(max_seqlen_q) * (q.shape[-2] // hk) <= DECODE_ROWS


def _by_head_groups(q, hk: int, out, run, decode: bool = False):
    """Serve a head ratio the tile cannot hold whole as ``ratio / r`` launches of ratio ``r``.

    Qwen3.6-27B's ratio 6 is grouped on prefill calls only; a decode-shaped call (``decode``) runs it in
    one launch of the decode leaf, which reads each KV head once for all six query heads.

    Launch ``g`` takes query heads ``[g*r, (g+1)*r)`` of every KV head, over the same keys. A row's bits do
    not depend on the ratio it was packed with (one row's program is its own QK, softmax and PV chain over
    the absolute segments; tests/test_seam_gpu.py), so these rows are the rows one launch at the full ratio
    would write. Returns ``None`` when one launch serves the ratio.
    """
    T, hq, d = q.shape
    ratio = hq // hk
    r = _launch_ratio(ratio, decode)
    if r == ratio:
        return None
    groups = ratio // r
    if out is None:
        out = q.new_empty(T, hq, d)
    lse = torch.empty(T, hq, device=q.device, dtype=torch.float32)
    qg = q.view(T, hk, groups, r, d)
    og, lg = out.view(T, hk, groups, r, d), lse.view(T, hk, groups, r)
    for g in range(groups):
        o, l = run(qg[:, :, g].reshape(T, hk * r, d).contiguous())
        og[:, :, g] = o.view(T, hk, r, d)
        lg[:, :, g] = l.view(T, hk, r)
    return out, lse


def _seqlens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Per-sequence lengths from cumulative offsets, on the device and without a host read."""
    return (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)


def varlen_fwd(q, k, v, cu_seqlens, max_seqlen: int, scale: float):
    """Packed THD causal self-attention: ``q`` [T, Hq, D], ``k``/``v`` [T, Hk, D] over ``cu_seqlens``
    -> ``(out [T, Hq, D] bf16, lse [T, Hq] float32, natural log of the scaled scores)``."""
    _check(q, k, v, "gqa_sm100.varlen_fwd")
    cu = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    lens = _seqlens(cu)

    def run(qq):
        return gqa_sm100_fwd(qq, k, v, cu, lens, softmax_scale=float(scale), seg_tokens=segment_span(q.shape[-1]),
                             max_seqlen_q=int(max_seqlen), max_seqlen_k=int(max_seqlen), cu_seqlens_k=cu)

    grouped = _by_head_groups(q, k.shape[-2], None, run)
    return run(q.contiguous()) if grouped is None else grouped


def paged_fwd(q, k_cache, v_cache, cu_seqlens_q, seqused_k, block_table, max_seqlen_q: int, max_seqlen_k: int,
              scale: float, *, out=None):
    """Causal forward over a paged cache ``[num_blocks, page, Hk, D]`` (``page % 128 == 0``; a strided
    token-major view is accepted) with queries right-aligned to each sequence's ``seqused_k`` keys ->
    ``(out, lse)``. ``max_seqlen_k`` must bound every ``seqused_k``: it sizes the plan. ``out``
    (contiguous bf16 ``[T, Hq, D]``) is written in place."""
    _check(q, k_cache, v_cache, "gqa_sm100.paged_fwd")
    cu, used, table = cu_seqlens_q.contiguous(), seqused_k.contiguous(), block_table.contiguous()

    def run(qq, dst=None):
        return gqa_sm100_fwd(qq, k_cache, v_cache, cu, used, softmax_scale=float(scale),
                             seg_tokens=segment_span(q.shape[-1]), max_seqlen_q=int(max_seqlen_q),
                             max_seqlen_k=int(max_seqlen_k), block_table=table, out=dst)

    hk = k_cache.shape[-2]
    grouped = _by_head_groups(q, hk, out, run, decode=_decode_shaped(q, hk, max_seqlen_q))
    return run(q.contiguous(), out) if grouped is None else grouped


def backward_kernel(head_dim: int) -> str:
    """Which FA4 SM100 backward serves ``head_dim`` (module docstring)."""
    if head_dim == 256:
        return "FA4-sm100-hd256-2cta"
    if head_dim == 128:
        return "FA4-sm100-deterministic"
    raise RuntimeError(f"gqa_sm100: no backward for head dim {head_dim} (forward head dims {HEAD_DIMS})")


#: The CuTe DSL wheels that each carry a complete compiler: the CUDA 13 build first, as torch cu130
#: needs, then the base (CUDA 12) build.
_CUTE_DSL_BUILDS = ("nvidia-cutlass-dsl-libs-cu13", "nvidia-cutlass-dsl-libs-base")


@lru_cache(None)
def require_consistent_cute_dsl() -> str:
    """Refuse a CuTe DSL whose files come from more than one compiler build; return the build's name.

    ``nvidia-cutlass-dsl-libs-base`` and ``-libs-cu13`` write many of the same paths (the compiler
    library, the generated MLIR bindings, the TVM-FFI provider) with different contents, so
    installing both leaves whichever file was written last -- a mixture that cannot compile FA4's
    backward. The first build that is installed must still own every file it recorded.
    """
    import base64
    import hashlib
    from importlib import metadata

    for name in _CUTE_DSL_BUILDS:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        stale = []
        for f in dist.files or ():
            if f.hash is None or f.hash.mode != "sha256":
                continue
            path = dist.locate_file(f)
            digest = base64.urlsafe_b64encode(hashlib.sha256(Path(path).read_bytes()).digest()).rstrip(b"=").decode()
            if digest != f.hash.value:
                stale.append(str(f))
        if stale:
            raise RuntimeError(
                f"gqa_sm100: the CuTe DSL install mixes compiler builds: {len(stale)} files {name} installed "
                f"were overwritten by another build (e.g. {stale[:3]}). FA4's SM100 backward cannot compile "
                f"against a mixture; reinstall {name}=={dist.version} so its files are its own.")
        return f"{name}=={dist.version}"
    raise RuntimeError(f"gqa_sm100: FA4's SM100 backward needs the CuTe DSL ({' or '.join(_CUTE_DSL_BUILDS)})")


def varlen_bwd(dout, q, k, v, out, lse, cu_seqlens, max_seqlen: int, scale: float):
    """FA4's SM100 varlen backward on this package's forward -> ``(dq, dk, dv)``.

    ``lse`` is the forward's ``[T, Hq]`` natural-log LSE of the scaled scores, which is FA4's own
    convention; FA4 takes it head-major (``[Hq, T]``) and its preprocess forms ``lse * log2(e)``.
    """
    require_consistent_cute_dsl()
    from isoexec.ops._vendor.flash_attn_cute.interface_bwd_sm100 import flash_attn_varlen_bwd_sm100

    kernel = backward_kernel(q.shape[-1])
    cu = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    m = int(max_seqlen)
    return flash_attn_varlen_bwd_sm100(
        q, k, v, out, dout.contiguous(), lse.t().contiguous(), cu, cu, m, m, softmax_scale=float(scale),
        causal=True, deterministic=kernel == "FA4-sm100-deterministic", disable_2cta=False)


class _VarlenAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_seqlen, scale):
        out, lse = varlen_fwd(q, k, v, cu_seqlens, max_seqlen, scale)
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens)
        ctx.args = (max_seqlen, scale)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout):
        q, k, v, out, lse, cu = ctx.saved_tensors
        dq, dk, dv = varlen_bwd(dout, q, k, v, out, lse, cu, *ctx.args)
        return dq, dk, dv, None, None, None


def varlen_attention(q, k, v, cu_seqlens, max_seqlen: int, scale: float):
    """Differentiable packed causal GQA (this forward + FA4's SM100 backward) -> ``out [T, Hq, D]``."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("gqa_sm100: softmax scale must be finite and positive")
    if cu_seqlens.dtype != torch.int32:
        raise TypeError("gqa_sm100: cumulative offsets must be int32")
    return _VarlenAttention.apply(q, k, v, cu_seqlens, int(max_seqlen), float(scale))


__all__ = [
    "DECODE_ROWS",
    "GRAN_AUTO",
    "GRAN_PAIRS",
    "GRAN_SINGLE",
    "HEAD_DIMS",
    "KEY_TILE",
    "LEAF_AUTO",
    "LEAF_DECODE",
    "LEAF_GENERAL",
    "SEG_TOKENS",
    "backward_kernel",
    "default_scale",
    "gqa_sm100_fwd",
    "paged_fwd",
    "require_consistent_cute_dsl",
    "seg_for",
    "segment_count",
    "segment_span",
    "varlen_attention",
    "varlen_bwd",
    "varlen_fwd",
]
