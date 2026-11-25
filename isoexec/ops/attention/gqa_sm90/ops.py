"""Python API over the vendored FA3 library: varlen forward/backward, paged forward, and the autograd core.

Every entry runs causal attention with an unlimited left window, no softcap, no PackGQA override (FA3's
own per-launch choice, which does not change a row's arithmetic), and ``sm_margin=0``. The KV reduction
is either one pass (``seg_tokens=None``: ``num_splits=1``, the ns1 recipe) or the segment split
(``seg_tokens=SEG``: fixed logical KV segments of SEG keys, right-folded by ``fa3/seg_merge.h``; spec.py).
The segment split has two execution paths with one numerical program, chosen per call (``seg_path``):
``fold`` (the unsplit kernel walks each work tile's segments and folds them in an L2-resident per-CTA slot)
and ``split`` (one work tile per segment; ``fa3/seg_combine.cu`` folds the partials). Callers pass explicit cumulative offsets and host maximum
lengths; under the segment split ``max_seqlen_k`` must bound every sequence's KV length (overstating it
changes no bit; understating it yields NaN rows for the sequences it cuts).

Backward runs FA3's own kernels, deterministic below head dim 256 (spec.DETERMINISTIC_BWD_MAX_HEAD_DIM).
At 256 FA3 has only its atomic backward. When the process requires determinism
(``require_deterministic_backward(True)``, from Megatron's ``deterministic_mode``), that head dim
instead runs torch's FlashAttention-2 backward in deterministic mode on FA3's saved output and LSE.
``allow_nondeterministic=True`` always selects FA3's kernel.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from isoexec.ops.attention.gqa_sm90 import spec

_REQUIRE_DETERMINISTIC_BWD = False


def require_deterministic_backward(required: bool) -> None:
    """Set the process-wide backward policy (the trainer install calls this once)."""
    global _REQUIRE_DETERMINISTIC_BWD
    if type(required) is not bool:
        raise TypeError("required must be a bool")
    _REQUIRE_DETERMINISTIC_BWD = required


def round_up_headdim(d: int) -> int:
    """FA3's ``round_up_headdim`` under this build (96 and 192 compiled out)."""
    for h in spec.HEAD_DIMS:
        if d <= h:
            return h
    return spec.HEAD_DIMS[-1]


def forward_kernel(d: int, dv: int, *, where: str = "gqa_sm90") -> tuple[int, int]:
    """The ``(D, DV)`` forward kernel FA3 runs for Q/K width ``d`` and V width ``dv`` in this build.

    FA3's dispatcher takes the first compiled kernel whose Q/K width covers ``d``; a call whose V is
    wider than that kernel's V would leave columns unwritten, so it is refused here.
    """
    if d <= 64 and dv > 64:
        kernel = (64, 256) if dv <= 256 else (64, 512)
    elif d <= 256:
        h = round_up_headdim(d)
        kernel = (h, h)
    else:
        kernel = None
    if kernel is None or d > kernel[0] or dv > kernel[1] or d % 8 or dv % 8:
        raise RuntimeError(f"{where}: no FA3 forward kernel in this build serves Q/K head dim {d} with V head dim {dv} "
                           f"(built: {spec.HEAD_DIMS} and {spec.HDIMDIFF64}, widths multiples of 8)")
    return kernel


def deterministic_backward(d: int, dv: int) -> bool:
    """Whether FA3's deterministic backward exists for this geometry (it is then always used)."""
    return round_up_headdim(max(d, dv)) <= spec.DETERMINISTIC_BWD_MAX_HEAD_DIM


@lru_cache(None)
def _is_sm90(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (9, 0)


def _require_sm90(t: torch.Tensor, where: str) -> None:
    if t.device.type != "cuda" or not _is_sm90(t.device.index if t.device.index is not None else torch.cuda.current_device()):
        raise RuntimeError(f"{where}: gqa_sm90 runs on SM90 CUDA devices only (got {t.device})")


def _contig(t):
    return t if t is None or t.stride(-1) == 1 else t.contiguous()


def _check(q, k, v, where):
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise RuntimeError(f"{where}: bf16 Q/K/V only (got {q.dtype}, {k.dtype}, {v.dtype})")
    if q.shape[-2] % k.shape[-2]:
        raise RuntimeError(f"{where}: {q.shape[-2]} query heads are not a multiple of {k.shape[-2]} kv heads")
    _require_sm90(q, where)
    forward_kernel(q.shape[-1], v.shape[-1], where=where)


def _ops():
    from isoexec.ops.attention.gqa_sm90._extension import extension

    return extension()


#: The segment split's execution paths (one numerical program): ``fold`` walks every segment of a work tile in
#: one CTA and folds them in an L2-resident slot (the saturated path); ``split`` runs one work tile per segment
#: and folds the partials in ``seg_combine.cu``; ``hybrid`` splits the decode-like sequences (queries fit one
#: m-block) and folds the others, in one launch; ``auto`` picks fold or hybrid per call.
SEG_PATHS = {"auto": 0, "split": 1, "fold": 2, "hybrid": 3}


def num_splits_arg(seg_tokens: int | None, seg_path: str = "auto") -> int:
    """FA3's ``num_splits`` for this build: 1 (one pass) or ``-(SEG + path)`` (segment split, flash_api_stable.cpp)."""
    if seg_path not in SEG_PATHS:
        raise ValueError(f"gqa_sm90: seg_path must be one of {sorted(SEG_PATHS)} (got {seg_path!r})")
    if seg_tokens is None:
        return spec.NUM_SPLITS
    if type(seg_tokens) is not int or seg_tokens <= 0 or seg_tokens % 640:
        raise ValueError(f"gqa_sm90: seg_tokens must be a positive multiple of 640 (got {seg_tokens!r})")
    return -(seg_tokens + SEG_PATHS[seg_path])


def workspace_floats(total_q: int, num_heads: int, num_seqs: int, head_dim_v: int, num_sm: int, *,
                     partial_floats: int = 0) -> int:
    """fp32 elements a ``seg_workspace`` needs for a call of this size (fa3/flash_api_stable.cpp carves it):
    the lse, the scheduler metadata, the split flags and the fold scratch (an upper bound over the kernels'
    tile heights), plus ``partial_floats`` for the split path's partials. A call whose partials do not fit
    what is left runs the fold path instead: the same bits, no partials."""
    r = lambda n: -(-int(n) // 64) * 64  # noqa: E731  (256-byte regions)
    b4 = -(-int(num_seqs) // 4) * 4
    return (r(num_heads * total_q) + r(4 * b4 + 4) + r(num_seqs) + r(num_sm * 192 * (head_dim_v + 8))
            + int(partial_floats) + 128)


def _fwd(q, k, v, *, out, q_v, cu_seqlens_q, cu_seqlens_k, seqused_k, max_seqlen_q, max_seqlen_k, block_table,
         scale, seg_tokens=None, seg_path="auto", workspace=None, rotary_interleaved=True):
    num_splits = num_splits_arg(seg_tokens, seg_path)
    if num_splits != spec.NUM_SPLITS:
        if max_seqlen_k is None:
            raise ValueError("gqa_sm90: the segment split needs max_seqlen_k (a bound on every KV length)")
        if q.shape[-1] != v.shape[-1] or q.shape[-1] not in spec.SEG_HEAD_DIMS:
            raise RuntimeError(f"gqa_sm90: the segment split is built for head dims {spec.SEG_HEAD_DIMS} (Q/K == V), "
                               f"got ({q.shape[-1]}, {v.shape[-1]})")
    o, lse, _, _ = _ops().fwd(
        q, k, v, None, None, q_v, out,
        cu_seqlens_q, cu_seqlens_k, None, None, seqused_k,
        max_seqlen_q, max_seqlen_k, block_table,
        None, None, None, None, None, None, None, None,
        float(scale), True, -1, 0, 0, 0.0, rotary_interleaved, None,
        num_splits, None, 0, workspace,
    )
    return o, lse


def varlen_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q: int, max_seqlen_k: int, scale: float, *, out=None,
               seg_tokens: int | None = None, seg_path: str = "auto"):
    """Packed THD causal forward: ``q`` [Tq, Hq, D], ``k`` [Tk, Hk, D], ``v`` [Tk, Hk, Dv] ->
    ``(out [Tq, Hq, Dv], lse [Hq, Tq] float32)``."""
    _check(q, k, v, "gqa_sm90.varlen_fwd")
    o, lse = _fwd(_contig(q), _contig(k), _contig(v), out=out, q_v=None, cu_seqlens_q=_contig(cu_seqlens_q),
                  cu_seqlens_k=_contig(cu_seqlens_k), seqused_k=None, max_seqlen_q=int(max_seqlen_q),
                  max_seqlen_k=int(max_seqlen_k), block_table=None, scale=scale, seg_tokens=seg_tokens,
                  seg_path=seg_path)
    return o, lse.contiguous()


def paged_fwd(q, k_cache, v_cache, cu_seqlens_q, seqused_k, block_table, max_seqlen_q: int, max_seqlen_k, scale: float,
              *, out=None, q_v=None, seg_tokens: int | None = None, seg_path: str = "auto", workspace=None):
    """Causal forward over a paged cache ``[num_blocks, page, Hk, D]`` (queries right-aligned to each
    sequence's ``seqused_k`` keys) -> ``(out, lse)``. ``max_seqlen_k=None`` lets FA3 bound it by the block table
    (one-pass only; the segment split needs the bound). With ``workspace`` (fp32, ``workspace_floats``) the
    call allocates nothing but what the caller passes (CUDA-graph capture) and ``lse`` is a flat view of it."""
    _check(q, k_cache, v_cache, "gqa_sm90.paged_fwd")
    o, lse = _fwd(_contig(q), _contig(k_cache), _contig(v_cache), out=out, q_v=q_v,
                  cu_seqlens_q=_contig(cu_seqlens_q), cu_seqlens_k=None, seqused_k=_contig(seqused_k),
                  max_seqlen_q=int(max_seqlen_q), max_seqlen_k=None if max_seqlen_k is None else int(max_seqlen_k),
                  block_table=_contig(block_table), scale=scale, seg_tokens=seg_tokens, seg_path=seg_path,
                  workspace=workspace)
    return o, lse


def varlen_bwd(dout, q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, max_seqlen_q: int, max_seqlen_k: int,
               scale: float, *, dq=None, dk=None, dv=None, allow_nondeterministic: bool = False):
    """FA3's varlen backward -> ``(dq, dk, dv)``; deterministic wherever FA3 provides it (module docstring)."""
    deterministic = deterministic_backward(q.shape[-1], v.shape[-1])
    if not deterministic and _REQUIRE_DETERMINISTIC_BWD and not allow_nondeterministic:
        return _fa2_deterministic_bwd(dout, q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                      scale)
    q, k, v, out, dout = map(_contig, (q, k, v, out, dout))
    dq = torch.empty_like(q) if dq is None else dq
    dk = torch.empty_like(k) if dk is None else dk
    dv = torch.empty_like(v) if dv is None else dv
    _ops().bwd(dout, q, k, v, out, _contig(lse), dq, dk, dv, cu_seqlens_q, cu_seqlens_k, None, None,
               int(max_seqlen_q), int(max_seqlen_k), float(scale), True, -1, 0, 0.0, deterministic, 0)
    return dq, dk, dv


def _fa2_deterministic_bwd(dout, q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale):
    """Torch's native FA2 varlen backward (deterministic mode) for a head dim FA3 cannot run deterministically."""
    where = "gqa_sm90.varlen_bwd"
    if q.shape[-1] != v.shape[-1]:
        raise RuntimeError(f"{where}: determinism is required but neither FA3 nor FA2 has a deterministic backward "
                           f"for Q/K head dim {q.shape[-1]} with V head dim {v.shape[-1]}")
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        raise RuntimeError(f"{where}: the FA2 deterministic backward needs torch.use_deterministic_algorithms(True)")
    seed = torch.zeros(2, dtype=torch.uint64, device=q.device)
    offset = torch.zeros((), dtype=torch.uint64, device=q.device)
    dq, dk, dv = torch.ops.aten._flash_attention_backward(
        dout.contiguous(), q, k, v, out, lse.contiguous(), cu_seqlens_q, cu_seqlens_k, int(max_seqlen_q),
        int(max_seqlen_k), 0.0, True, seed, offset, scale=float(scale), window_size_left=-1, window_size_right=0)
    return dq, dk, dv


class _VarlenAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale, seg_tokens):
        out, lse = varlen_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale,
                              seg_tokens=seg_tokens)
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.args = (max_seqlen_q, max_seqlen_k, scale)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout):
        q, k, v, out, lse, cu_q, cu_k = ctx.saved_tensors
        dq, dk, dv = varlen_bwd(dout, q, k, v, out, lse, cu_q, cu_k, *ctx.args)
        return dq, dk, dv, None, None, None, None, None, None


def varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q: int, max_seqlen_k: int, scale: float, *,
                     seg_tokens: int | None = None):
    """Differentiable packed causal GQA attention (forward + FA3 backward) -> ``out [Tq, Hq, Dv]``.

    The backward reads the forward's saved output and LSE; under the segment split that LSE is the combine's
    (``fa3/seg_combine.cu``), so the backward stays FA3's own and is correct, not a bitwise twin of anything."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("gqa_sm90: softmax scale must be finite and positive")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise TypeError("gqa_sm90: cumulative offsets must be int32")
    num_splits_arg(seg_tokens)  # validate before autograd records anything
    return _VarlenAttention.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, int(max_seqlen_q), int(max_seqlen_k),
                                  float(scale), seg_tokens)


def flash_attn_paged(*, q, k, v, q_v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, seqused_k, block_table,
                     softmax_scale, causal, window_size, num_splits):
    """Absorbed MLA's paged entry (``mla_absorbed.absorbed_kernel``): FA3 with ``qv`` and the (64, 512) kernel.

    Called the way ``flash_attn_with_kvcache`` calls FA3: the KV bound comes from the block table
    (``max_seqlen_k`` is not passed on), and the LSE is not returned.
    """
    del max_seqlen_k
    if not causal or tuple(window_size) != (-1, 0) or num_splits != spec.NUM_SPLITS:
        raise ValueError("gqa_sm90.flash_attn_paged: absorbed MLA requires causal, window (-1, 0), num_splits=1")
    forward_kernel(q.shape[-1], v.shape[-1], where="absorbed MLA (gqa_sm90 paged forward)")
    out, _ = paged_fwd(q, k, v, cu_seqlens_q, seqused_k, block_table, max_seqlen_q, None, softmax_scale,
                       q_v=_contig(q_v))
    return out


__all__ = ["deterministic_backward", "flash_attn_paged", "forward_kernel", "num_splits_arg", "paged_fwd",
           "require_deterministic_backward", "round_up_headdim", "varlen_attention", "varlen_bwd", "varlen_fwd",
           "workspace_floats"]
