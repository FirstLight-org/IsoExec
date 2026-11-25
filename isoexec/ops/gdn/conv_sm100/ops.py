"""SM100 width-4 causal convolution preserves tap order."""

from __future__ import annotations

import torch

# Read the recipe identity from import-light metadata.
from isoexec.ops.gdn.conv_sm100._extension import extension
from isoexec.ops.gdn.conv_sm100.spec import RECIPE_VERSION, RECIPE_VERSION_FMA, WIDTH

_FLOATS = (torch.bfloat16, torch.float32)


def _recipe(recipe: int) -> int:
    """The declared arithmetic version to run: 1 (default, ``sm100_cuda_v1``) or 2 (``sm100_conv_fma_v1``, csrc/recipe_v2.cuh)."""
    if recipe not in (RECIPE_VERSION, RECIPE_VERSION_FMA):
        raise ValueError(
            f"conv_sm100: recipe must be {RECIPE_VERSION} (the default, sm100_cuda_v1) or "
            f"{RECIPE_VERSION_FMA} (sm100_conv_fma_v1, explicit fma.rn taps), got {recipe!r}"
        )
    return int(recipe)


def _act(activation) -> int:
    if activation in (None, False):
        return 0
    if activation in (True, "silu", "swish"):
        return 1
    raise ValueError(f"conv_sm100: unsupported activation {activation!r}")


def _weight(weight: torch.Tensor, D: int) -> torch.Tensor:
    if weight.ndim == 3 and weight.shape[1] == 1:
        weight = weight.squeeze(1)
    if weight.ndim != 2 or weight.shape != (D, WIDTH):
        raise ValueError(f"conv_sm100: weight must be [D={D}, {WIDTH}] (or [D, 1, {WIDTH}]), got {tuple(weight.shape)}")
    if weight.dtype not in _FLOATS:
        raise ValueError(f"conv_sm100: weight dtype {weight.dtype} is not bf16/fp32")
    return _aligned(weight)


def _aligned(t: torch.Tensor) -> torch.Tensor:
    """Contiguous with a 16-byte aligned base (packed 128-bit loads); copies only when necessary."""
    if t.is_contiguous() and t.data_ptr() % 16 == 0:
        return t
    t = t.clone(memory_format=torch.contiguous_format)
    if t.data_ptr() % 16:  # pragma: no cover - the caching allocator aligns to 512 bytes
        raise RuntimeError("conv_sm100: could not obtain a 16-byte aligned buffer")
    return t


def _rows_aligned(t: torch.Tensor, cpt: int = 8) -> bool:
    """[N, D] with unit column stride, a 16-byte aligned base and rows a whole number of ``cpt``-wide
    packs apart: what the decode update kernel reads (and writes) in place."""
    if t.ndim != 2 or t.stride(1) != 1:
        return False
    rows_ok = t.shape[0] <= 1 or (t.stride(0) % cpt == 0 and (t.stride(0) * t.element_size()) % 16 == 0)
    return t.data_ptr() % 16 == 0 and rows_ok


def _check_x(x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError("conv_sm100 requires CUDA tensors")
    if x.ndim != 2:
        raise ValueError(f"conv_sm100: x must be [T, D], got {tuple(x.shape)}")
    if x.dtype not in _FLOATS:
        raise ValueError(f"conv_sm100: x dtype {x.dtype} is not bf16/fp32")
    if x.shape[1] % 8:
        raise ValueError(f"conv_sm100: channel count {x.shape[1]} must be a multiple of 8")


def fwd_plan(total: int, channels: int = 2048) -> tuple[int, int, int]:
    """(token tile, threads per block, channels per thread) for the forward kernels."""
    # Round 4 re-probe (packed-fp32 forward, `_tune_probe --rl --phases fwd`, full (tile, threads, cpt)
    # sweep incl. cpt = 8, i.e. 16-byte lanes): cpt = 4 (8-byte lanes) wins at all eight RL cells; the
    # only plan change is tile 16 at 4096 < T <= 8192 for D >= 2048 (16.54 vs 17.06 us).
    tile = 4 if total <= 2048 else 16 if (total > 8192 or (total > 4096 and channels >= 2048)) else 8
    return tile, 128, 4


#: G, the row-group count of the VJP's own fold kernel (csrc/conv_bwd.cu ``fold_dwdb``) -- part of
#: the fold tree's identity, so changing it moves ``dw``/``db`` bits. The tree, over the ``H``
#: partial rows (one per token tile) and independently for each (channel, tap|bias) column, every
#: add an fp32 ``add.rn``: group ``g`` sums rows ``g, g+G, g+2G, ...`` ascending, then the groups
#: that own a row are summed ascending ``g = 0, 1, ...``; one RNE cast into ``dw``/``db``.
#: (Measured in situ, kernel + fold minus kernel, at the RL shapes: G=16 costs 14-20 us, G=32
#: 10-15, G=64 4-13; more groups stop paying at 96-128.)
FOLD_GROUPS = 64


def bwd_plan(total: int, channels: int = 2048) -> tuple[int, int, int]:
    """(token tile, threads, channels per thread) for the VJP."""
    if total <= 8192:
        tile = 12
    elif total <= 16384:
        tile = 18
    else:
        tile = 20
    return tile, 64, 2


def varlen(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor,
    activation="silu",
    conv_state: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    null_block_id: int = 0,
    out: torch.Tensor | None = None,
    plan: tuple[int, int] | tuple[int, int, int] | None = None,
    recipe: int = RECIPE_VERSION,
) -> torch.Tensor:
    """Packed varlen forward ``x [T, D] -> y [T, D]`` (no autograd)."""
    _check_x(x)
    rec = _recipe(recipe)
    T, D = x.shape
    w = _weight(weight, D)
    if bias is not None:
        if bias.shape != (D,) or bias.dtype != w.dtype:
            raise ValueError("conv_sm100: bias must be [D] with the weight dtype")
        bias = bias.contiguous()
    if cu_seqlens.dtype != torch.int32 or cu_seqlens.ndim != 1 or not cu_seqlens.is_cuda:
        cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.int32)
    cu_seqlens = cu_seqlens.contiguous()
    nseq = cu_seqlens.shape[0] - 1
    if conv_state is not None:
        if cache_indices is None or has_initial_state is None:
            raise ValueError("conv_sm100: conv_state needs cache_indices and has_initial_state")
        if conv_state.ndim != 3 or conv_state.shape[1] != D or conv_state.shape[2] != WIDTH - 1:
            raise ValueError(f"conv_sm100: conv_state must be [lines, {D}, {WIDTH - 1}], got {tuple(conv_state.shape)}")
        if conv_state.dtype != x.dtype:
            raise ValueError("conv_sm100: conv_state dtype must equal x dtype")
        cache_indices = cache_indices.to(dtype=torch.int32).contiguous()
        has_initial_state = has_initial_state.to(dtype=torch.bool).contiguous()
        if cache_indices.shape[0] < nseq or has_initial_state.shape[0] < nseq:
            raise ValueError("conv_sm100: cache_indices/has_initial_state shorter than the batch")
    xa = _aligned(x)
    if out is not None and out.data_ptr() == x.data_ptr():
        # Forward is not in-place safe and never was: a tile reads three halo rows belonging to the
        # previous tile and, when it owns a sequence's cache line, reads that sequence's last three
        # inputs from wherever they are -- all of which another block may already have overwritten
        # with outputs. ``update`` supports aliasing (one token, no halo); ``varlen`` refuses it
        # rather than racing.
        raise ValueError("conv_sm100: varlen cannot write into its own input (out aliases x)")
    dst = out if out is not None and out.is_contiguous() and out.data_ptr() % 16 == 0 and out.dtype == x.dtype else None
    y = dst if dst is not None else torch.empty_like(xa)
    if T == 0 or nseq <= 0:
        return y if out is None else out
    cfg = tuple(plan) if plan else fwd_plan(T, D)
    tile, threads, cpt = cfg if len(cfg) == 3 else (*cfg, 8)  # a 2-tuple plan keeps the 128-bit width
    extension().varlen_fwd(
        y, xa, w, bias, cu_seqlens, conv_state, cache_indices, has_initial_state, int(null_block_id),
        _act(activation), tile, threads, cpt, rec
    )
    if out is not None and dst is None:
        out.copy_(y)
        return out
    return y


def decode_plan(N: int, D: int = 2048) -> tuple[int, int]:
    """(threads per block, channels per thread) for the decode kernel."""
    if N < 64:
        return 128, 2
    return 32, 8


def update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation="silu",
    cache_indices: torch.Tensor,
    null_block_id: int = 0,
    out: torch.Tensor | None = None,
    threads: int | None = None,
    cpt: int | None = None,
    recipe: int = RECIPE_VERSION,
) -> torch.Tensor:
    """Decode step ``x [N, D]``: one token per lane, the window slides in ``conv_state`` in place."""
    _check_x(x)
    rec = _recipe(recipe)
    N, D = x.shape
    if threads is None or cpt is None:
        dt, dc = decode_plan(N, D)
        threads = dt if threads is None else threads
        cpt = dc if cpt is None else cpt
    if D % cpt:
        cpt = 8 if D % 8 == 0 else 2
    w = _weight(weight, D)
    if bias is not None:
        bias = bias.contiguous()
    if conv_state.ndim != 3 or conv_state.shape[1] != D or conv_state.shape[2] != WIDTH - 1 or conv_state.dtype != x.dtype:
        raise ValueError("conv_sm100: conv_state must be [lines, D, 3] with the x dtype")
    idx = cache_indices.to(dtype=torch.int32).contiguous()
    if idx.shape[0] != N:
        raise ValueError("conv_sm100: cache_indices must have one entry per lane")
    if N == 0:
        return x if out is None else out
    # A row-strided x (the q|k|v column window of the in_proj GEMM output) is read in place: the kernel
    # takes both row strides, so only a misaligned or non-unit-column-stride x is copied.
    xa = x if _rows_aligned(x, cpt) else _aligned(x)
    if out is None:
        out = torch.empty(xa.shape, dtype=xa.dtype, device=xa.device)
    if out.data_ptr() == xa.data_ptr() and out.stride() == xa.stride():
        dst = out  # in place
    else:
        # The kernel writes every row of dst: live lanes the conv, padding lanes a byte copy of their
        # input (what the in-place form leaves there).
        dst = out if (_rows_aligned(out, cpt) and out.dtype == x.dtype) else torch.empty(
            xa.shape, dtype=xa.dtype, device=xa.device)
    extension().update_fwd(dst, xa, w, bias, conv_state, idx, int(null_block_id), _act(activation),
                           threads, cpt, rec)
    if dst.data_ptr() != out.data_ptr():
        out.copy_(dst)
    return out


class _ConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, cu_seqlens, activation):
        _check_x(x)
        D = x.shape[1]
        w = _weight(weight, D)
        cu = cu_seqlens.to(device=x.device, dtype=torch.int32).contiguous()
        xa = _aligned(x)
        y = varlen(xa, w, bias, cu_seqlens=cu, activation=activation)
        ctx.save_for_backward(xa, w, bias, cu)
        ctx.activation = activation
        ctx.weight_shape = weight.shape
        return y

    @staticmethod
    def backward(ctx, dy):
        xa, w, bias, cu = ctx.saved_tensors
        T, D = xa.shape
        dy = dy.to(xa.dtype) if dy.dtype != xa.dtype else dy
        dy = _aligned(dy)
        dx = torch.empty_like(xa)
        tile, threads, cpt = bwd_plan(T, D)
        part = extension().varlen_bwd(dx, xa, dy, w, bias, cu, _act(ctx.activation) == 1, tile,
                                      threads, cpt)
        # The fold is this package's own kernel (``FOLD_GROUPS`` documents its tree): one pass over
        # the partial rows that casts straight into dw [D, 4] and db [D].
        dw = torch.empty(D, WIDTH, dtype=w.dtype, device=w.device) if ctx.needs_input_grad[1] else None
        db = (torch.empty(D, dtype=bias.dtype, device=w.device)
              if (bias is not None and ctx.needs_input_grad[2]) else None)
        if dw is not None or db is not None:
            extension().fold_dwdb(part, dw, db, FOLD_GROUPS)
        if dw is not None:
            dw = dw.reshape(ctx.weight_shape)
        return dx if ctx.needs_input_grad[0] else None, dw, db, None, None


def conv(x, weight, bias=None, *, cu_seqlens, activation="silu"):
    """Differentiable packed varlen conv for the trainer: ``x [T, D] -> y [T, D]`` (zero initial state)."""
    needs_grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, weight, bias))
    if not needs_grad:
        return varlen(x, weight, bias, cu_seqlens=cu_seqlens, activation=activation)
    return _ConvFn.apply(x, weight, bias, cu_seqlens, activation)
