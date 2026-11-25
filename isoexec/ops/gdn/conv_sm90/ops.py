"""SM90 width-4 causal convolution: oldest-to-current fp32 RN tap accumulation, optional bias,
CUDA expf/__fdividef SiLU with FTZ disabled, then one RNE output cast.
History comes from the same sequence, carried state or +0; the final state keeps the last three inputs verbatim.
Forward is bitwise across execution plans; backward is checked against float64 autograd without a bitwise contract."""

from __future__ import annotations

import torch

# Read the recipe identity from import-light metadata.
from isoexec.ops.gdn.conv_sm90._extension import extension
from isoexec.ops.gdn.conv_sm90.spec import RECIPE_VERSION, WIDTH  # noqa: F401  (RECIPE_VERSION is re-exported, not used here)

_FLOATS = (torch.bfloat16, torch.float32)


def _act(activation) -> int:
    if activation in (None, False):
        return 0
    if activation in (True, "silu", "swish"):
        return 1
    raise ValueError(f"conv_sm90: unsupported activation {activation!r}")


def _weight(weight: torch.Tensor, D: int) -> torch.Tensor:
    if weight.ndim == 3 and weight.shape[1] == 1:
        weight = weight.squeeze(1)
    if weight.ndim != 2 or weight.shape != (D, WIDTH):
        raise ValueError(f"conv_sm90: weight must be [D={D}, {WIDTH}] (or [D, 1, {WIDTH}]), got {tuple(weight.shape)}")
    if weight.dtype not in _FLOATS:
        raise ValueError(f"conv_sm90: weight dtype {weight.dtype} is not bf16/fp32")
    return _aligned(weight)


def _aligned(t: torch.Tensor) -> torch.Tensor:
    """Contiguous with a 16-byte aligned base (packed 128-bit loads); copies only when necessary."""
    if t.is_contiguous() and t.data_ptr() % 16 == 0:
        return t
    t = t.clone(memory_format=torch.contiguous_format)
    if t.data_ptr() % 16:  # pragma: no cover - the caching allocator aligns to 512 bytes
        raise RuntimeError("conv_sm90: could not obtain a 16-byte aligned buffer")
    return t


def _rows_aligned(t: torch.Tensor) -> bool:
    """[N, D] with unit column stride and 16-byte aligned rows: what the update kernel reads in place."""
    if t.ndim != 2 or (t.stride(1) != 1 and t.shape[1] > 1):
        return False
    rows_16 = t.shape[0] <= 1 or (t.stride(0) * t.element_size()) % 16 == 0
    return t.data_ptr() % 16 == 0 and rows_16


def _row_window(t: torch.Tensor) -> bool:
    """[T, D] rows at a stride of whole 8-element packs (>= D), unit column stride, 16-byte aligned base:
    what the varlen kernels read in place (``x`` may be a column window of a wider row-major tensor)."""
    return (t.ndim == 2 and t.stride(1) == 1 and t.stride(0) % 8 == 0 and t.stride(0) >= t.shape[1]
            and t.data_ptr() % 16 == 0)


def _check_x(x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError("conv_sm90 requires CUDA tensors")
    if x.ndim != 2:
        raise ValueError(f"conv_sm90: x must be [T, D], got {tuple(x.shape)}")
    if x.dtype not in _FLOATS:
        raise ValueError(f"conv_sm90: x dtype {x.dtype} is not bf16/fp32")
    if x.shape[1] % 8:
        raise ValueError(f"conv_sm90: channel count {x.shape[1]} must be a multiple of 8")


def fwd_plan(total: int) -> tuple[int, int]:
    """(token tile, threads per block) for the forward kernels. Execution only: bits never depend on it."""
    # Measured (min of 3 x mean of 20 graph replays, D in {1024, 1280, 2048, 2560}): 8-token tiles win at
    # every production size once the state write-back rides the main launch.
    return (8, 64) if total <= 4096 else (8, 128)


def bwd_plan(total: int, channels: int = 2048) -> tuple[str, int, int]:
    """(layout, token tile, threads) for the VJP. ``varlen_bwd2`` (one CTA per channel group, one thread per
    tile, dw/db folded in shared memory) wins while tiles are few; ``varlen_bwd`` (one CTA per tile) wins at
    large token counts. Measured on H100 (min of 3 x mean of 20 graph replays); execution only."""
    work = total * channels
    if work <= 12_000_000:
        return "bwd2", (16 if total < 8192 else 32), 128
    if work <= 24_000_000:
        return "bwd", 64, 64
    return "bwd", 128, 128


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
    plan: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Packed varlen forward ``x [T, D] -> y [T, D]`` (no autograd).

    With ``conv_state [lines, D, 3]`` (any strides), sequence ``s`` resumes from line
    ``cache_indices[s]`` when ``has_initial_state[s]`` and writes its final window back; lines ``< 0``
    or ``== null_block_id`` are padding and are neither read nor written.
    """
    _check_x(x)
    T, D = x.shape
    w = _weight(weight, D)
    if bias is not None:
        if bias.shape != (D,) or bias.dtype != w.dtype:
            raise ValueError("conv_sm90: bias must be [D] with the weight dtype")
        bias = bias.contiguous()
    if cu_seqlens.dtype != torch.int32 or cu_seqlens.ndim != 1 or not cu_seqlens.is_cuda:
        cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.int32)
    cu_seqlens = cu_seqlens.contiguous()
    nseq = cu_seqlens.shape[0] - 1
    if conv_state is not None:
        if cache_indices is None or has_initial_state is None:
            raise ValueError("conv_sm90: conv_state needs cache_indices and has_initial_state")
        if conv_state.ndim != 3 or conv_state.shape[1] != D or conv_state.shape[2] != WIDTH - 1:
            raise ValueError(f"conv_sm90: conv_state must be [lines, {D}, {WIDTH - 1}], got {tuple(conv_state.shape)}")
        if conv_state.dtype != x.dtype:
            raise ValueError("conv_sm90: conv_state dtype must equal x dtype")
        cache_indices = cache_indices.to(dtype=torch.int32).contiguous()
        has_initial_state = has_initial_state.to(dtype=torch.bool).contiguous()
        if cache_indices.shape[0] < nseq or has_initial_state.shape[0] < nseq:
            raise ValueError("conv_sm90: cache_indices/has_initial_state shorter than the batch")
    # A row-strided x (the q|k|v column window of the in_proj GEMM output) is read in place.
    xa = x if _row_window(x) else _aligned(x)
    dst = out if out is not None and out.is_contiguous() and out.data_ptr() % 16 == 0 and out.dtype == x.dtype else None
    y = dst if dst is not None else torch.empty(xa.shape, dtype=xa.dtype, device=xa.device)
    if T == 0 or nseq <= 0:
        return y if out is None else out
    tile, threads = plan or fwd_plan(T)
    extension().varlen_fwd(
        y, xa, w, bias, cu_seqlens, conv_state, cache_indices, has_initial_state, int(null_block_id), _act(activation), tile, threads
    )
    if out is not None and dst is None:
        out.copy_(y)
        return out
    return y


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
    threads: int = 128,
) -> torch.Tensor:
    """Decode step ``x [N, D]``: one token per lane, the window slides in ``conv_state`` in place.

    ``out`` may alias ``x``. Padding lanes (line ``< 0`` or ``== null_block_id``) are skipped: their
    output rows and state lines are left untouched.
    """
    _check_x(x)
    N, D = x.shape
    w = _weight(weight, D)
    if bias is not None:
        bias = bias.contiguous()
    if conv_state.ndim != 3 or conv_state.shape[1] != D or conv_state.shape[2] != WIDTH - 1 or conv_state.dtype != x.dtype:
        raise ValueError("conv_sm90: conv_state must be [lines, D, 3] with the x dtype")
    idx = cache_indices.to(dtype=torch.int32).contiguous()
    if idx.shape[0] != N:
        raise ValueError("conv_sm90: cache_indices must have one entry per lane")
    if N == 0:
        return x if out is None else out
    # A row-strided x (a column window of the in_proj GEMM output) is read in place: the kernel takes
    # both row strides, so only a misaligned or non-unit-column-stride x is copied.
    xa = x if _rows_aligned(x) else _aligned(x)
    if out is None:
        out = torch.empty(xa.shape, dtype=xa.dtype, device=xa.device)
    if out.data_ptr() == xa.data_ptr() and out.stride() == xa.stride():
        dst = out  # in place
    else:
        # The kernel writes every row of dst: live lanes get the conv, padding lanes a byte copy of
        # their input (what the in-place form leaves there).
        dst = out if (_rows_aligned(out) and out.dtype == x.dtype) else torch.empty(
            xa.shape, dtype=xa.dtype, device=xa.device)
    extension().update_fwd(dst, xa, w, bias, conv_state, idx, int(null_block_id), _act(activation), threads)
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
        # read in place when row-strided (a column window of the in_proj output); saved as that view
        xa = x if _row_window(x) else _aligned(x)
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
        dx = torch.empty(xa.shape, dtype=xa.dtype, device=xa.device)
        kind, tile, threads = bwd_plan(T, D)
        fn = extension().varlen_bwd2 if kind == "bwd2" else extension().varlen_bwd
        part = fn(dx, xa, dy, w, bias, cu, _act(ctx.activation) == 1, tile, threads)
        folded = part.sum(0)  # [D, 5] fp32: dw taps 0..3, db
        dw = folded[:, :WIDTH].to(w.dtype).reshape(ctx.weight_shape) if ctx.needs_input_grad[1] else None
        db = folded[:, WIDTH].to(bias.dtype) if (bias is not None and ctx.needs_input_grad[2]) else None
        return dx if ctx.needs_input_grad[0] else None, dw, db, None, None


def conv(x, weight, bias=None, *, cu_seqlens, activation="silu"):
    """Differentiable packed varlen conv for the trainer: ``x [T, D] -> y [T, D]`` (zero initial state)."""
    needs_grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, weight, bias))
    if not needs_grad:
        return varlen(x, weight, bias, cu_seqlens=cu_seqlens, activation=activation)
    return _ConvFn.apply(x, weight, bias, cu_seqlens, activation)
