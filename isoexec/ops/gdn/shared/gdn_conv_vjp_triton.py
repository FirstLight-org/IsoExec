"""Fused Triton VJP for the packed varlen causal depthwise conv (``conv.short``).

The eager reference is :func:`..gdn_ops._native_conv_vjp_eager`: an explicit ascending tap loop
that materializes an ``index_select`` + fp32 ``mul`` + ``add`` per tap per row chunk. This module
computes the same three stages -- preactivation, activation derivative, ``dx``/``dw`` -- with the
same operations in the same order, but keeps every per-tap intermediate in registers:

* :func:`_preact_act_kernel`   stage 1 + stage 2, one launch per row chunk;
* :func:`_dx_kernel`           the ``dx`` tap loop, one launch per row chunk;
* :func:`_dw_products_kernel`  the ``dw`` per-tap products, one launch per row chunk.

**Bit-neutrality.** Every rounding point of the reference is reproduced, not approximated:

1. *No contraction.* Every launch passes ``_COMPILE_OPTS``, whose ``enable_fp_fusion=False`` is the
   first half. Triton contracts ``acc + a * b`` into an FMA by default, which drops the intermediate
   rounding the reference performs; the option turns that off, restoring one rounding per multiply
   and one per add.
2. *Sigmoid.* ATen's ``sigmoid_kernel_cuda`` for fp32 is ``scalar_t{1} / (scalar_t{1} + std::exp(-a))``
   compiled by nvcc at its default ``-ftz=false``. Emitted PTX, from ``nvcc -arch=sm_90a -ptx`` on
   that expression: ``ex2.approx.ftz.f32`` (inside ``expf``'s range reduction), ``fma.rm.f32`` and
   ``fma.rn.f32`` (the same reduction, NOT flush-to-zero) and ``rcp.rn.f32`` for the ``1.0f /``.
   Two things are needed to reproduce it, and this module got the second one wrong when it landed:

   * The right expression. ``tl.sigmoid``/``tl.exp`` apply ``ex2.approx.f32`` straight to
     ``z * log2(e)`` with no range reduction -- measured 64 ULP and 2.02M of 4M sweep points away
     from ``torch.sigmoid`` -- and Triton's ``/`` lowers to ``div.full.f32``, which is not a
     round-to-nearest divide (1 ULP, 391k of 4M points). libdevice ``expf`` over a correctly
     rounded divide removes both.
   * The right DIVIDE. Triton links libdevice with ``__CUDA_FTZ`` set, so ``libdevice.div_rn``
     compiles to ``div.rn.ftz.f32`` and returns ``+0.0`` where ATen returns an fp32 subnormal. For
     preactivations in ``(-88.7228, -87.3345)`` -- where ``1/(1+expf(-z))`` lands in
     ``[2.94e-39, 1.18e-38]`` -- that collapsed the derivative to a signed zero and ``dx`` lost the
     tap's whole contribution: 650 of 29.4M elements at the 9B width, reproducible at every
     production geometry. This module originally spelled the divide ``libdevice.div_rn``; it now
     calls :func:`...core.triton_nonftz.sigmoid`, the tree's audited non-FTZ primitive, which is an
     inline-asm ``div.rn.f32``. That file's audit (``ops/moe/moe_sm90/tests/_ftz_check.py``) had already
     recorded this exact trap for the MoE router. Measured after the change: 0 differing bit
     patterns against ``torch.sigmoid`` over 26.5M inputs -- the whole (-105, -80) band, EVERY
     mantissa of the exponent containing the window, the subnormals, both zeros, the infinities and
     4M random bit patterns.

   ``_COMPILE_OPTS`` then adds ``enable_reflect_ftz=False`` on top. The non-FTZ sigmoid alone is
   already bitwise over all 26.5M (measured both ways), so this is defence in depth rather than the
   fix: it clears the module's ``nvvm-reflect-ftz`` flag, so libdevice's ``expf`` uses the
   ``fma.rm.f32``/``fma.rn.f32`` reduction nvcc gives ATen instead of the ``.ftz`` one, and any f32
   op added to these kernels later is IEEE by default rather than by review. It costs nothing:
   the fused VJP at the 9B geometry is 0.8662 ms with it and 0.8666 ms without, and
   ``_preact_act_kernel`` alone is 0.1196 ms either way. ``ex2.approx.ftz.f32`` survives and is
   meant to -- it is inside ``expf``'s reduction, where the argument is bounded, and nvcc emits it
   for ATen too.
3. *Order.* The tap loop runs ascending with an fp32 accumulator, exactly as the reference's
   ``acc = acc + ...``; each output element still has exactly one owner, so there is no atomic and
   no cross-thread reduction anywhere in this file.
4. *Masked taps.* The reference multiplies the CLAMPED gather by a 0/1 ``valid`` flag rather than
   skipping it, so an infinite input yields ``inf * 0 = NaN``. The kernels load the same clamped row
   and multiply by the same flag, so that behaviour is preserved rather than optimized away.
5. *Reductions stay in ATen.* ``dw`` (and ``db``) fold with ``Tensor.sum(dim=0)``, whose reduction
   order is an internal ATen schedule that no hand-written kernel reproduces (measured: a sequential
   ascending sum differs, and the schedule is even sensitive to the operand's 16-byte alignment).
   The kernels therefore materialize the per-tap PRODUCT and hand the sum back to ATen unchanged --
   :func:`_rows_pad` keeps every tap slice 512-byte aligned so the schedule is the one the
   reference's freshly allocated product tensor gets.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ....core.triton_nonftz import sigmoid as _nonftz_sigmoid

# Launch shape only, nothing here rounds: 128 fp32 columns is a 512-byte contiguous run per row and
# 8 rows give the tap gathers (which read rows t-W+1..t) their reuse inside one program.
_BLOCK_R = 8
_BLOCK_C = 128

# torch's caching allocator hands out 512-byte aligned storage; `sum(dim=0)`'s reduction schedule is
# alignment-sensitive, so every product slice must start on the same boundary a fresh tensor would.
_ALLOC_ALIGN_BYTES = 512

# The compile options that make the kernels round exactly as the reference's ATen chain does, passed
# to EVERY launch as one dict so there is a single place to state them and a CPU test can compile the
# kernels with the very options the launches use
# (`tests/test_native_conv_backward_cpu.py::test_fused_conv_vjp_kernels_emit_the_ptx_aten_emits`).
# `enable_fp_fusion=False` keeps the tap accumulator two rounded operations instead of one FMA;
# `enable_reflect_ftz=False` keeps every remaining f32 op IEEE. See docstring points 1 and 2.
_COMPILE_OPTS = {"enable_fp_fusion": False, "enable_reflect_ftz": False}


@triton.jit
def _preact_act_kernel(
    x_ptr,
    dy_ptr,
    w_ptr,
    bias_ptr,
    seq_start_ptr,
    out_ptr,
    row0,
    rows,
    channels,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    TAPS: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    """``out[r, c] = dy * act'(sum_tap x[t-lag, c] * w[c, tap] + bias[c])`` for one row chunk.

    ``out`` is chunk-local (the caller passes the already-sliced destination); ``x``/``dy`` are
    indexed globally so the tap window may reach rows before ``row0``.
    """
    r = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    c = (tl.program_id(1) * BC + tl.arange(0, BC)).to(tl.int64)
    row_mask = r < rows
    col_mask = c < channels
    mask = row_mask[:, None] & col_mask[None, :]
    t = row0 + r
    seq_start = tl.load(seq_start_ptr + t, mask=row_mask, other=0)

    acc = tl.zeros((BR, BC), dtype=tl.float32)
    for tap in tl.static_range(TAPS):
        lag = TAPS - 1 - tap
        source = t - lag
        valid = (source >= seq_start).to(tl.float32)
        clamped = tl.maximum(source, 0)
        xv = tl.load(x_ptr + clamped[:, None] * channels + c[None, :], mask=mask, other=0.0)
        wv = tl.load(w_ptr + c * TAPS + tap, mask=col_mask, other=0.0)
        acc = acc + (xv.to(tl.float32) * valid[:, None]) * wv[None, :]
    if HAS_BIAS:
        acc = acc + tl.load(bias_ptr + c, mask=col_mask, other=0.0)

    dy = tl.load(dy_ptr + t[:, None] * channels + c[None, :], mask=mask, other=0.0).to(tl.float32)
    if SILU:
        sigmoid = _nonftz_sigmoid(acc)
        out = dy * (sigmoid * (1.0 + acc * (1.0 - sigmoid)))
    else:
        out = dy
    tl.store(out_ptr + r[:, None] * channels + c[None, :], out, mask=mask)


@triton.jit
def _dx_kernel(
    dz_ptr,
    w_ptr,
    seq_end_ptr,
    dx_ptr,
    row0,
    rows,
    total,
    channels,
    TAPS: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    """``dx[t, c] = sum_tap dz[t + lag, c] * w[c, tap]`` for one row chunk, taps ascending."""
    r = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    c = (tl.program_id(1) * BC + tl.arange(0, BC)).to(tl.int64)
    row_mask = r < rows
    col_mask = c < channels
    mask = row_mask[:, None] & col_mask[None, :]
    t = row0 + r
    seq_end = tl.load(seq_end_ptr + t, mask=row_mask, other=0)

    acc = tl.zeros((BR, BC), dtype=tl.float32)
    for tap in tl.static_range(TAPS):
        lag = TAPS - 1 - tap
        sink = t + lag
        valid = (sink < seq_end).to(tl.float32)
        clamped = tl.minimum(sink, total - 1)
        dzv = tl.load(dz_ptr + clamped[:, None] * channels + c[None, :], mask=mask, other=0.0)
        wv = tl.load(w_ptr + c * TAPS + tap, mask=col_mask, other=0.0)
        acc = acc + (dzv * valid[:, None]) * wv[None, :]
    tl.store(
        dx_ptr + t[:, None] * channels + c[None, :],
        acc.to(dx_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _dw_products_kernel(
    x_ptr,
    dz_ptr,
    seq_start_ptr,
    prod_ptr,
    row0,
    rows,
    channels,
    tap_stride,
    TAPS: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    """``prod[tap, r, c] = dz[t, c] * x[t - lag, c]`` for one row chunk; ATen owns the row sum."""
    r = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    c = (tl.program_id(1) * BC + tl.arange(0, BC)).to(tl.int64)
    row_mask = r < rows
    col_mask = c < channels
    mask = row_mask[:, None] & col_mask[None, :]
    t = row0 + r
    seq_start = tl.load(seq_start_ptr + t, mask=row_mask, other=0)
    dzv = tl.load(dz_ptr + t[:, None] * channels + c[None, :], mask=mask, other=0.0)

    for tap in tl.static_range(TAPS):
        lag = TAPS - 1 - tap
        source = t - lag
        valid = (source >= seq_start).to(tl.float32)
        clamped = tl.maximum(source, 0)
        xv = tl.load(x_ptr + clamped[:, None] * channels + c[None, :], mask=mask, other=0.0)
        tl.store(
            prod_ptr + tap * tap_stride + r[:, None] * channels + c[None, :],
            dzv * (xv.to(tl.float32) * valid[:, None]),
            mask=mask,
        )


def _rows_pad(rows: int, channels: int) -> int:
    """Rows per tap slice of the ``dw`` product buffer, so every slice base stays 512-byte aligned.

    ``buffer[tap, :rows]`` then starts on the same boundary as a freshly allocated
    ``[rows, channels]`` tensor, which is what makes its ``sum(dim=0)`` take the same ATen reduction
    schedule as the eager reference's product tensor.
    """
    floats_per_align = _ALLOC_ALIGN_BYTES // 4
    per = floats_per_align // math.gcd(channels, floats_per_align)
    return -(-rows // per) * per


def _grid(rows: int, channels: int) -> tuple[int, int]:
    return (triton.cdiv(rows, _BLOCK_R), triton.cdiv(channels, _BLOCK_C))


def conv_vjp_preact_act(
    out: torch.Tensor,
    x: torch.Tensor,
    dy: torch.Tensor,
    wf: torch.Tensor,
    bias_f: torch.Tensor | None,
    seq_start: torch.Tensor,
    row0: int,
    taps: int,
    silu: bool,
) -> None:
    """Write one row chunk of ``dL/d(preactivation)`` into ``out`` ``[rows, channels]`` fp32."""
    rows, channels = out.shape
    _preact_act_kernel[_grid(rows, channels)](
        x,
        dy,
        wf,
        bias_f,
        seq_start,
        out,
        row0,
        rows,
        channels,
        HAS_BIAS=bias_f is not None,
        SILU=silu,
        TAPS=taps,
        BR=_BLOCK_R,
        BC=_BLOCK_C,
        **_COMPILE_OPTS,
    )


def conv_vjp_dx(
    dx: torch.Tensor,
    dz: torch.Tensor,
    wf: torch.Tensor,
    seq_end: torch.Tensor,
    row0: int,
    rows: int,
    taps: int,
) -> None:
    """Write rows ``[row0, row0 + rows)`` of ``dx`` from the adjusted preactivation ``dz``."""
    total, channels = dz.shape
    _dx_kernel[_grid(rows, channels)](
        dz,
        wf,
        seq_end,
        dx,
        row0,
        rows,
        total,
        channels,
        TAPS=taps,
        BR=_BLOCK_R,
        BC=_BLOCK_C,
        **_COMPILE_OPTS,
    )


def conv_vjp_dw_products(
    prod: torch.Tensor,
    x: torch.Tensor,
    dz: torch.Tensor,
    seq_start: torch.Tensor,
    row0: int,
    rows: int,
    taps: int,
) -> None:
    """Fill ``prod[tap, :rows]`` with this chunk's ``dz * x[t - lag]`` products, one per tap."""
    channels = dz.shape[1]
    _dw_products_kernel[_grid(rows, channels)](
        x,
        dz,
        seq_start,
        prod,
        row0,
        rows,
        channels,
        prod.stride(0),
        TAPS=taps,
        BR=_BLOCK_R,
        BC=_BLOCK_C,
        **_COMPILE_OPTS,
    )


def native_conv_vjp_fused(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    seq_start: torch.Tensor,
    seq_end: torch.Tensor,
    chunk_rows: int,
    silu: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """``(dx, dw, db)`` for the packed conv, bitwise equal to the eager reference.

    ``seq_start``/``seq_end`` are the per-token sequence bounds the caller already derived from
    ``cu_seqlens``; ``chunk_rows`` is the reference's row-chunk size, kept because it is the fold
    granularity of the ``dw`` and ``db`` sums.
    """
    total, channels = x.shape
    taps = weight.shape[1]
    wf = weight.float().contiguous()  # the kernels index w[c * taps + tap]
    dz = torch.empty((total, channels), dtype=torch.float32, device=x.device)
    dx = torch.empty((total, channels), dtype=x.dtype, device=x.device)
    dwf = torch.zeros_like(wf)
    dbf = torch.zeros(channels, dtype=torch.float32, device=x.device) if bias is not None else None
    if total == 0:
        return dx, dwf.to(weight.dtype), (dbf.to(bias.dtype) if bias is not None else None)

    bias_f = bias.float() if bias is not None else None
    span = min(chunk_rows, total)
    # `db` sums a freshly allocated tensor in the reference, and dz[start:stop] is not always
    # 512-byte aligned, so the bias path stages the chunk exactly as the reference does.
    staged = torch.empty((span, channels), dtype=torch.float32, device=x.device) if bias is not None else None

    for start in range(0, total, chunk_rows):
        stop = min(start + chunk_rows, total)
        rows = stop - start
        out = dz[start:stop] if staged is None else staged[:rows]
        conv_vjp_preact_act(out, x, dy, wf, bias_f, seq_start, start, taps, silu)
        if staged is not None:
            dbf.add_(staged[:rows].sum(dim=0))
            dz[start:stop].copy_(staged[:rows])
    del staged

    prod = torch.empty((taps, _rows_pad(span, channels), channels), dtype=torch.float32, device=x.device)
    for start in range(0, total, chunk_rows):
        stop = min(start + chunk_rows, total)
        rows = stop - start
        conv_vjp_dx(dx, dz, wf, seq_end, start, rows, taps)
        conv_vjp_dw_products(prod, x, dz, seq_start, start, rows, taps)
        dwf.add_(torch.stack([prod[tap, :rows].sum(dim=0) for tap in range(taps)], dim=1))

    return dx, dwf.to(weight.dtype), (dbf.to(bias.dtype) if bias is not None else None)
