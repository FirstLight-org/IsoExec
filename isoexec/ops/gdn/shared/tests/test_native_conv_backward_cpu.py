"""CPU proofs for the packed, atomics-free native GDN convolution VJP."""

from __future__ import annotations

import inspect
import re
import sys
from types import ModuleType

import pytest
import torch

from isoexec.ops.gdn.shared import gdn_conv_vjp_triton as FUSED
from isoexec.ops.gdn.shared import gdn_ops as G

_ISOEXEC = "isoexec"


def _native_conv_backward_stats():
    return dict(G._NATIVE_CONV_BWD_COUNTS)


def _eager_reference(x, weight, bias, bounds, activation):
    outputs = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        outputs.append(G.gdn_causal_conv(x[start:stop], weight, bias, activation=activation))
    return torch.cat(outputs, dim=0)


def test_native_conv_channel_last_input_is_a_zero_copy_view_for_production_layout():
    x = torch.randn(19, 7)

    x_dt = G._native_conv_channel_last_input(x)

    assert x_dt.shape == (7, 19)
    assert x_dt.stride() == (1, 7)
    assert x_dt.data_ptr() == x.data_ptr()
    assert x_dt.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert torch.equal(x_dt, x.transpose(0, 1))


def test_native_conv_channel_last_input_normalizes_only_an_unusual_source_layout():
    base = torch.randn(13, 18)
    x = base[:, ::2]
    assert x.stride(1) != 1

    x_dt = G._native_conv_channel_last_input(x)

    assert x_dt.shape == (9, 13)
    assert x_dt.stride(0) == 1 and x_dt.stride(1) > 1
    assert torch.equal(x_dt, x.transpose(0, 1))


def test_native_conv_forward_source_does_not_compact_the_transpose():
    source = inspect.getsource(G._GdnNativeConvAutograd.forward)
    helper = inspect.getsource(G._native_conv_channel_last_input)
    assert "_native_conv_channel_last_input(x)" in source
    assert "transpose(0, 1).contiguous()" not in source
    assert "x = x.contiguous()" in helper
    assert "x_dt = x.transpose(0, 1)" in helper
    assert "metadata = causal_conv1d_metadata(cu_seqlens)" in source


def test_native_conv_cached_launch_args_remove_only_stateless_scratch_work():
    source = inspect.getsource(G._GdnNativeConvAutograd.forward)
    assert "metadata = causal_conv1d_metadata(cu_seqlens)" in source
    assert "scratch = x.new_empty" in source
    assert "idx = metadata.cache_indices" in source
    assert "has0 = metadata.has_initial_state" in source
    # Cache-off must retain zero-initialized scratch.
    assert "scratch = x.new_zeros" in source


@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_conv_vjp_matches_eager_autograd(monkeypatch, activation, with_bias, dtype):
    monkeypatch.setenv("ISOEXEC_GDN_CONV_BWD_ROWS", "5")
    torch.manual_seed(17)
    bounds = [0, 7, 8, 17]
    cu = torch.tensor(bounds, dtype=torch.int32)
    x = torch.randn(bounds[-1], 11, dtype=dtype, requires_grad=True)
    weight = torch.randn(11, 4, dtype=dtype, requires_grad=True)
    bias = torch.randn(11, dtype=dtype, requires_grad=True) if with_bias else None
    dy = torch.randn_like(x)

    y = _eager_reference(x, weight, bias, bounds, activation)
    leaves = (x, weight, bias) if with_bias else (x, weight)
    reference = torch.autograd.grad(y, leaves, dy)
    with torch.no_grad():
        got = G._native_conv_vjp(dy, x.detach(), weight.detach(), bias.detach() if with_bias else None, cu, activation)

    rtol, atol = (3e-2, 3e-2) if dtype == torch.bfloat16 else (2e-5, 2e-5)
    torch.testing.assert_close(got[0], reference[0], rtol=rtol, atol=atol)
    torch.testing.assert_close(got[1], reference[1], rtol=rtol, atol=atol)
    if with_bias:
        torch.testing.assert_close(got[2], reference[2], rtol=rtol, atol=atol)
    else:
        assert got[2] is None


def test_native_conv_vjp_is_repeatable_and_does_not_cross_sequence_boundaries(monkeypatch):
    monkeypatch.setenv("ISOEXEC_GDN_CONV_BWD_ROWS", "3")
    cu = torch.tensor([0, 5, 9], dtype=torch.int32)
    x = torch.arange(9 * 3, dtype=torch.float32).view(9, 3)
    weight = torch.arange(1, 13, dtype=torch.float32).view(3, 4)
    dy = torch.zeros_like(x)
    dy[5] = 1.0  # first output of sequence two must not reach sequence one's final inputs

    with torch.no_grad():
        first = G._native_conv_vjp(dy, x, weight, None, cu, None)
        second = G._native_conv_vjp(dy, x, weight, None, cu, None)
    assert all(torch.equal(a, b) for a, b in zip(first[:2], second[:2]))
    assert torch.count_nonzero(first[0][:5]) == 0


def test_native_conv_custom_function_backward_moves_served_census(monkeypatch):
    # Stub the vendored conv and disable runtime-built metadata to test CPU autograd wiring.
    monkeypatch.setenv("ISOEXEC_PACKED_META_CACHE", "0")
    leaf_name = f"{_ISOEXEC}.ops.gdn.conv_sm90.vendored.causal_conv1d"
    modules = {}
    leaf = ModuleType(leaf_name)

    def fake_causal_conv(x_t, weight, bias, *, query_start_loc, activation, **_):
        bounds = query_start_loc.tolist()
        outputs = [
            G.gdn_causal_conv(x_t.T[start:stop], weight, bias, activation=activation)
            for start, stop in zip(bounds[:-1], bounds[1:])
        ]
        return torch.cat(outputs, dim=0).T.contiguous()

    leaf.causal_conv1d_fn = fake_causal_conv
    modules[leaf_name] = leaf
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    cu = torch.tensor([0, 4, 9], dtype=torch.int32)
    x = torch.randn(9, 6, requires_grad=True)
    weight = torch.randn(6, 4, requires_grad=True)
    bias = torch.randn(6, requires_grad=True)
    before = _native_conv_backward_stats()
    monkeypatch.setattr(G, "_NATIVE_CONV_ANALYTIC_BWD_ENABLED", False)
    out = G.gdn_native_conv(x, weight, bias, cu_seqlens=cu, activation="silu")
    out.sum().backward()
    after_fallback = _native_conv_backward_stats()

    assert after_fallback["served"] == before["served"]
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (x, weight, bias))

    x2 = x.detach().clone().requires_grad_()
    weight2 = weight.detach().clone().requires_grad_()
    bias2 = bias.detach().clone().requires_grad_()
    monkeypatch.setattr(G, "_NATIVE_CONV_ANALYTIC_BWD_ENABLED", True)
    out2 = G.gdn_native_conv(x2, weight2, bias2, cu_seqlens=cu, activation="silu")
    out2.sum().backward()
    after_analytic = _native_conv_backward_stats()

    assert after_analytic["served"] == after_fallback["served"] + 1
    assert after_analytic["tokens"] == after_fallback["tokens"] + 9
    assert after_analytic["taps"] == after_fallback["taps"] + 4
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (x2, weight2, bias2))


def test_analytic_native_conv_branch_has_no_eager_graph_or_sequence_host_loop():
    source = inspect.getsource(G._GdnNativeConvAutograd.backward)
    assert "_native_conv_vjp" in source
    assert "if not _NATIVE_CONV_ANALYTIC_BWD_ENABLED:" in source
    # Compile each chunk body through the backward-region ledger.
    helper = inspect.getsource(G._native_conv_vjp_eager)
    chunks = "".join(
        inspect.getsource(fn) for fn in (G._conv_vjp_dz_chunk, G._conv_vjp_act_chunk, G._conv_vjp_dxdw_chunk)
    )
    whole = helper + chunks
    assert "for tap in range(taps)" in chunks
    assert "torch.enable_grad" not in whole
    assert "torch.autograd.grad" not in whole
    assert ".tolist()" not in whole
    assert ".index_add" not in whole and ".scatter" not in whole
    assert "dxf = torch.zeros((rows, channels)" in chunks
    assert "torch.zeros_like(dz)" not in whole, "dx accumulation must stay row-chunked"
    # Keep the outer chunk loop uncompiled to preserve the temporary-memory bound.
    assert "for start in range(0, total, chunk_rows)" in helper
    assert "range(0, total, chunk_rows)" not in chunks


def test_native_conv_vjp_dispatches_cuda_to_the_fused_kernels_and_keeps_eager_as_the_reference():
    source = inspect.getsource(G._native_conv_vjp)
    assert "if x.is_cuda and fused_conv_backward_enabled():" in source
    assert "native_conv_vjp_fused(" in source
    assert "return _native_conv_vjp_eager(dy, x, weight, bias, cu_seqlens, activation)" in source
    # The dispatcher itself must not do arithmetic: it validates, derives the bounds, and routes.
    assert "torch.sigmoid" not in source and "for tap in range" not in source


# Compile production specializations for SM90 using the launches' own _COMPILE_OPTS.
_PREACT_SIG = {
    "x_ptr": "*bf16",
    "dy_ptr": "*bf16",
    "w_ptr": "*fp32",
    "bias_ptr": "*fp32",
    "seq_start_ptr": "*i64",
    "out_ptr": "*fp32",
    "row0": "i32",
    "rows": "i32",
    "channels": "i32",
    "HAS_BIAS": "constexpr",
    "SILU": "constexpr",
    "TAPS": "constexpr",
    "BR": "constexpr",
    "BC": "constexpr",
}
_DX_SIG = {
    "dz_ptr": "*fp32",
    "w_ptr": "*fp32",
    "seq_end_ptr": "*i64",
    "dx_ptr": "*bf16",
    "row0": "i32",
    "rows": "i32",
    "total": "i32",
    "channels": "i32",
    "TAPS": "constexpr",
    "BR": "constexpr",
    "BC": "constexpr",
}
_DW_SIG = {
    "x_ptr": "*bf16",
    "dz_ptr": "*fp32",
    "seq_start_ptr": "*i64",
    "prod_ptr": "*fp32",
    "row0": "i32",
    "rows": "i32",
    "channels": "i32",
    "tap_stride": "i32",
    "TAPS": "constexpr",
    "BR": "constexpr",
    "BC": "constexpr",
}
_BLOCKS = {"TAPS": 4, "BR": FUSED._BLOCK_R, "BC": FUSED._BLOCK_C}


def _fused_kernel_ptx(kernel, signature, constexprs):
    """PTX for one fused VJP kernel at SM90, using the launches' own options. No GPU needed."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    compiled = triton.compile(
        ASTSource(fn=kernel, signature=signature, constexprs=constexprs),
        target=GPUTarget("cuda", 90, 32),
        options=dict(FUSED._COMPILE_OPTS),
    )
    return compiled.asm["ptx"]


def _f32_arithmetic(ptx):
    """The set of f32 arithmetic mnemonics the PTX contains, e.g. ``{'add.rn.f32', 'div.rn.f32'}``."""
    return set(re.findall(r"\b(?:add|sub|mul|div|fma|ex2|lg2|rcp|sqrt|rsqrt)\.[a-z0-9.]+f32\b", ptx))


def test_fused_conv_vjp_kernels_emit_the_ptx_aten_emits():
    """Check emitted PTX against ATen's sigmoid rounding with nvcc -ftz=false.
    Only expf's bounded range-reduction ex2 may use FTZ; division and other f32 operations must
    preserve subnormals, including preactivations in (-88.7228, -87.3345)."""
    ptx = _fused_kernel_ptx(FUSED._preact_act_kernel, _PREACT_SIG, {"HAS_BIAS": True, "SILU": True, **_BLOCKS})
    ops = _f32_arithmetic(ptx)

    # Division must preserve subnormal results with round-to-nearest.
    assert "div.rn.f32" in ops, f"the sigmoid divide must be an IEEE RN divide; got {sorted(ops)}"
    assert "div.rn.ftz.f32" not in ops, "a flush-to-zero divide returns +0.0 where ATen returns a subnormal"
    assert not any(
        op.startswith("div.full") or op.startswith("div.approx") for op in ops
    ), f"Triton's `/` lowers to div.full.f32, which is not round-to-nearest; got {sorted(ops)}"

    # Only expf's bounded range-reduction ex2 may use FTZ; bare ex2 bypasses that reduction.
    assert "ex2.approx.f32" not in ops, "tl.exp/tl.sigmoid's un-reduced ex2 is 64 ULP from torch.sigmoid"
    assert "ex2.approx.ftz.f32" in ops, "libdevice expf's range reduction should still be here"

    # Nothing else may flush. `expf`'s reduction FMAs must be the non-ftz ones nvcc emits.
    flushing = {op for op in ops if ".ftz." in op} - {"ex2.approx.ftz.f32"}
    assert not flushing, f"flush-to-zero f32 arithmetic outside libdevice expf's ex2: {sorted(flushing)}"
    assert {"fma.rm.f32", "fma.rn.f32"} <= ops, f"expf's range reduction should be non-ftz; got {sorted(ops)}"

    # The tap accumulator is a separate multiply and a separate add, as the reference's is.
    assert "mul.rn.f32" in ops and "add.rn.f32" in ops


def test_fused_conv_vjp_tap_loops_carry_no_fma_and_no_reduction():
    """`dx` and the `dw` products: no contraction, no flush, no atomic, no cross-thread reduction."""
    for kernel, signature in ((FUSED._dx_kernel, _DX_SIG), (FUSED._dw_products_kernel, _DW_SIG)):
        ptx = _fused_kernel_ptx(kernel, signature, dict(_BLOCKS))
        ops = _f32_arithmetic(ptx)
        assert not any(
            op.startswith("fma.") for op in ops
        ), f"{kernel.__name__} contracted its tap loop into an FMA, dropping a rounding: {sorted(ops)}"
        assert not any(".ftz." in op for op in ops), f"{kernel.__name__} flushes denormals: {sorted(ops)}"
        assert "mul.rn.f32" in ops
        assert "atom." not in ptx and "red." not in ptx, f"{kernel.__name__} must stay atomics-free"

    # dw/db reductions stay in ATen: no cross-thread reduction may appear in the kernels at all.
    fused = inspect.getsource(FUSED)
    assert "tl.sum(" not in fused and "tl.atomic" not in fused


def test_every_fused_launch_uses_the_one_compile_option_set():
    # AOT tests and actual launches must use identical compile options.
    fused = inspect.getsource(FUSED)
    launches = sum(1 for line in fused.splitlines() if line.strip() == "**_COMPILE_OPTS,")
    assert launches == fused.count("@triton.jit") == 3, "every kernel launch must pass _COMPILE_OPTS"
    assert "enable_fp_fusion=False," not in fused, "the options live in _COMPILE_OPTS, nowhere else"
    assert FUSED._COMPILE_OPTS == {"enable_fp_fusion": False, "enable_reflect_ftz": False}
    # Use the audited non-FTZ sigmoid; a local libdevice divide can flush subnormal quotients.
    assert "from ....core.triton_nonftz import sigmoid as _nonftz_sigmoid" in fused
    assert "_nonftz_sigmoid(acc)" in fused
    assert "libdevice" not in fused.split('"""', 3)[-1], "no local libdevice call may re-introduce an FTZ op"
    assert "tl.sigmoid(" not in fused and "tl.exp(" not in fused


def test_fused_conv_vjp_product_buffer_slices_keep_the_reference_reduction_alignment():
    # `sum(dim=0)` picks its schedule partly from the operand's alignment, so every tap slice must
    # start where a freshly allocated [rows, channels] tensor would: on a 512-byte boundary.
    for rows, channels in ((5, 11), (9, 6), (3584, 8192), (23, 11), (1, 3)):
        padded = FUSED._rows_pad(rows, channels)
        assert padded >= rows
        assert (padded * channels * 4) % 512 == 0


def test_native_conv_backward_rows_refuses_nonpositive(monkeypatch):
    monkeypatch.setenv("ISOEXEC_GDN_CONV_BWD_ROWS", "0")
    with pytest.raises(ValueError, match="must be positive"):
        G._native_conv_backward_rows()
