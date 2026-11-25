# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# [2025-07-04] Version in Cute-DSL, for Hopper and Blackwell. You'll need install nvidia-cutlass-dsl==4.2.0.
# Modified for IsoExec: the SM100 varlen backward of flash_attn/cute/interface.py (_flash_attn_bwd with
# _bwd_preprocess / _bwd_postprocess_convert), extracted. Other architectures, the forward, score_mod,
# mask_mod, block sparsity, softcap, dlse, seqused and the on-disk JIT cache are removed; compiled kernels
# are cached in process-local dicts, and the 2-CTA choice is the `disable_2cta` argument instead of the
# FA_DISABLE_2CTA environment variable. The kernel selection, tile sizes, buffer layouts and launch
# sequence are upstream's. See PROVENANCE.md and upstream_manifest.json beside this file.

import math
from typing import Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from .quack.compile_utils import make_fake_tensor as fake_tensor

from . import utils
from .cute_dsl_utils import to_cute_tensor, get_broadcast_dims
from .flash_bwd_preprocess import FlashAttentionBackwardPreprocess
from .flash_bwd_sm100 import FlashAttentionBackwardSm100
from .flash_bwd_postprocess import FlashAttentionBackwardPostprocess

# SM100 head_dim=256 2CTA kernel imports
from .sm100_hd256_2cta_fmha_backward import BlackwellFusedMultiHeadAttentionBackward

ARCH = 100


def _validate_head_dims(head_dim: int, head_dim_v: int, alignment: int) -> None:
    """Validate head dimension constraints for SM100."""
    is_deepseek_shape = head_dim == 192 and head_dim_v == 128
    is_dedicate_kernel_shape = head_dim == 256 and head_dim_v == 256
    is_standard_range = 8 <= head_dim <= 128 and 8 <= head_dim_v <= 128
    assert (is_standard_range or is_deepseek_shape or is_dedicate_kernel_shape) and head_dim % alignment == 0 and head_dim_v % alignment == 0, (
        f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported on SM100/SM110. "
        f"head_dim and head_dim_v must be between 8 and 128 and divisible by {alignment}, or (192, 128) for DeepSeek, or (256, 256) for hd256."
    )


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _validate_tensor(t, name, expected_shape, expected_dtype, expected_device):
    assert t.shape == expected_shape, f"{name} shape {t.shape} != expected {expected_shape}"
    assert t.dtype == expected_dtype, f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert t.device == expected_device, f"{name} device {t.device} != expected {expected_device}"
    assert t.is_cuda, f"{name} must be on CUDA"


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def make_fake_bwd_tensors(dtype, has_gqa, varlen_q, varlen_k):
    sym = cute.sym_int
    # divisibility in elements: assumed_align_bytes = divisibility * dtype.width // 8
    # For 16-byte align: fp16/bf16 → divisibility=8, float32 → divisibility=4
    div = 128 // dtype.width  # 8 for fp16/bf16
    # Shared sym_ints for dimensions that must match across tensors
    b, seqlen_q, seqlen_k, h_q, d, d_v = sym(), sym(), sym(), sym(), sym(), sym()
    h_kv = h_q if not has_gqa else sym()
    seqlen_q_rounded, seqlen_k_rounded = sym(), sym()
    seqlen_q_d_rounded, seqlen_k_d_rounded, seqlen_k_dv_rounded = sym(), sym(), sym()
    total_q, total_k, total_q_rounded, total_k_rounded = sym(), sym(), sym(), sym()
    total_q_d_rounded, total_k_d_rounded, total_k_dv_rounded = sym(), sym(), sym()
    b_seqlenq = (b, seqlen_q) if not varlen_q else (total_q,)
    b_seqlenk = (b, seqlen_k) if not varlen_k else (total_k,)
    mQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mdO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)
    mdQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mdK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mdV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)
    if not varlen_q:
        mLSE = fake_tensor(Float32, (b, h_q, seqlen_q), divisibility=1)
        mLSElog2 = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
        mPdPsum = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
        dQaccum = fake_tensor(Float32, (b, h_q, seqlen_q_d_rounded), divisibility=4)
    else:
        mLSE = fake_tensor(Float32, (h_q, total_q), divisibility=1)
        mLSElog2 = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
        mPdPsum = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
        dQaccum = fake_tensor(Float32, (h_q, total_q_d_rounded), divisibility=4)
    if not has_gqa:
        mdKaccum, mdVaccum = None, None
    else:
        if not varlen_k:
            mdKaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_dv_rounded), divisibility=4)
        else:
            mdKaccum = fake_tensor(Float32, (h_kv, total_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(Float32, (h_kv, total_k_dv_rounded), divisibility=4)
    return mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, dQaccum, mdKaccum, mdVaccum


def _compile_bwd_preprocess(
    dtype, head_dim, head_dim_v, m_block_size, has_cuseqlens_q, has_seqused_q, has_dlse, has_dq_accum,
    use_padded_offsets,
):
    """Compile bwd preprocess kernel using cute fake tensors (no real GPU tensors needed)."""
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum = make_fake_bwd_tensors(
        dtype, has_gqa=True, varlen_q=has_cuseqlens_q, varlen_k=False
    )
    batch = mQ.shape[0] if not has_cuseqlens_q else cute.sym_int()
    batchp1 = cute.sym_int()
    mCuSeqlensQ = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cuseqlens_q else None
    mSequsedQ = fake_tensor(Int32, (batch,), divisibility=1) if has_seqused_q else None
    mdLSE = fake_tensor(Float32, mLSE.shape, divisibility=1) if has_dlse else None
    mdQaccum = mdQaccum if has_dq_accum else None
    fa_bwd_pre = FlashAttentionBackwardPreprocess(
        dtype, head_dim, head_dim_v, m_block_size, use_padded_offsets=use_padded_offsets
    )
    return cute.compile(
        fa_bwd_pre, mO, mdO, mPdPsum, mLSE, mLSElog2, mdQaccum, mCuSeqlensQ, mSequsedQ, mdLSE,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_preprocess(
    out, dout, dpsum, lse, lse_log2, dq_accum,
    cu_seqlens_q, seqused_q, dlse,
    dtype, head_dim, head_dim_v, m_block_size,
    use_padded_offsets=True,
):
    """Backward preprocess: compute (o * dout).sum(dim=-1) - dLSE, lse * log2_e, and zero out dq_accum."""
    is_varlen = cu_seqlens_q is not None
    compile_key = (
        dtype, head_dim, head_dim_v, m_block_size, is_varlen, seqused_q is not None, dlse is not None, dq_accum is not None,
        use_padded_offsets,
    )
    if compile_key not in _bwd_preprocess.compile_cache:
        _bwd_preprocess.compile_cache[compile_key] = _compile_bwd_preprocess(*compile_key)
    _bwd_preprocess.compile_cache[compile_key](
        out, dout, dpsum, lse, lse_log2, dq_accum, cu_seqlens_q, seqused_q, dlse
    )


_bwd_preprocess.compile_cache = {}


def _compile_bwd_postprocess(
    dtype, hdim, block_size, num_threads, atom_layout, swap_ab,
    has_cuseqlens_q, has_seqused_q,
    use_2cta_instrs, cluster_size, arch,
):
    """Compile bwd postprocess kernel using cute fake tensors."""
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum = make_fake_bwd_tensors(
        dtype, has_gqa=True, varlen_q=has_cuseqlens_q, varlen_k=False
    )
    batch = mQ.shape[0] if not has_cuseqlens_q else cute.sym_int()
    batchp1 = cute.sym_int()
    mCuSeqlensQ = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cuseqlens_q else None
    mSeqUsedQ = fake_tensor(Int32, (batch,), divisibility=1) if has_seqused_q else None
    fa_bwd_post = FlashAttentionBackwardPostprocess(
        dtype, hdim, arch, block_size, num_threads, atom_layout, swap_ab,
        use_2cta_instrs=use_2cta_instrs,
        cluster_size=cluster_size,
    )
    return cute.compile(
        fa_bwd_post, mdQaccum, mdQ, Float32(0.0), mCuSeqlensQ, mSeqUsedQ,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_postprocess_convert(
    accum, output, scale,
    cu_seqlens, seqused,
    arch, dtype, hdim, block_size, num_threads,
    atom_layout, swap_ab,
    use_2cta_instrs=False, cluster_size=1,
):
    """Backward postprocess: convert float32 accumulator to bf16/fp16 output."""
    compile_key = (
        dtype, hdim, block_size, num_threads, atom_layout, swap_ab,
        cu_seqlens is not None, seqused is not None,
        use_2cta_instrs, cluster_size, arch,
    )
    if compile_key not in _bwd_postprocess_convert.compile_cache:
        _bwd_postprocess_convert.compile_cache[compile_key] = _compile_bwd_postprocess(*compile_key)
    _bwd_postprocess_convert.compile_cache[compile_key](
        accum, output, scale, cu_seqlens, seqused,
    )


_bwd_postprocess_convert.compile_cache = {}


def flash_attn_varlen_bwd_sm100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    deterministic: bool = False,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    disable_2cta: bool = False,
    stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    q_tile_mask: Optional[torch.Tensor] = None,
    kv_tile_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``_flash_attn_bwd`` on SM100 for packed varlen Q/K/V, ``lse`` [num_head, total_q] (natural log).

    IsoExec additions, head dim 256 only. ``stats`` is ``(lse_log2, dpsum)`` already in the preprocess's
    padded layout for ``cu_seqlens_q``; the preprocess is then skipped and ``out``/``lse`` are unused.
    ``q_tile_mask`` int32 ``[batch, ceil(max_seqlen_q / 256)]`` and ``kv_tile_mask`` int32
    ``[batch, ceil(max_seqlen_k / 128)]`` select the dQ and dK/dV tiles to compute (nonzero = compute);
    a computed tile runs exactly as without a mask, a dropped dQ tile stores zeros and a dropped dK/dV
    tile stores nothing.
    """
    # _resolve_causal_local_window(causal, None, None): no window either way.
    window_size_left = None
    window_size_right = None
    local = False
    num_head, head_dim = q.shape[-2:]
    head_dim_v = v.shape[-1]

    m_block_size = 128
    n_block_size = 128
    dQ_swapAB = False
    dKV_swapAB = False
    AtomLayoutMdQ = 1
    AtomLayoutNdKV = 1
    cluster_size = 2 if head_dim >= 128 and not disable_2cta else 1
    use_2cta_instrs = cluster_size == 2

    use_dedicated_hd256_kernel = head_dim == 256 and head_dim_v == 256
    use_2cta_instrs = use_2cta_instrs or use_dedicated_hd256_kernel
    if (stats is not None or q_tile_mask is not None or kv_tile_mask is not None) and not use_dedicated_hd256_kernel:
        raise ValueError("stats and tile masks are served by the head-dim-256 backward only")
    for mask in (q_tile_mask, kv_tile_mask):
        if mask is not None and (mask.dtype != torch.int32 or mask.dim() != 2 or mask.stride(1) != 1
                                 or mask.stride(0) % 4 or mask.shape[0] != cu_seqlens_q.shape[0] - 1):
            raise ValueError("a tile mask is int32 [batch, tiles] with unit inner stride and rows of 16-byte multiples")

    q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k = [
        maybe_contiguous(t)
        for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k)
    ]
    batch_size = cu_seqlens_q.shape[0] - 1
    total_q = q.shape[0]
    seqlen_q = max_seqlen_q if max_seqlen_q is not None else total_q

    batch_size = cu_seqlens_k.shape[0] - 1
    total_k = k.shape[0]
    seqlen_k = max_seqlen_k if max_seqlen_k is not None else total_k

    num_head_kv = k.shape[-2]

    seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
    seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size
    num_n_blocks = seqlen_k_rounded // n_block_size
    if cluster_size == 2 and num_n_blocks % cluster_size != 0:
        seqlen_k_rounded = seqlen_k_rounded + n_block_size

    assert k.shape == (total_k, num_head_kv, head_dim)
    assert v.shape == (total_k, num_head_kv, head_dim_v)
    assert cu_seqlens_k.shape == (batch_size + 1,), (
        "cu_seqlens_k must have shape (batch_size + 1,)"
    )
    assert cu_seqlens_q.shape == (batch_size + 1,), (
        "cu_seqlens_q must have shape (batch_size + 1,)"
    )
    assert dout.shape == (total_q, num_head, head_dim_v)
    if stats is None:
        assert out.shape == (total_q, num_head, head_dim_v)
        assert lse.shape == (num_head, total_q), "lse must have shape (num_head, total_q)"

    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype == dout.dtype, (
        "inputs must have the same dtype"
    )
    for t in [cu_seqlens_q, cu_seqlens_k]:
        assert t.dtype == torch.int32, "cu_seqlens_q, cu_seqlens_k must be int32"
    if stats is None:
        assert out.dtype == q.dtype, "inputs must have the same dtype"
        assert lse.dtype == torch.float32, "lse must be float32"
    assert all(
        t is None or t.is_cuda for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k)
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    alignment = 16 // q.element_size()
    _validate_head_dims(head_dim, head_dim_v, alignment)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv
    # pack_gqa backward not yet supported in bwd
    pack_gqa = False

    device = q.device
    out_torch_dtype = q.dtype

    if dq is None:
        dq = torch.empty_like(q)
    else:
        _validate_tensor(dq, "dq", q.shape, out_torch_dtype, device)

    if dk is None:
        dk = torch.empty_like(k)
    else:
        _validate_tensor(dk, "dk", k.shape, out_torch_dtype, device)

    if dv is None:
        dv = torch.empty_like(v)
    else:
        _validate_tensor(dv, "dv", v.shape, out_torch_dtype, device)

    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32

    total_q_rounded_padded = (
        (total_q + cu_seqlens_q.shape[0] * m_block_size - 1) // m_block_size * m_block_size
    )
    dq_accum = (
        None
        if use_dedicated_hd256_kernel
        else torch.empty(
            num_head, total_q_rounded_padded * head_dim_rounded, dtype=torch.float32, device=device
        )
    )
    if stats is None:
        dpsum = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)
        lse_log2 = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)
    else:
        lse_log2, dpsum = stats
        assert lse_log2.shape == dpsum.shape == (num_head, total_q_rounded_padded), "stats must be padded [H, rows]"

    # GQA (qhead_per_kvhead > 1) needs dK/dV accum+postprocess since multiple Q heads
    # accumulate into the same dK/dV.
    # hd=256 2CTA backward has its own internal postprocess for dK/dV.
    dKV_postprocess = qhead_per_kvhead > 1 and not use_dedicated_hd256_kernel
    if dKV_postprocess:
        head_dim_v_rounded = (head_dim_v + 32 - 1) // 32 * 32
        cluster_tile_n = cluster_size * n_block_size
        total_k_rounded_padded = (
            (total_k + cu_seqlens_k.shape[0] * cluster_tile_n - 1) // cluster_tile_n * cluster_tile_n
        )
        dk_accum = torch.zeros(
            num_head_kv,
            total_k_rounded_padded * head_dim_rounded,
            dtype=torch.float32,
            device=device,
        )
        dv_accum = torch.zeros(
            num_head_kv,
            total_k_rounded_padded * head_dim_v_rounded,
            dtype=torch.float32,
            device=device,
        )

    dtype = torch2cute_dtype_map[q.dtype]
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    if deterministic:
        dQ_semaphore = torch.zeros(batch_size, num_head, seqlen_q_rounded // m_block_size, cluster_size, dtype=torch.int32, device=device)
    else:
        dQ_semaphore = None

    if deterministic and qhead_per_kvhead > 1:
        dK_semaphore = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device=device)
        dV_semaphore = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device=device)
    else:
        dK_semaphore = None
        dV_semaphore = None

    # Preprocess kernel: compute (o * dout).sum(dim=-1) - dLSE, lse * log2_e, and zero out dq_accum.
    # For hd=256 dedicated path, dq_accum is None so preprocess only fills dpsum/lse_log2.
    if stats is None:
        _bwd_preprocess(
            out, dout, dpsum, lse, lse_log2, dq_accum,
            cu_seqlens_q, None, None,
            dtype, head_dim, head_dim_v, m_block_size,
            use_padded_offsets=use_dedicated_hd256_kernel,
        )
    num_threads = 384

    spt = (causal or local) and deterministic

    compile_key = (
        ARCH,
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        window_size_left is not None,
        window_size_right is not None,
        m_block_size,
        n_block_size,
        num_threads,
        pack_gqa,
        cluster_size,
        use_2cta_instrs,
        deterministic,
        spt,
        get_broadcast_dims(q),
        get_broadcast_dims(k),
        get_broadcast_dims(v),
        get_broadcast_dims(dout),
        # Prevent TVM stride poisoning when only one block is present.
        (seqlen_q_rounded // m_block_size == 1),
        (seqlen_k_rounded // n_block_size == 1),
        q_tile_mask is not None,
        kv_tile_mask is not None,
    )

    if compile_key not in flash_attn_varlen_bwd_sm100.compile_cache:
        q_tensor, k_tensor, v_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [
            to_cute_tensor(t) for t in (q, k, v, dout, dq, dk, dv)
        ]
        lse_log2_tensor, dpsum_tensor = [to_cute_tensor(t) for t in (lse_log2, dpsum)]
        q_tile_mask_tensor, kv_tile_mask_tensor = [
            None if t is None else to_cute_tensor(t, assumed_align=4) for t in (q_tile_mask, kv_tile_mask)
        ]
        dq_accum_tensor = to_cute_tensor(dq_accum) if dq_accum is not None else None
        if dKV_postprocess:
            dk_accum_tensor, dv_accum_tensor = [
                to_cute_tensor(t) for t in (dk_accum, dv_accum)
            ]
        cu_seqlens_q_tensor, cu_seqlens_k_tensor = [
            to_cute_tensor(t, assumed_align=4) for t in (cu_seqlens_q, cu_seqlens_k)
        ]
        dQ_semaphore_tensor, dK_semaphore_tensor, dV_semaphore_tensor = [
            utils.convert_from_dlpack_leading_static(t.detach(), leading_dim=3, alignment=4, stride_order=t.dim_order())
            if t is not None else None
            for t in (dQ_semaphore, dK_semaphore, dV_semaphore)
        ]
        if use_dedicated_hd256_kernel:
            dq_tile_mn = (128, 128)
            dkdv_tile_mn = (128, 64)
            fa_bwd_obj = BlackwellFusedMultiHeadAttentionBackward(
                head_dim,
                head_dim_v,
                is_causal=causal,
                is_local=local,
                qhead_per_kvhead=qhead_per_kvhead,
                is_persistent=False,
                deterministic=deterministic,
                cluster_size=cluster_size,
                use_2cta_instrs=use_2cta_instrs,
                tile_m_dq=dq_tile_mn[0],
                tile_n_dq=dq_tile_mn[1],
                tile_m_dkdv=dkdv_tile_mn[0],
                tile_n_dkdv=dkdv_tile_mn[1],
            )
        else:
            fa_bwd_obj = FlashAttentionBackwardSm100(
                head_dim,
                head_dim_v,
                is_causal=causal,
                is_local=local,
                qhead_per_kvhead=qhead_per_kvhead,
                tile_m=m_block_size,
                tile_n=n_block_size,
                cluster_size=cluster_size,
                use_2cta_instrs=use_2cta_instrs,
                deterministic=deterministic,
                spt=spt,
            )

        dq_accum_tensor = dq_tensor if use_dedicated_hd256_kernel else dq_accum_tensor

        flash_attn_varlen_bwd_sm100.compile_cache[compile_key] = cute.compile(
            fa_bwd_obj,
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_accum_tensor,
            dk_tensor if not dKV_postprocess else dk_accum_tensor,
            dv_tensor if not dKV_postprocess else dv_accum_tensor,
            softmax_scale,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            None,
            None,
            window_size_left,
            window_size_right,
            dQ_semaphore_tensor,
            dK_semaphore_tensor,
            dV_semaphore_tensor,
            None,
            None,
            *((q_tile_mask_tensor, kv_tile_mask_tensor) if use_dedicated_hd256_kernel else ()),
            current_stream,
            options="--enable-tvm-ffi",
        )
    dq_accum = dq if use_dedicated_hd256_kernel else dq_accum
    flash_attn_varlen_bwd_sm100.compile_cache[compile_key](
        q.detach(),
        k.detach(),
        v.detach(),
        dout,
        lse_log2,
        dpsum,
        dq_accum,
        dk if not dKV_postprocess else dk_accum,
        dv if not dKV_postprocess else dv_accum,
        softmax_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        None,
        window_size_left,
        window_size_right,
        dQ_semaphore,
        dK_semaphore,
        dV_semaphore,
        None,
        None,
        *((q_tile_mask, kv_tile_mask) if use_dedicated_hd256_kernel else ()),
    )
    # Postprocess: convert dq_accum from float32 to dq in bf16/fp16
    # hd=256 2CTA backward has its own internal postprocess, skip here.
    if not use_dedicated_hd256_kernel:
        num_threads_post_dQ = 128
        num_threads_post_dKV = 128

        _bwd_postprocess_convert(
            dq_accum, dq, softmax_scale,
            cu_seqlens_q, None,
            ARCH, dtype, head_dim, m_block_size, num_threads_post_dQ,
            AtomLayoutMdQ, dQ_swapAB,
            use_2cta_instrs=use_2cta_instrs, cluster_size=1,
        )

        if dKV_postprocess:
            # Postprocess: convert dk_accum from float32 to dk in bf16/fp16
            _bwd_postprocess_convert(
                dk_accum, dk, softmax_scale,
                cu_seqlens_k, None,
                ARCH, dtype, head_dim, n_block_size, num_threads_post_dKV,
                AtomLayoutNdKV, dKV_swapAB,
                cluster_size=cluster_size,
            )
            # Postprocess: convert dv_accum from float32 to dv in bf16/fp16
            _bwd_postprocess_convert(
                dv_accum, dv, 1.0,
                cu_seqlens_k, None,
                ARCH, dtype, head_dim_v, n_block_size, num_threads_post_dKV,
                AtomLayoutNdKV, dKV_swapAB,
                cluster_size=cluster_size,
            )

    return dq, dk, dv


flash_attn_varlen_bwd_sm100.compile_cache = {}


def backward_stats_sm100(
    out: torch.Tensor, dout: torch.Tensor, lse: torch.Tensor, cu_seqlens_q: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """IsoExec addition: the head-dim-256 backward's preprocess alone -> ``(lse_log2, dpsum)`` fp32
    ``[num_head, total_q_rounded_padded]`` in the padded per-request layout its kernels read, exactly the
    statistics ``flash_attn_varlen_bwd_sm100`` computes before its launch."""
    out, dout, lse, cu_seqlens_q = [maybe_contiguous(t) for t in (out, dout, lse, cu_seqlens_q)]
    total_q, num_head, head_dim_v = out.shape
    assert head_dim_v == 256 and dout.shape == out.shape and lse.shape == (num_head, total_q)
    assert cu_seqlens_q.dtype == torch.int32 and lse.dtype == torch.float32
    m_block_size = 128
    rows = (total_q + cu_seqlens_q.shape[0] * m_block_size - 1) // m_block_size * m_block_size
    dpsum = torch.empty(num_head, rows, dtype=torch.float32, device=out.device)
    lse_log2 = torch.empty(num_head, rows, dtype=torch.float32, device=out.device)
    _bwd_preprocess(
        out, dout, dpsum, lse, lse_log2, None,
        cu_seqlens_q, None, None,
        torch2cute_dtype_map[out.dtype], head_dim_v, head_dim_v, m_block_size,
        use_padded_offsets=True,
    )
    return lse_log2, dpsum
