"""Execute the scalefma128.e4m3_e4m3 policy schedule on GLM checkpoint codes."""
import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(X, Codes, Scales, K: tl.constexpr):
    row, group = tl.program_id(0), tl.program_id(1)
    i = row * K + group * 128 + tl.arange(0, 128)
    x = tl.load(X + i).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1.e-4)
    scale = amax * (1. / 448.)
    q = tl.div_rn(x, scale)
    tl.store(Codes + i, tl.minimum(448., tl.maximum(-448., q)))
    tl.store(Scales + row * (K // 128) + group, scale)


@triton.jit
def _project(A, SA, W, SW, Out, M, N: tl.constexpr, K: tl.constexpr,
             LEAVES: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    leaf = tl.program_id(2)
    groups: tl.constexpr = K // 128
    g0 = leaf * (groups // LEAVES)
    total = tl.full((BM, BN), 0., tl.float32)
    for g in range(g0, g0 + groups // LEAVES):
        # SM100 lowers this K128 dot to four ordered tcgen05 K32 instructions
        # accumulating into the same tensor-memory tile.
        k = g * 128 + tl.arange(0, 128)
        a = tl.load(A + row[:, None] * K + k[None, :], row[:, None] < M, 0.)
        b = tl.load(W + col[:, None] * K + k[None, :], col[:, None] < N, 0.)
        dot = tl.dot(a, tl.trans(b), max_num_imprecise_acc=128)
        sa = tl.load(SA + row * groups + g, row < M, 0.)
        sw = tl.load(SW + (col // 128) * groups + g, col < N, 0.)
        total = tl.fma(dot, sa[:, None] * sw[None, :], total)
    dst = (leaf * M + row[:, None]) * N + col[None, :]
    tl.store(Out + dst, total, (row[:, None] < M) & (col[None, :] < N))


def _device(x):
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] != 10:
        raise ValueError("GLM scaled FP8 projection requires SM100")


def quantize(x):
    """BF16 [M,K] to E4M3 codes and one FP32 scale per row per 128 channels."""
    _device(x)
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[1] < 128 or x.shape[1] % 128:
        raise ValueError("expected BF16 [M,K] and K divisible by 128")
    m, k = x.shape
    codes = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((m, k // 128), device=x.device, dtype=torch.float32)
    if m:
        _quantize[(m, k // 128)](x.contiguous(), codes, scales, k, num_warps=1, enable_fp_fusion=False)
    return codes, scales


def merge_partials(partials):
    """Merge FP32 rank or segment roots in increasing K order."""
    count = partials.shape[0]
    if partials.dtype != torch.float32 or count < 1 or count & (count - 1):
        raise ValueError("expected a power of two FP32 partials in K order")
    while partials.shape[0] > 1:
        partials = partials[::2] + partials[1::2]
    return partials[0]


def project_codes(a, sa, w, sw, *, schedule, tp_degree=1, dtype=torch.float32):
    """Execute the compiled segments on this K shard. Partial roots stay FP32."""
    _device(a)
    if a.ndim != 2 or w.ndim != 2 or a.shape[1] != w.shape[1]:
        raise ValueError("expected A[M,K] and W[N,K]")
    m, k = a.shape
    n = w.shape[0]
    leaves = schedule.local_segments(n, k, tp_degree)
    if tp_degree > 1 and dtype != torch.float32:
        raise ValueError("cast after merging the FP32 K shards")
    if (sa.shape != (m, k // 128) or sw.shape != (triton.cdiv(n, 128), k // 128)
            or a.dtype != torch.float8_e4m3fn or w.dtype != a.dtype
            or sa.dtype != torch.float32 or sw.dtype != torch.float32
            or dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError("expected E4M3 codes and FP32 scales for activation rows and 128x128 weight blocks")
    if any(t.device != a.device for t in (sa, w, sw)):
        raise ValueError("codes and scales must share a device")
    if not m or not n:
        return torch.empty((m, n), dtype=dtype, device=a.device)
    root = torch.empty((leaves, m, n), dtype=dtype if leaves == 1 else torch.float32, device=a.device)
    _project[(triton.cdiv(m, 64), triton.cdiv(n, 64), leaves)](
        a.contiguous(), sa.contiguous(), w.contiguous(), sw.contiguous(), root, m, n, k, leaves, 64, 64,
        num_stages=1, enable_fp_fusion=False)
    if leaves == 1:
        return root[0]
    return merge_partials(root).to(dtype)


def project(x, w, sw, *, schedule, tp_degree=1, dtype=torch.float32):
    """Project BF16 rows with the checkpoint's E4M3 weight bytes and FP32 scales."""
    a, sa = quantize(x)
    return project_codes(a, sa, w, sw, schedule=schedule, tp_degree=tp_degree, dtype=dtype)
