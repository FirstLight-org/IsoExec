"""Torch-facing surface of the sm90 segment-tree GEMM.

Conventions: ``mm_sm90(a, b)`` takes ``a[M, K]`` and ``b = W.t()`` (``W[N, K]`` row-major), the
``matmul_persistent`` convention. The pik entry points take ``x[M, K_local]`` and ``w[N, K_local]``
(pik's layout) and write a caller-owned buffer in place.

Capture safety: no entry point synchronises the host, and the only allocations are the output of
``mm_sm90`` (from PyTorch's caching allocator, which graph capture supports) and the split plan's
workspace, which is allocated eagerly and refused during capture (``_workspace``).
"""

from __future__ import annotations

from functools import lru_cache

import torch

from . import segments as seg
from ._extension import extension
from .spec import RECIPE_VERSION

__all__ = [
    "RECIPE_VERSION",
    "UnsupportedShape",
    "config_ids",
    "leaf_node",
    "leaves",
    "mm",
    "mm_sm90",
    "require_sm90",
    "run_plan",
    "supported_shape",
]


class UnsupportedShape(ValueError):
    """The sm90 kernel declines this shape; the caller must fall through to another provider."""


@lru_cache(None)
def config_ids() -> dict[str, int]:
    """Configuration name -> index in the compiled table, checked against ``segments._CONFIGS`` both ways."""
    ext = extension()
    ids = {}
    for i in range(ext.num_configs()):
        info = ext.config_info(i)
        name = info["name"]
        ids[name] = i
        spec = seg._CONFIGS.get(name)
        if spec is None:
            raise RuntimeError(f"compiled configuration {name!r} is not planned by segments._CONFIGS; drift")
        got = seg.ConfigSpec(info["block_m"], info["block_n"], info["cluster_m"], info["cluster_n"],
                             info["schedule"], info["out_fp32"], info["levels"], info["swap"])
        if got != spec or info["block_k"] != seg.SEGMENT_ALIGN:
            raise RuntimeError(f"configuration {name!r} is compiled as {got} (block_k {info['block_k']}) "
                               f"but segments.py plans {spec}")
    missing = sorted(set(seg._CONFIGS) - set(ids))
    if missing:
        raise RuntimeError(f"segments._CONFIGS plans {missing}, which the compiled module lacks; drift")
    return ids


def require_sm90() -> None:
    from isoexec.core.arch import live_device_arch

    arch = live_device_arch()
    if arch != "sm90":
        raise RuntimeError(f"mm_sm90 requires an sm90 device, not {arch}")


def supported_shape(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether :func:`mm_sm90` will serve ``a @ b`` (``b = W.t()``), without raising."""
    if a.dtype is not torch.bfloat16 or b.dtype is not torch.bfloat16:
        return False
    if a.dim() != 2 or b.dim() != 2 or not (a.is_cuda and b.is_cuda):
        return False
    M, K = a.shape
    if b.shape[0] != K or M < 1:
        return False
    N = b.shape[1]
    # b must view a [N, K] weight with unit stride on K and a 16-byte row pitch
    if b.stride(0) != 1 or b.stride(1) % 8 or b.stride(1) < K:
        return False
    if a.stride(1) != 1 or a.stride(0) % 8 or a.stride(0) < K:
        return False
    if a.data_ptr() % 16 or b.data_ptr() % 16:
        return False
    return seg.supported(K, N)


#: Persistent-scheduler CTA cap (0 = every SM). Schedule only: a tile's K chain, segments and epilogue do not
#: depend on which CTA runs it or how many run, so the bits are the same at every cap. pik's pipelined
#: row-parallel site lowers it while a producer GEMM chunk overlaps a reduce, leaving SMs to the reduce.
_SM_LIMIT = 0


def set_sm_limit(n: int) -> int:
    """Set the persistent CTA cap (0 = the device's SM count); returns the previous cap."""
    global _SM_LIMIT
    prev, _SM_LIMIT = _SM_LIMIT, max(0, int(n))
    return prev


def device_sm_count() -> int:
    return int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)


def _launch(cid: int, x: torch.Tensor, w: torch.Tensor, d: torch.Tensor, M: int, N: int, kseg: int, L: int,
            d_batch: int, seg_k_tiles: int, round_leaf: bool) -> None:
    extension().run(cid, x, w, d, M, N, kseg, L, N, kseg, kseg, d_batch, seg_k_tiles, round_leaf, _SM_LIMIT)


_WORKSPACE: dict = {}
_WORKSPACE_RETIRED: list = []


def _workspace(numel: int, device: torch.device) -> torch.Tensor:
    """A growable fp32 scratch buffer per device; growth during CUDA capture is refused."""
    key = device.index
    buf = _WORKSPACE.get(key)
    if buf is None or buf.numel() < numel:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("mm_sm90 split workspace growth during CUDA capture: warm this shape eagerly first")
        if buf is not None:
            _WORKSPACE_RETIRED.append(buf)  # a captured graph may still hold its pointer
        buf = torch.empty(max(numel, 2 * (buf.numel() if buf is not None else 0)), dtype=torch.float32,
                          device=device)
        _WORKSPACE[key] = buf
    return buf[:numel]


def run_plan(x: torch.Tensor, w: torch.Tensor, plan: seg.Plan, *, out: torch.Tensor | None = None,
             round_leaf: bool = False) -> torch.Tensor:
    """Run one explicit plan of ``x[M, K] @ w[N, K]^T`` into ``out`` (bf16 or fp32 per the plan).

    ``round_leaf`` rounds every segment root to bf16 before the tree (pik's bf16-leaf contract).
    The output dtype is the config's (bf16 root for bf16 configs, fp32 root for fp32 ones); a SPLIT plan
    writes fp32 segment roots to the workspace and the combine kernel rounds to ``out``'s dtype.
    """
    M, K = x.shape
    N = w.shape[0]
    ids = config_ids()
    spec = plan.spec
    if plan.mode == seg.ONCHIP:
        odt = torch.float32 if spec.out_fp32 else torch.bfloat16
        if out is None:
            out = torch.empty((M, N), dtype=odt, device=x.device)
        if out.dtype is not odt or tuple(out.shape) != (M, N) or not out.is_contiguous():
            raise ValueError(f"{plan.config} writes a contiguous [M, N] {odt}")
        if plan.split > 1 and (1 << spec.levels) < plan.split:
            raise ValueError(f"{plan.config} folds at most {1 << spec.levels} segments, not {plan.split}")
        tiles = plan.segment_k // seg.SEGMENT_ALIGN if plan.split > 1 else 0
        _launch(ids[plan.config], x, w, out, M, N, K, 1, M * N, tiles, round_leaf and plan.split > 1)
        return out
    # SPLIT: one batch per segment, fp32 roots, then the tree
    if not spec.out_fp32 or spec.levels:
        raise ValueError("a SPLIT plan runs an fp32, single-chain configuration")
    S = plan.split
    ws = _workspace(S * M * N, x.device)
    _launch(ids[plan.config], x, w, ws, M, N, plan.segment_k, S, M * N, 0, False)
    if round_leaf:
        ws = ws.view(S, M, N).to(torch.bfloat16)  # RN, the same rounding the onchip fold applies
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    extension().combine(ws, out, S, M * N, M * N)
    return out


# Host fast path: (shape, strides, device) -> config id; the same plan the slow path picks.
_FAST: dict = {}


def mm_sm90(a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """``a[M,K] @ b[K,N]`` with ``b = W.t()``; bf16 in, fp32 accumulate, bf16 out."""
    key = (a.shape, b.shape, a.stride(), b.stride(), a.dtype, b.dtype, a.device)
    cid = _FAST.get(key)
    if cid is None:
        if not supported_shape(a, b):
            raise UnsupportedShape(f"mm_sm90 declines a={tuple(a.shape)}/{a.dtype}/{a.stride()} "
                                   f"b={tuple(b.shape)}/{b.dtype}/{b.stride()}")
        M, K = a.shape
        seg.plan_for(M, K, b.shape[1], a.dtype)  # the policy: S = 1 (raises on anything else)
        name, swz, ras = seg.choose_s1(M, b.shape[1], K)
        cid = (config_ids()[name], swz, ras)
        _FAST[key] = cid
    M, K = a.shape
    N = b.shape[1]
    w = b.t()
    pitch = -(-N // seg.PITCH_ALIGN_N) * seg.PITCH_ALIGN_N
    if pitch != N:
        # the output's TMA row pitch must be 16 bytes: write [M, pitch], keep the first N columns
        if out is not None:
            raise UnsupportedShape(f"N={N} is served through a padded pitch; pass out=None")
        buf = torch.empty((M, pitch), dtype=torch.bfloat16, device=a.device)
        extension().run(cid[0], a, w, buf, M, N, K, 1, pitch, K, K, M * pitch, 0, False, _SM_LIMIT, cid[1], cid[2])
        return buf[:, :N].contiguous()
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    extension().run(cid[0], a, w, out, M, N, K, 1, N, K, K, M * N, 0, False, _SM_LIMIT, cid[1], cid[2])
    return out


def mm(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``matmul_persistent``-shaped entry point. Bias is not served here."""
    if bias is not None:
        raise UnsupportedShape("mm_sm90 has no fused bias epilogue")
    return mm_sm90(a, b)


# --------------------------------------------------------------------------------- pik leaf mode
def _check_leaf_operands(x: torch.Tensor, w: torch.Tensor, leaf_k: int) -> tuple[int, int, int, int]:
    if x.dtype is not torch.bfloat16 or w.dtype is not torch.bfloat16:
        raise UnsupportedShape("pik leaves on mm_sm90 are bf16 operands only")
    if x.dim() != 2 or w.dim() != 2 or x.shape[1] != w.shape[1]:
        raise UnsupportedShape("x [M, K_local] and w [N, K_local] expected")
    M, K = x.shape
    N = w.shape[0]
    if x.stride(1) != 1 or w.stride(1) != 1 or x.stride(0) % 8 or w.stride(0) % 8:
        raise UnsupportedShape("unit inner stride and 16-byte row pitch required")
    if not seg.execution_supported(K, N, leaf_k):
        raise UnsupportedShape(f"leaf_k={leaf_k} is not a multiple of {seg.EXECUTION_ALIGN} that cuts "
                               f"K_local={K} into 1/2/4/8 leaves, or N={N} is not a multiple of 8")
    return M, K, N, K // leaf_k


def leaf_supported(K_local: int, N: int, leaf_k: int) -> bool:
    return seg.execution_supported(K_local, N, leaf_k)


@lru_cache(None)
def available() -> bool:
    """A Hopper device and a loadable extension whose table matches segments.py. Never raises."""
    try:
        require_sm90()
        config_ids()
        return True
    except Exception:  # noqa: BLE001 -- unavailability is an answer here, not an error
        return False


def choose_leaf_config(M: int, N: int, m: int, K: int = 0) -> str:
    """bf16 single-chain configuration for the batched leaf launch (L = m). Speed only."""
    hit = seg.tuned_config("leaves", (K, N, K // m), M) if K else None
    if hit is not None:
        return hit
    if M <= 16:
        return "sw_64x8_s6" if M <= 8 else "sw_64x16_s6"
    if M <= 32:
        return "sw_64x32_s6"
    if M <= 64:
        return "ws_64x64" if -(-N // 64) * m >= seg.H100_SM_COUNT else "ws_64x32"
    if M <= 128:
        return "ws_64x128" if _tiles2(M, N, 64, 128) * m >= seg.H100_SM_COUNT else "ws_64x64"
    if M <= 512:
        return "co_128x128"
    return "co_128x256"


def _tiles2(M, N, bm, bn):
    return -(-M // bm) * -(-N // bn)


def leaves(x: torch.Tensor, w: torch.Tensor, leaf_k: int, out: torch.Tensor, *,
           config: str | None = None) -> torch.Tensor:
    """pik LEAF wire: ``out[j] = round_bf16(chain over leaf j's K slice)`` for the m local leaves.

    ``out`` is ``[m, M, N]`` bf16, contiguous (plane stride ``M*N``); written in place on the current
    stream, nothing allocated -- ``owner.stage(...)``'s symmetric view can be passed directly.
    """
    M, K, N, m = _check_leaf_operands(x, w, leaf_k)
    if out.dtype is not torch.bfloat16 or tuple(out.shape) != (m, M, N) or not out.is_contiguous():
        raise ValueError(f"leaves writes a contiguous [{m}, {M}, {N}] bf16 buffer")
    if M == 0:
        return out
    name = config or choose_leaf_config(M, N, m, K)
    if seg._CONFIGS[name].out_fp32 or seg._CONFIGS[name].levels:
        raise ValueError(f"{name} is not a bf16 single-chain configuration")
    _launch(config_ids()[name], x, w, out, M, N, leaf_k, m, M * N, 0, False)
    return out


#: A NODE plan is either an onchip-fold configuration (one launch, tree in registers) or
#: ``"split:<config>"``: a batched leaf launch of that single-chain configuration into a workspace, then the
#: combine kernel (two launches; the same bits -- the tree, the adds and the leaf rounding are the same).
SPLIT_PREFIX = "split:"


def _node_plan_ok(name: str, m: int, leaf_dtype: torch.dtype, leaf_k: int = 0) -> bool:
    if name.startswith(SPLIT_PREFIX):
        c = seg._CONFIGS.get(name[len(SPLIT_PREFIX):])
        return c is not None and c.levels == 0 and bool(c.out_fp32) == (leaf_dtype is torch.float32)
    c = seg._CONFIGS.get(name)
    # the fold walks consecutive leaves in one K extent: each leaf must be whole K blocks
    whole = m == 1 or leaf_k % seg.SEGMENT_ALIGN == 0
    return c is not None and bool(c.out_fp32) and (m == 1 or (1 << c.levels) >= m) and whole


def choose_node_config(M: int, N: int, m: int, K: int = 0, leaf_dtype: torch.dtype = torch.bfloat16) -> str:
    """The NODE plan. Speed only: tuned table first, then a rule from the Qwen3.8 sweeps."""
    leaf_k = K // m if K else 0
    hit = seg.tuned_config("node", (K, N, leaf_k), M) if K else None
    if hit is not None and _node_plan_ok(hit, m, leaf_dtype, leaf_k):
        return hit
    if m > 1 and K and leaf_k % seg.SEGMENT_ALIGN:
        # partial K blocks: each leaf runs as its own batch
        return SPLIT_PREFIX + (("co_128x128_f32" if M > 64 else "ws_64x64_f32") if leaf_dtype is torch.float32
                               else choose_leaf_config(M, N, m, K))
    if leaf_dtype is torch.float32:
        if M > 128 or m > 2:
            return SPLIT_PREFIX + ("co_128x128_f32" if M > 64 else "ws_64x64_f32")
    elif M > 192 or (m > 2 and M > 32):
        return SPLIT_PREFIX + choose_leaf_config(M, N, m, K)
    if m <= 2:
        if M <= 32:
            return "sw_64x16_s10_t1_f32" if M <= 16 else "swpp_64x32_t1_f32"
        return "ws_64x128_t1_f32"
    return "sw_64x16_s6_t3_f32" if M <= 16 else "sw_64x32_s6_t3_f32"


_TYPED_WS: dict = {}


def _typed_workspace(numel: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Growable scratch per (device, dtype); growth during CUDA capture is refused (warm eagerly)."""
    key = (device.index, dtype)
    buf = _TYPED_WS.get(key)
    if buf is None or buf.numel() < numel:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("mm_sm90 node workspace growth during CUDA capture: warm this shape eagerly first")
        if buf is not None:
            _WORKSPACE_RETIRED.append(buf)
        buf = torch.empty(max(numel, 2 * (buf.numel() if buf is not None else 0)), dtype=dtype, device=device)
        _TYPED_WS[key] = buf
    return buf[:numel]


def leaf_node(x: torch.Tensor, w: torch.Tensor, leaf_k: int, out: torch.Tensor, *,
              leaf_dtype: torch.dtype = torch.bfloat16, config: str | None = None) -> torch.Tensor:
    """pik NODE wire: this rank's subtree root over its m leaves, in fp32 (leaves rounded to bf16 first
    when ``leaf_dtype`` is bf16). At m == 1 the node IS the leaf: bf16 for bf16 plans, fp32 otherwise.

    Either one launch (one CTA walks the m leaves of its tile and folds the tree in registers) or, where
    measured faster, the batched leaf launch plus the combine kernel (``choose_node_config``).
    """
    M, K, N, m = _check_leaf_operands(x, w, leaf_k)
    if leaf_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("leaf_dtype is bf16 or fp32")
    want = torch.bfloat16 if (m == 1 and leaf_dtype is torch.bfloat16) else torch.float32
    if out.dtype is not want or tuple(out.shape) != (M, N) or not out.is_contiguous():
        raise ValueError(f"leaf_node writes a contiguous [{M}, {N}] {want} buffer for m={m}")
    if M == 0:
        return out
    if want is torch.bfloat16:
        leaves(x, w, leaf_k, out.view(1, M, N))
        return out
    name = config or choose_node_config(M, N, m, K, leaf_dtype)
    if not _node_plan_ok(name, m, leaf_dtype, leaf_k):
        raise ValueError(f"{name} cannot produce an fp32 node over {m} {leaf_dtype} leaves")
    if name.startswith(SPLIT_PREFIX):
        ws = _typed_workspace(m * M * N, leaf_dtype, x.device)
        _launch(config_ids()[name[len(SPLIT_PREFIX):]], x, w, ws, M, N, leaf_k, m, M * N, 0, False)
        extension().combine(ws, out, m, M * N, M * N)
        return out
    _launch(config_ids()[name], x, w, out, M, N, K, 1, M * N, leaf_k // seg.SEGMENT_ALIGN if m > 1 else 0,
            leaf_dtype is torch.bfloat16 and m > 1)
    return out
