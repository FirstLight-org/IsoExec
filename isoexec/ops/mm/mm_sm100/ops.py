"""Torch-facing surface of the SM100 deterministic dense GEMM."""

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
    "config_info",
    "counter_pool_bytes",
    "mm",
    "mm_sm100",
    "require_sm100",
    "run_plan",
    "supported_shape",
]


class UnsupportedShape(ValueError):
    """The sm100 leaf declines this shape; the caller must fall through to another provider."""


@lru_cache(None)
def config_ids() -> dict[str, int]:
    """Configuration name -> index in the compiled table."""
    ext = extension()
    ids = {ext.config_info(i)["name"]: i for i in range(ext.num_configs())}
    for (fam, rows, spc, mode), name in seg._CONFIGS.items():
        if name not in ids:
            raise RuntimeError(
                f"segments.py plans configuration {name!r}, which the compiled module does not "
                f"define; csrc/mm_sm100_gemm.cu and segments._CONFIGS have drifted"
            )
        info = ext.config_info(ids[name])
        expected_rows = rows or info["block_m"]
        if (info["a_rows"], info["segments_per_cta"], int(info["partial"])) != (
            expected_rows, spc, mode
        ):
            raise RuntimeError(
                f"configuration {name!r} is compiled as a_rows={info['a_rows']} "
                f"spc={info['segments_per_cta']} publish_mode={info['partial']} but segments.py "
                f"plans it as a_rows={expected_rows} spc={spc} publish_mode={mode}"
            )
    # And the other direction: a compiled configuration the planner does not know about is either a
    # declared measurement control or a drift, and the difference must be written down.
    stray = sorted(set(ids) - set(seg._CONFIGS.values()) - seg._MEASUREMENT_ONLY)
    if stray:
        raise RuntimeError(
            f"the compiled module defines configuration(s) {stray} that segments.py neither plans "
            f"nor declares in _MEASUREMENT_ONLY; csrc/mm_sm100_gemm.cu and segments.py have drifted"
        )
    return ids


@lru_cache(None)
def config_info(name: str) -> dict:
    return extension().config_info(config_ids()[name])


def require_sm100() -> None:
    """Refuse anywhere but a Blackwell device whose SM count the segment policy was written for."""
    from isoexec.core.arch import live_device_arch

    arch = live_device_arch()
    if arch != "sm100":
        raise RuntimeError(f"mm_sm100 requires an sm100 device, not {arch}")
    live = extension().sm_count()
    if live != seg.TARGET_SM_COUNT:
        raise RuntimeError(
            f"the sm100 mm segment policy is published for {seg.TARGET_SM_COUNT} SMs and this "
            f"device has {live}; S would differ, so this is refused rather than served"
        )


def supported_shape(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether :func:`mm` will serve this call, without raising."""
    if a.dtype is not torch.bfloat16 or b.dtype is not torch.bfloat16:
        return False
    if a.dim() != 2 or b.dim() != 2:
        return False
    if not (a.is_cuda and b.is_cuda):
        return False
    M, K = a.shape
    if b.shape[0] != K:
        return False
    N = b.shape[1]
    # b must be W.t(): unit stride on K, so the underlying weight is [N, K] row-major.
    if b.stride(0) != 1 or b.stride(1) != K:
        return False
    if a.stride(1) != 1 or a.stride(0) != K:
        return False
    return seg.supported(K, N) and M >= 1


def _weight_view(b: torch.Tensor) -> torch.Tensor:
    """``b`` is ``[K, N]`` with unit stride on K; the weight it views is ``[N, K]`` row-major."""
    w = b.t()
    assert w.stride(1) == 1 and w.stride(0) == b.shape[0]
    return w


#: Per-(device, stream) arrival-counter buffers for the in-kernel fold, keyed so two streams
#: never share one. The kernel leaves every word it touched at zero, so a buffer allocated once
#: with ``zeros`` stays valid for every later call -- there is no per-call memset, which is the
#: whole point of the fold, and the pointer is stable so a CUDA graph can capture it.
_COUNTERS: dict[tuple[int, int], torch.Tensor] = {}


def _counter_for(device: torch.device, words: int) -> torch.Tensor:
    """A zeroed int32 buffer of at least ``words`` words for this device and current stream."""
    key = (device.index if device.index is not None else torch.cuda.current_device(),
           torch.cuda.current_stream(device).cuda_stream)
    buf = _COUNTERS.get(key)
    if buf is None or buf.numel() < words:
        buf = torch.zeros(max(words, 1024), dtype=torch.int32, device=device)
        _COUNTERS[key] = buf
    return buf


def counter_pool_bytes() -> int:
    """Bytes the fold's arrival-counter pool currently holds. Diagnostics only."""
    return sum(t.numel() * 4 for t in _COUNTERS.values())


def run_plan(a: torch.Tensor, b: torch.Tensor, plan: seg.Plan, *,
             out: torch.Tensor | None = None,
             workspace: torch.Tensor | None = None,
             slots: int = 0,
             root_dtype: torch.dtype = torch.bfloat16,
             combine: bool = True,
             leaf_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Run one explicit execution plan."""
    ext = extension()
    M, K = a.shape
    N = b.shape[1]
    w = _weight_view(b)
    if root_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"root_dtype must be bfloat16 or float32, not {root_dtype}")
    if leaf_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"leaf_dtype must be float32 or bfloat16, not {leaf_dtype}")
    publish_bf16 = leaf_dtype is torch.bfloat16
    if root_dtype is torch.float32:
        if publish_bf16:
            raise ValueError("root_dtype=float32 and leaf_dtype=bfloat16 contradict each other")
        plan = seg.publishing_twin(plan)
        if out is not None:
            if (out.dtype is not torch.float32 or tuple(out.shape) != (M, N)
                    or not out.is_contiguous()):
                raise ValueError("root_dtype=float32 writes a contiguous [M, N] float32 `out`")
            workspace = out
        combine = False
    if not combine and plan.mode != seg.COMBINE:
        raise ValueError(
            f"combine=False publishes subtree roots and needs a COMBINE-mode plan, not "
            f"{plan.config} (mode {plan.mode})")
    if publish_bf16:
        if combine:
            raise ValueError("leaf_dtype=bfloat16 is a publish-only mode: pass combine=False")
        if plan.spc != 1:
            raise ValueError(
                f"leaf_dtype=bfloat16 rounds each published root, which is a segment root only at "
                f"spc == 1; {plan.config} owns {plan.spc} segments per CTA")
    groups = plan.groups
    need = groups * M * N if plan.mode == seg.COMBINE and not combine else \
        plan.workspace_floats(M, N)
    ws_dtype = torch.bfloat16 if publish_bf16 else torch.float32
    if need:
        if workspace is None:
            workspace = torch.empty(need, dtype=ws_dtype, device=a.device)
        elif workspace.numel() < need or workspace.dtype is not ws_dtype:
            raise ValueError(f"workspace has {workspace.numel()} {workspace.dtype}, plan needs "
                             f"{need} {ws_dtype}")
    else:
        workspace = torch.empty(0, dtype=torch.float32, device=a.device)
    if combine:
        c = torch.empty((M, N), dtype=a.dtype, device=a.device) if out is None else out
    else:
        # Publish only: the leaf never touches C, and the binding does not read this tensor.
        c = torch.empty(0, dtype=a.dtype, device=a.device)
    cid = config_ids()[plan.config]
    # Zero words for every configuration but the in-kernel fold; the buffer is still passed so the
    # binding has one signature, and the pool's minimum allocation covers it.
    counter = _counter_for(a.device, ext.counter_words(cid, M, N))
    ext.run(cid, a, w, c, workspace, counter, plan.segment_k, plan.groups, slots, combine,
            publish_bf16)
    if combine:
        return c
    view = workspace.view(-1)[: groups * M * N].view(groups, M, N)
    if root_dtype is torch.float32:
        return view[0]
    return view


#: Host fast path (round 8): ``(shape, strides, dtype, device) -> (config id, segment_k, groups)``
#: for plans that own the whole tree on chip or cut it for the combine kernel (no counter words), so a
#: repeated call skips
#: ``supported_shape``/``plan_for``/``run_plan``'s validation and allocations. Pure host caching: the
#: same plan, the same launch, the same bits. A key is only ever filled from the slow path's verdict.
_FAST: dict = {}
_EMPTY: dict = {}


def _empty_ws(device: torch.device) -> torch.Tensor:
    idx = device.index
    t = _EMPTY.get(idx)
    if t is None:
        t = _EMPTY[idx] = torch.empty(0, dtype=torch.float32, device=device)
    return t


def mm_sm100(a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """``a[M,K] @ b[K,N]`` with ``b = W.t()``; bf16 in, fp32 accumulate, bf16 out."""
    key = (a.shape, b.shape, a.stride(), b.stride(), a.dtype, b.dtype, a.device, b.device)
    ent = _FAST.get(key)
    if ent is not None and out is None:
        cid, segk, groups = ent
        dev = a.device
        stream = torch._C._cuda_getCurrentRawStream(dev.index)
        counter = _COUNTERS.get((dev.index, stream))
        if counter is not None:
            M, N = a.shape[0], b.shape[1]
            c = torch.empty((M, N), dtype=a.dtype, device=dev)
            # a cut tree (a COMBINE plan) publishes `groups` fp32 subtree roots
            ws = (torch.empty(groups * M * N, dtype=torch.float32, device=dev) if groups > 1
                  else _empty_ws(dev))
            extension().run(cid, a, b.t(), c, ws, counter, segk, groups, 0, True, False)
            return c
    if not supported_shape(a, b):
        raise UnsupportedShape(
            f"mm_sm100 declines a={tuple(a.shape)}/{a.dtype}/{a.stride()} "
            f"b={tuple(b.shape)}/{b.dtype}/{b.stride()}"
        )
    M, K = a.shape
    N = b.shape[1]
    plan = seg.plan_for(M, K, N, a.dtype)
    if (plan.mode == seg.ONCHIP and plan.groups == 1 or plan.mode == seg.COMBINE) and \
            extension().counter_words(config_ids()[plan.config], M, N) == 0:
        _FAST[key] = (config_ids()[plan.config], plan.segment_k, plan.groups)
    return run_plan(a, b, plan, out=out)


def mm(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``matmul_persistent``-shaped entry point. Bias is not served here; the caller falls through."""
    if bias is not None:
        raise UnsupportedShape("mm_sm100 has no fused bias epilogue")
    return mm_sm100(a, b)
