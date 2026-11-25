"""The trainer backward's FLA building blocks: the one place this family touches FLA internals.

The backward (integration.py) runs FLA's chunked GDN backward kernels on this family's saved forward
intermediates. FLA does not export the pieces it needs -- the backward's stages, and the two Triton kernels
launched without their wrappers' per-value-head group sums (``csrc/backward.cu`` does those) -- so they are
taken from the namespaces of the FLA functions that use them. The source is IsoExec's vendored copy of
flash-linear-attention 0.5.2 (``ops/gdn/shared/vendored_fla_bwd``), loaded and pinned by the shared GDN
loader; no ``fla`` package is imported. :func:`load` checks every name and every kernel parameter this
module relies on and names what moved, so a re-vendor that drops one fails here, not inside a launch.

Loading is lazy on purpose: it compiles and pins Triton kernels, and only a backward needs it (an
engine never does).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import torch

from isoexec.ops.gdn.shared.gdn_fla_backward import get_fla_chunk_backward

#: Stages of ``chunk_gated_delta_rule_bwd`` and helpers, read from the vendored ``ops/gated_delta_rule/chunk.py``.
_CHUNK_NAMES = ("fused_beta_sigmoid", "fused_beta_sigmoid_bwd", "prepare_chunk_indices", "gdn_gate_chunk_cumsum",
                "gdn_gate_bwd", "chunk_local_cumsum", "recompute_w_u_fwd", "chunk_gated_delta_rule_fwd_h",
                "chunk_bwd_dv_local", "chunk_gated_delta_rule_bwd_dhu", "chunk_bwd_dqkwg", "prepare_wy_repr_bwd",
                "RCP_LN2")
#: The Triton kernels launched directly, with the parameters passed to them.
_KERNELS = {
    ("chunk_bwd_dqkwg", "chunk_bwd_kernel_dqkwg"): {
        "q", "k", "v", "g", "g_gamma", "h", "do", "dh", "dw", "dq", "dk", "dv", "dg", "cu_seqlens", "chunk_indices",
        "scale", "B", "T", "H", "HV", "K", "V", "BT", "BK", "BV", "STATE_V_FIRST"},
    ("prepare_wy_repr_bwd", "prepare_wy_repr_bwd_kernel"): {
        "k", "v", "beta", "g", "A", "dw", "du", "dk", "dv", "db", "dg", "cu_seqlens", "chunk_indices", "T", "H", "HV",
        "K", "V", "BT", "BK", "BV"},
}
#: Keyword arguments integration.py passes to the stages.
_STAGE_KWARGS = {
    "chunk_bwd_dv_local": {"q", "k", "g", "do", "scale", "cu_seqlens", "chunk_indices", "chunk_size"},
    "chunk_gated_delta_rule_bwd_dhu": {"q", "k", "w", "g", "h0", "dht", "do", "dv", "scale", "cu_seqlens",
                                       "chunk_indices", "state_v_first", "chunk_size"},
    "chunk_gated_delta_rule_fwd_h": {"k", "w", "u", "g", "initial_state", "output_final_state", "cu_seqlens",
                                     "chunk_indices", "state_v_first", "chunk_size"},
    "recompute_w_u_fwd": {"k", "v", "beta", "A", "g", "cu_seqlens", "chunk_indices"},
    "gdn_gate_chunk_cumsum": {"g", "A_log", "chunk_size", "scale", "dt_bias", "cu_seqlens", "chunk_indices"},
    "gdn_gate_bwd": {"g", "A_log", "dt_bias", "dyg"},
}


class FlaMoved(RuntimeError):
    """The vendored FLA tree does not provide what this backward was written against."""


@dataclass(frozen=True)
class _ReverseScanSpec:
    chunk_size: int = 64
    num_warps: int = 1
    num_stages: int = 3
    num_ctas: int = 1
    reverse: bool = True
    head_first: bool = False
    has_scale: bool = False
    is_varlen: bool = True

    @property
    def recipe(self) -> str:
        return f"fla_scalar_reverse_bt{self.chunk_size}_w{self.num_warps}_s{self.num_stages}_v1"

    def launch(self) -> dict[str, Any]:
        return {"BT": self.chunk_size, "REVERSE": self.reverse, "HEAD_FIRST": self.head_first,
                "HAS_SCALE": self.has_scale, "IS_VARLEN": self.is_varlen,
                **self.geometry()}

    def geometry(self) -> dict[str, int]:
        return {"num_warps": self.num_warps, "num_stages": self.num_stages, "num_ctas": self.num_ctas}


_REVERSE_SPEC = _ReverseScanSpec()
REVERSE_CUMSUM_RECIPE = _REVERSE_SPEC.recipe
_REVERSE_ARGS = frozenset({"s", "o", "scale", "cu_seqlens", "chunk_indices", "T", "B", "H", "BT",
                           "REVERSE", "HAS_SCALE", "IS_VARLEN", "HEAD_FIRST"})


def _reverse_kernel(function: Any) -> Any:
    """Admit the existing native candidate, without changing any autotuner."""
    wrapper = inspect.unwrap(function)
    kernel = wrapper.__globals__.get("chunk_local_cumsum_scalar_kernel")
    autotuner = None
    for _ in range(8):
        if kernel is None:
            break
        if hasattr(kernel, "configs"):
            autotuner = kernel
        if hasattr(kernel, "arg_names") and inspect.isfunction(getattr(kernel, "fn", None)):
            break
        kernel = getattr(kernel, "fn", None)
    if kernel is None or _REVERSE_ARGS != set(getattr(kernel, "arg_names", ())):
        raise FlaMoved("chunk_local_cumsum_scalar_kernel lost the CPR reverse-cumsum launch ABI")
    source = getattr(kernel, "fn", None)
    if (not inspect.isfunction(source) or source.__name__ != "chunk_local_cumsum_scalar_kernel"
            or source.__code__.co_filename != wrapper.__code__.co_filename):
        raise FlaMoved("chunk_local_cumsum_scalar_kernel came from a different native source module")
    configs = () if autotuner is None else autotuner.configs
    if not any(all(getattr(config, name) == value for name, value in _REVERSE_SPEC.geometry().items())
               and not config.kwargs and getattr(config, "maxnreg", None) is None for config in configs):
        raise FlaMoved(f"chunk_local_cumsum_scalar_kernel lacks the declared native {_REVERSE_SPEC.geometry()} candidate")
    return kernel


def reverse_cumsum_identity() -> dict[str, Any]:
    """The backend's local numerical recipe, independent of timing-based caches."""
    return {"recipe": _REVERSE_SPEC.recipe, "kernel": "chunk_local_cumsum_scalar_kernel",
            **vars(_REVERSE_SPEC)}


def _compiled_reverse(compiled: Any) -> dict[str, Any]:
    metadata = getattr(compiled, "metadata", None)
    geometry = {name: getattr(metadata, name, None) for name in _REVERSE_SPEC.geometry()}
    kernel_hash = getattr(compiled, "hash", None)
    if geometry != _REVERSE_SPEC.geometry() or not isinstance(kernel_hash, str) or not kernel_hash:
        raise FlaMoved("Compiled reverse cumsum geometry/hash differs from the backend launch recipe")
    return {**geometry, "hash": kernel_hash}


def _params(fn) -> set:
    return set(inspect.signature(inspect.unwrap(fn)).parameters)


@lru_cache(None)
def load() -> SimpleNamespace:
    """FLA's backward building blocks from the vendored tree, loaded and checked once."""
    chunk_ns = inspect.unwrap(get_fla_chunk_backward()).__globals__
    missing = [name for name in _CHUNK_NAMES if name not in chunk_ns]
    ns = SimpleNamespace(**{name: chunk_ns[name] for name in _CHUNK_NAMES if name in chunk_ns})
    problems = [f"vendored_fla_bwd.ops.gated_delta_rule.chunk has no {name!r}" for name in missing]
    for (owner, kernel), passed in _KERNELS.items():
        if owner not in chunk_ns:
            continue
        glb = inspect.unwrap(chunk_ns[owner]).__globals__
        jit = glb.get(kernel)
        names = set(getattr(jit, "arg_names", ()) or ())
        if jit is None:
            problems.append(f"{owner}'s module has no kernel {kernel!r}")
        elif not passed <= names:
            problems.append(f"{kernel} lost parameters {sorted(passed - names)}")
        else:
            setattr(ns, kernel, jit)
        for helper in ("check_shared_mem", "triton"):
            if helper not in glb:
                problems.append(f"{owner}'s module has no {helper!r}")
            else:
                setattr(ns, helper, glb[helper])
    for stage, kwargs in _STAGE_KWARGS.items():
        fn = chunk_ns.get(stage)
        if fn is not None and not kwargs <= _params(fn):
            problems.append(f"{stage} lost keyword arguments {sorted(kwargs - _params(fn))}")
    if "chunk_local_cumsum" in chunk_ns:
        try:
            ns.reverse_cumsum_kernel = _reverse_kernel(chunk_ns["chunk_local_cumsum"])
        except FlaMoved as error:
            problems.append(str(error))
    if problems:
        raise FlaMoved(
            "gdn_cpr_sm90's trainer backward was written against flash-linear-attention 0.5.2 as vendored in "
            "ops/gdn/shared/vendored_fla_bwd; the loaded tree differs:\n  - " + "\n  - ".join(problems))
    return ns


def reverse_cumsum(ns: SimpleNamespace, gradient: torch.Tensor,
                   cu: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
    """Run only CPR's packed FP32 scalar reverse scan with one native recipe.

    This scan participates in the gate gradient's reduction order. Selecting
    its warp count by timing changes da bits; forward and other FLA callers
    retain their own kernels/configs. No global tuning cache is modified.
    """
    if (gradient.device.type != "cuda" or gradient.dtype != torch.float32
            or gradient.ndim != 3 or gradient.shape[0] != 1 or not gradient.is_contiguous()
            or gradient.shape[1] < 1 or gradient.shape[2] < 1
            or cu.device != gradient.device or cu.dtype != torch.int64 or cu.ndim != 1
            or cu.numel() < 2 or not cu.is_contiguous() or chunks.device != gradient.device
            or chunks.dtype != torch.int64 or chunks.ndim != 2 or chunks.shape[1] != 2
            or chunks.shape[0] < 1 or not chunks.is_contiguous()):
        raise ValueError("CPR reverse cumsum requires contiguous CUDA FP32[1,T,H] and same-device int64 packed offsets/chunk pairs")
    # FLA's input_guard selects the input's device before launching. Preserve
    # that ownership when bypassing only its timing-based autotuner.
    with torch.cuda.device(gradient.device):
        output = torch.empty_like(gradient)
        compiled = ns.reverse_cumsum_kernel[(len(chunks), gradient.shape[2])](
            s=gradient, o=output, scale=None, cu_seqlens=cu, chunk_indices=chunks,
            T=gradient.shape[1], B=1, H=gradient.shape[2], **_REVERSE_SPEC.launch())
        ns.reverse_cumsum_compiled = _compiled_reverse(compiled)
    return output


def _tiling(ns, arch_levels, device: torch.device) -> int:
    for arch, size in arch_levels:
        if ns.check_shared_mem(arch, device.index or 0):
            return size
    return 32


def dqkwg(ns, q, k, v, w, g, h, dv, do, dh, scale, cu, chunks):
    """FLA's ``chunk_bwd_dqkwg`` (state_v_first, its own grid and block sizes) returning dq, dk per value head
    [1, T, HV, K] -- without the wrapper's sums over each key head's value heads."""
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    tiling = _tiling(ns, (("hopper", 128), ("ada", 64)), k.device)
    BK = min(max(ns.triton.next_power_of_2(K), 16), tiling)
    BV = min(max(ns.triton.next_power_of_2(V), 16), tiling)
    NK = ns.triton.cdiv(K, BK)
    dq, dk = q.new_empty(B, T, HV, K), k.new_empty(B, T, HV, K)
    dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device)
    dw = torch.empty_like(w)
    ns.chunk_bwd_kernel_dqkwg[(NK, len(chunks), B * HV)](
        q=q, k=k, v=v, g=g, g_gamma=None, h=h, do=do, dh=dh, dw=dw, dq=dq, dk=dk, dv=dv, dg=dg, cu_seqlens=cu,
        chunk_indices=chunks, scale=scale, B=B, T=T, H=H, HV=HV, K=K, V=V, BT=64, BK=BK, BV=BV, STATE_V_FIRST=True)
    return dq, dk, dw, dg.sum(0)


def wy_bwd(ns, k, v, beta, A, dw, du, g, cu, chunks):
    """FLA's ``prepare_wy_repr_bwd`` (its grid and block sizes) returning dk per value head [1, T, HV, K] --
    without the wrapper's sum over each key head's value heads."""
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    tiling = _tiling(ns, (("none", 64),), k.device)
    BK = min(max(ns.triton.next_power_of_2(K), 16), tiling)
    BV = min(max(ns.triton.next_power_of_2(V), 16), tiling)
    dk, dv = k.new_empty(B, T, HV, K), torch.empty_like(v)
    dg, db = torch.empty_like(g), torch.empty_like(beta)
    ns.prepare_wy_repr_bwd_kernel[(len(chunks), B * HV)](
        k=k, v=v, beta=beta, g=g, A=A, dw=dw, du=du, dk=dk, dv=dv, db=db, dg=dg, cu_seqlens=cu, chunk_indices=chunks,
        T=T, H=H, HV=HV, K=K, V=V, BT=A.shape[-1], BK=BK, BV=BV)
    return dk, dv, db, dg
