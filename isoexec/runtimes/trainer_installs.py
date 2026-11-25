"""Explicit process-wide kernel installation, environment defaults and live reduction-plan access.
Environment defaults never overwrite deployment values, and recorded leaf counts come from the
plan kernels actually use.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch

from isoexec.ops.logprobs.temperature import temperature_divide

if TYPE_CHECKING:
    from isoexec.compiler.source import ModelSource
    from isoexec.ops.mm.mm_shapes import GemmCensus


class _FullLogprobsShard(Protocol):
    def __call__(self, logits: torch.Tensor, *, vocab_start_index: int, vocab_end_index: int,
                 group: Any) -> torch.Tensor: ...

TRAINER_ENV: dict[str, str] = {
    # ATen caches these on first native GEMM, including the installer's self-check.
    # Fresh framework actors must inherit them before importing numerical runtimes.
    "CUBLAS_WORKSPACE_CONFIG": ":16:8",
    "CUBLASLT_WORKSPACE_SIZE": "1",
    "ISOEXEC": "1",
    "ISOEXEC_PIK": "1",
    "ISOEXEC_PIK_FASTLAUNCH": "1",
    "ISOEXEC_MM_FWD_ONLY": "1",
    "ISOEXEC_MM_FWD_ONLY_BMM": "1",
    "ISOEXEC_NATIVE_NORM_MEMO": "1",
    "ISOEXEC_GDN": "1",
    "ISOEXEC_GDN_KERNEL": "cpr",
    "ISOEXEC_GDN_NATIVE_KERNELS": "1",
    "ISOEXEC_GDN_CHUNKED_PREFILL": "1",
    "ISOEXEC_GDN_CPR_MIN_PAGES": "1",
    "ISOEXEC_GDN_CPR_SLEEP": "1",
    "ISOEXEC_GDN_FLA_BACKWARD": "1",
    "ISOEXEC_GDN_ANALYTIC_CONV_BWD": "1",
    "ISOEXEC_MOE_ROUTER_O2": "1",
    "ISOEXEC_MOE_SHARED_OWNER_FUSION": "1",
    "ISOEXEC_MOE_PIK_OWNER_COMBINE": "1",
    "ISOEXEC_MOE_COMBINE_SORT": "1",
    "VLLM_BATCH_INVARIANT": "1",
}
PIK_LEAVES = 8  # ISOEXEC_PIK_LEAVES default; the fc2 leaf count at ETP=1 (== leaves // etp)


def pik_leaves() -> int:
    """Read the leaf count from the live reduction plan, including ISOEXEC_PIK_LEAVES overrides.
    Binding inputs and placements must describe that plan rather than the module default.
    """
    from isoexec.ops.collectives.pik_tp_invariant import get_plan

    return int(get_plan().num_leaves)
_INSTALLED: dict[str, Any] = {}

# Install the trainer's process-wide kernel bindings.


def apply_trainer_env(model_path: str | None) -> dict[str, str]:
    """``setdefault`` the production flag set (import-time readers such as
    the kernels consult it) and the model path the mm census derives shapes from."""
    applied = {}
    for k, v in TRAINER_ENV.items():
        os.environ.setdefault(k, v)
        applied[k] = os.environ[k]
    if model_path is not None:
        os.environ.setdefault("ISOEXEC_MODEL_PATH", model_path)
        applied["ISOEXEC_MODEL_PATH"] = os.environ["ISOEXEC_MODEL_PATH"]
    return applied


def enable_trainer_batch_invariant() -> bool:
    """Install owned batch-invariant reductions and forward-scoped GEMMs."""
    from isoexec.ops.mm.batch_invariant_install import install
    from isoexec.ops.mm.mm_fwd_scope import install_bmm_scope

    install()
    install_bmm_scope()
    return True


def install_attention(deterministic_backward: bool) -> dict[str, Any]:
    """Load the arch-selected GQA attention package (gqa_sm90 on SM90, gqa_sm100 on SM100) and set its
    backward policy.

    ``deterministic_backward`` (Megatron ``deterministic_mode``) also turns on torch's deterministic
    algorithms. On SM90 it makes a backward FA3 cannot run deterministically (head dim 256) take
    torch's deterministic FA2 backward instead of FA3's atomic kernel; SM100's FA4 backward is
    deterministic at every head dim it serves, so there it is recorded and changes nothing.
    """
    from isoexec.ops._dispatch import backend_module

    if deterministic_backward:
        torch.use_deterministic_algorithms(True, warn_only=False)
    receipt = dict(backend_module("gqa", "integration").install(deterministic_backward=deterministic_backward))
    receipt["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    receipt["deterministic_warn_only"] = torch.is_deterministic_algorithms_warn_only_enabled()
    return receipt


def model_gemm_census(source: ModelSource | str, raw_config: Mapping[str, object]) -> GemmCensus:
    """The dense GEMM census of the model ``source`` builds from ``raw_config`` (on meta).

    A function of the package and the effective HF config alone, so the trainer and the engine,
    given the same pair, register equal censuses and reach the same shape-admission decisions.
    """
    from isoexec.compiler.source import ModelSource
    from isoexec.nn import gemm_census

    source = source if isinstance(source, ModelSource) else ModelSource(package=source)
    return gemm_census(source.build_meta(raw_config))


def _has_moe(source: ModelSource | str, raw_config: Mapping[str, object] | None) -> bool:
    from isoexec.compiler.source import ModelSource
    from isoexec.nn import MoE

    source = source if isinstance(source, ModelSource) else ModelSource(package=source)
    return any(isinstance(module, MoE) for module in source.build_meta(raw_config).modules())


def install_trainer_process(model_path: str, *, source: Any, raw_config: Mapping[str, object],
                            deterministic_attention_backward: bool) -> dict[str, Any]:
    """Everything one trainer process installs before it builds its model: the kernels and the PIK
    transport intent. ``source``/``raw_config`` are the pair the engine builds its model from, so both
    sides derive the same GEMM census and kernel decisions. Shared by the verl and SkyRL integrations."""
    installs = install_trainer_kernels(
        model_path,
        source=source,
        raw_config=raw_config,
        deterministic_attention_backward=deterministic_attention_backward,
    )
    from isoexec.ops.collectives.pik_tp_invariant import select_trainer_transport

    installs["pik_transport_intent"] = select_trainer_transport("TRAINER")
    return installs


def install_trainer_kernels(
    model_path: str | None = None,
    *,
    source: ModelSource | str | None = None,
    raw_config: Mapping[str, object] | None = None,
    deterministic_attention_backward: bool = False,
) -> dict[str, Any]:
    """Perform the trainer's process-wide kernel installs once and return what was installed.

    ``source`` and ``raw_config`` (the effective HF config the model is built from) select the
    model whose GEMM census the installed GEMM providers report coverage against; ``model_path`` labels it. A process
    that names a model must give both, so it cannot install a census-free composition its peer
    does not. Kernel tests that name no model install without a census.
    """
    if type(deterministic_attention_backward) is not bool:
        raise TypeError("deterministic_attention_backward must be a bool")
    if (source is None) != (raw_config is None) or (model_path is not None and source is None):
        raise ValueError("install_trainer_kernels needs both source and raw_config (the effective HF config) "
                         "to derive the model's GEMM census")
    if _INSTALLED and _INSTALLED.get("deterministic_attention_backward", False) != deterministic_attention_backward:
        raise RuntimeError("One trainer worker cannot mix native attention backward profiles")
    from isoexec.runtimes.precision import pin_fp32_ieee

    # Another native owner may reset process flags after the kernel install.
    pin_fp32_ieee()
    if _INSTALLED:
        # Another owner may have reset torch's algorithm policy; restore it and refresh the record.
        if deterministic_attention_backward:
            torch.use_deterministic_algorithms(True, warn_only=False)
        _INSTALLED["attention_recipe"] = {**_INSTALLED["attention_recipe"],
                                          "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                          "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}
        return dict(_INSTALLED)
    if not torch.cuda.is_available():
        raise RuntimeError("the trainer kernel installs bind CUDA kernels; no CUDA device is available")
    applied = apply_trainer_env(model_path)
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik
    from isoexec.ops.mm import mm_shapes
    rec: dict[str, Any] = {"env": applied}
    # Memoize only the device-derived part of normalization admission; preserve kernel selection.
    from isoexec.ops.norms import install_native_rmsnorm_memo

    rec["native_norm_memo"] = bool(cast(Callable[[], bool], install_native_rmsnorm_memo)())
    from isoexec.ops.moe import moe_batch_invariant as _mbi

    # Router and expert GEMMs reach aten::mm; route it to the batch-invariant kernel exactly when the
    # engine does (runtimes/vllm/factory: MoE models only). A dense model's GEMMs are bound ops and never
    # reach aten::mm, so the global override would only slow the eager leftovers and differ from the
    # engine's dispatcher. A process that names no model (kernel tests) keeps the conservative install.
    if source is None or _has_moe(source, raw_config):
        _mbi._install_moe_matmul_invariance()  # aten::mm / aten::addmm -> mm_fwd_scope (returns None)
    rec["moe_matmul_invariance"] = getattr(_mbi, "_matmul_invariance_lib", None) is not None
    rec["trainer_batch_invariant"] = bool(enable_trainer_batch_invariant())
    if source is not None and raw_config is not None:
        census = model_gemm_census(source, raw_config)
        mm_shapes.set_live_context(census=census, model_path=model_path or "", tp=1, etp=1, side="TRAINER")
        rec["gemm_census"] = census.digest()
    from isoexec.ops.mm.segtree import install_mm_segtree

    # The sm90 mm provider, outermost on matmul_persistent: the segment-tree GEMM serves every shape it
    # expresses and the vendored Triton kernel the rest; the engine installs the same chain
    # (runtimes/vllm/factory._install_engine_gemms).
    rec["mm_segtree"] = bool(install_mm_segtree())
    from isoexec.ops.mm.wgrad import install_column_assembler

    # Backward only: split_columns' gradient assembly in one launch (same bits as the eager loop).
    rec["column_assembler"] = install_column_assembler()
    rec["attention_recipe"] = install_attention(deterministic_attention_backward)
    rec["deterministic_attention_backward"] = deterministic_attention_backward
    ensure_pik()
    rec["pik"] = True
    from isoexec.core.local_memory import install_autotune_guard

    rec["autotune_lmem_guard"] = install_autotune_guard()
    # Install architecture-specific families that bind through module attributes.
    from isoexec.ops._dispatch import install_arch_impls, resolved

    rec["arch_impls"] = resolved()  # the whole resolution, in the launch receipt
    rec.update(install_arch_impls())
    _INSTALLED.update(rec)
    return dict(rec)
@torch.no_grad()
def compute_full_logprobs_for_check(
    logits_fp32: torch.Tensor, temperature: float | torch.Tensor = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the trainer's full `[N, V]` logprob rows and exact fp32 temperature scalar."""
    temperature_fp32 = torch.as_tensor(temperature, dtype=torch.float32, device=logits_fp32.device)
    if temperature_fp32.ndim != 0:
        raise ValueError(f"temperature must be a scalar, got shape={tuple(temperature_fp32.shape)}")
    if not bool(torch.isfinite(temperature_fp32).item()) or not bool((temperature_fp32 > 0.0).item()):
        raise ValueError(f"temperature must be finite and positive, got {float(temperature_fp32.item())!r}")
    tempered = temperature_divide(logits_fp32, temperature_fp32).contiguous()

    from isoexec.ops.logprobs import full_logprobs_for_check

    return cast(Callable[[torch.Tensor], torch.Tensor], full_logprobs_for_check)(tempered), temperature_fp32


def compute_full_logprobs_for_check_shard(
    logits_fp32: torch.Tensor,
    temperature: float | torch.Tensor = 1.0,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    group,
) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's `[N, V_local]` slice of the full logprob rows, plus the exact temperature scalar."""
    temperature_fp32 = torch.as_tensor(temperature, dtype=torch.float32, device=logits_fp32.device)
    if temperature_fp32.ndim != 0:
        raise ValueError(f"temperature must be a scalar, got shape={tuple(temperature_fp32.shape)}")
    if not bool(torch.isfinite(temperature_fp32).item()) or not bool((temperature_fp32 > 0.0).item()):
        raise ValueError(f"temperature must be finite and positive, got {float(temperature_fp32.item())!r}")
    tempered = temperature_divide(logits_fp32, temperature_fp32).contiguous()
    from isoexec.ops.logprobs import full_logprobs_for_check_sharded

    rows = cast(_FullLogprobsShard, full_logprobs_for_check_sharded)(
        tempered, vocab_start_index=vocab_start_index, vocab_end_index=vocab_end_index, group=group
    )
    return rows, temperature_fp32


__all__ = [
    "PIK_LEAVES",
    "TRAINER_ENV",
    "apply_trainer_env",
    "install_trainer_process",
    "compute_full_logprobs_for_check",
    "compute_full_logprobs_for_check_shard",
    "install_trainer_kernels",
    "model_gemm_census",
    "pik_leaves",
]
