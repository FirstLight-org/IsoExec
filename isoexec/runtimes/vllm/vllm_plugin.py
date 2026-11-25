"""vLLM general plugin that installs the IsoExec engine stack in every vLLM process.

Each TP rank is its own process, so everything must be installed per worker. vLLM may load plugins
several times per process, so installs are idempotent. A job opts in with ``VLLM_ISOEXEC_REGISTERED=1``,
which ``registration.register_plugin`` exports in the driver and
``runtimes.environment.resolved_environment`` carries into fresh engine actors; every other vLLM job
is untouched.
"""

from __future__ import annotations

import logging
import os
from importlib import import_module
from typing import Any, Callable, cast

logger = logging.getLogger(__name__)

_installed = False


def alloc_sleepable(shape, dtype, device):
    """Allocate an aliased-parameter buffer inside vLLM's sleepable CuMem pool when sleep mode is live.

    A plain ``torch.empty`` lands outside the sleepable pool, so level-1 sleep cannot offload it and
    the expert weights stay resident through the whole training window. ``use_memory_pool`` also gives
    the virtual-address stability vLLM's graph-captured weights rely on. ``CuMemAllocator.instance`` is
    read without ``get_instance()`` on purpose: the getter would create the singleton (and its
    free-callback hooks) in processes that never enabled sleep mode.

    Injected through the neutral family's buffer allocator before any model is built.
    """
    import torch

    try:
        CuMemAllocator = import_module("vllm.device_allocator.cumem").CuMemAllocator

        alloc = CuMemAllocator.instance
        if alloc is not None:
            with alloc.use_memory_pool(tag="weights"):
                return torch.empty(shape, dtype=dtype, device=device)
    except Exception as e:  # pragma: no cover - vLLM absent or pool unavailable
        logger.info("[isoexec-moe-fw] sleepable-pool alloc unavailable (%s); plain alloc", e)
    return torch.empty(shape, dtype=dtype, device=device)


def engine_token_bound(tokens: int) -> int:
    """The most tokens any data-parallel peer feeds the current forward (host values only)."""
    forward_context = import_module("vllm.forward_context")
    if not forward_context.is_forward_context_available():
        return tokens
    metadata = forward_context.get_forward_context().dp_metadata
    if metadata is None:
        return tokens
    return max(tokens, int(metadata.num_tokens_across_dp_cpu.max().item()))


def register() -> None:
    """vLLM general-plugin entry point: install the IsoExec engine stack in this process.

    Idempotent; a job without ``VLLM_ISOEXEC_REGISTERED=1`` is untouched.
    """
    global _installed

    if _installed:
        return
    if os.environ.get("VLLM_ISOEXEC_REGISTERED") != "1":
        return
    # Autotune sweeps in the engine (vLLM's own FLA copies warm up at profiling) must not leave the
    # context's local-memory reservation inflated beside the KV cache.
    from ...core.local_memory import install_autotune_guard

    install_autotune_guard()
    # Engine memory defaults; an explicit user setting wins.
    # Note (Alexander): expandable segments on. Fixed-size allocator segments fragmented ~5 GiB/GPU
    # outside vLLM's gpu_memory_utilization budget (35B DAPO). No numerics or measurable perf change.
    if not (os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or os.environ.get("PYTORCH_ALLOC_CONF")):
        import torch

        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    # Note (Alexander): NVLS off. It costs 1 GiB per engine NCCL communicator, mostly outside the budget,
    # and buys nothing here: engine NCCL only all-gathers bytes (pik owns the reductions).
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    # Aliased expert-weight buffers (ops/moe/shared/packed_experts.py)
    # must come from vLLM's sleepable CuMem pool: a plain torch.empty lands outside it, so level-1
    # sleep cannot offload the expert weights and they stay resident through the whole training
    # window. That policy needs `vllm.device_allocator.cumem`, which `ops/` may not import, so it
    # is injected here rather than called -- and here rather than at either module's import, because
    # a layer packs on its first forward.
    from ...ops.moe import set_buffer_allocator

    cast(Callable[[Any], None], set_buffer_allocator)(alloc_sleepable)

    # vLLM re-pins NCCL_MIN/MAX_NCHANNELS=1 inside each worker before its communicator exists.
    # Default ON where Placement owns the row-parallel reduce (ISOEXEC_PIK=1);
    # ISOEXEC_ENGINE_NCCL_UNPIN still overrides in both directions. See
    # ops/collectives/nccl_identity.engine_channel_pin_neutralized.
    # Engine EP dispatch sizes static buffers from the tokens every data-parallel peer feeds this
    # forward, which vLLM's forward context already carries on the host.
    from ...ops.moe.expert_parallel import set_engine_token_bound

    set_engine_token_bound(engine_token_bound)

    from .vllm_patches import neutralize_vllm_nccl_channel_pin

    neutralize_vllm_nccl_channel_pin()

    from .registration import register_plugin

    register_plugin()
    # A failed nested installation must not poison this worker's retry path.
    _installed = True
    print(f"[ISOEXEC-PLUGIN] native model registered in pid {os.getpid()}", flush=True)
