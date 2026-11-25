"""Construct the model through vLLM's native configuration entry point."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsMRoPE, SupportsPP

from isoexec.compiler.capture.capture import capture
from isoexec.compiler.partition import define_stage
from isoexec.compiler.source import ModelSource, ModelSourceError
from isoexec.compiler.text import text_inputs
from isoexec.nn import MoE
from isoexec.ops._dispatch import install_arch_impls
from isoexec.runtimes.precision import pin_fp32_ieee

from . import model
from .scoring_temperature import install as install_scoring_temperature
from .validation import kv_cache_refusal, parallel_config_context_parallel_refusal, structured_outputs_refusal

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _install_engine_gemms(source: ModelSource, raw: Mapping[str, object]) -> None:
    """Install the trainer's GEMM provider (the segment-tree GEMM over the vendored Triton kernel) in the engine.

    ``install_trainer_kernels`` does the same on the trainer, so both sides run the same provider chain;
    which shapes the segment-tree GEMM serves is a function of the shape alone. The census only feeds the
    coverage banner.
    """
    from isoexec.ops.mm import mm_shapes
    from isoexec.runtimes.trainer_installs import model_gemm_census

    mm_shapes.set_live_context(census=model_gemm_census(source, raw), tp=1, etp=1, side="ENGINE")
    from isoexec.ops.mm.segtree import install_mm_segtree

    install_mm_segtree()  # the trainer's provider, from the same census
    # The self-checks run before vLLM profiles memory; hand their blocks back to the driver now.
    import torch

    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


class DenseVllmModel(model.VllmModel, SupportsMRoPE, SupportsPP):
    """A native vLLM model built from the configured author package.

    ``additional_config.isoexec`` supplies ``model_source`` and the original
    ``raw_model_config``. The execution graph and stage definition are temporary;
    registered state and execution belong to the constructed model.
    ``checkpoint_path`` optionally selects a filename for the native loader.
    ``score_temperature`` sets the logprob reporting temperature, defaulting to one.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        for refusal in (structured_outputs_refusal(vllm_config), kv_cache_refusal(vllm_config),
                        parallel_config_context_parallel_refusal(vllm_config.parallel_config)):
            if refusal is not None:
                raise ValueError(refusal)
        settings = (vllm_config.additional_config or {}).get("isoexec") or {}
        source = ModelSource.from_dict(settings.get("model_source"))
        raw = settings.get("raw_model_config")
        if not isinstance(raw, Mapping):
            raise ModelSourceError("VllmModel requires additional_config.isoexec.raw_model_config")
        # Targeted installs only. The trainer's full install_trainer_kernels runs GEMM self-checks that
        # allocate and free device memory during construction, which vLLM's startup memory profiling
        # reads as another process releasing memory and refuses.
        pin_fp32_ieee()
        install_arch_impls()
        author = source.build_meta(raw).eval()
        if any(isinstance(module, MoE) for module in author.modules()):
            # Router and expert GEMMs reach aten::mm; install_trainer_kernels routes it to
            # the same batch-invariant kernel on the trainer.
            from isoexec.ops.moe import moe_batch_invariant

            moe_batch_invariant._install_moe_matmul_invariance()
        _install_engine_gemms(source, raw)
        examples = text_inputs(author, device=vllm_config.device_config.device,
                               max_tokens=vllm_config.scheduler_config.max_num_batched_tokens, profile="engine")
        execution_graph = capture(author, examples.args, examples.kwargs, dynamic_shapes=examples.dynamic_shapes,
                                  root_layouts=examples.root_layouts, seq=examples.seq)
        from vllm.distributed import get_pp_group
        from vllm.distributed.utils import get_pp_indices

        group = get_pp_group().device_group
        count, pp = len(execution_graph.layers), group.size()
        # vLLM's own stage layout, which honours VLLM_PP_LAYER_PARTITION.
        assignments = tuple(get_pp_indices(count, rank, pp) for rank in range(pp))
        definition = define_stage(execution_graph, assignments, pp_rank=group.rank())
        super().__init__(definition, source=source, hf_config=source.build_config(raw),
                         vllm_config=vllm_config, prefix=prefix)
        install_scoring_temperature(self, settings.get("score_temperature", 1.0))
        if settings.get("checkpoint_path") is not None:
            self.allow_patterns_overrides = [Path(settings["checkpoint_path"]).name]


class VllmModel(DenseVllmModel, HasInnerState, IsHybrid):
    """The native vLLM model for graphs with GDN state layers, which vLLM treats as hybrid."""
