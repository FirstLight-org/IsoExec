"""Construct and load Megatron models, then attach native training objects."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import closing
from functools import partial
from typing import TYPE_CHECKING, Any

import torch

from isoexec.compiler.capture.capture import capture
from isoexec.compiler.partition import define_stage
from isoexec.compiler.text import text_inputs
from isoexec.storage import CheckpointStore

from ._config import make_transformer_config
from .model import MegatronModel

if TYPE_CHECKING:
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import MegatronOptimizer, OptimizerConfig
    from megatron.core.process_groups_config import ProcessGroupCollection

    from isoexec.compiler.source import ModelSource


def build_model(
    source: ModelSource,
    raw_config: Mapping[str, Any],
    *,
    checkpoint_path: str,
    training: bool,
    max_tokens: int,
    pg_collection: ProcessGroupCollection,
    transformer_config_kwargs: Mapping[str, Any],
    cp_layout: str | None = None,
) -> MegatronModel:
    """Return a native model with every registered parameter loaded.

    The caller installs trainer kernels and initializes process groups before
    construction. Capture uses the requested training mode and token bound.
    Native configuration uses captured component dimensions and the supplied
    parallelism and determinism settings.

    Raises:
        ValueError: The checkpoint does not cover every registered parameter.
    """
    author = source.build_meta(raw_config).train(training)
    examples = text_inputs(author, device=torch.device("cuda", torch.cuda.current_device()),
                           max_tokens=max_tokens, profile="trainer")
    execution_graph = capture(author, examples.args, examples.kwargs, dynamic_shapes=examples.dynamic_shapes,
                              root_layouts=examples.root_layouts, seq=examples.seq)
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    config = make_transformer_config(
        execution_graph, **transformer_config_kwargs, bf16=True,
        params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16,
        gradient_accumulation_fusion=False, variable_seq_lengths=True,
        calculate_per_token_loss=False, perform_initialization=False,
    )
    # Megatron's own pipeline stage layout: each rank's [first, last) decoder rows.
    offsets = [get_transformer_layer_offset(config, vp_stage=None, pp_rank=rank)
               for rank in range(pg_collection.pp.size())]
    assignments = tuple((offset, offset + get_num_layers_to_build(config, vp_stage=None, pp_rank=rank))
                        for rank, offset in enumerate(offsets))
    definition = define_stage(execution_graph, assignments, pp_rank=pg_collection.pp.rank())
    model = MegatronModel(definition, source=source, hf_config=source.build_config(raw_config), config=config,
                          pg_collection=pg_collection, cp_layout=cp_layout)
    with closing(CheckpointStore(checkpoint_path)) as store:
        keys = store.keys()
        loaded = model.load_weights((name, store.reader(name)[:]) for name in keys)
    loaded_parameters = {model.get_parameter(name) for name in loaded}
    missing = {name for name, parameter in model.named_parameters() if parameter not in loaded_parameters}
    if missing:
        raise ValueError(f"Initial checkpoint is missing model parameters: {sorted(missing)}")
    return model


def wrap_ddp_and_optimizer(
    model: MegatronModel,
    *,
    ddp_config: DistributedDataParallelConfig,
    optimizer_config: OptimizerConfig,
) -> tuple[DistributedDataParallel, MegatronOptimizer]:
    """Return native DDP and optimizer objects using the model's registered parameters.

    Both objects borrow the model's process groups. Install the native gradient
    finalizer and DDP no-sync callback on the model's configuration.
    """
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.optimizer import get_megatron_optimizer

    from .optimizer import dtype_uniform_param_group_overrides, leaf_fp32_optimizer_shards

    config, groups = model.config, model.pg_collection
    sites = _bucket_pack_modules(model)
    layout = None
    if sites:
        from .layout import full_param_layout, packed_sites

        # Each packed projection's members as adjacent rows, in row order, of one bucket: the GEMM reads
        # them as one view instead of a per-forward cat, and its wgrad accumulates into one main_grad view.
        layout = full_param_layout(model, packed_sites(sites), config=config, ddp_config=ddp_config,
                                   pg_collection=groups)
    extra = {} if layout is None else {"full_param_layout": layout}
    ddp = DistributedDataParallel(config, ddp_config, model, pg_collection=groups, **extra)
    if layout is not None:
        from isoexec.ops.mm.packed_proj import attach_bucket_alias

        attach_bucket_alias(sites)
    if sites:
        print(f"[ISOEXEC-PROJ-PACK] trainer DDP layout: {len(sites)} packed sites, "
              f"{'packed layout' if layout is not None else 'Megatron default layout (cat path)'}", flush=True)
    if optimizer_config.use_precision_aware_optimizer:
        from .grad_kernels import install_mixed_dtype_grad_kernels

        install_mixed_dtype_grad_kernels()
    with leaf_fp32_optimizer_shards(optimizer_config.optimizer_cpu_offload):
        optimizer = get_megatron_optimizer(
            optimizer_config, [ddp], config_overrides=dtype_uniform_param_group_overrides(optimizer_config),
            use_gloo_process_groups=False, pg_collection=groups,
        )
    config.no_sync_func = ddp.no_sync
    config.finalize_model_grads_func = partial(finalize_model_grads, pg_collection=groups)
    return ddp, optimizer


def _bucket_pack_modules(model: MegatronModel) -> list:
    """The components whose projection members the DDP layout packs: GDN in_proj and attention
    q|k|v|gate. MLP gate|up is not packed -- measured slower packed (``ops/mm/packed_proj``).
    ``ISOEXEC_TRAINER_BUCKET_PACK=0`` keeps Megatron's own layout and the per-forward ``cat``."""
    import os

    from isoexec.nn import GDN, Attention

    if os.environ.get("ISOEXEC_TRAINER_BUCKET_PACK", "1") == "0":
        return []
    return [m for m in model.modules() if isinstance(m, (GDN, Attention))
            and len(tuple(m.packed_proj_roles())) >= 2]
