"""DDP buffer layout that keeps each packed projection's members adjacent, in row order.

Megatron's DDP lays every trainable parameter into one flat buffer per ``BufferKey``, in REVERSE
registration order (backprop order), splitting buckets wherever the running size crosses
``bucket_size``. A packed projection site (GDN ``in_proj`` = q|k|v|z|beta|alpha, attention
q|k|v|gate) therefore lands reversed, possibly split across a bucket boundary, and its GEMM has to
``torch.cat`` the members on every forward and recompute. This module computes the same layout
Megatron would -- same padding, same bucket rule, same shared-embedding rule -- except that each
packed site is placed as one unit, its members in forward row order, and never split by a bucket
boundary. ``ops/mm/packed_proj.bucket_weight`` then reads the site's operand as a view of those rows,
and ``main_grad`` (same offsets in the grad buffer) is one ``[sum(N_i), H]`` fp32 view as well, which
the packed weight-gradient GEMM accumulates into directly.

Layout is storage placement only: no value, gradient or optimizer arithmetic changes (the distributed
optimizer shards whatever the layout is; buckets keep their sizes up to one site).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch


def packed_sites(modules: Iterable[torch.nn.Module]) -> list[tuple[torch.nn.Parameter, ...]]:
    """Each module's packed members in row order (duck-typed on ``packed_proj_roles``)."""
    sites = []
    for mod in modules:
        roles = tuple(mod.packed_proj_roles()) if hasattr(mod, "packed_proj_roles") else ()
        params = tuple(getattr(mod, role) for role in roles)
        if len(params) >= 2 and all(isinstance(p, torch.nn.Parameter) and p.requires_grad for p in params):
            sites.append(params)
    return sites


def _placement_order(params: Sequence[torch.nn.Parameter], sites) -> list[tuple[torch.nn.Parameter, ...]]:
    """Megatron's buffer order (``params[::-1]``) as units: a site's members together, in row order, at
    the position of its first member Megatron would place; every other parameter a unit of one."""
    present = {id(p) for p in params}
    site_of = {}
    for site in sites:
        if all(id(p) in present for p in site):
            for p in site:
                site_of[id(p)] = site
    units, placed = [], set()
    for p in params[::-1]:
        site = site_of.get(id(p))
        if site is None:
            units.append((p,))
        elif id(site) not in placed:
            placed.add(id(site))
            units.append(site)
    return units


def per_buffer_layout(params: Sequence[torch.nn.Parameter], param_indices: list[int], sites, *,
                      bucket_size: int | None, distributed_optimizer: bool, dp_size: int, ddp_config):
    """Megatron's ``_compute_per_buffer_param_layout`` (distributed optimizer) or
    ``_compute_default_per_buffer_param_layout`` over placement units instead of parameters."""
    from megatron.core.optimizer.param_layout import PerBufferParamLayout, pad_bucket_end, pad_param_start

    pad = pad_param_start if distributed_optimizer else (lambda i: i)
    index_map, bucket_indices, unpadded = {}, [], []
    start = bucket_start = 0
    bucket_id, bucket_params, end = 0, 0, 0

    def finalize(end_index, bucket_start, bucket_id):
        unpadded.append(end_index - bucket_start)
        bucket_end = (pad_bucket_end(end_index, dp_size, ddp_config.pad_buckets_for_high_nccl_busbw)
                      if distributed_optimizer else end_index)
        bucket_indices.append((bucket_start, bucket_end))
        return bucket_end, bucket_id + 1

    for unit in _placement_order(params, sites):
        own_bucket = distributed_optimizer and any(getattr(p, "shared_embedding", False) for p in unit)
        start = pad(start)
        if own_bucket and bucket_params:
            bucket_start, bucket_id = finalize(start, bucket_start, bucket_id)
            bucket_params, start = 0, bucket_start
        for p in unit:
            start = pad(start)
            end = start + p.data.nelement()
            index_map[p] = (start, end, bucket_id)
            bucket_params += 1
            start = end
        if (bucket_size is not None and end - bucket_start >= bucket_size) or own_bucket:
            bucket_start, bucket_id = finalize(end, bucket_start, bucket_id)
            bucket_params, start = 0, bucket_start
    if bucket_params:
        finalize(end, bucket_start, bucket_id)
    kwargs = {"num_optimizer_shards": dp_size} if distributed_optimizer else {}
    return PerBufferParamLayout(param_index_map=index_map, bucket_indices=bucket_indices,
                                per_bucket_numel_unpadded=unpadded, param_indices=list(param_indices), **kwargs)


def full_param_layout(model: torch.nn.Module, sites, *, config, ddp_config, pg_collection):
    """The ``FullParamLayout`` to hand Megatron's DDP, or ``None`` to let it lay out on its own (no
    packed site, or a buffer kind -- layer-wise optimizer -- whose layout this does not mirror)."""
    if not sites:
        return None
    from megatron.core.distributed.param_and_grad_buffer import group_params_for_buffers
    from megatron.core.optimizer.param_layout import FullParamLayout
    from megatron.core.process_groups_config import ProcessGroupCollection

    try:
        groups = ProcessGroupCollection.setup_process_groups_for_ddp(pg_collection, config, ddp_config)
    except ValueError:
        return None  # a collection DDP itself would refuse; leave the layout (and the refusal) to DDP
    # DDP's own bucket-size rule (distributed_data_parallel.py), evaluated without mutating ddp_config.
    bucket_size = ddp_config.bucket_size
    if bucket_size is None:
        bucket_size = max(40000000, 1000000 * groups["dp_group"].size())
    if not ddp_config.overlap_grad_reduce:
        bucket_size = None
    pp = groups["pp_group"]
    if (pp[0] if isinstance(pp, list) else pp).rank() > 0:
        bucket_size = None
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    layouts = {}
    for key, (members, indices) in group_params_for_buffers(params, ddp_config.grad_reduce_in_fp32).items():
        if getattr(key, "is_managed_by_layer_wise_optimizer", False):
            return None
        dp = groups["intra_expt_dp_group" if key.is_expert_parallel else "intra_dp_cp_group"].size()
        layouts[key] = per_buffer_layout(members, indices, sites, bucket_size=bucket_size,
                                         distributed_optimizer=ddp_config.use_distributed_optimizer,
                                         dp_size=dp, ddp_config=ddp_config)
    return FullParamLayout(layouts=layouts)
