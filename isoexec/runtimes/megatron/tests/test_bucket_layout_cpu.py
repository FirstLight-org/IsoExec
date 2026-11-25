"""The packed-site DDP layout is Megatron's layout, except packed members are adjacent rows in row order."""

from __future__ import annotations

import pytest
import torch

from isoexec.runtimes.megatron.layout import per_buffer_layout


def _params(shapes):
    return [torch.nn.Parameter(torch.empty(s, dtype=torch.bfloat16)) for s in shapes]


def _cfg(**kw):
    from megatron.core.distributed import DistributedDataParallelConfig

    return DistributedDataParallelConfig(**kw)


SHAPES = [(64, 128), (8, 64), (512, 128), (512, 128), (1536, 128), (1536, 128), (12, 128), (12, 128), (128,),
          (300, 128), (7,), (4120, 128)]


@pytest.mark.parametrize("distopt", [True, False])
@pytest.mark.parametrize("bucket_size", [None, 100_000, 250_000])
def test_no_site_is_megatrons_layout(distopt, bucket_size):
    from megatron.core.distributed.param_and_grad_buffer import _compute_default_per_buffer_param_layout
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    params = _params(SHAPES)
    params[3].shared_embedding = True
    cfg = _cfg(use_distributed_optimizer=distopt)
    indices = list(range(len(params)))
    if distopt:
        want = DistributedOptimizer._compute_per_buffer_param_layout(params, bucket_size, 4, cfg, indices)
    else:
        want = _compute_default_per_buffer_param_layout(params, bucket_size)
        want.param_indices = indices
    got = per_buffer_layout(params, indices, [], bucket_size=bucket_size, distributed_optimizer=distopt, dp_size=4,
                            ddp_config=cfg)
    assert got.param_index_map == want.param_index_map
    assert got.bucket_indices == want.bucket_indices
    assert got.per_bucket_numel_unpadded == want.per_bucket_numel_unpadded
    assert got.param_indices == want.param_indices
    assert got.num_optimizer_shards == want.num_optimizer_shards


@pytest.mark.parametrize("distopt", [True, False])
@pytest.mark.parametrize("bucket_size", [None, 50_000, 200_000])
def test_site_members_adjacent_in_row_order_and_never_split(distopt, bucket_size):
    params = _params(SHAPES)
    site = tuple(params[2:8])  # q|k|v|z|beta|alpha
    other = (params[0], params[1])  # a second, two-member site
    cfg = _cfg(use_distributed_optimizer=distopt)
    got = per_buffer_layout(params, list(range(len(params))), [site, other], bucket_size=bucket_size,
                            distributed_optimizer=distopt, dp_size=2, ddp_config=cfg)
    for members in (site, other):
        spans = [got.param_index_map[p] for p in members]
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))  # adjacent, in row order
        assert len({s[2] for s in spans}) == 1  # one bucket
    # every parameter placed exactly once, no overlaps
    spans = sorted(v[:2] for v in got.param_index_map.values())
    assert len(spans) == len(params) and all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    # bucket boundaries are respected by every parameter
    for start, end, b in got.param_index_map.values():
        lo, hi = got.bucket_indices[b]
        assert lo <= start and end <= hi


def test_adjacent_rows_view_and_refusals():
    from isoexec.ops.mm.packed_proj import adjacent_rows

    flat = torch.arange(10 * 4, dtype=torch.float32)
    a, b, c = flat[:8].view(2, 4), flat[8:20].view(3, 4), flat[20:40].view(5, 4)
    v = adjacent_rows([a, b, c])
    assert v is not None and v.shape == (10, 4) and v.data_ptr() == a.data_ptr()
    assert torch.equal(v, torch.cat([a, b, c]))
    assert adjacent_rows([a, c]) is None  # gap
    assert adjacent_rows([b, a]) is None  # order
    assert adjacent_rows([a, torch.zeros(3, 4)]) is None  # other storage
    assert adjacent_rows([a, flat[8:20].view(2, 6)]) is None  # width
