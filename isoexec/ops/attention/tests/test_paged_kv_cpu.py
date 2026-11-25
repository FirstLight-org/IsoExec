import pytest
import torch

from isoexec.ops.attention.paged_kv import split_paged_kv_cache


def test_split_packed_vllm_028_layout():
    blocks, heads, block_size, head_size = 3, 2, 16, 4
    packed = torch.arange(blocks * heads * block_size * 2 * head_size).reshape(
        blocks, heads, block_size, 2 * head_size
    )

    key, value = split_paged_kv_cache(packed, num_kv_heads=heads, head_size=head_size)

    assert key.shape == (blocks, block_size, heads, head_size)
    assert value.shape == (blocks, block_size, heads, head_size)
    assert torch.equal(key, packed.transpose(1, 2)[..., :head_size])
    assert torch.equal(value, packed.transpose(1, 2)[..., head_size:])


def test_split_legacy_explicit_kv_axis_layout():
    cache = torch.arange(3 * 2 * 16 * 2 * 4).reshape(3, 2, 16, 2, 4)

    key, value = split_paged_kv_cache(cache, num_kv_heads=2, head_size=4)

    assert torch.equal(key, cache[:, 0])
    assert torch.equal(value, cache[:, 1])


def test_split_rejects_unknown_layout():
    with pytest.raises(ValueError, match="unsupported paged KV cache layout"):
        split_paged_kv_cache(torch.empty(3, 16, 2, 4), num_kv_heads=2, head_size=4)
