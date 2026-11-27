"""Memory-only slot initialization and invalid ownership refusal."""

import pytest
import torch

from isoexec.ops.kda.tests._fixtures import require_sm100


@pytest.fixture(autouse=True)
def _gpu():
    require_sm100()


@pytest.mark.parametrize('index_dtype', [torch.int32, torch.int64])
@torch.no_grad()
def test_strided_reset_preserves_continuations_null_padding_and_gaps(index_dtype):
    from isoexec.ops.kda.slot_reset_sm100 import reset_fresh_slots

    conv_storage = torch.full((6, 11, 4), 3.0, device='cuda', dtype=torch.bfloat16)
    rec_storage = torch.full((6, 3, 71), 123, device='cuda', dtype=torch.uint8)
    conv, rec = conv_storage[:, :7, :3], rec_storage[:, :2, :67]
    expected_conv, expected_rec = conv_storage.clone(), rec_storage.clone()
    expected_conv[4, :7, :3] = 0
    expected_rec[4, :2, :67] = 0
    slots = torch.tensor([0, -1, 4, 2], dtype=index_dtype, device='cuda')
    pos = torch.tensor([0, 0, 0, 7], dtype=index_dtype, device='cuda')
    reset_fresh_slots(conv, rec, slots, pos)
    assert torch.equal(conv_storage.view(torch.int16), expected_conv.view(torch.int16))
    assert torch.equal(rec_storage, expected_rec)


@pytest.mark.parametrize('bad_slots', ['[1, 1]', '[1, 9]'])
def test_invalid_live_slot_ownership_refuses_in_isolated_cuda_context(bad_slots):
    import subprocess
    import sys

    # A device assertion deliberately poisons its CUDA context; use a fresh
    # process so the rest of the numerical suite remains meaningful.
    code = f'''
import torch
from isoexec.ops.kda.slot_reset_sm100 import reset_fresh_slots
conv = torch.zeros(3, 7, 3, dtype=torch.bfloat16, device='cuda')
rec = torch.zeros(3, 2, 67, dtype=torch.uint8, device='cuda')
slots = torch.tensor({bad_slots}, dtype=torch.int32, device='cuda')
positions = torch.zeros(2, dtype=torch.int64, device='cuda')
reset_fresh_slots(conv, rec, slots, positions)
torch.cuda.synchronize()
'''
    run = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
    assert run.returncode != 0
    assert 'device-side assert' in run.stdout + run.stderr




@pytest.mark.parametrize('layers', [1, 3, 34, 64])
@torch.no_grad()
def test_layers_reset_equals_per_layer_resets(layers):
    """One launch over many layers' states makes exactly the per-layer launches' stores, in replay too."""
    from isoexec.ops.kda.slot_reset_sm100 import reset_fresh_slots, reset_fresh_slots_layers

    g = torch.Generator(device='cuda').manual_seed(layers)
    def states():
        conv_storage = torch.randn(layers, 9, 11, 4, device='cuda', generator=g).to(torch.bfloat16)
        rec_storage = torch.randint(0, 255, (layers, 9, 3, 71), device='cuda', generator=g, dtype=torch.uint8)
        return conv_storage, rec_storage
    conv_a, rec_a = states()
    conv_b, rec_b = conv_a.clone(), rec_a.clone()
    slots = torch.tensor([0, -1, 4, 2, 7, 8], dtype=torch.int32, device='cuda')
    pos = torch.tensor([0, 0, 0, 7, 0, 3], dtype=torch.int64, device='cuda')
    view = lambda c, r, l: (c[l, :, :7, :3], r[l, :, :2, :67])
    for l in range(layers):
        reset_fresh_slots(*view(conv_a, rec_a, l), slots, pos)
    convs, recs = zip(*(view(conv_b, rec_b, l) for l in range(layers)))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        reset_fresh_slots_layers(convs, recs, slots, pos)
    graph.replay()
    assert torch.equal(conv_a.view(torch.int16), conv_b.view(torch.int16))
    assert torch.equal(rec_a, rec_b)
    # a replay with other slots/positions resets other rows, as the per-layer launches would
    slots.copy_(torch.tensor([3, 0, 5, 1, 6, 8], dtype=torch.int32, device='cuda'))
    pos.copy_(torch.tensor([0, 0, 2, 0, 0, 0], dtype=torch.int64, device='cuda'))
    for l in range(layers):
        reset_fresh_slots(*view(conv_a, rec_a, l), slots, pos)
    graph.replay()
    assert torch.equal(conv_a.view(torch.int16), conv_b.view(torch.int16))
    assert torch.equal(rec_a, rec_b)


def test_layers_reset_refuses_mixed_geometry():
    from isoexec.ops.kda.slot_reset_sm100 import reset_fresh_slots_layers
    convs = [torch.zeros(3, 7, 3, dtype=torch.bfloat16, device='cuda'), torch.zeros(3, 8, 3, dtype=torch.bfloat16, device='cuda')]
    recs = [torch.zeros(3, 2, 67, dtype=torch.uint8, device='cuda')] * 2
    slots = torch.tensor([1], dtype=torch.int32, device='cuda')
    with pytest.raises(RuntimeError):
        reset_fresh_slots_layers(convs, recs, slots, torch.zeros(1, dtype=torch.int32, device='cuda'))
