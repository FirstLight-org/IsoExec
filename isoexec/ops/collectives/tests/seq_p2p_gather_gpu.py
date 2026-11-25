"""Two-TP-group byte and shared-buffer lifetime checks.

Launch with torchrun on an even number of ranks >= 4 (8 ranks: two TP4 groups; 4 ranks: two TP2 groups).
Skipped on an NCCL-only arch (sm100): the checks here are of the P2P transport."""
import json
import os

import torch
import torch.distributed as dist

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
device = torch.device('cuda', rank)
from isoexec.ops.collectives.pik_tp_invariant import p2p_arch_refusal

if (refusal := p2p_arch_refusal()) is not None:
    if rank == 0:
        print(f"SKIPPED: {refusal}", flush=True)
    raise SystemExit(0)
dist.init_process_group('nccl', device_id=device)
world = dist.get_world_size()
assert world >= 4 and world % 2 == 0, world
tp = world // 2
groups = [dist.new_group(ranks=list(range(start, start + tp)), backend='nccl')
          for start in (0, tp)]
group = groups[rank // tp]
from isoexec.ops.collectives import tp_bytes as module
gather = module._gather_dim0
from isoexec.ops.collectives.pik import allreduce as ar


def bits(x):
    return x.contiguous().view(torch.uint8)


def reference(x):
    x = x.contiguous()
    out = x.new_empty((tp * x.shape[0], *x.shape[1:]))
    dist.all_gather_into_tensor(out, x, group=group)
    return out


cases = []
# Complete BF16 encoding space, with rank-specific rotations. This includes
# signed zero, every subnormal, infinities and all quiet/signaling NaN payloads.
enc = ((torch.arange(65536, device=device, dtype=torch.int64) + 733 * rank)
       % 65536).to(torch.int16)
cases.append(enc.view(torch.bfloat16).reshape(512, 128))
edge = torch.tensor([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
                     0x7f800000, 0xff800000, 0x7fc00001, 0xffc12345,
                     0x7f800001, 0x7f7fffff], device=device, dtype=torch.int64)
cases.append(edge.roll(rank).to(torch.int32).view(torch.float32).reshape(3, 4))
torch.manual_seed(7100 + rank)
for dtype in (torch.bfloat16, torch.float32):
    for shape in ((1, 3), (17, 31), (1024, 128), (8192, 2048), (5, 7)):
        cases.append(torch.randn(shape, device=device, dtype=dtype))
    cases.append(torch.randn((33, 257), device=device, dtype=dtype).t())
    cases.append(torch.empty((0, 19), device=device, dtype=dtype))

checked = 0
for x in cases:
    expected = reference(x)
    actual = gather(x, tp, group)
    assert torch.equal(bits(actual), bits(expected)), (rank, x.shape, x.dtype)
    checked += 1

# No host synchronization between movements: alternate gather and tree AR on
# the same dtype-keyed symmetric input pools, then inspect retained outputs.
pending = []
for i in range(48):
    dtype = torch.bfloat16 if i % 2 else torch.float32
    rows = (17, 1024, 33, 4096)[i % 4]
    x = torch.full((rows, 32), rank * 64 + i, dtype=dtype, device=device)
    actual = gather(x, tp, group)
    pending.append((actual, i, rows, dtype))
    partial = torch.full((rows * tp, 32), i + rank, dtype=dtype, device=device)
    ar.tree_all_reduce(partial, group=group)
for actual, i, rows, dtype in pending:
    expected = torch.cat([
        torch.full((rows, 32), peer * 64 + i, dtype=dtype, device=device)
        for peer in range((rank // tp) * tp, (rank // tp) * tp + tp)
    ])
    assert torch.equal(bits(actual), bits(expected)), (rank, i)
    checked += 1
dist.barrier()
transport = module.seq_gather_transport_counts()
assert transport['p2p_calls'] >= checked - 2, (rank, transport)
assert transport['p2p_fallbacks'] == 0, (rank, transport)
if rank == 0:
    print(json.dumps({'status': 'passed', 'cases_per_rank': checked,
                      'ranks': world, 'tp_groups': 2, 'tp': tp,
                      'gather': f'{module.__name__}.{gather.__name__}',
                      'transport': transport}), flush=True)
dist.destroy_process_group()
