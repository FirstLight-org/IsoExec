"""Run a CP world's ranks as processes sharing ONE GPU, with byte-exact Gloo collectives.

The context-parallel seams call ``torch.distributed.all_gather_into_tensor`` and ``all_to_all_single`` on
the CP group. NCCL refuses two ranks on one device, so :func:`install_gloo_cuda_shim` routes those two calls
through Gloo on host copies of the CUDA payloads. Both only move bytes, so every rank runs the real
kernels and the real movement code; what the harness cannot show is NCCL itself.
"""

from __future__ import annotations

import torch


def install_gloo_cuda_shim() -> None:
    import torch.distributed as dist

    native_gather, native_a2a = dist.all_gather_into_tensor, dist.all_to_all_single

    def as_bytes(t):
        return t.detach().contiguous().view(-1).view(torch.uint8)

    def all_gather_into_tensor(out, inp, group=None, async_op=False):
        host = torch.empty(out.numel() * out.element_size(), dtype=torch.uint8)
        native_gather(host, as_bytes(inp).cpu(), group=group)
        out.view(-1).view(torch.uint8).copy_(host)

    def all_to_all_single(out, inp, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False):
        # Variable splits (expert dispatch) count rows of dim 0; Gloo moves them as byte rows.
        row = lambda t: (t.numel() // t.shape[0] if t.dim() and t.shape[0] else 1) * t.element_size()
        osz = None if output_split_sizes is None else [int(n) * row(out) for n in output_split_sizes]
        isz = None if input_split_sizes is None else [int(n) * row(inp) for n in input_split_sizes]
        host = torch.empty(out.numel() * out.element_size(), dtype=torch.uint8)
        native_a2a(host, as_bytes(inp).cpu(), osz, isz, group=group)
        out.view(-1).view(torch.uint8).copy_(host)

    dist.all_gather_into_tensor, dist.all_to_all_single = all_gather_into_tensor, all_to_all_single


def init_world(rank: int, world: int, init_method: str):
    """Gloo world on the shared GPU; returns the group the CP placement carries."""
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world)
    install_gloo_cuda_shim()
    torch.cuda.set_device(0)
    return dist.group.WORLD
