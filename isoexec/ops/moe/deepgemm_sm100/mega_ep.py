"""Symmetric output region and gather protocol of MegaMoE over a real EP process group.

Token layout. Replicated TP rows: rank r feeds only its canonical home slice ``home_slice`` (so the router and
the shared expert run on M / W rows, not M), and the home slices are reassembled by the byte gather of
``csrc/mega_ep_io.cu``. Partitioned rows (sequence parallel) are fed as they are, without the gather.

Gather. MegaMoE stores its output rows into this rank's region of a second symmetric buffer (``OutRegion``); the
gather kernel publishes "call t done" to every peer (st.release.sys), waits for every peer's flag (local
polling) and reads the peers' rows over NVLink into the output. The tag is a per-rank call counter advanced by
every prep launch (identical on all ranks: every call is collective); a flag ahead of the tag is a protocol
violation and traps (fail closed).

Reuse of the output region across calls needs no extra barrier: a peer overwrites its region only in the combine
of its NEXT MegaMoE launch, which starts after that launch's pre-dispatch NVLink barrier, i.e. after every rank
entered the next launch -- and each rank enters it only after its own gather of this call, which is
stream-ordered before it.

Uniform launch geometry. MegaMoE derives its tile configuration (block_m, ...) from the CALLING rank's token count,
so every rank must pass the same count: each call feeds ``width`` rows per rank (the widest home slice, or a fixed
chunk of it, ``chunks``), padding with zero rows whose top-k ids are -1 (never dispatched; their shared-expert
output is discarded). Real rows are row-independent, so padding never changes their bits.

Progress. MegaMoE is a persistent grid of num_SMs CTAs with in-kernel grid syncs and NVLink barriers: it needs all
of its CTAs co-resident, which holds when it runs alone on the device (IsoExec's one-runtime-stream invariant,
the same assumption native vLLM makes); it is not safe under MPS or a foreign kernel pinned on an SM. Calls are
serialized on the stream (one workspace for all layers), like every IsoExec collective.
"""
from __future__ import annotations

import torch


def home_slice(M: int, world: int, rank: int) -> tuple[int, int]:
    """Canonical contiguous partition of M replicated rows into EP homes (ragged and empty homes)."""
    return M * rank // world, M * (rank + 1) // world


def chunks(width: int, cap: int) -> list[int]:
    """Equal per-rank row counts covering ``width`` (identical on every rank: a function of width and cap)."""
    if width <= cap:
        return [max(width, 1)]
    n = -(-width // cap)
    per = -(-width // n)
    return [per] * n


class OutRegion:
    """Second symmetric buffer of a MegaMoE workspace: [W u64 gather flags | u64 call counter | cap x H BF16 MegaMoE
    output rows]. Every rank's region has the same layout; ``ptrs`` holds all ranks' base addresses (device int64
    [W]). Used by ``prep`` / ``prep_bf16`` (counter) and ``gather`` (flags, rows) of ``csrc/mega_ep_io.cu``.
    Collective at W > 1 (symmetric-memory rendezvous through ``moe.symm_rendezvous``: a rank-skewed setup sequence
    raises ``RendezvousSkewError`` on every rank); allocate outside CUDA graph capture. At W = 1 nothing is
    rendezvoused."""

    def __init__(self, group, world, device, cap, hidden):
        import torch.distributed as dist
        self.flags_off = 0
        self.counter_off = -(-8 * world // 128) * 128
        self.y_off = -(-(self.counter_off + 8) // 4096) * 4096
        nbytes = self.y_off + cap * hidden * 2
        self.handle = None
        with torch.cuda.device(device):
            if world == 1:
                self.buf = torch.zeros(nbytes, dtype=torch.int8, device=device)
                ptrs = [self.buf.data_ptr()]
            else:
                import torch.distributed._symmetric_memory as symm_mem

                from isoexec.ops.moe import symm_rendezvous
                self.buf = symm_mem.empty(nbytes, dtype=torch.int8, device=device)
                self.handle = symm_rendezvous.rendezvous(self.buf, group, site="moe.mega_ep.out_region")
                ptrs = list(self.handle.buffer_ptrs)
                self.buf.zero_()
            self.ptrs = torch.tensor(ptrs, dtype=torch.int64, device=device)
            torch.cuda.synchronize(device)
        if world > 1:
            dist.barrier(group=group)
        self.counter = self.buf[self.counter_off:self.counter_off + 8].view(torch.int64)
        self.y = self.buf[self.y_off:].view(torch.bfloat16).view(cap, hidden)


__all__ = ["OutRegion", "chunks", "home_slice"]
