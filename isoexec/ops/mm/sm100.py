"""Strict SM100 dense forward with the trainer's established GEMM backward."""

import torch

from isoexec.ops.mm.trainer import _MmFn


class _SM100Mm(_MmFn):
    @staticmethod
    def forward(ctx, x, w):
        from isoexec.ops.mm.mm_sm100.ops import mm_sm100
        from isoexec.ops.mm.mm_sm100.segments import PITCH_ALIGN_N

        ctx.save_for_backward(x, w)
        n = w.shape[0]
        pad = -n % PITCH_ALIGN_N
        if pad:
            # The output's TMA pitch must be 16 bytes. Every output column is its own K chain, so the
            # zero rows appended past N leave the first N columns' bits untouched (e.g. Qwen3.5-0.8B's
            # GDN in_proj at TP 8, N = 1028).
            w = torch.nn.functional.pad(w, (0, 0, 0, pad))
            return mm_sm100(x, w.t())[:, :n].contiguous()
        return mm_sm100(x, w.t())


def mm_linear(x, w):
    """Unsupported shapes refuse; they never change the recorded forward provider."""
    return _SM100Mm.apply(x.contiguous(), w)

