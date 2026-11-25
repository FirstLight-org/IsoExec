"""Pack projection weights and split output columns with a bit-exact backward. Read live parameters
on each call because distributed wrappers may replace their storage.
"""

from __future__ import annotations

import torch


def packed_view(*weights: torch.Tensor) -> torch.Tensor:
    """Concatenate live logical weights along dimension zero when no packed storage alias is
    installed.
    """
    return torch.cat([w for w in weights], dim=0)


#: Optional one-launch assembler for :class:`_SplitColumns`'s backward, installed by an op provider
#: (``ops/mm/wgrad.install_column_assembler``; ``core`` may not import ``ops``). It must compute exactly
#: the loop below -- ``cat([g + 0 for g in grads], dim=1)`` -- and return ``None`` for operands it declines.
_COLUMN_ASSEMBLER = None


def set_column_assembler(fn) -> None:
    """Install (or with ``None`` remove) the split-columns gradient assembler."""
    global _COLUMN_ASSEMBLER
    _COLUMN_ASSEMBLER = fn


class _SplitColumns(torch.autograd.Function):
    """Return column views and assemble their gradients into one buffer. Explicit addition of zero
    matches autograd slice accumulation, including conversion of negative zero to positive zero.
    """

    @staticmethod
    def forward(ctx, out, offsets):  # noqa: D102
        ctx.offsets = offsets
        return tuple(out[:, offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1))

    @staticmethod
    def backward(ctx, *grads):  # noqa: D102
        off = ctx.offsets
        if _COLUMN_ASSEMBLER is not None:
            buf = _COLUMN_ASSEMBLER(grads)
            if buf is not None:
                return buf, None
        g0 = grads[0]
        buf = torch.empty((g0.shape[0], off[-1]), dtype=g0.dtype, device=g0.device)
        for i, g in enumerate(grads):
            torch.add(g, 0, out=buf[:, off[i] : off[i + 1]])
        return buf, None


def split_widths(out: torch.Tensor, widths: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    """Split ``out`` into adjacent column views of the given widths. With gradients enabled the
    gradient is assembled by :class:`_SplitColumns` (one buffer, ``g + 0`` per block) -- the bits autograd's
    slice accumulation produces, without a zero-filled ``[T, sum(widths)]`` buffer and an add per slice."""
    offsets = [0]
    for w in widths:
        offsets.append(offsets[-1] + int(w))
    if not (torch.is_grad_enabled() and out.requires_grad):
        return tuple(out[:, offsets[i] : offsets[i + 1]] for i in range(len(widths)))
    return _SplitColumns.apply(out, tuple(offsets))


def split_columns(out: torch.Tensor, weights: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Split projected columns into adjacent storage views in weight order. With gradients enabled,
    use the shared-buffer backward to match slice-accumulation bits.
    """
    return split_widths(out, tuple(int(w.shape[0]) for w in weights))
