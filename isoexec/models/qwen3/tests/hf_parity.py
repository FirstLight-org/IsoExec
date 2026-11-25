"""Shared CPU harness: an IsoExec model package against Transformers on a tiny random checkpoint.

The IsoExec model is captured, loaded from the HF state dict through the package's ``weight.py`` and
placed with the reference op table; the HF model runs each packed sequence on its own.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch

import isoexec.nn as inn
from isoexec.compiler.capture.capture import capture
from isoexec.compiler.realization.storage import allocate_tensors
from isoexec.compiler.realization.weights import load_weights
from isoexec.compiler.source import ModelSource
from isoexec.compiler.text import text_inputs
from isoexec.ops.table import OpTable
from isoexec.runtimes.parallel import trivial_placement
from isoexec.schema import make_sequence_context


def randomize(hf: torch.nn.Module, seed: int = 0) -> None:
    """Gammas around one (so a +-1 convention error is visible), projections small, all bf16."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in hf.parameters():
            noise = torch.randn(parameter.shape, generator=generator)
            value = noise * 0.3 + 1.0 if parameter.ndim == 1 else noise * 0.08
            parameter.copy_(value.to(parameter.dtype))


class Harness:
    """A captured, loaded and reference-placed IsoExec model plus a callable over packed lengths."""

    def __init__(self, source: ModelSource, raw: Mapping, state: Mapping[str, torch.Tensor]) -> None:
        self.model = source.build_meta(raw).eval()
        examples = text_inputs(self.model, device="cpu", max_tokens=64)
        self.captured = capture(self.model, examples.args, examples.kwargs, dynamic_shapes=examples.dynamic_shapes,
                                root_layouts=examples.root_layouts, seq=examples.seq)
        allocate_tensors(self.model, device=torch.device("cpu"))
        self.loaded = load_weights(self.model, state.items(), source=source, hf_config=source.build_config(raw))
        inn.place(self.model, trivial_placement(), op_table=OpTable(mode="reference"))

    def logits(self, ids: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
        positions = torch.cat([torch.arange(n) for n in lengths])
        seq = make_sequence_context(positions, cu.clone(), max_q=max(lengths), max_kv=max(lengths), input_ids=ids,
                                    cu_seqlens_cpu=cu)
        with torch.no_grad():
            out = self.captured.program(ids, seq=seq)
        return out if torch.is_tensor(out) else out[0]


def hf_logits(hf: torch.nn.Module, ids: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    rows, start = [], 0
    with torch.no_grad():
        for n in lengths:
            rows.append(hf(ids[start:start + n][None]).logits[0].float())
            start += n
    return torch.cat(rows)


def plain_gamma_read_as_zero_centred(model: torch.nn.Module) -> Callable[[], None]:
    """Negative control: shift every held q/k gamma by +1, i.e. what a zero-centred kernel reading a
    plain-gamma checkpoint computes. Returns the undo."""
    norms = [p for n, p in model.named_parameters() if n.endswith(("mixer.q_norm", "mixer.k_norm"))]
    with torch.no_grad():
        for p in norms:
            p.add_(1.0)

    def undo() -> None:
        with torch.no_grad():
            for p in norms:
                p.sub_(1.0)

    return undo
