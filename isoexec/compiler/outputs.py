"""The authored logits output and optional named auxiliary tensors."""

from typing import Any

import torch
from torch import fx


def model_outputs(value: Any) -> tuple[Any, dict[str, Any]]:
    """Accept logits or (logits, named tensors), without detaching any output."""
    match value:
        case torch.Tensor() | fx.Node():
            return value, {}
        case tuple((torch.Tensor() | fx.Node(), dict() as aux)) if all(
            isinstance(name, str) and isinstance(tensor, (torch.Tensor, fx.Node))
            for name, tensor in aux.items()
        ):
            return value
        case _:
            raise ValueError(
                "The model must return terminal LMHead logits or (logits, dict[str, Tensor])"
            )
