"""Closed token-flow declarations shared by components and graph analysis."""

from dataclasses import dataclass
from enum import Enum


class TokenLayoutEffect(Enum):
    # Inputs/outputs describe the component boundary, before adapter movement.
    EMBEDDING = "replicated token IDs to replicated hidden rows"
    DECODER = "sequence chunks to sequence chunks when SP is active"
    PRESERVE = "preserve the input token layout"
    HEAD = "trainer chunks or engine selected rows to complete token rows"


@dataclass(frozen=True)
class TokenLayoutContract:
    """A component's token-flow effect and the actual tensor axis it preserves.

    DECODER owns its existing gather/reduce-scatter seams. HEAD owns the trainer
    input gather; engines supply already selected complete rows. No declaration
    implements a collective or assigns model layers to pipeline ranks.
    """

    effect: TokenLayoutEffect
    token_axis: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.effect, TokenLayoutEffect):
            raise ValueError("Token layout effect must be a declared TokenLayoutEffect")
        if type(self.token_axis) is not int or self.token_axis < 0:
            raise ValueError("Token axis must be a nonnegative integer")
