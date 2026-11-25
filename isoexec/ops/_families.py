"""Import-light catalog shared by declarations, reference bindings and selection."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OpFamily:
    package: str
    prefixes: tuple[str, ...]
    placement_owned: bool = False


# Stable family order shared by registry assembly and reference selection.
FAMILIES = (
    OpFamily("mm", ("mm",)),
    OpFamily("norms", ("norms",)),
    OpFamily("attention", ("attention", "rope")),
    OpFamily("logprobs", ("logprobs",)),
    OpFamily("gdn", ("gdn", "conv")),
    OpFamily("moe", ("moe",)),
    OpFamily("collectives", ("collectives",), placement_owned=True),
    OpFamily("embed", ("embed",)),
    OpFamily("residual", ("residual",)),
    OpFamily("mlp", ("mlp",)),
)
