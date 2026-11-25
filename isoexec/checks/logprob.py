"""Check 5 -- the logprob gate: rollout logprobs vs the trainer's, before backward.

The other seven checks compare declarations. This one compares numbers, and it is the only check
that would still catch a divergence produced by something none of the declarations model.

The bound is a property of WHO OWNS the scoring chain, and the deployment declares it:

* ``contracted`` -- SkyRL under ``IsoExec`` routes scoring through
  ``IsoExecModel.compute_logprobs``, so the rollout and the trainer run the SAME
  ``logprobs.log_softmax`` binding on the same logits. The bound is ``0.0``: not a tolerance, an
  identity. Anything else is a bug in something.
* ``framework_owned`` -- the bridge path, where the framework's own scoring chain produces the
  rollout number. The bound is the certified floor the bridge was measured at (mean 1e-5, max 1e-4).

What is gone, deliberately: the tolerance was read from a contract artifact through a five-attempt
retry loop with a sleep, with a module-level fallback, and a demoted refusal silently fell back to
1e-5. A bound that is fetched over a filesystem race and degrades to a looser bound on failure is
not a bound. It is a literal in the run config now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from . import Refusal

SCORING = ("contracted", "framework_owned")

#: The bridge path's certified floor, measured on the shipped recipes.
FRAMEWORK_OWNED_MEAN = 1e-5
FRAMEWORK_OWNED_MAX = 1e-4


@dataclass(frozen=True)
class LogprobPolicy:
    """Who owns the scoring chain, the bound that follows from it, and how often it is audited.

    ``interval`` is a declaration, not a switch: an interval of 4 means the gate runs on steps 1, 5,
    9 ... and a step it does not run on records no receipt. Step 1 always runs.
    """

    scoring: Literal["contracted", "framework_owned"]
    bound: float
    interval: int = 1

    def __post_init__(self) -> None:
        if self.scoring not in SCORING:
            raise ValueError(f"scoring={self.scoring!r}; the vocabulary is {list(SCORING)}")
        if self.bound < 0.0:
            raise ValueError(f"bound={self.bound!r}: a bound is a non-negative distance")
        if self.interval < 1:
            raise ValueError(f"interval={self.interval!r}: the gate runs at least every {self.interval}th step")

    @property
    def max_bound(self) -> float:
        """The bound applied to the MAX statistic. Contracted scoring holds both at ``0.0``."""
        return self.bound if self.scoring == "contracted" else FRAMEWORK_OWNED_MAX

    def audits(self, step: int) -> bool:
        """Does the declared interval audit this (1-based) step? Step 1 always does."""
        return step <= 1 or (step - 1) % self.interval == 0


#: The policy an IsoExec deployment runs: we own both chains, so the answer is identity or nothing.
CONTRACTED = LogprobPolicy("contracted", 0.0)
#: The policy the bridge path runs.
FRAMEWORK_OWNED = LogprobPolicy("framework_owned", FRAMEWORK_OWNED_MEAN)

MEAN_KEY = "isoexec_gate/mean_abs_diff"
MAX_KEY = "isoexec_gate/max_abs_diff"


def check_logprob_gate(
    metrics: Mapping[str, float],
    policy: LogprobPolicy,
    *,
    audited: bool,
    step: int = 1,
    mean_key: str = MEAN_KEY,
    max_key: str = MAX_KEY,
) -> dict:
    """Judge one step's rollout-vs-trainer logprob distance; returns the receipt, raises on failure.

    ``audited`` says whether this step actually produced the comparison. An unaudited step returns a
    receipt saying so; a step that claims to be audited and carries no statistic is a refusal, not a
    pass, because a missing measurement and a passing measurement must never look alike.
    """
    receipt: dict = {"scoring": policy.scoring, "bound": policy.bound, "step": step, "audited": bool(audited)}
    if not audited:
        return receipt
    if mean_key not in metrics or max_key not in metrics:
        raise Refusal(
            f"scoring: step {step} is audited but carries no logprob statistic "
            f"({mean_key!r}/{max_key!r} absent from {sorted(metrics)}); a gate with no measurement is not a pass"
        )
    mean = float(metrics[mean_key])
    maximum = float(metrics[max_key])
    receipt["mean"] = mean
    receipt["max"] = maximum
    if mean > policy.bound:
        raise Refusal(
            f"scoring: mean |rollout - trainer_score| = {mean:.3e} > bound {policy.bound:.3e} "
            f"(scoring={policy.scoring!r}, step {step})"
        )
    if maximum > policy.max_bound:
        raise Refusal(
            f"scoring: max |rollout - trainer_score| = {maximum:.3e} > bound {policy.max_bound:.3e} "
            f"(scoring={policy.scoring!r}, step {step})"
        )
    return receipt


__all__ = [
    "CONTRACTED",
    "FRAMEWORK_OWNED",
    "FRAMEWORK_OWNED_MAX",
    "FRAMEWORK_OWNED_MEAN",
    "LogprobPolicy",
    "MAX_KEY",
    "MEAN_KEY",
    "SCORING",
    "check_logprob_gate",
]
