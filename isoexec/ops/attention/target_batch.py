"""The deployment's optional target batch, and the record two parties compare before they start.

The target batch is the decode batch (sequences per engine replica) the numerical schedule is tuned for. It is
optional. Absent (the default), the schedule policy takes ``p = workers(sms) = 1``: one segment per shard of the
declared longest context, the schedule for throughput. Declared, the policy cuts a row into as many segments as
the device has workers free at that batch. Every batch size, including 1, runs under either and gives the same
bits as any other batch size under the same declaration; the declaration only says which batch is fast. Two
different declarations are two different numerical programs.

One place states it: ``ISOEXEC_TARGET_BATCH`` in the environment of the launch (declared in ``core/flags.py``,
FUNCTION). ``runtimes/environment.resolved_environment`` carries it with the other flags: SkyRL puts that
environment into the job's Ray runtime environment (trainer and engine actors alike), and verl's
``isoexec.enable`` expansion forwards the driver's value to every Ray worker the same way
(``integrations/verl/config.py``). The operators read it through :func:`declared` when their schedule is bound
(import), never on a call.

:func:`schedule_record` is what a process bound; :func:`require_same_schedule` compares two of them and refuses by
name. In the verl integration every training worker records what it bound when its model is built and every
engine worker reports it in its weight acknowledgement; each weight sync compares them all, the first one before
a token is generated (``integrations/verl/schedule.py``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from isoexec.checks import Refusal

#: The environment key of the declaration. Empty or unset: no target batch.
FLAG = "ISOEXEC_TARGET_BATCH"


def parse(text: str | None) -> int | None:
    """``None`` for an absent declaration (unset or empty), else the positive batch it states."""
    if text is None or not text.strip():
        return None
    try:
        batch = int(text.strip())
    except ValueError:
        raise ValueError(f"{FLAG} must be a positive integer or empty (no target batch), got {text!r}") from None
    if batch < 1:
        raise ValueError(f"{FLAG} must be a positive integer or empty (no target batch), got {text!r}")
    return batch


def declared() -> int | None:
    """The target batch this process was launched with; ``None`` when the deployment declares none."""
    return parse(os.environ.get(FLAG))


def schedule_record() -> dict[str, Any]:
    """The attention schedules this process bound, as plain data: the policy version, the target batch and, per
    SM100 operator, its declarations and the segment span every call hands the kernel."""
    from isoexec.ops import schedule_policy
    from isoexec.ops.attention.gqa_sm100 import ops as gqa
    from isoexec.ops.attention.gqa_sm100 import spec as gqa_spec
    from isoexec.ops.attention.mla_sm100 import fwd as mla
    from isoexec.ops.attention.mla_sm100 import spec as mla_spec

    return {
        "policy": schedule_policy.VERSION,
        "gqa_sm100": {
            "target_batch": gqa_spec.TARGET_BATCH,
            "longest_context": gqa_spec.LONGEST_CONTEXT,
            "model_kv_heads": gqa_spec.MODEL_KV_HEADS,
            "seg_tokens": {str(d): gqa.segment_span(d) for d in gqa_spec.HEAD_DIMS},
        },
        "mla_sm100": {
            "target_batch": mla_spec.TARGET_BATCH,
            "longest_context": mla_spec.LONGEST_CONTEXT,
            "model_heads": mla_spec.MODEL_HEADS,
            "seg_tokens": mla.segment_span(),
        },
    }


def require_same_schedule(mine: Mapping[str, Any], theirs: Mapping[str, Any] | None, *, who: str,
                          here: str = "here") -> None:
    """Refuse unless ``theirs`` (the record of ``who``) is the record ``mine`` (of ``here``); the message names the
    first field that differs. A party that reports no record is refused: it cannot be shown to run the same
    schedule."""
    if theirs is None:
        raise Refusal(f"attention_schedule.missing: {who} reported no attention schedule record")

    def differ(a: Any, b: Any, path: str) -> str | None:
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for key in sorted(set(a) | set(b)):
                found = differ(a.get(key, "<absent>"), b.get(key, "<absent>"), f"{path}.{key}" if path else str(key))
                if found:
                    return found
            return None
        return None if type(a) is type(b) and a == b else f"{path}: {here} {a!r}, {who} {b!r}"

    found = differ(mine, theirs, "")
    if found:
        raise Refusal(
            f"attention_schedule.{found}. The two parties were launched with different schedule declarations "
            f"({FLAG} or the operator's spec.py); they would compute different bits, so they do not start."
        )


__all__ = ["FLAG", "declared", "parse", "require_same_schedule", "schedule_record"]
