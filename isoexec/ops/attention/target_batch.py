"""The deployment's optional target batch, and the record two parties compare before they start.

The target batch is the decode batch (sequences per engine replica) the numerical schedule is tuned for. It is
optional. Absent (the default), the schedule policy takes ``p = workers(sms) = 1``: one segment per shard of the
declared longest context, the schedule for throughput. Declared, the policy cuts a row into as many segments as
the device has workers free at that batch. Every batch size, including 1, runs under either and gives the same
bits as any other batch size under the same declaration; the declaration only says which batch is fast. Two
different declarations are two different numerical programs.

One place states it: ``ISOEXEC_TARGET_BATCH`` in the environment of the launch (declared in ``core/flags.py``,
FUNCTION, forwarded to the trainer actors and the engine actors by ``runtimes/environment.resolved_environment``).
The operators read it through :func:`declared` when their schedule is bound (import), never on a call.

:func:`schedule_record` is what a process bound; :func:`require_same_schedule` compares two of them and refuses by
name. The verl integration puts the record in every engine worker's weight acknowledgement and compares it at the
first weight sync, before a token is generated (``integrations/verl``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from isoexec.checks import Refusal

#: The environment key of the declaration. Empty or unset: no target batch.
FLAG = "ISOEXEC_TARGET_BATCH"

#: SMs of the device generation the SM100 attention operators compile for (B200). Hardware: the workers a declared
#: target batch is divided into (``gqa_sm100.ops.DEVICE_SMS`` and ``mla_sm100.fwd.DEVICE_SMS`` are this value).
DEVICE_SMS = 148


def parse(text: str | None) -> int | None:
    """``None`` for an absent declaration (unset or empty), else the positive batch it states in ASCII decimal
    digits (no sign, no underscore, no other digit set)."""
    if text is None or not text.strip():
        return None
    digits = text.strip()
    if not (digits.isascii() and digits.isdigit()):
        raise ValueError(f"{FLAG} must be a positive integer or empty (no target batch), got {text!r}")
    batch = int(digits)
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


def require_same_schedule(mine: Mapping[str, Any], theirs: Mapping[str, Any] | None, *, who: str) -> None:
    """Refuse unless ``theirs`` (the record of ``who``) is the record ``mine``; the message names the first field
    that differs. A party that reports no record is refused: it cannot be shown to run the same schedule."""
    if theirs is None:
        raise Refusal(f"attention_schedule.missing: {who} reported no attention schedule record; the trainer and "
                      f"the engine must run the same IsoExec version with the same {FLAG}")

    def differ(a: Any, b: Any, path: str) -> str | None:
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for key in sorted(set(a) | set(b)):
                found = differ(a.get(key, "<absent>"), b.get(key, "<absent>"), f"{path}.{key}" if path else str(key))
                if found:
                    return found
            return None
        return None if type(a) is type(b) and a == b else f"{path}: here {a!r}, {who} {b!r}"

    found = differ(mine, theirs, "")
    if found:
        raise Refusal(
            f"attention_schedule.{found}. The two parties were launched with different schedule declarations "
            f"({FLAG} or the operator's spec.py); they would compute different bits, so they do not start. "
            f"Run the trainer and the engine with the same IsoExec version and the same {FLAG}."
        )


__all__ = ["DEVICE_SMS", "FLAG", "declared", "parse", "require_same_schedule", "schedule_record"]
