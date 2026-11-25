"""The numerical schedule of the logprobs_sm100 log-sum-exp: what the policy gives, and how it reaches the kernel.

One reduction: ``lse[r] = log sum_v exp(x[r, v])`` over the vocabulary, with the (max, sum) reducer -- partial
``(m, s)``, ``m`` the maximum, ``s = sum exp(v - m)``; merge: rescale both sides to the larger maximum and add.
The row entropy rides on the same partial with a third component and the same segments and trees.

Rule 3 of :mod:`isoexec.ops.schedule_policy` for a scalar step (the reducer publishes a shortest segment and no
occupancy): the vocabulary axis is divisible into ``d_max`` shards (a TP rank holds whole shards); inside a shard,
segments of ``S = L_r`` consecutive columns, the last one short, under the adjacent tree; then the adjacent tree
over the ``d_max`` shards. Nothing is keyed on the vocabulary size or the logits dtype.
"""

from __future__ import annotations

from isoexec.ops import schedule_policy as policy
from isoexec.ops.logprobs.logprobs_sm100.spec import LEAVES

#: L_r of the (max, sum) reducer: the shortest segment worth a partial of its own. Measured on B200; source file
#: ``policy_validation/logprobs_sm100/ROUND_B.md`` (bench6; ONEPAGE.md, inputs table: "(max, sum) 32").
LSE_SHORTEST_SEGMENT = 32

#: d_max of the vocabulary axis: the largest tensor-parallel degree a vocabulary may be split over. It is the
#: tree's existing declaration ``spec.LEAVES`` (``ops.supports``: ``LEAVES % world == 0``), read here by rule 3.
VOCAB_SHARDS = LEAVES


def shard_columns(vocab: int) -> int:
    """n of rule 3: the columns of one shard."""
    if vocab % VOCAB_SHARDS:
        raise ValueError(f"logprobs_sm100: vocabulary {vocab} is not {VOCAB_SHARDS} equal shards")
    return vocab // VOCAB_SHARDS


def vocab_sum(vocab: int) -> policy.Segments:
    """Segments of the log-sum-exp inside one of the ``VOCAB_SHARDS`` shards of a ``vocab``-column row."""
    return policy.reduce(shard_columns(vocab), b=1, L=LSE_SHORTEST_SEGMENT)


def cuda_defines() -> tuple[str, ...]:
    """The segment length and d_max as compiler definitions: the only place the kernel gets them from.

    A scalar step's ``S`` does not depend on the axis length, so one value serves every vocabulary;
    ``tests/test_schedule_policy_cpu.py`` checks that on the deployed vocabularies.
    """
    return (
        f"-DLOGPROBS_SM100_SEGMENT={policy.reduce(LSE_SHORTEST_SEGMENT, b=1, L=LSE_SHORTEST_SEGMENT).S}",
        f"-DLOGPROBS_SM100_SHARDS={VOCAB_SHARDS}",
    )


__all__ = ["LSE_SHORTEST_SEGMENT", "VOCAB_SHARDS", "cuda_defines", "shard_columns", "vocab_sum"]
