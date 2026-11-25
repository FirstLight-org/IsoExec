"""The numerical schedule of the norms_sm100 sums: what the policy gives, and how it reaches the kernels.

Every reduction in this package is a plain fp32 sum (one ``add.rn`` per term from ``+0``, merge ``add.rn``):

    hidden axis   the sum of squares of a row (RMSNorm, fused add + RMSNorm, gated out-norm) and the backward's
                  row dot ``sum_h xhat * gy``
    row axis      the weight gradient ``dW[h] = sum_rows dy * xhat``

Rule 3 of :mod:`isoexec.ops.schedule_policy` for a scalar step (the reducer publishes a shortest segment and no
occupancy): segments of ``S = L_r`` consecutive positions, the last one short, combined by the adjacent tree.
Neither axis is split across devices (``isoexec/nn/norm.py`` runs the whole hidden axis on one device and the
row axis is one device's rows), so ``d_max = 1`` and no deployment declaration is read.
"""

from __future__ import annotations

from isoexec.ops import schedule_policy as policy

#: L_r of the plain fp32 sum: the shortest segment worth a partial of its own. Measured on B200; source file
#: ``policy_validation/rmsnorm_sm100/kernel_next/round_d_*.log`` (ONEPAGE.md, inputs table: "plain sum 8").
SUM_SHORTEST_SEGMENT = 8


def hidden_sum(width: int) -> policy.Segments:
    """Segments of a sum over the ``width`` columns of one row."""
    return policy.reduce(width, b=1, L=SUM_SHORTEST_SEGMENT)


def row_sum(rows: int) -> policy.Segments:
    """Segments of the weight gradient's sum over ``rows`` rows."""
    return policy.reduce(rows, b=1, L=SUM_SHORTEST_SEGMENT)


def cuda_defines() -> tuple[str, ...]:
    """The segment lengths as compiler definitions: the only place the kernels get them from.

    A scalar step's ``S`` does not depend on the axis length, so one value per axis serves every width and
    row count; ``tests/test_schedule_policy_cpu.py`` checks that on the deployed shapes.
    """
    return (
        f"-DNORMS_SM100_HIDDEN_SEGMENT={hidden_sum(SUM_SHORTEST_SEGMENT).S}",
        f"-DNORMS_SM100_ROW_SEGMENT={row_sum(SUM_SHORTEST_SEGMENT).S}",
    )


__all__ = ["SUM_SHORTEST_SEGMENT", "cuda_defines", "hidden_sum", "row_sum"]
