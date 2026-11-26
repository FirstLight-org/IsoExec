"""The group table and the fold rule of the sparse attention (README.md section 4.4).

``G = LIST_GROUPS[(H_Q, K)]`` is a table over deployment-fixed axes only: never over batch, list
length, grid or SM count.  Execution may evaluate the G groups with G workers plus a combine, or with
one worker folding in place; both evaluate ``reference.tree_shape(G)``.  A worker that continues its
accumulator across a group boundary when G > 1 has produced the flat chain, which is a different
program, and is refused here (dtir.md sections 3 and 4).
"""

from __future__ import annotations

from . import spec

__all__ = ["groups_for", "check_fold", "FOLDS"]

#: The fold forms an execution plan may request.  "tree": the group tree (G workers + combine, or one
#: worker with a pending-node stack); "chain": one accumulator across the whole list.
FOLDS = ("tree", "chain")


def groups_for(h_q: int, topk: int) -> int:
    """The recipe's group count for (h_q, K); KeyError for a geometry the table does not declare."""
    return spec.LIST_GROUPS[(int(h_q), int(topk))]


def check_fold(groups: int, fold: str) -> None:
    """Refuse the flat chain over G > 1 groups.  At G = 1 the chain IS the tree."""
    if fold not in FOLDS:
        raise ValueError(f"unknown fold {fold!r}; expected one of {FOLDS}")
    if fold == "chain" and groups > 1:
        raise ValueError(
            f"refusing a flat chain over G = {groups} groups: continuing one accumulator across a "
            "group boundary is a different program from the declared group tree (README.md 4.4)")
