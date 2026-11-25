"""One model's CPR decode group: the pools whose decode steps share one slot-id tensor (gdn_cpr_sm90, gdn_cpr_sm100).

The first pool of the group to decode in a step resolves every member's rows, NULL lanes and LRU stamp in one
``decode_rows`` launch of its family's extension (index arithmetic only); each member then takes its own entry.
"""

from __future__ import annotations


class DecodeGroup:
    """The CPR pools of one model, which every decode step calls with the same slot-id tensor.

    The first pool of the group to decode in a step resolves every member's rows, null lanes and LRU
    stamp in one ``decode_rows`` launch; each member then takes its own entry. A pool not in a group (or
    a member whose entry is already taken for this slot tensor) launches ``decode_rows`` for itself
    alone: either way each decode call stamps its pool exactly once.
    """

    def __init__(self) -> None:
        self.pools: list = []

    def add(self, pool) -> None:
        if pool.decode_group is not None and pool.decode_group is not self:
            raise ValueError("[isoexec-gdn] a CPR pool belongs to one decode group")
        pool.decode_group = self
        if all(p is not pool for p in self.pools):
            self.pools.append(pool)


__all__ = ["DecodeGroup"]
