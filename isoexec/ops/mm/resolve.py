"""Resolve GEMM identity from the installed provider and pin that object for the execution record.
The first call after an incompatible provider replacement refuses, because the recorded and the installed providers make no general twin claim."""

from __future__ import annotations

from typing import Any

from isoexec.ops._resolve import refuse, resolution


def resolve(op: str, *, site: str, **unused: Any):
    """The installed batch-invariant GEMM, under the impl id read off the installed object."""
    if unused:
        raise refuse(op, site=site, asked=unused, on_offer=(),
                     why="which provider serves mm is not something a caller supplies; it IS the op.")
    if op != "mm":
        raise refuse(op, site=site, asked={}, on_offer=("mm",), why="the mm family states one op id.")
    from isoexec.ops._dispatch import current_arch
    if current_arch() == "sm100":
        from isoexec.ops.mm.mm_sm100 import segments
        from isoexec.ops.mm.sm100 import mm_linear
        return resolution(op, "sm100_tcgen05_segtree", mm_linear, provider="ops.mm.mm_sm100", site=site,
                          inputs={"schedule_digest": segments.schedule_digest(),
                                  "schedule_policy_version": segments.schedule_policy.VERSION})
    from isoexec.ops.mm.trainer import mm_trainer, pin_recorded_impl

    impl = pin_recorded_impl()
    if impl == "sm90_wgmma_segtree":
        from isoexec.ops.mm.mm_sm90 import segments

        return resolution(op, impl, mm_trainer, provider="ops.mm.mm_sm90", site=site,
                          inputs={"segment_table_digest": segments.table_digest(),
                                  "segment_policy_version": segments.POLICY_VERSION})
    return resolution(op, impl, mm_trainer, provider="ops.mm", site=site)


__all__ = ["resolve"]
