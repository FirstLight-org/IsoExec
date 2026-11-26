"""Graph-safe initialization of vLLM-owned KDA slots; no numerical operations."""

from functools import lru_cache

from isoexec.ops.build_cache import build_sm100_extension


@lru_cache(None)
def extension():
    return build_sm100_extension('kda_slot_reset', __file__, ('reset.cu',))


def reset_fresh_slots(conv, records, slots, positions, *, null_slot=0):
    """Zero fresh live slots in place, including recycled slots and recompute at0.

    Each decode row owns one distinct live slot. Null/negative slots are padding;
    invalid duplicate/out-of-range live ownership poisons the CUDA context and
    refuses the result (other blocks may already have reset their own slots).
    Shapes/strides are static graph metadata; positions and slot IDs remain GPU
    inputs and can change on every replay. Neither state tensor is reallocated.
    """
    extension().reset(conv, records, slots, positions, int(null_slot))


def reset_fresh_slots_layers(convs, records, slots, positions, *, null_slot=0):
    """``reset_fresh_slots`` for every KDA layer of a step in ONE launch: same stores per layer. All layers share
    the step's slots/positions and one state geometry (sizes and strides); at most 64 layers."""
    extension().reset_layers(list(convs), list(records), slots, positions, int(null_slot))


__all__ = ['reset_fresh_slots', 'reset_fresh_slots_layers']
