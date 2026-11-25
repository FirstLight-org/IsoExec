"""Finalize GDN slot-map capacity at vLLM's existing cache initialization boundary."""

import logging
from contextlib import nullcontext
from functools import wraps

logger = logging.getLogger(__name__)


def wrap_cache_initialization(initialize):
    @wraps(initialize)
    def wrapped(
        runner,
        kv_cache_config,
        is_profiling=False,
        kv_cache_allocation_context=None,
    ):
        if not is_profiling:
            from ...ops.gdn import CPR_LAYERS

            pools = [pool for pool in CPR_LAYERS if not pool._native]
            allocation = nullcontext()
            if any(pool.slot2row.is_cuda and kv_cache_config.num_blocks >= pool.slot2row.numel() for pool in pools):
                from vllm.device_allocator.cumem import CuMemAllocator

                if CuMemAllocator.instance is not None:
                    # initialize_kv_cache runs inside the native "kv_cache"
                    # pool, whose bytes are discarded on level-1 sleep. These
                    # registered lifecycle maps must retain the "weights" tag.
                    allocation = CuMemAllocator.instance.use_memory_pool(tag="weights")
            copied = 0
            maps = []
            with allocation:
                for pool in pools:
                    copied += pool.finalize_slot_map(kv_cache_config.num_blocks)
                    maps.append(pool.slot2row.numel())
            runner._isoexec_slot_map_receipt = {
                "num_blocks": kv_cache_config.num_blocks,
                "map_entries": maps,
                "initialization_copied_bytes": copied,
            }
            logger.info("IsoExec GDN slot-map capacity: %s", runner._isoexec_slot_map_receipt)
        return initialize(
            runner,
            kv_cache_config,
            is_profiling=is_profiling,
            kv_cache_allocation_context=kv_cache_allocation_context,
        )

    return wrapped
