"""Architecture-neutral logprobs API; importing the facade does not load kernel providers."""

from isoexec.ops._dispatch import facade as _facade

_EXPORTS = {
    "LANES": ("spec", "LANES"),
    "LEAVES": ("spec", "LEAVES"),
    "RECIPE_VERSION": ("ops", "RECIPE_VERSION"),
    "STRIP": ("spec", "STRIP"),
    "TEMPERATURE_FUSION_POLICY": ("integration", "TEMPERATURE_FUSION_POLICY"),
    "full_logprobs_for_check": ("integration", "full_logprobs_for_check"),
    "full_logprobs_for_check_sharded": ("integration", "full_logprobs_for_check_sharded"),
    "full_logprobs": ("integration", "full_logprobs"),
    "integration": ("integration", None),
    "leaf_states": ("ops", "leaf_states"),
    "ops": ("ops", None),
    "sampled_logprobs": ("integration", "sampled_logprobs"),
    "sampler": ("sampler", None),
    "shard_sample": ("shard", "shard_sample"),
    "shard_sample_build": ("shard", "build"),
    "stats": ("integration", "stats"),
    "supports": ("ops", "supports"),
}

__all__ = list(_EXPORTS)

__getattr__, __dir__ = _facade("logprobs", _EXPORTS, globals())
