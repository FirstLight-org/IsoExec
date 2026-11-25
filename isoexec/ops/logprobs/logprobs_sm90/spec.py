"""Import-light scoring declarations for row selection, the bf16 head/fp32 boundary and SM90 log-softmax.
SM90 full-row, sampled and TP-sharded paths share a fixed leaf recipe with no ATen twin claim;
changing its numerical constants requires a version bump."""


from isoexec.core.registry import ImplSpec

#: The row's leaf count, the columns one lane consumes per step, and the lanes per leaf. ``csrc/
#: logprobs.cu`` states the same three numbers as ``G``, ``STRIP`` and ``LANES``; a change on
#: either side without the other is a silently different reduction, so both move with a version bump.
LEAVES = 8
STRIP = 8
LANES = 256

#: Leaf states and the row merge accumulate in fp32 (the merge widens to double for the weights).
ACCUM = "fp32"

#: The recipe's identity. ``ops.py`` imports it rather than restating it: the version is the only
#: declared signal that a kernel's rounding schedule changed, and a second copy of that signal is
#: the one drift a reader cannot see.
RECIPE_VERSION = 1
IMPL = "sm90_online_leaf8_v1"
_SM90 = frozenset({"sm90"})
#: The vocab-parallel sampling head. Its noise reproduces vLLM's ``gumbel_sample`` (Triton 3.8 lowering of
#: v1/worker/gpu/sample/gumbel.py at vLLM 0.33): a vLLM change to that expression is a version bump here.
SHARD_SAMPLE_IMPL = "sm90_shard_gumbel_leaf8_v1"
SHARD_SAMPLE_VERSION = 1

IMPLS: tuple[ImplSpec, ...] = (
    ImplSpec(
        op="logprobs.head",
        impl_id="eager",
        version=1,
        supported_archs=_SM90,
    ),
    ImplSpec(
        op="logprobs.lm_head_slice",
        impl_id="identity",
        version=1,
        supported_archs=_SM90,
        twin_class="logprobs.row_select",
    ),
    ImplSpec(
        op="logprobs.lm_head_slice",
        impl_id="sampled_rows",
        version=1,
        supported_archs=_SM90,
        twin_class="logprobs.row_select",
    ),
    # The recipe on the table seam both hosts bind.
    ImplSpec(op="logprobs.log_softmax", impl_id=IMPL, version=RECIPE_VERSION, supported_archs=_SM90),
    # Engine vocab-parallel sampling head (ops.shard_sample): the leaf8 recipe's leaf states and merge plus
    # vLLM Model Runner V2's fp32 Gumbel-max, per rank shard; bitwise the gathered path's token, logprob, rank.
    ImplSpec(op="logprobs.shard_sample", impl_id=SHARD_SAMPLE_IMPL, version=SHARD_SAMPLE_VERSION,
             supported_archs=_SM90),
)
