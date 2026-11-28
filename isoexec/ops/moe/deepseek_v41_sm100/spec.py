"""Declarations of the SM100 DeepSeek-V4.1 MoE BF16-actor recipe. Imports nothing that builds.

The impl id is the recipe: decoded BF16 masters through native BF16 grouped GEMMs for BOTH rollout and
training (exact train/rollout identity), router_sm100 ``sqrtsoftplus_bias_select`` routing (FP32
sqrt(softplus) scores, selection-only bias, exact normalization, routed scale; the shared expert as
the shared-as-routed slot), FC1 rounded to bf16 (moe_sm100 one K chain), DeepSeek's FP32 activation
rule with the route weight BEFORE the down projection, FC2 one K segment rounded to bf16, fp32
combine in ascending expert id (shared last), one bf16 rounding. The released FP8/FP4 checkpoint is
decoded exactly to the BF16 masters at load.
"""
from __future__ import annotations

from isoexec.core.registry import ImplSpec

ARCH = "sm100"
OP = "moe.deepseek_v41"
ROUTE_OP = "moe.deepseek_v41.route"
#: v2 of the impl: the same recipe (``RECIPE`` names the program family) on the schedule policy's schedule
#: (``isoexec/ops/schedule_policy.py``): router logits ONE chain over the hidden axis (v1: 4 segments and a tree),
#: renormaliser the rule-2 chain in ascending expert id from +0 (v1: a half fold in selection-rank order).
#: FC1 / FC2 were one chain already, and the combine is v1's: the rule-2 chain in ascending expert id that
#: starts from the first contribution, taken as is.
IMPL = "sm100_deepseek_v41_bf16_actor_v2"
RECIPE = "deepseek_v41_bf16_actor_v1"
ROUTING = "sqrtsoftplus_bias_select"
#: Leaves of FC2's tree: d_max shards of the policy's one segment. The intermediate axis is not sharded in this
#: recipe (TP shards FC2's output columns, EP shards experts), so d_max = 1.
FC2_LEAVES = 1

IMPLS: tuple[ImplSpec, ...] = (
    ImplSpec(op=OP, impl_id=IMPL, version=2, supported_archs=frozenset({ARCH})),
)

__all__ = ["ARCH", "FC2_LEAVES", "IMPL", "IMPLS", "OP", "RECIPE", "ROUTE_OP", "ROUTING"]
