"""Primitives for cloning stage graph modules with their registered state."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from torch import fx


def clone_graph_module(source: fx.GraphModule) -> fx.GraphModule:
    """Clone an FX graph module with its registered state.

    The clone preserves shared Parameter and buffer object, without sharing
    those objects with ``source``. But, FX node metadata are copied shallowly;
    their annotated values must remain read-only.

    Args:
        source: FX graph module to clone.

    Returns:
        The cloned graph module.
    """
    # fx.GraphModule deepcopy rebuilds the root module without pre-hooks.
    # Use share memo to avoid repeatedly copying the same objects.
    # TODO(jiangjiawei1103): Think about if capture really needs forward pre-hooks.
    memo: dict[int, Any] = {}
    cloned = deepcopy(source, memo)
    cloned._forward_pre_hooks = deepcopy(source._forward_pre_hooks, memo)
    cloned._forward_pre_hooks_with_kwargs = deepcopy(source._forward_pre_hooks_with_kwargs, memo)

    # PyTorch <= 2.13 GraphModule deepcopy re-registers get_attr buffers as persistent; 2.14's
    # fx _copy_attr keeps the source's persistence. Remove this once 2.13 leaves checks/dependencies.py.
    for path, module in cloned.named_modules():
        module._non_persistent_buffers_set = source.get_submodule(path)._non_persistent_buffers_set.copy()
    return cloned


def drop_auxiliary_outputs(module: fx.GraphModule) -> None:
    """Return only the stage's primary output, e.g., logits without MoE router scores.

    ``define_stage`` returns ``(primary, aux)`` when local operations produce
    auxiliary outputs. Operations that only fed those outputs are removed.
    """
    output = next(node for node in module.graph.nodes if node.op == "output")
    value = output.args[0]
    if not (isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict)):
        return
    output.args = (value[0],)
    module.graph.eliminate_dead_code()
    module.graph.lint()
    module.recompile()
