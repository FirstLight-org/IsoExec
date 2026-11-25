"""Utilities for splitting graph body and head for native vLLM calls."""

import inspect
import operator
from typing import cast

from torch import fx

from isoexec.compiler.realization.graph import drop_auxiliary_outputs
from isoexec.nn import LMHead, MoE


def _elide_unused_moe_aux(body: fx.GraphModule) -> None:
    """Remove auxiliary production only when all surviving users select the value."""
    for node in list(body.graph.nodes):
        if node.op != "call_module" or not isinstance(body.get_submodule(str(node.target)), MoE):
            continue
        module = body.get_submodule(str(node.target))
        arguments = inspect.signature(module.forward).bind_partial(*node.args, **node.kwargs).arguments
        if arguments.get("return_aux") is not True or arguments.get("aux") is not None:
            continue
        users = list(node.users)
        if not users or any(user.op != "call_function" or user.target is not operator.getitem
                            or user.args != (node, 0) or user.kwargs for user in users):
            continue
        node.kwargs = {**node.kwargs, "return_aux": False}
        node.meta = dict(users[0].meta)
        for user in users:
            user.replace_all_uses_with(node)
            body.graph.erase_node(user)

def _split_body_and_head(module: fx.GraphModule) -> tuple[fx.GraphModule, LMHead | None]:
    """Split a captured graph into body and terminal LMHead in place.

    ``module`` is modified and returned as the body; pass a clone to keep the
    source unchanged. The head is removed from its children and shares local
    tied parameters with the body. The body returns the head's hidden input;
    auxiliary outputs are omitted because inference does not consume them.
    A PP stage without an LMHead returns its hidden tensor alone, also without
    auxiliary outputs, so native PP crosses one tensor per cut; its head is None.
    """
    body = module
    output = next(node for node in body.graph.nodes if node.op == "output")
    # Capture guarantees one terminal LMHead, regardless of the output container.
    head_node = next((node for node in body.graph.nodes if node.op == "call_module"
                      and isinstance(body.get_submodule(str(node.target)), LMHead)), None)
    if head_node is None:
        drop_auxiliary_outputs(body)
        _elide_unused_moe_aux(body)
        # Keep every registered parameter: weight sync and loading still address them by name.
        body.graph.lint()
        body.recompile()
        return body, None
    head = cast(LMHead, body.get_submodule(str(head_node.target)))
    arguments = inspect.signature(head.forward).bind(*head_node.args, **head_node.kwargs)
    hidden = cast(fx.Node, arguments.arguments["hidden"])
    output.args = (hidden,)
    output.meta = dict(hidden.meta)
    body.graph.eliminate_dead_code()
    _elide_unused_moe_aux(body)
    body.delete_all_unused_submodules()
    body.graph.lint()
    body.recompile()
    return body, head
