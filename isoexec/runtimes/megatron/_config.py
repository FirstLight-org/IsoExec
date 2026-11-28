"""Native scheduler and DDP configuration from captured components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isoexec.nn import MLP, Attention, LMHead, MoE
from isoexec.nn.component import Component

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

    from isoexec.compiler.capture.graph import ExecutionGraph


def make_transformer_config(execution_graph: ExecutionGraph, **options: Any) -> TransformerConfig:
    """Return a native configuration using the captured model's component dimensions."""
    from megatron.core.transformer.transformer_config import TransformerConfig

    graph_module = execution_graph.program
    head = next(module for module in graph_module.modules() if isinstance(module, LMHead))
    attention = next((module for module in graph_module.modules() if isinstance(module, Attention)), None)
    mlp = next((module for module in graph_module.modules() if isinstance(module, MLP)), None)
    moe = next((module for module in graph_module.modules() if isinstance(module, MoE)), None)
    # A placement-bound decoder row owns its attention and experts; its facts state their dimensions.
    bound = next((module.facts() for module in graph_module.modules()
                  if isinstance(module, Component) and module.PLACEMENT_BOUND), {})
    # MCore requires an expert count for expert parallelism. With experts it refuses
    # the default allgather dispatcher under variable sequence lengths and linear
    # biases under ETP>1, so select alltoall without biases.
    count, width = ((moe.num_experts, moe.intermediate) if moe is not None
                    else (bound.get("experts"), bound.get("intermediate")))
    experts = {} if count is None else {
        "num_moe_experts": count, "moe_ffn_hidden_size": width,
        "moe_token_dispatcher_type": "alltoall", "add_bias_linear": False,
    }
    # MCore requires attention dimensions even when the graph has no Attention.
    heads = attention.heads if attention is not None else bound.get("heads", 1)
    return TransformerConfig(
        num_layers=len(execution_graph.layers), hidden_size=head.hidden,
        num_attention_heads=heads,
        num_query_groups=attention.kv_heads if attention is not None else None,
        kv_channels=attention.head_dim if attention is not None else bound.get("head_dim"),
        ffn_hidden_size=mlp.intermediate if mlp is not None else None,
        **experts,
        **options,
    )
