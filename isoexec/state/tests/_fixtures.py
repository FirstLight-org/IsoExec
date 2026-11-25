"""Module definitions and metadata for state contract tests."""

import torch
from torch import nn

from isoexec.core.registry_build import build_registry
from isoexec.compiler.contract.contract import ModelContract
from isoexec.schema import GDN as GdnSpec
from isoexec.schema import MLA
from isoexec.schema import Attn as AttnSpec
from isoexec.state.contract import StateDecl, StateRef, StateSpec

REGISTERED_OP_IDS = frozenset(build_registry().ops)


def attention_spec(**kwargs) -> AttnSpec:
    return AttnSpec(**{"num_q_heads": 8, "num_kv_heads": 4, "head_dim": 32, "rope_theta": 10000.0} | kwargs)


def mla_spec(**kwargs) -> MLA:
    return MLA(
        **{
            "num_q_heads": 8,
            "q_lora_rank": 16,
            "kv_lora_rank": 32,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "rope_theta": 10000.0,
        }
        | kwargs
    )


def gdn_spec(**kwargs) -> GdnSpec:
    return GdnSpec(
        **{
            "num_key_heads": 4,
            "num_value_heads": 8,
            "key_head_dim": 16,
            "value_head_dim": 32,
            "conv_kernel": 4,
        }
        | kwargs
    )


class Attention(nn.Module):
    """Minimal module containing stateless work and one stateful op."""

    stateful_op_id = "attention.core.gqa"

    def __init__(self, spec: AttnSpec | None = None):
        super().__init__()
        self.spec = attention_spec() if spec is None else spec
        self.mm = nn.Identity()
        self.mm.op_id = "mm"

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.mm(hidden)
        return self._gqa(hidden)

    @staticmethod
    def _gqa(hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class LatentAttention(nn.Module):
    stateful_op_id = "attention.core.mla"

    def __init__(self, spec: MLA | None = None):
        super().__init__()
        self.spec = mla_spec() if spec is None else spec

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class GDN(nn.Module):
    """A component containing separate convolution and recurrence invocations."""

    def __init__(self, spec: GdnSpec | None = None):
        super().__init__()
        self.spec = gdn_spec() if spec is None else spec
        self.conv = nn.Identity()
        self.conv.stateful_op_id = "conv.short"
        self.conv.spec = self.spec
        self.core = nn.Identity()
        self.core.stateful_op_id = "gdn.core"
        self.core.spec = self.spec

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.core(self.conv(hidden))


class Block(nn.Module):
    def __init__(self, attention: Attention):
        super().__init__()
        self.attention = attention
        self.mlp = nn.Identity()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.attention(hidden))


class Model(nn.Module):
    def __init__(self, *attentions: Attention):
        super().__init__()
        self.layers = nn.ModuleList(Block(attention) for attention in attentions)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def single_spec_contract(spec: StateSpec) -> ModelContract:
    return ModelContract(
        (StateDecl("state", (spec,)),),
        (StateRef("block", "attention.core.gqa", "state"),),
    )
