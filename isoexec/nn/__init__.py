"""Composable model components with meta construction and per-build operation bindings. Placement
resolves kernels; runtime adapters complete resource-dependent slots before execution
validation.
"""

from isoexec.nn._seams import NO_MOVEMENT, SeamError, Seams, bind_seam
from isoexec.nn._slots import BoundOp, OpSlots, SlotError, Unbound, UnfilledSlotError
from isoexec.nn.attention import Attention
from isoexec.nn.block import DecoderBlock
from isoexec.nn.component import Component, PlacementError
from isoexec.nn.embed import Embedding
from isoexec.nn.gdn import GDN
from isoexec.nn.gemm_census import gemm_census
from isoexec.nn.head import LMHead
from isoexec.nn.inspect import Inventory, InventoryError, executed_modules, inventory, place, resolved_ops, verify_placed
from isoexec.nn.mla import LatentAttention
from isoexec.nn.mlp import MLP
from isoexec.nn.moe import MoE
from isoexec.nn.norm import RMSNorm
from isoexec.nn.unexecuted import Unexecuted, is_unexecuted_subtree

#: The components a model author composes. Everything else exported here is the machinery they
#: compose against -- slots, the placement protocol, and the walks a host runs.
COMPONENTS = ("Attention", "DecoderBlock", "Embedding", "GDN", "LMHead", "LatentAttention", "MLP", "MoE", "RMSNorm")

# Held tensors and subtrees remain in weight inventory while execution walks skip them.

__all__ = [
    "COMPONENTS",
    "GDN",
    "MLP",
    "NO_MOVEMENT",
    "Attention",
    "BoundOp",
    "Component",
    "DecoderBlock",
    "Embedding",
    "Inventory",
    "InventoryError",
    "LMHead",
    "LatentAttention",
    "MoE",
    "OpSlots",
    "PlacementError",
    "RMSNorm",
    "SeamError",
    "Seams",
    "SlotError",
    "Unbound",
    "Unexecuted",
    "UnfilledSlotError",
    "bind_seam",
    "executed_modules",
    "gemm_census",
    "inventory",
    "is_unexecuted_subtree",
    "place",
    "resolved_ops",
    "verify_placed",
]
