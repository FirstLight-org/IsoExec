"""Shared model configuration, tensor, placement, and sequence declarations. Importing the schema
requires only the standard library; torch conversion and pytree registration are explicit calls.
"""

from isoexec.schema.model import (
    GDN,
    MLA,
    Attn,
    Mlp,
    ModelConfig,
    Mtp,
    Row,
    layer_ns,
    n_decoder,
)
from isoexec.schema.params import (
    SHARD_KINDS,
    TORCH_DTYPES,
    HeldDtypeError,
    LogicalTensor,
    ShardAxis,
    TensorPlanMetadata,
    StateSpec,
    TieDecl,
    col,
    expert,
    replicated,
    row,
    torch_dtype,
)
from isoexec.schema.run import SIDES, Axis, Placement, ShardError
from isoexec.schema.sequence import SequenceContext, make_sequence_context, num_tokens, register_pytrees
from isoexec.schema.state import attention_state_specs, gdn_state_specs
from isoexec.schema.vocabulary import (
    EXPERT,
    REPL,
    ROW1,
    SUM_OVER_TP,
    attention_vocabulary,
    dense_vocabulary,
    gdn_vocabulary,
    moe_vocabulary,
    stack_vocabulary,
    ties_for,
)

__all__ = [
    "MLA",
    # model
    "Attn",
    "GDN",
    "Mlp",
    "Mtp",
    "ModelConfig",
    "Row",
    "n_decoder",
    "layer_ns",
    # params
    "SHARD_KINDS",
    "ShardAxis",
    "TensorPlanMetadata",
    "col",
    "row",
    "replicated",
    "expert",
    "StateSpec",
    "TieDecl",
    "LogicalTensor",
    "TORCH_DTYPES",
    "torch_dtype",
    "HeldDtypeError",
    # sequence
    "SequenceContext",
    "make_sequence_context",
    "num_tokens",
    "register_pytrees",
    # vocabulary: the config -> logical tensor fold
    "REPL",
    "SUM_OVER_TP",
    "ROW1",
    "EXPERT",
    "attention_vocabulary",
    "gdn_vocabulary",
    "moe_vocabulary",
    "dense_vocabulary",
    "stack_vocabulary",
    "ties_for",
    # state: the config -> declared cache fold
    "attention_state_specs",
    "gdn_state_specs",
    # run / placement
    "Axis",
    "Placement",
    "SIDES",
    "ShardError",
]
