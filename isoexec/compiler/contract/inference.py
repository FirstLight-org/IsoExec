"""Map stateful op IDs to op-owned facts inference and validation."""

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

from isoexec.ops.attention.facts import (
    infer_gqa_op_facts,
    infer_mla_op_facts,
    validate_gqa_op_facts,
    validate_mla_op_facts,
)
from isoexec.ops.gdn.facts import infer_gdn_op_facts, validate_gdn_op_facts


@dataclass(frozen=True)
class OpFactsRule:
    infer: Callable[[object], object]
    validate: Callable[[object], None]


OP_FACTS_BY_OP = MappingProxyType(
    {
        "attention.core.gqa": OpFactsRule(
            infer=infer_gqa_op_facts,
            validate=validate_gqa_op_facts,
        ),
        "attention.core.mla": OpFactsRule(
            infer=infer_mla_op_facts,
            validate=validate_mla_op_facts,
        ),
        "gdn.core": OpFactsRule(
            infer=infer_gdn_op_facts,
            validate=validate_gdn_op_facts,
        ),
    }
)


__all__ = ["OP_FACTS_BY_OP", "OpFactsRule"]
