"""Map stateful op IDs to op-owned state inference."""

from isoexec.ops.attention.state import infer_attention_state_specs, infer_mla_state_specs
from isoexec.ops.gdn.conv_state import infer_conv_state_specs
from isoexec.ops.gdn.state import infer_gdn_state_specs

# Op registration will supply this mapping when its inference interface is ready.
STATE_SPECS_BY_OP = {
    "attention.core.gqa": infer_attention_state_specs,
    "attention.core.mla": infer_mla_state_specs,
    "conv.short": infer_conv_state_specs,
    "gdn.core": infer_gdn_state_specs,
}
