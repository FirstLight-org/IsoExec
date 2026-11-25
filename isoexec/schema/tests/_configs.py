"""The two tiny row tables the schema tests fold: three GDN rows, one attention row and one MTP
attention row, over an 8-expert top-2 mixture (``tiny_moe``) or a dense MLP (``tiny_dense``)."""

from __future__ import annotations

from isoexec.schema import GDN, Attn, Mlp, ModelConfig, Mtp

_GDN = GDN(num_key_heads=4, num_value_heads=8, key_head_dim=32, value_head_dim=32, conv_kernel=4, chunk_size=64,
           out_norm="gated_rms_zero_centered", output_activation="silu")
_ATTN = Attn(num_q_heads=4, num_kv_heads=2, head_dim=64, rope_theta=10000000.0, partial_rotary=0.25,
             rope_style="neox", mrope_section=(3, 3, 2), mrope_interleaved=True, qk_norm="rms_zero_centered",
             qkv_bias=False, out_gate="sigmoid")
_MLPS = {
    "tiny_moe": Mlp(intermediate_size=128, num_experts=8, top_k=2, shared_intermediate=128,
                    routing="softmax_renorm", routed_scale=1.0, shared_expert_gate=True, activation="silu_mul"),
    "tiny_dense": Mlp(intermediate_size=512, activation="silu_mul"),
}


def tiny_configs() -> dict[str, ModelConfig]:
    return {
        name: ModelConfig(
            model_id=f"{name}/4x256", hidden=256, vocab=2048, padded_vocab=2048, norm="rms_zero_centered",
            rms_eps=1e-6, tied_head=False, mtp=Mtp(depth=1, proj="fused_eh", share_weights=False),
            layers=((_GDN, mlp),) * 3 + ((_ATTN, mlp),) * 2,
        )
        for name, mlp in _MLPS.items()
    }
