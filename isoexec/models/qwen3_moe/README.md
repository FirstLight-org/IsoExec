# Qwen3 MoE

Set `isoexec.model_package=isoexec.models.qwen3_moe` (verl selects it from `model_type: qwen3_moe`).
For SkyRL, set `trainer.policy.model.path` to the Hugging Face Hub model ID `Qwen/Qwen3-30B-A3B`, or to a compatible local checkpoint directory.
IsoExec reads that checkpoint's Hugging Face `config.json` and safetensors.
Every decoder block must be an expert block (`decoder_sparse_step=1`, empty `mlp_only_layers`) with `norm_topk_prob=true`.
The q/k norms hold plain gamma (`qk_norm="rms"`); SM90 runs them in the fused q/k-norm + RoPE kernel, SM100 refuses them until that path is verified there.
