# Qwen3

Set `isoexec.model_package=isoexec.models.qwen3` (verl selects it from `model_type: qwen3`).
For SkyRL, set `trainer.policy.model.path` to the Hugging Face Hub model ID `Qwen/Qwen3-8B`, or to a compatible local checkpoint directory.
IsoExec reads that checkpoint's Hugging Face `config.json` and safetensors.
The q/k norms hold plain gamma (`qk_norm="rms"`); SM90 runs them in the fused q/k-norm + RoPE kernel, SM100 refuses them until that path is verified there.
