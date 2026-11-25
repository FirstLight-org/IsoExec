# Qwen2 MoE

Set `isoexec.model_package=isoexec.models.qwen2_moe`.
For SkyRL, set `trainer.policy.model.path` to the Hugging Face Hub model ID `Qwen/Qwen1.5-MoE-A2.7B`, or to a compatible local checkpoint directory.
IsoExec reads that checkpoint's Hugging Face `config.json` and safetensors.
