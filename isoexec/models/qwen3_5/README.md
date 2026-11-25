# Qwen3.5

Set `isoexec.model_package=isoexec.models.qwen3_5`.
For SkyRL, set `trainer.policy.model.path` to the Hugging Face Hub model ID `Qwen/Qwen3.5-0.8B` or `Qwen/Qwen3.5-9B`, or to a compatible local checkpoint directory.
IsoExec reads that checkpoint's Hugging Face `config.json` and safetensors.
