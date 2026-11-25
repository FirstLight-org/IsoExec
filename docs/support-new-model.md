# Support a new model

## 1. Models available on Hugging Face

For a checkpoint supported by Transformers, reuse its Hugging Face `config.json`
and weights. Add a Python package that every worker can import:

```text
models/
  qwen/
    model.py
    weight.py
```

### `model.py`

- Define the model and its blocks as PyTorch `nn.Module`s.
- Build their layers from `isoexec.nn` components.

```python
from torch import nn
import isoexec.nn as inn
```

The constructor receives the HF config. The model class name must appear in
`config.json`'s `architectures`.

### `weight.py`

Define `checkpoint_conversion_mapping(config)` using Hugging Face
`WeightRenaming` and `WeightConverter`:

- `WeightRenaming` changes a checkpoint key, for example
  `model.embed_tokens.weight` to `embed.weight`, without changing the tensor.
- `WeightConverter` transforms a tensor, for example splitting a packed weight
  or transposing its dimensions.

Return an empty list when checkpoint names and tensors already match the model.

See complete implementations in [Qwen3.5](../isoexec/models/qwen3_5/),
[Qwen3.5 MoE](../isoexec/models/qwen3_5_moe/) and [Qwen2 MoE](../isoexec/models/qwen2_moe/).

### Use in SkyRL

In a SkyRL Megatron/vLLM recipe (see the README), select your package and HF model ID:

```yaml
isoexec:
  model_package: models.qwen

trainer:
  policy:
    model:
      path: Qwen/Qwen3.5-0.8B
```

`model_package` names the importable Python package. A built-in architecture is selected
the same way, for example `model_package: isoexec.models.qwen3_5`.

SkyRL uses IsoExec to obtain the files from the same HF commit:

```text
HF model ID
    |
    v
Resolve a commit, then download / reuse its files
    |
    v
Local HF cache
    +-- config.json        -> model configuration
    +-- *.safetensors      -> weights
    +-- tokenizer files
```

## 2. Models not available on Hugging Face

Write `model.py` and `weight.py` the same way, and provide your own local
checkpoint:

```text
models/
  my_model/
    model.py
    weight.py

checkpoints/
  my_model/
    config.json
    model.safetensors
    ...                 # tokenizer files
```

Use the Hugging Face format for `config.json`:

- `model_type` selects the Transformers config class.
- `architectures` names your model class in `model.py`.
- Other fields provide the configuration values your model needs.

Use the local directory in the same SkyRL recipe:

```yaml
isoexec:
  model_package: models.my_model

trainer:
  policy:
    model:
      path: /path/to/checkpoints/my_model
```

SkyRL reads the files directly from your directory:

```text
Local checkpoint directory
    |
    +-- config.json        -> model configuration
    +-- *.safetensors      -> weights
    +-- tokenizer files
```

Make the Python package and checkpoint files available to every worker.
