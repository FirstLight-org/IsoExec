# Qwen3.5 MoE text

Use `isoexec.models.qwen3_5_moe` as the existing IsoExec model source package,
with `Qwen/Qwen3.5-35B-A3B-Base` or a local checkpoint directory. The HF
`architectures` entry selects `Qwen3_5MoeForConditionalGeneration`.

This definition uses native IsoExec GDN/attention and MoE components, normalized
softmax top-k routing, a sigmoid-gated shared expert, and HF checkpoint conversion
rules for fused expert gate/up and down projections. Every decoder block must be
MoE. Vision and MTP checkpoint namespaces are ignored; neither executes.
Router scores are explicit auxiliary outputs; this package does not implicitly
enable an auxiliary training loss.

Checkpoint conversion currently supports loading only: the fused expert
converter does not provide an inverse for Hugging Face export. Conversion rules
use Transformers `WeightRenaming` and
`WeightConverter`; `isoexec.compiler.realization.weights` loads converted tensors
into the current native runtime and skips experts owned by other EP ranks.

The package does not select a GDN kernel backend or enable activation
checkpointing. Those remain runtime capabilities. In particular, the current
upstream H100 dispatcher selects CPR, so this package alone does not provide a
recurrent-only production configuration.

CPU tests cover the released 40-layer geometry and checkpoint key coverage,
HF fused-expert conversion into native rank slices, hybrid graph capture and
pipeline partitioning, and MoE outputs against Transformers. These are not a
GPU numerical or distributed-training qualification.
