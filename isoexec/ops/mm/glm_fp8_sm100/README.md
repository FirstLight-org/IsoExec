# GLM checkpoint FP8 projection

The FirstLight implementation executes the FP32-scale GLM projection emitted by the SM100 low-bit policy. The checkpoint stores E4M3 codes and one FP32 multiplier per 128 × 128 weight block. For each BF16 input row, the quantizer computes one FP32 scale per 128 channels, using the maximum absolute value with a floor of 1e−4 times the FP32 reciprocal of 448. It divides with round-to-nearest and rounds the clamped result to E4M3.

Each group starts a fresh FP32 dot product. Four K32 tensor-core instructions accumulate into it. The kernel multiplies the two binary32 scales and uses one FP32 fused multiply-add to promote that group's dot into the segment sum. A deployment that divides K across ranks uses equal segments and an adjacent-pair FP32 tree. The caller selects the segment count for the deployment. The operator rounds to BF16 at the root when requested.

To reproduce the four-B200 forward check on the first four release layers, run:

    torchrun --standalone --nproc-per-node=4 examples/check_glm53_fp8.py CHECKPOINT OUTPUT

Each file written to OUTPUT contains the checkpoint rows and scale words needed to evaluate eight sampled FP32 outputs with the policy interpreter. Keep these local with the checkpoint. From project-x run:

    bash tirx_ext/run.sh policy_validation/lowbit/glm_fp8_schedule.py --probe OUTPUT/3.self_attn.o_proj.inputs.json

The checker also emits the numerical schedule with --k, --n and --max-tp. For the attention output projection the four-GPU test declares TP up to four and compares the single-device tree against four physical K shards.

This forward operator consumes the original FP8 bytes at each call. The four-GPU check exercises batch invariance, the BF16 output cast and the TP4 merge. Optimizer updates and the SkyRL weight transfer need a model binding to exercise them.
