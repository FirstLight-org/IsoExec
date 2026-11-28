# GLM checkpoint FP8 projection

The FirstLight kernel executes the GLM projection emitted by the SM100 low-bit policy. The checkpoint stores E4M3 codes and one FP32 multiplier per 128 × 128 weight block. For each BF16 input row, the quantizer computes one FP32 scale per 128 channels by flooring the absolute maximum at 1e−4 and multiplying by the FP32 reciprocal of 448. It divides with round-to-nearest and rounds the clamped result to E4M3.

Each scale group starts a fresh FP32 dot product. Four K32 tensor-core instructions accumulate into it. A rounded product of the two binary32 scales and one FP32 fused multiply-add promote the dot into the segment sum. The kernel loader reads the model dimensions and segment length from the project-x manifest and checks the emitted program's SHA-256 against the implemented programs. A K shard returns FP32. The caller gathers the shards in K order and uses `merge_partials` for the adjacent-pair tree. The complete projection can round to BF16 at the root.

## Reproduce on four B200s

In the project-x checkout, compile deployment manifests for the eight distinct FP8 projection geometries of the four-layer GLM-5.3-Flash release slice. Use an absolute output path accessible to both checkouts.

    bash tirx_ext/run.sh policy_validation/lowbit/glm_fp8_schedule.py \
        --max-tp 4 --release-manifests /path/to/manifests.json

In the FirstLight checkout, run the checkpoint test.

    torchrun --standalone --nproc-per-node=4 examples/check_glm53_fp8.py \
        CHECKPOINT /path/to/manifests.json /path/to/output

The test compares the same complete FP32 row at batches 1 and 64 and checks BF16 rounding on all 64 output rows. For the dense MLP down projection and sparse attention output projection it also compares the single-device four-segment tree with four physical K shards over every output. Each local probe file includes the manifest hash, sampled checkpoint rows and scales. Keep these files with the checkpoint. Score all eight using the independent project-x interpreter.

    for probe in /path/to/output/*.inputs.json; do
        bash tirx_ext/run.sh policy_validation/lowbit/glm_fp8_schedule.py --probe "$probe" || exit 1
    done

These checks exercise projection forward on released weights. The model-level SkyRL trainer, rollout and optimizer weight transfer require separate qualification.
