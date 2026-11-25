#!/usr/bin/env bash
# One verl GRPO qualification case: native verl sync trainer, IsoExec MegatronModel, vLLM Model Runner V2.
#
# isoexec.enable=true selects IsoExec's engine, vLLM V2 server and trainer (isoexec.integrations.verl.config).
# Lifecycle (enforced by isoexec.integrations.verl.trainer): W0 sync -> worker acks -> rollout ->
# Megatron rescoring compares every complete FP32 row -> one optimizer update -> W1 sync -> acks ->
# W1 rollout -> rescoring compares every row -> stop. Negative controls are not run.
#
# Required environment:
#   CASE            1 | 2 | 3
#   MODEL_PATH      local snapshot directory of the case's pinned checkpoint
#   FIXTURE_DIR     output of make_fixture.py (gsm8k-4.parquet, fixture.json)
#   RUN_DIR         node-local directory for this attempt (rows, receipts, Ray temp, logs)
#   ATTEMPT_ID      unique attempt name (also the W&B experiment name)
#   VERL_ROOT       verl checkout with the companion diff (branch experiment/isoexec-mrv2)
#   WANDB_ENTITY, WANDB_PROJECT, and a W&B credential in the environment (never printed)
# vLLM shares each GPU with the resident Megatron trainer (actor + ref): the MoE cases leave ~27 GiB
# free on an 80 GiB H100, so their engines take 0.25 (inference weights are ~8 GB per GPU for case 2
# and ~14 GB for case 3; KV for 20 x 2048 tokens is small).
# In both MoE topologies the expert data-parallel size is 1, so each GPU holds the full FP32 Adam state
# of its ~3.1B expert parameters (~43 GB with the dense share) on top of BF16 weights, FP32 gradients
# and the reference model. Those cases therefore use Megatron's native optimizer CPU offload
# (HybridDeviceOptimizer, offload fraction 1.0): parameters and gradients stay on the GPU and the
# Adam step runs on the host. This changes where the update runs, not which weights both sides compare.
# Sampling: cases 1 and 2 use temperature 1.0 and n=5. Case 3 uses temperature 0.8 and n=16 so its single
# GRPO batch has a reward spread: the base Qwen1.5-MoE checkpoint rarely writes the "#### <answer>" line the
# strict GSM8K reward needs. A stock vLLM 0.33 reference on the same four prompts (200 samples per
# temperature) scored 1.0% at T=1.0 and 3.0% at T=0.8; the estimated chance that a 4-prompt batch has at
# least one mixed group is 0.19 (T=1.0, n=5), 0.47 (T=0.8, n=5) and 0.87 (T=0.8, n=16).
# An all-equal batch has zero GRPO advantage, so the update would not change any weight.
# DRY_RUN=1 composes the config, expands isoexec.enable and runs IsoExec's validators (no Ray, no GPU).
# Optional: RAY_PORT (default 6379 + 10*CASE), WANDB_RUN_GROUP, ISOEXEC_SHA, VERL_SHA
set -euo pipefail

: "${CASE:?}" "${MODEL_PATH:?}" "${FIXTURE_DIR:?}" "${RUN_DIR:?}" "${ATTEMPT_ID:?}" "${WANDB_ENTITY:?}" "${WANDB_PROJECT:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISOEXEC_ROOT="$(cd "$HERE/../.." && pwd)"
: "${VERL_ROOT:?set VERL_ROOT to the verl checkout (branch experiment/isoexec-mrv2)}"
mkdir -p "$RUN_DIR/evidence" "$RUN_DIR/ray" "$RUN_DIR/logs"

case "$CASE" in
  1) NGPU=1; MODEL_ID=Qwen/Qwen3.5-0.8B; GPU_MEM_DEFAULT=0.35; OPTIM_CPU_OFFLOAD=False; TEMPERATURE=1.0; ROLLOUT_N=5
     A_TP=1 A_PP=1 A_EP=1 A_SP=False
     R_TP=1 R_DP=1 R_PP=1 R_EP=1 PCP=1 DCP=1 ;;
  2) NGPU=4; MODEL_ID=Qwen/Qwen1.5-MoE-A2.7B; GPU_MEM_DEFAULT=0.25; OPTIM_CPU_OFFLOAD=True; TEMPERATURE=1.0; ROLLOUT_N=5
     A_TP=2 A_PP=1 A_EP=4 A_SP=True
     R_TP=2 R_DP=2 R_PP=1 R_EP=4 PCP=1 DCP=1 ;;
  3) NGPU=4; MODEL_ID=Qwen/Qwen1.5-MoE-A2.7B; GPU_MEM_DEFAULT=0.25; OPTIM_CPU_OFFLOAD=True; TEMPERATURE=0.8; ROLLOUT_N=16
     A_TP=2 A_PP=2 A_EP=2 A_SP=True
     R_TP=1 R_DP=1 R_PP=2 R_EP=1 PCP=2 DCP=2 ;;
  *) echo "CASE must be 1, 2 or 3" >&2; exit 2 ;;
esac

# Native environment: workers inherit it from the Ray head started below.
export CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ISOEXEC_ROOT:$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ISOEXEC_FULL_DISTRIBUTION_EVIDENCE_DIR="$RUN_DIR/evidence"
export WANDB_MODE=online WANDB_ENTITY WANDB_PROJECT WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-verl-mrv2-paper}"
if [ "${DRY_RUN:-0}" != 1 ]; then
eval "$(python "$HERE/env.py" --model "$MODEL_PATH")"   # IsoExec engine/trainer flags, V2 runner

RAY_PORT="${RAY_PORT:-$((6379 + 10 * CASE))}"
ray stop --force >/dev/null 2>&1 || true
ray start --head --port "$RAY_PORT" --num-gpus "$NGPU" --temp-dir "$RUN_DIR/ray" \
  --include-dashboard=false --object-store-memory="${RAY_OBJECT_STORE_BYTES:-34359738368}" \
  --min-worker-port $((20000 + 1000 * CASE)) --max-worker-port $((20999 + 1000 * CASE)) >"$RUN_DIR/logs/ray-start.log"
export RAY_ADDRESS="127.0.0.1:$RAY_PORT"
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT
fi

ENGINE_KWARGS=()
if [ "$PCP" -gt 1 ] || [ "$DCP" -gt 1 ]; then
  ENGINE_KWARGS+=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.prefill_context_parallel_size=$PCP
    +actor_rollout_ref.rollout.engine_kwargs.vllm.decode_context_parallel_size=$DCP
  )
fi
if [ "$R_EP" -gt 1 ]; then
  # Recorded as vLLM's use_sequence_parallel_moe; IsoExec's own dispatch moves the experts' tokens.
  ENGINE_KWARGS+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.all2all_backend=allgather_reducescatter)
fi

MEGATRON=()
for role in actor ref; do
  MEGATRON+=(
    actor_rollout_ref.$role.megatron.tensor_model_parallel_size=$A_TP
    actor_rollout_ref.$role.megatron.pipeline_model_parallel_size=$A_PP
    actor_rollout_ref.$role.megatron.expert_model_parallel_size=$A_EP
    actor_rollout_ref.$role.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.$role.megatron.context_parallel_size=1
    actor_rollout_ref.$role.megatron.sequence_parallel=$A_SP
  )
done

OPTIM=()
if [ "$OPTIM_CPU_OFFLOAD" = True ]; then
  OPTIM=(+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
         +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0)
fi

set -x +e
ENTRY=(-m verl.trainer.main_ppo)
[ "${DRY_RUN:-0}" = 1 ] && ENTRY=("$HERE/dry_check.py")
python "${ENTRY[@]}" \
  model_engine=megatron isoexec.enable=true \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  data.train_files="$FIXTURE_DIR/gsm8k-4.parquet" data.val_files="$FIXTURE_DIR/gsm8k-4.parquet" \
  data.train_batch_size=4 data.max_prompt_length=512 data.max_response_length=1024 \
  data.filter_overlong_prompts=False data.truncation=error data.shuffle=False data.seed=1 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
  "${MEGATRON[@]}" \
  ${OPTIM[@]+"${OPTIM[@]}"} \
  actor_rollout_ref.rollout.mode=async actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.temperature=$TEMPERATURE actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$R_TP actor_rollout_ref.rollout.data_parallel_size=$R_DP \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=$R_PP actor_rollout_ref.rollout.expert_parallel_size=$R_EP \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.max_model_len=2048 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.rollout.prompt_length=512 actor_rollout_ref.rollout.response_length=1024 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-$GPU_MEM_DEFAULT}" \
  ${ENGINE_KWARGS[@]+"${ENGINE_KWARGS[@]}"} \
  trainer.total_training_steps=1 trainer.total_epochs=1 trainer.test_freq=-1 trainer.save_freq=-1 trainer.critic_warmup=0 \
  trainer.n_gpus_per_node=$NGPU trainer.nnodes=1 \
  trainer.logger='["console","wandb"]' trainer.project_name="$WANDB_PROJECT" trainer.experiment_name="$ATTEMPT_ID" \
  +isoexec_meta.case=$CASE +isoexec_meta.model="$MODEL_ID" +isoexec_meta.attempt="$ATTEMPT_ID" \
  +isoexec_meta.isoexec_sha="${ISOEXEC_SHA:-unset}" +isoexec_meta.verl_sha="${VERL_SHA:-unset}" \
  +isoexec_meta.trainer="tp$A_TP-pp$A_PP-ep$A_EP-etp1-cp1-sp$A_SP-optimcpu$OPTIM_CPU_OFFLOAD" \
  +isoexec_meta.inference="tp$R_TP-dp$R_DP-pp$R_PP-ep$R_EP-pcp$PCP-dcp$DCP" \
  +isoexec_meta.sampling="t$TEMPERATURE-n$ROLLOUT_N" \
  2>&1 | tee "$RUN_DIR/logs/main_ppo.log"
status=${PIPESTATUS[0]}
set +x
mkdir -p "$RUN_DIR/evidence/receipts"
printf '{"exit_code": %d, "attempt": "%s", "case": %d}\n' "$status" "$ATTEMPT_ID" "$CASE" >"$RUN_DIR/evidence/receipts/terminal.json"
exit "$status"
