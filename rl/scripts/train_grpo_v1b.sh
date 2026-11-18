#!/usr/bin/env bash
# GRPO for Nanbeige4.2-3B on KSAT math with veRL (FSDP2 actor + SGLang async rollout).
# Runs INSIDE the ksat-rl/verl-sgl image. Mounts expected: /models, /data, /rl, /runs.
# Every knob is an env var so experiments are reproducible from the launch line.
set -euo pipefail
NGPUS=${NGPUS:-8}
TRAIN_FILE=${TRAIN_FILE:?}
VAL_FILE=${VAL_FILE:-$TRAIN_FILE}
EXP=${EXP:-grpo_dev}
MODEL=${MODEL:-/models/Nanbeige4.2-3B-train}
BS=${BS:-16}                 # prompts per step
MINI_BS=${MINI_BS:-$BS}      # prompts per optimizer step (BS/MINI_BS updates per rollout batch)
N=${N:-8}                    # rollouts per prompt
MAX_PROMPT=${MAX_PROMPT:-2048}
MAX_RESP=${MAX_RESP:-4096}
LR=${LR:-1e-6}
CLIP_HIGH=${CLIP_HIGH:-0.28}
TEMP=${TEMP:-0.8}
MAX_TOK_GPU=${MAX_TOK_GPU:-65536}
ROLLOUT_TP=${ROLLOUT_TP:-1}
GPU_UTIL=${GPU_UTIL:-0.6}
ATTN_BACKEND=${ATTN_BACKEND:-trtllm_mha}  # fastest decode on B300 (bench_rollout.py); image sglang reports 0.0.0.dev -> veRL would pick fa3
STEPS=${STEPS:-2}
EPOCHS=${EPOCHS:-100}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:--1}
VAL_BEFORE=${VAL_BEFORE:-False}
FILTER_GROUPS=${FILTER_GROUPS:-False}
mkdir -p /runs/$EXP
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  algorithm.filter_groups.enable=${FILTER_GROUPS} \
  data.train_files=${TRAIN_FILE} data.val_files=${VAL_FILE} \
  data.train_batch_size=${BS} \
  data.max_prompt_length=${MAX_PROMPT} data.max_response_length=${MAX_RESP} \
  data.filter_overlong_prompts=True data.truncation=error \
  actor_rollout_ref.model.path=${MODEL} \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=${LR} \
  actor_rollout_ref.actor.optim.weight_decay=0.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BS} \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOK_GPU} \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=${CLIP_HIGH} \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_UTIL} \
  actor_rollout_ref.rollout.n=${N} \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=${ATTN_BACKEND} \
  actor_rollout_ref.rollout.temperature=${TEMP} \
  actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOK_GPU} \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=20 actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  reward.custom_reward_function.path=/rl/ksat_rl/reward.py \
  reward.custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node=${NGPUS} trainer.nnodes=1 \
  trainer.logger='["console"]' \
  trainer.project_name=ksat_math trainer.experiment_name=${EXP} \
  trainer.default_local_dir=/runs/${EXP}/ckpt \
  trainer.total_training_steps=${STEPS} trainer.total_epochs=${EPOCHS} \
  trainer.save_freq=${SAVE_FREQ} trainer.test_freq=${TEST_FREQ} \
  trainer.val_before_train=${VAL_BEFORE} \
  ray_kwargs.ray_init.runtime_env.py_executable=null \
  "$@" 2>&1 | tee /runs/${EXP}/train.log
