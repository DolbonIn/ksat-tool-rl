#!/usr/bin/env bash
# Intervention: continue math GRPO from a v1 checkpoint with a more conservative update to stop entropy growth.
#   clip_ratio_high 0.28 -> 0.2 (symmetric clip), lr 1e-6 -> 5e-7, fresh optimizer (weights only).
# Usage: restart_v1b.sh <step> <remaining_steps>   e.g. restart_v1b.sh 30 14
set -euo pipefail
STEP=${1:?checkpoint step}; REMAIN=${2:?steps to run}
R=/raid/cache/b300-07/ksat-rl; S=/home/b300-07/ksat-tool-rl/rl/scripts
CK=$R/runs/math_v1/ckpt/global_step_$STEP/actor/huggingface
[ "$(cat $R/runs/math_v1/ckpt/latest_checkpointed_iteration.txt)" -ge "$STEP" ] || { echo "checkpoint $STEP not finished"; exit 1; }
docker stop -t 60 ksat-train-math-v1 >/dev/null 2>&1 || true
$S/ops/prepare_ckpt_model.sh $CK math_v1_step$STEP
cd $S && ENTRY=/rl/scripts/train_grpo_v1b.sh CLIP_HIGH=0.2 IMAGE_TAG=v4 GPUS=0,1,2,3,4,5,6,7 NAME=ksat-train-math-v1b \
  TRAIN_FILE=/data/ksat_math/rl/rl_train.parquet VAL_FILE=/data/ksat_math/rl/rl_val.parquet \
  EXP=math_v1b MODEL=/models/math_v1_step$STEP BS=64 MINI_BS=16 N=8 MAX_PROMPT=2304 MAX_RESP=32768 \
  LR=5e-7 TEMP=0.8 GPU_UTIL=0.8 MAX_TOK_GPU=65536 STEPS=$REMAIN EPOCHS=10 SAVE_FREQ=7 TEST_FREQ=7 VAL_BEFORE=False \
  ./docker_train.sh \
    actor_rollout_ref.model.use_fused_kernels=True actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    'actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra","hf_model"]'
echo "launched ksat-train-math-v1b from step $STEP for $REMAIN steps"
