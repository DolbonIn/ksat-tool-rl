#!/usr/bin/env bash
# Math GRPO phase 2 (from math_v1 step 40 weights, fresh optimizer).
# Changes vs v1: response budget 32k -> 48k (long killers such as 2026 공통 21 needed 45k), clip_ratio_high 0.28 -> 0.2
# (v1 entropy rose 0.117 -> 0.20), set re-selected by the step-40 policy's pass rates (build_rl_set_v2.py),
# hardest learnable items duplicated, level-1 hinted versions of still-unsolved items, Korean math-term drills.
set -euo pipefail
cd "$(dirname "$0")"
NTRAIN=${NTRAIN:?rows in rl_train.parquet}; EPOCHS=${EPOCHS:-3}; BS=${BS:-64}
STEPS=$(( NTRAIN / BS * EPOCHS ))
echo "rows $NTRAIN, epochs $EPOCHS, bs $BS -> $STEPS steps (drop_last)"
ENTRY=/rl/scripts/train_grpo_v1b.sh CLIP_HIGH=${CLIP_HIGH:-0.2} \
IMAGE_TAG=v4 GPUS=${GPUS:-0,1,2,3,4,5,6,7} NAME=${NAME:-ksat-train-math-v2} \
TRAIN_FILE=/data/ksat_math/rl_v2/rl_train.parquet VAL_FILE=/data/ksat_math/rl_v2/rl_val.parquet \
EXP=${EXP:-math_v2} MODEL=${MODEL:-/models/math_v1_step40} \
BS=$BS MINI_BS=${MINI_BS:-16} N=8 MAX_PROMPT=3072 MAX_RESP=${MAX_RESP:-49152} LR=${LR:-1e-6} TEMP=0.8 \
GPU_UTIL=0.85 MAX_TOK_GPU=65536 STEPS=$STEPS EPOCHS=$EPOCHS SAVE_FREQ=10 TEST_FREQ=10 VAL_BEFORE=False \
./docker_train.sh \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  'actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra","hf_model"]' \
  "$@"
