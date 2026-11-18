#!/usr/bin/env bash
# Math GRPO v1 for Nanbeige4.2-3B (8x B300). Data: /data/ksat_math/rl/{rl_train,rl_val}.parquet from build_rl_set.py.
#   - no-tool GRPO, rule reward (ksat_rl/reward.py), n=8 rollouts at T=0.8, 32k response budget
#   - 32 prompts/step, 2 optimizer updates/step (mini-batch 16), lr 1e-6 constant, clip 0.2/0.28, no KL
#   - token-level truncated IS rollout correction (cap 2.0); old log-probs recomputed by the trainer
#   - val = held-out 2026학년도 9월 모평 (46 items, n=4, T=0.6/top-p 0.95/top-k 20) every 10 steps
#   - checkpoints (incl. HF weights for SGLang eval) every 10 steps
set -euo pipefail
cd "$(dirname "$0")"
NTRAIN=${NTRAIN:?number of rows in rl_train.parquet}
EPOCHS=${EPOCHS:-4}
BS=${BS:-32}
STEPS=$(( (NTRAIN + BS - 1) / BS * EPOCHS ))
echo "train rows $NTRAIN, epochs $EPOCHS, bs $BS -> $STEPS steps"
IMAGE_TAG=v4 GPUS=${GPUS:-0,1,2,3,4,5,6,7} NAME=${NAME:-ksat-train-math-v1} \
TRAIN_FILE=/data/ksat_math/rl/rl_train.parquet VAL_FILE=/data/ksat_math/rl/rl_val.parquet \
EXP=${EXP:-math_v1} MODEL=/models/Nanbeige4.2-3B-train-fused \
BS=$BS MINI_BS=${MINI_BS:-16} N=8 MAX_PROMPT=2304 MAX_RESP=32768 LR=${LR:-1e-6} TEMP=0.8 \
GPU_UTIL=0.8 MAX_TOK_GPU=65536 STEPS=$STEPS EPOCHS=$EPOCHS SAVE_FREQ=10 TEST_FREQ=10 VAL_BEFORE=True \
./docker_train.sh \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  'actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra","hf_model"]' \
  trainer.max_actor_ckpt_to_keep=null \
  "$@"
