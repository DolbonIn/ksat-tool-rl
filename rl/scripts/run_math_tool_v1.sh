#!/usr/bin/env bash
# Math tool-calling GRPO v1 for Nanbeige4.2-3B (8x B300): python sandbox tool, veRL multi-turn agent loop.
# NOT launched automatically.  Prerequisites:
#   WORKERS=64 CPUS=64 rl/ksat_rl/tools/sandbox/sandbox.sh restart   (CPU sandbox on the internal network
#        ksat-tool-sbx-net; 64 workers recommended: a step has 256 concurrent episodes, and a burst of 10 s programs
#        on 16 workers would queue past the tool's 120 s HTTP timeout and be masked as infra faults)
#   tool parquets in /data/ksat_math/rl_tool/          (python -m ksat_rl.tools.make_tool_parquet, see rl/ksat_rl/tools/README.md)
#
# Settings (ksat_tool_rl_v2/README.md §4-§6; start values, not tuned):
#   - same data rows / reward rule / optimizer as math_v1: 32 prompts x n=8, T=0.8, lr 1e-6, clip 0.2/0.28, no KL,
#     token-level truncated-IS rollout correction (cap 2.0, from train_grpo.sh), old log-probs recomputed
#   - one tool `python` (sandbox: fresh process, 10 s, 1 GB, no network); <= 6 tool calls per episode
#     (max_user_turns=6, one call per turn: generation stops at </tool_call>), <= 8 assistant turns,
#     tool output <= 1000 chars (sandbox) + max_tool_response_length=1000 (veRL guard)
#   - response budget 32768 tokens per episode INCLUDING tool-response tokens (same as math_v1)
#   - reward = final assistant turn only (ksat_rl/tools/reward_tool.py): no </think>, an unexecuted tool call
#     (budget exhausted) or a wrong \boxed{} -> 0; sandbox infra faults are masked out of the loss
#   - ICRL demo curriculum as dataset phases (PHASE=k2 -> k1 -> k0): same rows in the same order, so advance a
#     phase by re-running this script with the next PHASE and a larger PHASE_END_STEP; the run resumes from the
#     latest checkpoint of EXP (trainer.resume_mode=auto).  Advance on metrics (valid-call rate, val acc,
#     completion rate), not on a fixed step (§4-C).  Val = held-out 2026학년도 9월 모평 with ZERO demos.
#
# Usage:  PHASE=k2 PHASE_END_STEP=30 rl/scripts/run_math_tool_v1.sh [extra hydra overrides]
#         PHASE=k1 PHASE_END_STEP=60 rl/scripts/run_math_tool_v1.sh
#         PHASE=k0 PHASE_END_STEP=120 rl/scripts/run_math_tool_v1.sh
set -euo pipefail
cd "$(dirname "$0")"
PHASE=${PHASE:-k2}
PHASE_END_STEP=${PHASE_END_STEP:?cumulative global step at which this phase stops}
case "$PHASE" in   # prompt budget per phase: max tool-prompt length (system+tools+demos+problem) + margin
  k2) MAX_PROMPT=${MAX_PROMPT:-7168} ;;   # max 6723 tokens
  k1) MAX_PROMPT=${MAX_PROMPT:-5632} ;;   # max 5325
  k0) MAX_PROMPT=${MAX_PROMPT:-3584} ;;   # max 3161
  *) echo "PHASE must be k2|k1|k0" >&2; exit 1 ;;
esac
TOOLDATA=/data/ksat_math/rl_tool
test -f "/raid/cache/b300-07/ksat-rl/data/ksat_math/rl_tool/rl_train_tool_${PHASE}.parquet" || { echo "missing tool parquet for $PHASE" >&2; exit 1; }
[ "$(docker inspect -f '{{.State.Running}}' ksat-tool-sandbox 2>/dev/null)" = "true" ] || { echo "sandbox not running: rl/ksat_rl/tools/sandbox/sandbox.sh start" >&2; exit 1; }
EXP=${EXP:-math_tool_v1}
IMAGE_TAG=${IMAGE_TAG:-v4} GPUS=${GPUS:-0,1,2,3,4,5,6,7} NAME=${NAME:-ksat-train-math-tool-v1} \
ENTRY=/rl/scripts/train_grpo_tool.sh SANDBOX_NET=${SANDBOX_NET:-ksat-tool-sbx-net} \
TRAIN_FILE=$TOOLDATA/rl_train_tool_${PHASE}.parquet VAL_FILE=$TOOLDATA/rl_val_tool_k0.parquet \
EXP=$EXP MODEL=${MODEL:-/models/Nanbeige4.2-3B-train-fused} \
BS=${BS:-32} MINI_BS=${MINI_BS:-16} N=${N:-8} MAX_PROMPT=$MAX_PROMPT MAX_RESP=${MAX_RESP:-32768} LR=${LR:-1e-6} TEMP=${TEMP:-0.8} \
GPU_UTIL=${GPU_UTIL:-0.8} MAX_TOK_GPU=${MAX_TOK_GPU:-65536} STEPS=$PHASE_END_STEP EPOCHS=${EPOCHS:-100} \
SAVE_FREQ=${SAVE_FREQ:-10} TEST_FREQ=${TEST_FREQ:-10} VAL_BEFORE=${VAL_BEFORE:-True} \
./docker_train.sh \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  'actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra","hf_model"]' \
  trainer.max_actor_ckpt_to_keep=null \
  trainer.resume_mode=auto \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=nanbeige \
  actor_rollout_ref.rollout.multi_turn.tool_config_path=/rl/configs/tool/python_tool.yaml \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1000 \
  actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=strict \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=/rl/configs/tool/agent_loop.yaml \
  actor_rollout_ref.rollout.agent.default_agent_loop=ksat_tool_agent \
  +data.apply_chat_template_kwargs.preserve_thinking=True \
  reward.custom_reward_function.path=/rl/ksat_rl/tools/reward_tool.py \
  reward.custom_reward_function.name=compute_score \
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=/rl \
  "$@"
