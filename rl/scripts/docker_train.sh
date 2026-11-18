#!/usr/bin/env bash
# Launch train_grpo.sh in a container. Usage: GPUS=4,5,6,7 NAME=ksat-train-smoke [ENV=...] docker_train.sh [hydra overrides]
# Tool RL: SANDBOX_NET=ksat-tool-sbx-net additionally attaches the container to the sandbox's --internal network
# (docker create -> docker network connect -> docker start, so the network exists before training starts; the
# default bridge stays the primary network) and sets KSAT_SANDBOX_URL (default http://ksat-tool-sandbox:8080).
# Start the sandbox first: rl/ksat_rl/tools/sandbox/sandbox.sh start
# ENTRY selects the in-container script (default train_grpo.sh; tool RL uses train_grpo_tool.sh).
set -euo pipefail
GPUS=${GPUS:-0,1,2,3,4,5,6,7}; NAME=${NAME:-ksat-train}
NG=$(echo $GPUS | tr ',' '\n' | wc -l)
ENVS=(); for v in TRAIN_FILE VAL_FILE EXP MODEL BS MINI_BS N MAX_PROMPT MAX_RESP LR TEMP MAX_TOK_GPU ROLLOUT_TP GPU_UTIL STEPS EPOCHS SAVE_FREQ TEST_FREQ VAL_BEFORE FILTER_GROUPS ATTN_BACKEND CLIP_HIGH; do [ -n "${!v:-}" ] && ENVS+=(-e "$v=${!v}"); done
DOCKER_CMD=(run -d)
if [ -n "${SANDBOX_NET:-}" ]; then
  [ "$(docker network inspect -f '{{.Internal}}' "$SANDBOX_NET" 2>/dev/null)" = "true" ] || { echo "SANDBOX_NET=$SANDBOX_NET is not an existing --internal network (start it with rl/ksat_rl/tools/sandbox/sandbox.sh start)" >&2; exit 1; }
  DOCKER_CMD=(create)
  ENVS+=(-e "KSAT_SANDBOX_URL=${KSAT_SANDBOX_URL:-http://ksat-tool-sandbox:8080}")
fi
docker "${DOCKER_CMD[@]}" --name $NAME --gpus "\"device=$GPUS\"" --shm-size 128g --ipc host --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
  -v /raid/cache/b300-07/ksat-rl/models:/models -v /raid/cache/b300-07/ksat-rl/data:/data \
  -v /home/b300-07/ksat-tool-rl/rl:/rl -v /raid/cache/b300-07/ksat-rl/runs:/runs \
  -v /raid/cache/b300-07/huggingface:/root/.cache/huggingface \
  -e NGPUS=$NG -e PYTHONPATH=/rl -e HF_HUB_OFFLINE=1 "${ENVS[@]}" \
  --entrypoint bash ksat-rl/verl-sgl:${IMAGE_TAG:-v2} ${ENTRY:-/rl/scripts/train_grpo.sh} "$@"
if [ -n "${SANDBOX_NET:-}" ]; then
  docker network connect "$SANDBOX_NET" "$NAME"
  docker start "$NAME"
fi
