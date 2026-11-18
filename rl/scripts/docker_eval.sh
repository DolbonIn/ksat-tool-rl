#!/usr/bin/env bash
# Serve a checkpoint with SGLang (data-parallel over GPUS) and run eval_csat.py against it.
# Usage: GPUS=0,1,2,3 MODEL=/models/Nanbeige4.2-3B TAG=base [N=8] [PROBLEMS=/data/ksat_math/test_2026.jsonl] docker_eval.sh
set -euo pipefail
GPUS=${GPUS:?}; MODEL=${MODEL:?}; TAG=${TAG:?}; N=${N:-8}; PORT=${PORT:-30600}
PROBLEMS=${PROBLEMS:-/data/ksat_math/test_2026.jsonl}; MAX_TOKENS=${MAX_TOKENS:-65536}
DP=$(echo $GPUS | tr ',' '\n' | wc -l); NAME=ksat-eval-$TAG
OUT=/raid/cache/b300-07/ksat-rl/runs/eval; mkdir -p $OUT
docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --name $NAME --gpus "\"device=$GPUS\"" --shm-size 64g --ipc host -p $PORT:$PORT \
  -v /raid/cache/b300-07/ksat-rl/models:/models:ro -v /raid/cache/b300-07/ksat-rl/runs:/runs:ro \
  --entrypoint python ksat-rl/verl-sgl:${IMAGE_TAG:-v3} -m sglang.launch_server --model-path $MODEL \
  --served-model-name nanbeige --trust-remote-code --host 0.0.0.0 --port $PORT --dp-size $DP \
  --reasoning-parser nanbeige --attention-backend trtllm_mha --mem-fraction-static 0.85 ${SGL_EXTRA:-} >/dev/null
until curl -s -m 2 localhost:$PORT/v1/models >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -q "^$NAME$"; then docker logs $NAME 2>&1 | tail -30; exit 1; fi; sleep 5; done
docker run --rm --network host -v /home/b300-07/ksat-tool-rl/rl:/rl -v /raid/cache/b300-07/ksat-rl/data:/data:ro -v $OUT:/out \
  --entrypoint python ksat-rl/verl-sgl:${IMAGE_TAG:-v3} /rl/scripts/eval_csat.py --problems $PROBLEMS \
  --url http://127.0.0.1:$PORT/v1 --n $N --max-tokens $MAX_TOKENS --concurrency ${CONCURRENCY:-$((128 * DP))} --out /out/$TAG.jsonl ${EVAL_EXTRA:-}
docker rm -f $NAME >/dev/null
