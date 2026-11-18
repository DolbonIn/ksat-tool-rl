#!/usr/bin/env bash
# Tool-mode CSAT eval: serve a checkpoint with SGLang (data-parallel over GPUS) and run eval_csat_tool.py against
# its native /generate endpoint, executing the model's python calls in the sandbox.
# Usage: GPUS=0,1,2,3 MODEL=/models/Nanbeige4.2-3B TAG=tool_base [N=8] [TOOL_DEMOS=0] [PROBLEMS=...] docker_eval_tool.sh
# Needs: rl/ksat_rl/tools/sandbox/sandbox.sh start.  The client container is on the default bridge (reaches the
# SGLang container by its bridge IP) and is attached to the internal sandbox network; the SGLang container is not.
set -euo pipefail
GPUS=${GPUS:?}; MODEL=${MODEL:?}; TAG=${TAG:?}; N=${N:-8}; PORT=${PORT:-30600}; TOOL_DEMOS=${TOOL_DEMOS:-0}
PROBLEMS=${PROBLEMS:-/data/ksat_math/test_2026.jsonl}; MAX_TOKENS=${MAX_TOKENS:-65536}
SANDBOX_NET=${SANDBOX_NET:-ksat-tool-sbx-net}; IMAGE=ksat-rl/verl-sgl:${IMAGE_TAG:-v4}
DP=$(echo $GPUS | tr ',' '\n' | wc -l); NAME=ksat-eval-$TAG; CLIENT=ksat-eval-$TAG-client
OUT=/raid/cache/b300-07/ksat-rl/runs/eval; mkdir -p $OUT
[ "$(docker network inspect -f '{{.Internal}}' "$SANDBOX_NET" 2>/dev/null)" = "true" ] || { echo "sandbox network missing: rl/ksat_rl/tools/sandbox/sandbox.sh start" >&2; exit 1; }
docker rm -f $NAME $CLIENT >/dev/null 2>&1 || true
docker run -d --name $NAME --gpus "\"device=$GPUS\"" --shm-size 64g --ipc host -p $PORT:$PORT \
  -v /raid/cache/b300-07/ksat-rl/models:/models:ro -v /raid/cache/b300-07/ksat-rl/runs:/runs:ro \
  --entrypoint python $IMAGE -m sglang.launch_server --model-path $MODEL \
  --served-model-name nanbeige --trust-remote-code --host 0.0.0.0 --port $PORT --dp-size $DP \
  --attention-backend trtllm_mha --mem-fraction-static 0.85 ${SGL_EXTRA:-} >/dev/null
until curl -s -m 2 localhost:$PORT/v1/models >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -q "^$NAME$"; then docker logs $NAME 2>&1 | tail -30; exit 1; fi; sleep 5; done
SERVER_IP=$(docker inspect -f '{{.NetworkSettings.IPAddress}}' $NAME)
docker create --name $CLIENT --network bridge -e PYTHONPATH=/rl -e HF_HUB_OFFLINE=1 \
  -e KSAT_SANDBOX_URL=${KSAT_SANDBOX_URL:-http://ksat-tool-sandbox:8080} \
  -v /home/b300-07/ksat-tool-rl/rl:/rl:ro -v /raid/cache/b300-07/ksat-rl/models:/models:ro \
  -v /raid/cache/b300-07/ksat-rl/data:/data:ro -v $OUT:/out \
  --entrypoint bash $IMAGE -c "python /rl/docker/patches/patch_verl_nanbeige_tools.py /opt/verl >/dev/null && \
    python /rl/scripts/eval_csat_tool.py --problems $PROBLEMS --url http://$SERVER_IP:$PORT --tokenizer $MODEL \
    --n $N --max-tokens $MAX_TOKENS --tool-demos $TOOL_DEMOS --concurrency ${CONCURRENCY:-$((128 * DP))} \
    --out /out/$TAG.jsonl ${EVAL_EXTRA:-}" >/dev/null
docker network connect "$SANDBOX_NET" $CLIENT
docker start -a $CLIENT
docker rm -f $NAME $CLIENT >/dev/null
