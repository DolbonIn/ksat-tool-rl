#!/usr/bin/env bash
# Launch the transcription VLM (Qwen3.5-397B-A17B, official FP8 checkpoint) with SGLang on GPUs 0-3 only.
# OpenAI-compatible endpoint: http://127.0.0.1:${PORT}/v1, served model name "vlm".
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}
PORT=${PORT:-31801}
NAME=${NAME:-ksat-data-vlm}
IMAGE=${IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20261030-302b78cc}
HF_HOME_HOST=${HF_HOME_HOST:-/raid/cache/b300-07/huggingface}

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus '"device=0,1,2,3"' --ipc=host --shm-size 32g \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -v "$HF_HOME_HOST":/hf \
  -p 127.0.0.1:${PORT}:${PORT} \
  "$IMAGE" \
  python3 -m sglang.launch_server --model-path "$MODEL" --served-model-name vlm \
    --tp 4 --host 0.0.0.0 --port "$PORT" \
    --context-length 65536 --mem-fraction-static 0.85 \
    --reasoning-parser qwen3 --trust-remote-code \
    --max-running-requests ${MAX_RUNNING:-128} --cuda-graph-max-bs-decode ${MAX_RUNNING:-128}
echo "started $NAME on 127.0.0.1:$PORT (GPUs 0-3)"
