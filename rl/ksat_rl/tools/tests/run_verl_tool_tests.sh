#!/usr/bin/env bash
# CPU-only veRL tool-integration tests (no --gpus).  The test container is wired exactly like a training
# container launched by docker_train.sh with SANDBOX_NET set: default bridge + `docker network connect` to the
# internal sandbox network, KSAT_SANDBOX_URL pointing at the sandbox, veRL patched at start (KSAT_TOOL_PATCH).
set -euo pipefail
RL="$(cd "$(dirname "$0")/../../.." && pwd)"
NET="${NET:-ksat-tool-sbx-net}"; NAME="${NAME:-ksat-tool-verltest}"; IMAGE="${IMAGE:-ksat-rl/verl-sgl:v4}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker create --name "$NAME" --network bridge --cpus 16 --memory 32g \
  -e PYTHONPATH=/rl -e HF_HUB_OFFLINE=1 -e KSAT_SANDBOX_URL=http://ksat-tool-sandbox:8080 -e TOKENIZERS_PARALLELISM=false \
  -v "$RL":/rl:ro -v /raid/cache/b300-07/ksat-rl/models:/models:ro -v /raid/cache/b300-07/ksat-rl/data:/data:ro \
  --entrypoint bash "$IMAGE" -c "python /rl/docker/patches/patch_verl_nanbeige_tools.py /opt/verl && python /rl/ksat_rl/tools/tests/test_verl_tool.py" >/dev/null
docker network connect "$NET" "$NAME"
docker start -a "$NAME"; rc=$?
docker rm -f "$NAME" >/dev/null 2>&1 || true
exit $rc
