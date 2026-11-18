#!/usr/bin/env bash
# Launch gen_tir.py in a CPU-only container that can reach the gateway (default bridge) and the sandbox
# (internal network ksat-tool-sbx-net).  Extra args go to gen_tir.py.  Never uses GPUs.
#   NAME=ksat-tool-gen data_pipeline/tool/run_gen.sh --limit 40
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
NAME="${NAME:-ksat-tool-gen}"
NET="${NET:-ksat-tool-sbx-net}"
IMAGE="${IMAGE:-ksat-data-tools:local}"
DATA=/raid/cache/b300-07/ksat-rl/data
mkdir -p "$DATA/tool/gen"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker create --name "$NAME" --user "$(id -u):$(id -g)" --network bridge \
  --env-file "$REPO/.env" -e KSAT_SANDBOX_URL=http://ksat-tool-sandbox:8080 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO/rl":/rl:ro -v "$REPO/data_pipeline/tool":/gen:ro -v "$DATA":/data \
  --cpus 8 --memory 8g \
  "$IMAGE" python -u /gen/gen_tir.py "$@" >/dev/null
docker network connect "$NET" "$NAME"
docker start "$NAME" >/dev/null
echo "started $NAME; logs: docker logs -f $NAME"
