#!/usr/bin/env bash
# CPU-only (no --gpus, --network none) runs of the Nanbeige rendering / format checks in ksat-rl/verl-sgl:v4.
#   data_pipeline/tool/run_format.sh render [args]   -> tir_nanbeige.jsonl
#   data_pipeline/tool/run_format.sh check  [args]   -> format_check_report.json
#   data_pipeline/tool/run_format.sh fewshot [args]  -> fewshot_demos.json
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATA=/raid/cache/b300-07/ksat-rl/data
MODEL=/raid/cache/b300-07/ksat-rl/models/Nanbeige4.2-3B
cmd="$1"; shift
case "$cmd" in
  render) script=render_nanbeige.py ;;
  check) script=check_format.py ;;
  fewshot) script=make_fewshot.py ;;
  *) echo "usage: $0 render|check|fewshot"; exit 1 ;;
esac
docker run --rm --name "ksat-tool-fmt-$cmd" --network none --user "$(id -u):$(id -g)" --cpus 8 --memory 16g \
  -e HF_HUB_OFFLINE=1 -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e TOKENIZERS_PARALLELISM=false \
  -v "$REPO/rl":/rl:ro -v "$REPO/data_pipeline/tool":/gen:ro -v "$DATA":/data -v "$MODEL":/models/Nanbeige4.2-3B:ro \
  --entrypoint python ksat-rl/verl-sgl:v4 "/gen/$script" "$@"
