#!/usr/bin/env bash
# CPU test of rl/scripts/eval_csat_tool.py against a mock SGLang /generate server (no GPU, no model weights used
# beyond the tokenizer); tool calls run in the real sandbox.  Writes into a temp dir (TMPDIR honoured).
set -euo pipefail
RL="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT=$(mktemp -d); NET=${NET:-ksat-tool-sbx-net}; NAME=ksat-tool-evalmock
docker rm -f $NAME >/dev/null 2>&1 || true
docker create --name $NAME --network bridge --cpus 16 --memory 32g -e PYTHONPATH=/rl -e HF_HUB_OFFLINE=1 \
  -e KSAT_SANDBOX_URL=http://ksat-tool-sandbox:8080 \
  -v "$RL":/rl:ro -v /raid/cache/b300-07/ksat-rl/models:/models:ro -v /raid/cache/b300-07/ksat-rl/data:/data:ro -v "$OUT":/out \
  --entrypoint bash ksat-rl/verl-sgl:v4 -c "
    python /rl/docker/patches/patch_verl_nanbeige_tools.py /opt/verl >/dev/null
    python /rl/ksat_rl/tools/tests/mock_sglang_server.py --problems /data/ksat_math/test_2026.jsonl --port 31000 > /out/mock.log 2>&1 &
    until grep -q ready /out/mock.log; do sleep 1; done
    python /rl/scripts/eval_csat_tool.py --problems /data/ksat_math/test_2026.jsonl --url http://127.0.0.1:31000 \
      --n 2 --max-tokens 4096 --concurrency 64 --out /out/mock_tool.jsonl" >/dev/null
docker network connect $NET $NAME
docker start -a $NAME
docker rm -f $NAME >/dev/null
python3 - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
rows = [json.loads(l) for l in open(f"{out}/mock_tool.jsonl")]
s = json.load(open(f"{out}/mock_tool.summary.json"))
ok = (len(rows) == 92 and all(r["n_calls"] == 1 for r in rows) and all(r["finish"] == "stop" for r in rows)
      and 0.5 < s["acc"] < 1.0 and s["tool"]["tool_status"].get("ok") == 92)
print(f"[{'PASS' if ok else 'FAIL'}] eval_csat_tool.py end-to-end vs mock SGLang: {len(rows)} episodes, acc={s['acc']:.3f}, "
      f"tool={s['tool']['tool_status']}, report sections={[k for k in s if k in ('prob_stat','calculus','geometry')]}")
sys.exit(0 if ok else 1)
PY
echo "outputs in $OUT"
