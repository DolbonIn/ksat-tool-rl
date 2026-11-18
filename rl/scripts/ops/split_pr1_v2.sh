#!/usr/bin/env bash
# v2 meet rule (pr1b already running): stop the forward run pr1 when, of the problems at/after pr1's in-flight
# window (index >= F-70), at most 15 are incomplete in the reverse run, or when pr1 has submitted ~everything.
set -uo pipefail
R=/raid/cache/b300-07/ksat-rl; E=$R/runs/eval; S=/home/b300-07/ksat-tool-rl/rl/scripts
log(){ echo "[$(date +%H:%M:%S)] $*"; }
NPROB=$(wc -l < $R/data/synth/synth_pr1.jsonl)
while true; do
  sleep 30
  if grep -q -E "^wall" $E/passrate_synth_pr1.log 2>/dev/null; then log "pr1 finished on its own"; break; fi
  STARTED=$(docker logs ksat-eval-passrate_synth_pr1 2>&1 | grep "Prefill batch" | grep -o "#new-seq: [0-9]*" | awk '{s+=$2} END {print s+0}')
  F=$(python3 -c "print(int($STARTED/8/1.07))")
  INC=$(python3 - "$R/data/synth/synth_pr1.jsonl" "$E/passrate_synth_pr1b.jsonl.partial.jsonl" "$F" <<'PY'
import collections, json, sys
ids = [json.loads(l)["id"] for l in open(sys.argv[1]) if l.strip()]
F = int(sys.argv[3])
c = collections.Counter()
try:
    for l in open(sys.argv[2]):
        r = json.loads(l)
        if not (str(r["finish"]).startswith("error") or r["finish"] == "abort"):
            c[r["id"]] += 1
except FileNotFoundError:
    pass
lo = max(0, min(F, len(ids)) - 70)
print(sum(c[i] < 8 for i in ids[lo:]))
PY
)
  log "v2: forward frontier ~$F, incomplete in reverse run for index>=F-70: $INC"
  if [ "$INC" -le 15 ] || [ "$F" -ge $((NPROB * 98 / 100)) ]; then break; fi
done
if ! grep -q -E "^wall" $E/passrate_synth_pr1.log 2>/dev/null; then
  log "runs met: stopping pr1 server so its client writes completed results"
  docker rm -f ksat-eval-passrate_synth_pr1 >/dev/null
  until grep -q -E "^wall|Traceback" $E/passrate_synth_pr1.log 2>/dev/null; do sleep 10; done
  log "pr1 client done: $(tail -1 $E/passrate_synth_pr1.log)"
fi
docker rm -f ksat-eval-passrate_synth_pr1b >/dev/null 2>&1
until grep -q -E "^wall|Traceback" $E/passrate_synth_pr1b.log 2>/dev/null; do sleep 10; done
python3 $S/ops/merge_split_run.py $R/data/synth/synth_pr1.jsonl $E/passrate_synth_pr1.jsonl $E/passrate_synth_pr1b.jsonl.partial.jsonl 8 \
  $E/passrate_synth_pr1_merged.jsonl $R/data/synth/synth_pr1_residual.jsonl
NRES=$(wc -l < $R/data/synth/synth_pr1_residual.jsonl)
if [ "$NRES" -gt 0 ]; then
  log "residual $NRES problems -> running on all 8 GPUs"
  (cd $S && IMAGE_TAG=v4 GPUS=0,1,2,3,4,5,6,7 PORT=30670 MODEL=/models/Nanbeige4.2-3B TAG=passrate_synth_pr1c N=8 MAX_TOKENS=32768 \
    PROBLEMS=/data/synth/synth_pr1_residual.jsonl EVAL_EXTRA="--temperature 0.8 --top-p 1.0 --top-k -1 --no-report" ./docker_eval.sh > $E/passrate_synth_pr1c.log 2>&1)
  log "residual done: $(tail -1 $E/passrate_synth_pr1c.log)"
  cat $E/passrate_synth_pr1c.jsonl >> $E/passrate_synth_pr1_merged.jsonl
fi
log "ALL_DONE merged samples: $(wc -l < $E/passrate_synth_pr1_merged.jsonl)"
