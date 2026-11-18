#!/usr/bin/env bash
# Split the remaining synth pass-rate work (pr1, forward on GPUs 4-7) with a reverse run on GPUs 0-3 once pr2 ends.
set -uo pipefail
R=/raid/cache/b300-07/ksat-rl; E=$R/runs/eval; S=/home/b300-07/ksat-tool-rl/rl/scripts
log(){ echo "[$(date +%H:%M:%S)] $*"; }
until grep -q -E "^wall|Traceback" $E/passrate_synth_pr2.log 2>/dev/null; do sleep 20; done
log "pr2 finished: $(tail -1 $E/passrate_synth_pr2.log)"
NPROB=$(wc -l < $R/data/synth/synth_pr1.jsonl)
(cd $S && IMAGE_TAG=v4 GPUS=0,1,2,3 PORT=30660 MODEL=/models/Nanbeige4.2-3B TAG=passrate_synth_pr1b N=8 MAX_TOKENS=32768 \
  PROBLEMS=/data/synth/synth_pr1.jsonl EVAL_EXTRA="--temperature 0.8 --top-p 1.0 --top-k -1 --no-report --reverse" \
  nohup ./docker_eval.sh > $E/passrate_synth_pr1b.log 2>&1 &)
log "pr1b (reverse) launched on GPUs 0-3"
while true; do
  sleep 30
  STARTED=$(docker logs ksat-eval-passrate_synth_pr1 2>&1 | grep "Prefill batch" | grep -o "#new-seq: [0-9]*" | awk '{s+=$2} END {print s+0}')
  # forward frontier (problem index); prefill counts over-count ~7% (chunked prefill), so deflate
  F=$(python3 -c "print(int($STARTED/8/1.07))")
  B=$(python3 - "$R/data/synth/synth_pr1.jsonl" "$E/passrate_synth_pr1b.jsonl.partial.jsonl" <<'PY'
import collections, json, sys
ids=[json.loads(l)["id"] for l in open(sys.argv[1]) if l.strip()]
c=collections.Counter()
try:
    for l in open(sys.argv[2]):
        r=json.loads(l)
        if not (str(r["finish"]).startswith("error") or r["finish"]=="abort"): c[r["id"]]+=1
except FileNotFoundError: pass
b=len(ids)
for i in range(len(ids)-1,-1,-1):
    if c[ids[i]]>=8: b=i
    else: break
print(b)
PY
)
  log "forward frontier ~$F, reverse complete from index $B (of $NPROB)"
  if [ "$B" -le "$F" ]; then break; fi
  if grep -q -E "^wall" $E/passrate_synth_pr1.log 2>/dev/null; then log "pr1 finished on its own"; break; fi
done
if ! grep -q -E "^wall" $E/passrate_synth_pr1.log 2>/dev/null; then
  log "runs met: stopping pr1 server so its client writes completed results"
  docker rm -f ksat-eval-passrate_synth_pr1 >/dev/null
  until grep -q -E "^wall|Traceback" $E/passrate_synth_pr1.log 2>/dev/null; do sleep 10; done
  log "pr1 client done: $(tail -1 $E/passrate_synth_pr1.log)"
fi
python3 $S/ops/merge_split_run.py $R/data/synth/synth_pr1.jsonl $E/passrate_synth_pr1.jsonl $E/passrate_synth_pr1b.jsonl.partial.jsonl 8 \
  $E/passrate_synth_pr1_merged.jsonl $R/data/synth/synth_pr1_residual.jsonl
docker rm -f ksat-eval-passrate_synth_pr1b >/dev/null 2>&1
until grep -q -E "^wall|Traceback" $E/passrate_synth_pr1b.log 2>/dev/null; do sleep 10; done
NRES=$(wc -l < $R/data/synth/synth_pr1_residual.jsonl)
if [ "$NRES" -gt 0 ]; then
  log "residual $NRES problems -> running on all 8 GPUs"
  (cd $S && IMAGE_TAG=v4 GPUS=0,1,2,3,4,5,6,7 PORT=30670 MODEL=/models/Nanbeige4.2-3B TAG=passrate_synth_pr1c N=8 MAX_TOKENS=32768 \
    PROBLEMS=/data/synth/synth_pr1_residual.jsonl EVAL_EXTRA="--temperature 0.8 --top-p 1.0 --top-k -1 --no-report" ./docker_eval.sh > $E/passrate_synth_pr1c.log 2>&1)
  log "residual done: $(tail -1 $E/passrate_synth_pr1c.log)"
  cat $E/passrate_synth_pr1c.jsonl >> $E/passrate_synth_pr1_merged.jsonl
fi
log "ALL_DONE merged samples: $(wc -l < $E/passrate_synth_pr1_merged.jsonl)"
