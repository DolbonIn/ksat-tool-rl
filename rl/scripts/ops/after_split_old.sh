#!/usr/bin/env bash
# After the pr1 split finishes, difficulty-filter the 2017-2021 hard set on all 8 GPUs.
set -uo pipefail
R=/raid/cache/b300-07/ksat-rl; E=$R/runs/eval; S=/home/b300-07/ksat-tool-rl/rl/scripts
until grep -q "ALL_DONE" $E/split_pr1.log 2>/dev/null; do sleep 15; done
cp $R/data/ksat_math_old/train_2017_2021_hard.jsonl $R/data/ksat_math_old/hard_snapshot.jsonl
echo "[$(date +%H:%M:%S)] starting 2017-2021 hard filtering on 8 GPUs ($(wc -l < $R/data/ksat_math_old/hard_snapshot.jsonl) items)"
cd $S && IMAGE_TAG=v4 GPUS=0,1,2,3,4,5,6,7 PORT=30680 MODEL=/models/Nanbeige4.2-3B TAG=passrate_old_hard N=8 MAX_TOKENS=32768 \
  PROBLEMS=/data/ksat_math_old/hard_snapshot.jsonl EVAL_EXTRA="--temperature 0.8 --top-p 1.0 --top-k -1 --no-report" ./docker_eval.sh > $E/passrate_old_hard.log 2>&1
echo "[$(date +%H:%M:%S)] OLD_DONE $(tail -1 $E/passrate_old_hard.log)"
