#!/usr/bin/env bash
# After math_v1 finishes: evaluate base + checkpoints on the 2026 수능 math test with the baseline protocol
# (T=0.6, top_p 0.95, top_k 20, thinking on, max_tokens 65536) at n=16, four models in parallel on 2 GPUs each.
set -uo pipefail
R=/raid/cache/b300-07/ksat-rl; S=/home/b300-07/ksat-tool-rl/rl/scripts; E=$R/runs/eval
log(){ echo "[$(date +%H:%M:%S)] $*"; }
while docker ps --format '{{.Names}}' | grep -q '^ksat-train-math-v1$'; do sleep 30; done
log "training container exited; last checkpoint $(cat $R/runs/math_v1/ckpt/latest_checkpointed_iteration.txt)"
LAST=$(cat $R/runs/math_v1/ckpt/latest_checkpointed_iteration.txt)
for st in 30 40 $LAST; do
  [ -d $R/runs/math_v1/ckpt/global_step_$st/actor/huggingface ] && $S/ops/prepare_ckpt_model.sh $R/runs/math_v1/ckpt/global_step_$st/actor/huggingface math_v1_step$st >/dev/null && log "prepared math_v1_step$st"
done
run(){ (cd $S && IMAGE_TAG=v4 GPUS=$1 PORT=$2 MODEL=$3 TAG=$4 N=16 MAX_TOKENS=65536 PROBLEMS=/data/ksat_math/test_2026.jsonl \
        SGL_EXTRA="--tokenizer-path /models/Nanbeige4.2-3B" ./docker_eval.sh > $E/$4.log 2>&1; log "done $4: $(grep -E '^\[|전체' $E/$4.log | tr '\n' ' ')") & }
run 0,1 30710 /models/Nanbeige4.2-3B final_base_n16
run 2,3 30720 /models/math_v1_step30 final_step30_n16
run 4,5 30730 /models/math_v1_step40 final_step40_n16
run 6,7 30740 /models/math_v1_step$LAST final_step${LAST}_n16
wait
log "ALL_EVAL_DONE"
