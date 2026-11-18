#!/usr/bin/env bash
# Phase-2 chain: wait for the phase-2 pool measurement, then (in parallel) re-measure phase-1 real items + term drills
# on GPUs 0-3 and finish the 0/8 re-measure (--resume) on GPUs 4-7, then build the v2 RL set and launch GRPO v2.
set -uo pipefail
R=/raid/cache/b300-07/ksat-rl; E=$R/runs/eval; S=/home/b300-07/ksat-tool-rl/rl/scripts
log(){ echo "[$(date +%H:%M:%S)] $*"; }
until grep -q -E "^wall|Traceback" $E/passrate_phase2_step40.log 2>/dev/null; do sleep 30; done
log "phase-2 pool measured: $(tail -1 $E/passrate_phase2_step40.log)"
# stop the slow 2-GPU 0/8 re-measure; completed samples are kept in its .partial.jsonl
docker rm -f ksat-eval-zero_remeasure_step40 >/dev/null 2>&1; sleep 20
EV="--temperature 0.8 --top-p 1.0 --top-k -1 --no-report"
(cd $S && IMAGE_TAG=v4 GPUS=0,1,2,3 PORT=30790 MODEL=/models/math_v1_step40 TAG=remeasure_real327_terms N=8 MAX_TOKENS=32768 \
   PROBLEMS=/data/synth/remeasure_real327_terms.jsonl SGL_EXTRA="--tokenizer-path /models/Nanbeige4.2-3B" EVAL_EXTRA="$EV" \
   bash ./docker_eval.sh > $E/remeasure_real327_terms.log 2>&1; log "real327+terms done: $(tail -1 $E/remeasure_real327_terms.log)") &
(cd $S && IMAGE_TAG=v4 GPUS=4,5,6,7 PORT=30791 MODEL=/models/math_v1_step40 TAG=zero_remeasure_step40 N=16 MAX_TOKENS=65536 \
   PROBLEMS=/data/synth/zero_pass_112.jsonl SGL_EXTRA="--tokenizer-path /models/Nanbeige4.2-3B" EVAL_EXTRA="$EV --resume" \
   bash ./docker_eval.sh > $E/zero_remeasure_step40_resume.log 2>&1; log "0/8 re-measure done: $(tail -1 $E/zero_remeasure_step40_resume.log)") &
wait
mkdir -p $R/data/ksat_math/rl_v2
docker run --rm -v /home/b300-07/ksat-tool-rl/rl:/rl -v $R:/k -w /rl --entrypoint python ksat-rl/verl-sgl:v4 -m ksat_rl.build_rl_set_v2 \
  --items /k/data/synth/phase2_pool.jsonl /k/data/synth/remeasure_real327_terms.jsonl /k/data/synth/zero_pass_112.jsonl \
  --rates /k/runs/eval/passrate_phase2_step40.jsonl /k/runs/eval/remeasure_real327_terms.jsonl /k/runs/eval/zero_remeasure_step40.jsonl.partial.jsonl \
  --hinted /k/data/star/hint_prompts.jsonl --hint-rates /k/runs/eval/star_rollouts_step40.jsonl \
  --test /k/data/ksat_math/test_2026.jsonl --val /k/data/ksat_math/rl/rl_val.parquet --out-dir /k/data/ksat_math/rl_v2 > $R/data/ksat_math/rl_v2/build.log 2>&1
ROWS=$(grep -o '"rows": [0-9]*' $R/data/ksat_math/rl_v2/build.log | grep -o '[0-9]*')
log "RL v2 set built: rows=$ROWS"
[ -n "$ROWS" ] && [ "$ROWS" -ge 128 ] || { log "BUILD_FAILED"; exit 1; }
cd $S && NTRAIN=$ROWS ./run_math_v2.sh && log "TRAINING_V2_LAUNCHED"
