#!/usr/bin/env bash
# After STaR hint rollouts finish: build the STaR SFT parquet, then run SFT on GPUs 2-7 from math_v1_step40.
# (GPUs 0-1 keep running the 0/8 re-measure.)
set -uo pipefail
R=/raid/cache/b300-07/ksat-rl; E=$R/runs/eval
log(){ echo "[$(date +%H:%M:%S)] $*"; }
until grep -q -E "^wall|Traceback" $E/star_rollouts_step40.log 2>/dev/null; do sleep 30; done
log "STaR rollouts done: $(tail -1 $E/star_rollouts_step40.log)"
docker run --rm -v /home/b300-07/ksat-tool-rl/rl:/rl -v $R/data:/data -v $E:/out -w /rl --entrypoint python ksat-rl/verl-sgl:v4 \
  -m ksat_rl.star build-sft --samples /out/star_rollouts_step40.jsonl --pool /data/synth/seed_pool_all.jsonl \
  --test /data/ksat_math/test_2026.jsonl --out /data/star/star_sft.parquet --per-item 2 2>&1 | tail -2
log "SFT data built"
docker rm -f ksat-train-star-sft >/dev/null 2>&1 || true
docker run -d --name ksat-train-star-sft --gpus '"device=2,3,4,5,6,7"' --shm-size 128g --ipc host --ulimit memlock=-1 \
  -v $R/models:/models -v $R/data:/data -v /home/b300-07/ksat-tool-rl/rl:/rl -v $R/runs:/runs \
  -e NGPUS=6 -e MODEL=/models/math_v1_step40 -e DATA=/data/star/star_sft.parquet -e EXP=star_sft_v1 -e PYTHONPATH=/rl -e HF_HUB_OFFLINE=1 \
  --entrypoint bash ksat-rl/verl-sgl:v4 /rl/scripts/run_star_sft.sh >/dev/null
log "STaR SFT launched (ksat-train-star-sft, GPUs 2-7)"
