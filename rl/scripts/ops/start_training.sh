#!/usr/bin/env bash
# After the 2017-2021 hard-set filtering: build the final RL set and launch math GRPO v1 on 8 GPUs.
set -uo pipefail
R=/raid/cache/b300-07/ksat-rl; E=$R/runs/eval; S=/home/b300-07/ksat-tool-rl/rl/scripts
log(){ echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "OLD_DONE" $E/after_split_old.log 2>/dev/null; do sleep 15; done
mkdir -p $R/data/ksat_math/rl
docker run --rm -v /home/b300-07/ksat-tool-rl/rl:/rl -v $R:/k -w /rl --entrypoint python ksat-rl/verl-sgl:v4 -m ksat_rl.build_rl_set \
  --train /k/data/ksat_math/train.jsonl /k/data/ksat_math_old/hard_snapshot.jsonl \
  --synth /k/data/synth/synth_pr1.jsonl /k/data/synth/synth_pr2.jsonl \
  --rates /k/runs/eval/passrate_train.jsonl /k/runs/eval/passrate_old_hard.jsonl /k/runs/eval/passrate_synth_pr1_merged.jsonl /k/runs/eval/passrate_synth_pr2.jsonl \
  --test /k/data/ksat_math/test_2026.jsonl --out-dir /k/data/ksat_math/rl > $R/data/ksat_math/rl/build.log 2>&1
NTRAIN=$(grep -o '"train_real": [0-9]*' $R/data/ksat_math/rl/build.log | grep -o '[0-9]*'); NSYN=$(grep -o '"train_synth": [0-9]*' $R/data/ksat_math/rl/build.log | grep -o '[0-9]*')
N=$((NTRAIN + NSYN)); log "RL set built: real $NTRAIN + synth $NSYN = $N"
cd $S && NTRAIN=$N BS=64 MINI_BS=16 EPOCHS=4 ./run_math_v1.sh
log "TRAINING_LAUNCHED container ksat-train-math-v1"
