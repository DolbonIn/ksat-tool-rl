#!/usr/bin/env bash
# second batch: mixed-precision / K-quant variants + KV-cache study (see run_quant_eval.sh)
set -euo pipefail
R=/home/b300-07/ksat-tool-rl/mobile/scripts/run_lp.sh
G=/ksat/mobile/gguf/nanbeige42-3b-math-v1-s40; S=/ksat/mobile/eval/seqs; O=/ksat/mobile/eval/lc
TH=${TH:-40}
q() { $R "q-$1" $G-$1.gguf $S/seqs_quant.txt $O/quant_$1 -t $TH --ref $O/quant_F16_ref; }
# wait for batch 1
while docker ps --format '{{.Names}}' | grep -q '^ksat-mobile-lp-q-'; do sleep 20; done
for v in Q4_0-d8o8 IQ4_NL-a8o8 Q5_K-pure-o8 Q6_K-pure-o8; do q $v & done; wait
/home/b300-07/ksat-tool-rl/mobile/scripts/run_quant_eval.sh long
