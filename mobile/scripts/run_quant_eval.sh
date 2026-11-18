#!/usr/bin/env bash
# Quant quality vs the F16 GGUF on held-out RL rollouts (2026 CSAT, chat-formatted, assistant tokens scored)
# and KV-cache quantization on long (6k-token) rollouts. Needs the F16 --full reference dumps:
#   run_lp.sh ref-quant .../F16.gguf seqs_quant.txt eval/lc/quant_F16_ref -t 64 --full
#   run_lp.sh ref-long  .../F16.gguf seqs_long.txt  eval/lc/long_F16_ref  -t 64 --full
set -euo pipefail
R=/home/b300-07/ksat-tool-rl/mobile/scripts/run_lp.sh
G=/ksat/mobile/gguf/nanbeige42-3b-math-v1-s40; S=/ksat/mobile/eval/seqs; O=/ksat/mobile/eval/lc
TH=${TH:-40}
q() { $R "q-$1" $G-$1.gguf $S/seqs_quant.txt $O/quant_$1 -t $TH --ref $O/quant_F16_ref; }
l() { local tag=$1 w=$2; shift 2; $R "l-$tag" $G-$w.gguf $S/seqs_long.txt $O/long_$tag -t $TH --ref $O/long_F16_ref "$@"; }
MODE=${1:-quant}
if [[ $MODE == quant ]]; then
  for v in Q8_0 Q4_0-o8 Q4_0 ; do q $v & done; wait
  for v in Q4_0-pure Q4_0-a8o8 IQ4_NL-o8 Q4_K_M; do q $v & done; wait
else
  l F16-kvq8 F16 -ctk q8_0 -ctv q8_0 &
  l F16-kvq4 F16 -ctk q4_0 -ctv q4_0 &
  l Q4_0-o8-kvf16 Q4_0-o8 &
  wait
  l Q4_0-o8-kvq8 Q4_0-o8 -ctk q8_0 -ctv q8_0 &
  l Q8_0-kvq8 Q8_0 -ctk q8_0 -ctv q8_0 &
  LLAMA_ATTN_ROT_DISABLE=1 l F16-kvq8-norot F16 -ctk q8_0 -ctv q8_0 &
  wait
fi
