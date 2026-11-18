#!/usr/bin/env bash
# imatrix variants (calibration: 48x1024 tokens of chat-formatted *training-set* solutions, eval/make_calib.py),
# quantized from the F16 GGUF and scored against the same F16 reference as run_quant_eval.sh.
set -euo pipefail
P=/ksat/mobile/gguf/nanbeige42-3b-math-v1-s40
IM="--imatrix /ksat/mobile/gguf/imatrix-train-math.gguf"
A8="--tensor-type attn_q=q8_0 --tensor-type attn_k=q8_0 --tensor-type attn_v=q8_0 --tensor-type attn_output=q8_0"
quant() { local name=$1 type=$2; shift 2
  docker run --rm --name "ksat-mobile-quant-$(echo $name | tr 'A-Z_' 'a-z-')" -u "$(id -u):$(id -g)" -v /raid/cache/b300-07/ksat-rl:/ksat \
    ksat-rl/verl-sgl:v4 /ksat/mobile/builds/x86-cpu/bin/llama-quantize $IM "$@" $P-F16.gguf $P-$name.gguf $type 16 2>&1 | grep -E "quant size|rror"; }
quant Q4_0-o8-imat      Q4_0   --output-tensor-type q8_0
quant Q4_0-a8o8-imat    Q4_0   --output-tensor-type q8_0 $A8
quant IQ4_NL-a8o8-imat  IQ4_NL --output-tensor-type q8_0 $A8
quant Q5_K-pure-o8-imat Q5_K   --pure --output-tensor-type q8_0
R=/home/b300-07/ksat-tool-rl/mobile/scripts/run_lp.sh; S=/ksat/mobile/eval/seqs; O=/ksat/mobile/eval/lc
for v in Q4_0-o8-imat Q4_0-a8o8-imat IQ4_NL-a8o8-imat Q5_K-pure-o8-imat; do
  $R "q-$v" $P-$v.gguf $S/seqs_quant.txt $O/quant_$v -t ${TH:-32} --ref $O/quant_F16_ref &
done; wait
