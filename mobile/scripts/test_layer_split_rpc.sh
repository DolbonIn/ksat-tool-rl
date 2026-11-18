#!/usr/bin/env bash
# x86 emulation of a 2-device layer split (two local ggml RPC servers stand in for HTP0:0/HTP0:1 or GPU+NPU)
# to check how llama.cpp places the looped Nanbeige layers, with and without
# llama-looped-layer-device-map.patch. Prints per-device model/KV buffer sizes, graph splits and PPL.
#   usage: test_layer_split_rpc.sh <orig|patched> [gguf] [extra llama-perplexity args]
set -euo pipefail
V=${1:-orig}; M=${2:-/ksat/mobile/gguf/nanbeige42-3b-math-v1-s40-Q4_0-o8.gguf}
shift $(( $# < 2 ? $# : 2 ))
RPC="--rpc 127.0.0.1:50052,127.0.0.1:50053"; SUF=""
if [[ "${NO_RPC:-0}" == 1 ]]; then RPC=""; SUF="-cpuonly"; fi
docker run --rm --name "ksat-mobile-rpc-$V" -u "$(id -u):$(id -g)" -e HOME=/tmp -v /raid/cache/b300-07/ksat-rl:/ksat \
  ksat-rl/verl-sgl:v4 bash -c "
    B=/ksat/mobile/builds/x86-rpc-$V/bin
    \$B/ggml-rpc-server -H 127.0.0.1 -p 50052 -t 12 > /tmp/rpc0.log 2>&1 &
    \$B/ggml-rpc-server -H 127.0.0.1 -p 50053 -t 12 > /tmp/rpc1.log 2>&1 &
    sleep 2
    # plain text without special tokens, so the tokenizer patch cannot change the token sequence
    head -c 60000 /ksat/mobile/eval/calib_train_math.txt | sed -e 's/<|im_start|>//g; s/<|im_end|>//g; s/<think>//g; s#</think>##g' > /tmp/ppl.txt
    \$B/llama-perplexity -m $M -f /tmp/ppl.txt -c 512 --chunks 6 -b 512 -t 12 -ngl 99 -fa on \
       $RPC -lv 4 $* > /ksat/mobile/logs/rpc-split-$V$SUF.log 2>&1
    grep -i -E 'model buffer size|KV buffer size|graph splits|Final estimate|offloaded|error|failed' /ksat/mobile/logs/rpc-split-$V$SUF.log
    grep -E 'layer +[0-9]+ assigned to device' /ksat/mobile/logs/rpc-split-$V$SUF.log | sed -E 's/.*layer +([0-9]+) assigned to device ([^,]+),.*/\1:\2/' | tr '\n' ' '; echo
  "
