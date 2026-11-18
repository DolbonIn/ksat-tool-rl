#!/usr/bin/env bash
# Run nb_logprobs in a ksat-mobile-* container.  usage: run_lp.sh <tag> <gguf> <seqs.txt> <outdir> [extra nb_logprobs args]
set -euo pipefail
TAG=$1; GGUF=$2; SEQS=$3; OUT=$4; shift 4
docker run --rm --name "ksat-mobile-lp-$TAG" -u "$(id -u):$(id -g)" -e HOME=/tmp -e LLAMA_ATTN_ROT_DISABLE="${LLAMA_ATTN_ROT_DISABLE:-0}" \
  -v /raid/cache/b300-07/ksat-rl:/ksat ksat-rl/verl-sgl:v4 bash -c \
  "mkdir -p $OUT && SECONDS=0 && /ksat/mobile/builds/tools/nb_logprobs -m $GGUF -i $SEQS -o $OUT $* > $OUT/run.log 2>&1; echo exit=\$? secs=\$SECONDS >> $OUT/run.log; tail -3 $OUT/run.log"
