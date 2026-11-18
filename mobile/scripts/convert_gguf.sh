#!/usr/bin/env bash
# Convert HF Nanbeige4.2 checkpoints to GGUF with the llama.cpp converter pinned in GenieX.
# Runs inside ksat-rl/verl-sgl:v4 (torch/transformers/sentencepiece; CPU only) with
# /raid/cache/b300-07/ksat-rl mounted at /ksat. The model dirs hold relative symlinks into
# ../../runs, so the whole ksat-rl tree must be mounted. The RL checkpoint shards are written by
# the (root) trainer container with mode 0600, so the converter runs as root and hands the output
# back to the calling user with chown; the checkpoint itself is only read.
#   usage (host): mobile/scripts/convert_gguf.sh <model-name> <outtype: f16|bf16|f32> [tag]
set -euo pipefail
MODEL=${1:?model dir name under /raid/cache/b300-07/ksat-rl/models}
OUTTYPE=${2:-f16}
TAG=${3:-$MODEL}
ROOT=/raid/cache/b300-07/ksat-rl
docker run --rm --name "ksat-mobile-convert-${TAG}-${OUTTYPE}" -e HOME=/tmp \
  -v "$ROOT:/ksat" -w /ksat/mobile ksat-rl/verl-sgl:v4 bash -c "
    set -e
    LCPP=/ksat/mobile/GenieX/third-party/llama.cpp
    PYTHONPATH=\$LCPP/gguf-py python \$LCPP/convert_hf_to_gguf.py /ksat/models/$MODEL \
      --outtype $OUTTYPE --outfile /ksat/mobile/gguf/${TAG}-${OUTTYPE^^}.gguf
    chown $(id -u):$(id -g) /ksat/mobile/gguf/${TAG}-${OUTTYPE^^}.gguf"
