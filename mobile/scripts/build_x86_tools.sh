#!/usr/bin/env bash
# Build the host-side helper tools (mobile/tools/*.cpp) against the x86 CPU build of the pinned
# llama.cpp (builds/x86-cpu, made by `cmake -S GenieX/third-party/llama.cpp -B builds/x86-cpu`).
set -euo pipefail
ROOT=/raid/cache/b300-07/ksat-rl
REPO=/home/b300-07/ksat-tool-rl/mobile
docker run --rm --name ksat-mobile-tools -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$ROOT:/ksat" -v "$REPO:/repo:ro" ksat-rl/verl-sgl:v4 bash -c '
    set -e
    L=/ksat/mobile/GenieX/third-party/llama.cpp
    B=/ksat/mobile/builds/x86-cpu
    mkdir -p /ksat/mobile/builds/tools
    g++ -O2 -std=c++17 -I$L/include -I$L/ggml/include /repo/tools/nb_logprobs.cpp \
        -L$B/bin -lllama -lggml -lggml-base -Wl,-rpath,$B/bin -o /ksat/mobile/builds/tools/nb_logprobs
    echo built nb_logprobs
    g++ -O2 -std=c++17 -I$L/include -I$L/ggml/include -I$L/common -I$L/vendor /repo/tools/chat_tok_check.cpp \
        -L$B/bin -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,$B/bin -o /ksat/mobile/builds/tools/chat_tok_check
    echo built chat_tok_check'
