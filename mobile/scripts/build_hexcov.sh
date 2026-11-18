#!/usr/bin/env bash
# Build the x86 Hexagon op-coverage simulator (mobile/tools/hexcov.cpp) from the unmodified host-side
# Hexagon sources (htp-kparams/fusion/opnode/repack.cpp) of a llama.cpp tree.
#   usage: build_hexcov.sh [llama.cpp dir under /raid/cache/b300-07/ksat-rl, default: pinned GenieX copy] [out name]
set -euo pipefail
SRC=${1:-/ksat/mobile/GenieX/third-party/llama.cpp}
OUT=${2:-hexcov}
docker run --rm --name ksat-mobile-hexcov-build -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /raid/cache/b300-07/ksat-rl:/ksat -v /home/b300-07/ksat-tool-rl/mobile:/repo:ro ksat-rl/verl-sgl:v4 bash -c "
    set -e
    L=$SRC; B=/ksat/mobile/builds/x86-cpu; H=\$L/ggml/src/ggml-hexagon
    mkdir -p /ksat/mobile/builds/tools
    g++ -O1 -g -std=c++17 -w -D__fp16=_Float16 -I\$L/include -I\$L/ggml/include -I\$L/ggml/src -I\$H -I\$H/htp \
        /repo/tools/hexcov.cpp \$H/htp-kparams.cpp \$H/htp-fusion.cpp \$H/htp-opnode.cpp \
        -L\$B/bin -lllama -lggml -lggml-base -Wl,-rpath,\$B/bin -o /ksat/mobile/builds/tools/$OUT
    echo built $OUT"
