#!/usr/bin/env bash
# Op-coverage matrix for the Nanbeige quants on the simulated HTP session.
# usage: run_hexcov_matrix.sh [hexcov binary name] [out dir]
set -euo pipefail
BIN=${1:-hexcov}; OUT=${2:-/ksat/mobile/eval/hexcov}
G=/ksat/mobile/gguf/nanbeige42-3b-math-v1-s40
run() { # tag gguf args...
  local tag=$1 m=$2; shift 2
  docker run --rm --name "ksat-mobile-hexcov-$tag" -u "$(id -u):$(id -g)" -e HOME=/tmp -v /raid/cache/b300-07/ksat-rl:/ksat \
    ksat-rl/verl-sgl:v4 bash -c "/ksat/mobile/builds/tools/$BIN -m $m $* > $OUT/$tag.txt 2> $OUT/$tag.err; echo $tag exit=\$?"
}
mkdir -p /raid/cache/b300-07/ksat-rl/mobile/eval/hexcov
run q40o8-f16-ub128   $G-Q4_0-o8.gguf   -ub 128  --nhvx 4,6,8 --vtcm 8,4 &
run q40o8-f16-ub1024  $G-Q4_0-o8.gguf   -ub 1024 --nhvx 4,6 --vtcm 8 &
run q40o8-q80-ub128   $G-Q4_0-o8.gguf   -ub 128  -ctk q8_0 -ctv q8_0 --nhvx 4,6 --vtcm 8 &
run q40o8-q40-ub128   $G-Q4_0-o8.gguf   -ub 128  -ctk q4_0 -ctv q4_0 --nhvx 4 --vtcm 8 &
run q40o8-f16-nofa    $G-Q4_0-o8.gguf   -ub 128  -fa off --nhvx 4 --vtcm 8 &
wait
run q40-f16-ub128     $G-Q4_0.gguf      -ub 128  --nhvx 4 --vtcm 8 &
run q40pure-f16-ub128 $G-Q4_0-pure.gguf -ub 128  --nhvx 4 --vtcm 8 &
run q80-f16-ub128     $G-Q8_0.gguf      -ub 128  --nhvx 4 --vtcm 8 &
run iq4nl-f16-ub128   $G-IQ4_NL-o8.gguf -ub 128  --nhvx 4 --vtcm 8 &
run q4km-f16-ub128    $G-Q4_K_M.gguf    -ub 128  --nhvx 4 --vtcm 8 &
run q40a8o8-f16-ub128 $G-Q4_0-a8o8.gguf -ub 128  --nhvx 4 --vtcm 8 &
wait
