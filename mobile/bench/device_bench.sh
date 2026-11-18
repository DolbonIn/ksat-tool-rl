#!/usr/bin/env bash
# Speed benchmark for Nanbeige4.2-3B (RL, math_v1_step40) GGUFs on a Snapdragon phone over adb.
#
#   mobile/bench/device_bench.sh push   <pkg-dir> <gguf> [more ggufs...]   # one-time: libs, tools, models
#   mobile/bench/device_bench.sh run    <gguf-basename> [out-dir]          # pp/tg sweep, NPU / GPU / CPU
#
# What "run" measures (results land in <out-dir>, default ./bench-<model>-<date>):
#   1. geniex-bench (GenieX C API, same binary/JSON schema as notes/bench.md / QDC):
#      prompt 512/1024/4096 random tokens + 128 generated, 1 warmup + 3 reps, per compute unit.
#   2. llama-bench (pinned llama.cpp) decode speed *at depth*: tg64 after a KV cache of 0/4k/8k/16k tokens
#      (-d), which is what a 10-30k-token math solution actually experiences, for F16 and Q8_0 KV.
# Environment knobs (all optional):
#   SERIAL=<adb serial>  COMPUTES="npu gpu cpu"  POWER=sustained_high_performance  THREADS=6
#   KV="f16 q8_0"  HEX_DEVICES=3  (3 virtual NPU sessions HTP0..HTP2, see README §5.3)
set -euo pipefail
ADB=(adb ${SERIAL:+-s "$SERIAL"})
DEV=/data/local/tmp/geniex
ENVS="cd $DEV && LD_LIBRARY_PATH=./lib:./lib/llama_cpp ADSP_LIBRARY_PATH=./lib/llama_cpp GENIEX_PLUGIN_PATH=./lib"

cmd=${1:-}; shift || true
case "$cmd" in
push)
  PKG=${1:?pkg dir (artifacts/android-arm64)}; shift
  "${ADB[@]}" shell mkdir -p $DEV/models
  "${ADB[@]}" push "$PKG"/. $DEV/
  for g in "$@"; do "${ADB[@]}" push "$g" $DEV/models/; done
  "${ADB[@]}" shell "chmod 755 $DEV/bin/*"
  ;;
run)
  M=${1:?gguf basename under $DEV/models}; OUT=${2:-bench-${M%.gguf}-$(date +%Y%m%d-%H%M)}
  mkdir -p "$OUT"
  POWER=${POWER:-sustained_high_performance}; THREADS=${THREADS:-6}; COMPUTES=${COMPUTES:-"npu gpu cpu"}
  "${ADB[@]}" shell "getprop ro.soc.model; getprop ro.product.model; cat /proc/meminfo | head -3" > "$OUT/device.txt" || true
  DEVIDS=""
  if [[ "${HEX_DEVICES:-}" =~ ^[0-9]+$ ]]; then DEVIDS=$(seq -s, -f 'HTP%g' 0 $((HEX_DEVICES - 1))); fi
  for c in $COMPUTES; do
    extra=""; [[ $c == npu && -n $DEVIDS ]] && extra="--device-id $DEVIDS"
    for p in 512 1024 4096; do
      "${ADB[@]}" shell "$ENVS GENIEX_LLAMA_CACHE_TYPE_K=q8_0 GENIEX_LLAMA_CACHE_TYPE_V=q8_0 \
        ${HEX_DEVICES:+GGML_HEXAGON_DEVICES=$HEX_DEVICES} ./bin/geniex-bench --plugin llama_cpp --device $c \
        -m models/$M -p $p -n 128 -c $((p + 512)) -r 3 --power-mode $POWER -t $THREADS $extra \
        --output-json $DEV/bench_${c}_p${p}.json" | tee "$OUT/geniex_${c}_p${p}.txt"
      "${ADB[@]}" pull $DEV/bench_${c}_p${p}.json "$OUT/" >/dev/null || true
    done
  done
  # decode at depth with llama-bench: -dev HTP0 (NPU) / GPUOpenCL / none (CPU)
  for kv in ${KV:-"f16 q8_0"}; do
    for dev in HTP0 GPUOpenCL none; do
      ngl=99; [[ $dev == none ]] && ngl=0
      fa=on; [[ $dev == GPUOpenCL && $kv == f16 ]] && fa=off   # GenieX default for Adreno: FA off with F16 KV
      "${ADB[@]}" shell "$ENVS GGML_HEXAGON_POWER_MODE=$POWER ./bin/llama-bench -m models/$M \
        -dev $dev -ngl $ngl -fa $fa -ctk $kv -ctv $kv -t $THREADS -p 1024 -n 64 -d 0,4096,8192,16384 \
        -ub 1024 -r 2 -o md" | tee "$OUT/llamabench_${dev}_kv${kv}.md"
    done
  done
  echo "results in $OUT"
  ;;
*)
  sed -n '2,20p' "$0"; exit 1;;
esac
