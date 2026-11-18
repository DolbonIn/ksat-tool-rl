#!/usr/bin/env bash
# Produce the mobile quant variants of the RL model from the F16 GGUF with the pinned llama-quantize.
# The F16 source already carries tokenizer.ggml.add_bos_token=false (set_add_bos_false.sh), which the
# quants inherit.
#   usage: quantize_all.sh            (all variants evaluated in README §3)
#          run_imat.sh                (imatrix variants; needs gguf/imatrix-train-math.gguf from llama-imatrix)
# Naming: -o8 = output.weight (lm_head) Q8_0, -a8 = attn_q/k/v/output Q8_0, -d8 = ffn_down Q8_0,
#         -pure = one type everywhere (no llama-quantize mixture rules).
#   Q8_0          all Q8_0 (reference, ~lossless; HTP-resident 3.6 GiB > one NPU session's address window)
#   Q6_K-pure-o8  Q6_K body, lm_head Q8_0
#   Q5_K-pure-o8  Q5_K body, lm_head Q8_0                       <- recommended (quality)
#   Q4_0-a8o8     Q4_0 FFN, attention + lm_head Q8_0            <- recommended (simplest HTP kernels)
#   IQ4_NL-a8o8   IQ4_NL FFN, attention + lm_head Q8_0
#   Q4_0-d8o8     Q4_0, ffn_down + lm_head Q8_0
#   IQ4_NL-o8     IQ4_NL mixture, lm_head Q8_0 (attn_v becomes Q5_K -> breaks QKV fusion on HTP)
#   Q4_0-o8       Q4_0, lm_head Q8_0
#   Q4_0          llama-quantize default mixture (lm_head Q6_K -> CPU fallback on Adreno 7xx)
#   Q4_0-pure     everything Q4_0
#   Q4_K_M        reference K-quant mixture
set -euo pipefail
G=/ksat/mobile/gguf
SRC=$G/nanbeige42-3b-math-v1-s40-F16.gguf
P=$G/nanbeige42-3b-math-v1-s40
A8=(--tensor-type attn_q=q8_0 --tensor-type attn_k=q8_0 --tensor-type attn_v=q8_0 --tensor-type attn_output=q8_0)
run() { # out type [args...]
  local out=$1 type=$2; shift 2
  docker run --rm --name "ksat-mobile-quant-$(basename $out .gguf | tr 'A-Z_.' 'a-z--' | tail -c 40)" -u "$(id -u):$(id -g)" \
    -v /raid/cache/b300-07/ksat-rl:/ksat ksat-rl/verl-sgl:v4 \
    /ksat/mobile/builds/x86-cpu/bin/llama-quantize "$@" "$SRC" "$out" "$type" 16 2>&1 | grep -E "^llama_model_quantize_impl: quant size|rror"
}
run $P-Q8_0.gguf         Q8_0
run $P-Q6_K-pure-o8.gguf Q6_K   --pure --output-tensor-type q8_0
run $P-Q5_K-pure-o8.gguf Q5_K   --pure --output-tensor-type q8_0
run $P-Q4_0-a8o8.gguf    Q4_0   --output-tensor-type q8_0 "${A8[@]}"
run $P-IQ4_NL-a8o8.gguf  IQ4_NL --output-tensor-type q8_0 "${A8[@]}"
run $P-Q4_0-d8o8.gguf    Q4_0   --output-tensor-type q8_0 --tensor-type ffn_down=q8_0
run $P-IQ4_NL-o8.gguf    IQ4_NL --output-tensor-type q8_0
run $P-Q4_0-o8.gguf      Q4_0   --output-tensor-type q8_0
run $P-Q4_0.gguf         Q4_0
run $P-Q4_0-pure.gguf    Q4_0   --pure
run $P-Q4_K_M.gguf       Q4_K_M
