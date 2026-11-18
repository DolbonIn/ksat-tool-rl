#!/usr/bin/env bash
# Make a usable model dir from a veRL HF checkpoint (actor/huggingface).
# Why: veRL saves the FSDP-wrapped class, so transformers' custom_object_save copies torch FSDP internals
# (_fsdp_*.py) instead of modeling_nanbeige_train.py, and only a minimal tokenizer is written.
# Result: models/<name> with RELATIVE symlinks (../../runs/...) to the checkpoint weights/config — these resolve
# on the host and in containers that mount models at /models and runs at /runs — plus our training modeling
# file (GRPO/SFT restarts) and the original tokenizer files (SGLang eval).
set -euo pipefail
CKPT=${1:?path under /raid/cache/b300-07/ksat-rl/runs/.../actor/huggingface}; NAME=${2:?output model name}
R=/raid/cache/b300-07/ksat-rl; REL=${CKPT#$R/runs/}
[ "$REL" != "$CKPT" ] || { echo "checkpoint must live under $R/runs"; exit 1; }
docker run --rm -v $R:/k --entrypoint bash ksat-rl/verl-sgl:v4 -c "
set -e; O=/k/models/$NAME; rm -rf \$O; mkdir -p \$O; cd \$O
for f in /k/runs/$REL/*.safetensors /k/runs/$REL/model.safetensors.index.json /k/runs/$REL/config.json /k/runs/$REL/generation_config.json; do
  ln -s ../../runs/$REL/\$(basename \$f) \$(basename \$f); done
cp /k/models/Nanbeige4.2-3B-train-fused/modeling_nanbeige_train.py /k/models/Nanbeige4.2-3B/configuration_nanbeige.py .
for f in tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json added_tokens.json; do cp /k/models/Nanbeige4.2-3B/\$f .; done
chown -R 1001:1001 \$O; test -e config.json && echo OK"
