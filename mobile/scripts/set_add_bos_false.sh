#!/usr/bin/env bash
# Nanbeige's chat template already starts every turn with <|im_start|> (= the BOS id). With the converter's
# add_bos_token=True, llama.cpp strips that literal from the rendered template and re-adds it as BOS, which
# makes the following "system" the first text fragment and (with the SPM prepend fix) gives it a U+2581
# prefix the model never saw. Training/eval (SGLang, veRL) used apply_chat_template + add_special_tokens=False,
# i.e. no extra BOS, so the device GGUFs carry add_bos_token=False. In-place metadata edit, no tensor change.
#   usage: set_add_bos_false.sh file.gguf [...]
set -euo pipefail
for f in "$@"; do
  docker run --rm --name ksat-mobile-setmeta -u "$(id -u):$(id -g)" -e HOME=/tmp -v /raid/cache/b300-07/ksat-rl:/ksat \
    ksat-rl/verl-sgl:v4 bash -c "cd /tmp && PYTHONPATH=/ksat/mobile/GenieX/third-party/llama.cpp/gguf-py python -m gguf.scripts.gguf_set_metadata --force $f tokenizer.ggml.add_bos_token ''" 2>&1 | grep -E "change|already" || true
done
