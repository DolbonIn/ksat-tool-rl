"""Check that the device path tokenizes our eval prompts exactly like training/eval did.

For every 2026 CSAT math prompt (instruction + problem, same as eval_csat.py) render the chat template
and tokenize it (a) with the HF tokenizer (apply_chat_template + add_special_tokens=False, which is what
SGLang/veRL fed the model) and (b) through the GGUF exactly like GenieX's llama_cpp plugin
(chat_tok_check: common_chat_session + common_tokenize(add_special=true, parse_special=true)).
Prints mismatches; problem text is read from /raid at run time and never written to the repo.
"""
import json, subprocess, sys
from transformers import AutoTokenizer
sys.path.insert(0, "/rl")
from ksat_rl.make_parquet import instruction

GGUF = sys.argv[1] if len(sys.argv) > 1 else "/ksat/mobile/gguf/nanbeige42-3b-math-v1-s40-Q4_0-o8.gguf"
TOOL = sys.argv[2] if len(sys.argv) > 2 else "/ksat/mobile/builds/tools/chat_tok_check"
tok = AutoTokenizer.from_pretrained("/ksat/models/math_v1_step40")
probs = [json.loads(l) for l in open("/ksat/data/ksat_math/test_2026.jsonl")]
n_bad = 0
for p in probs:
    user = f"{instruction(p)}\n\n{p['text']}"
    hf = tok(tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True), add_special_tokens=False)["input_ids"]
    out = subprocess.run([TOOL, GGUF, user, "1"], capture_output=True, text=True).stdout
    lc = [int(x) for x in out.split("IDS", 1)[1].split()]
    if lc != hf:
        n_bad += 1
        i = next((k for k in range(min(len(lc), len(hf))) if lc[k] != hf[k]), min(len(lc), len(hf)))
        print(f"MISMATCH {p['id']}: len hf {len(hf)} lc {len(lc)} first diff at {i}: hf {hf[i:i+5]} lc {lc[i:i+5]}")
print(f"{len(probs) - n_bad}/{len(probs)} prompts tokenize identically (first ids lc {lc[:4]} hf {hf[:4]})")

# raw-text edge cases for the SPM prepend rule (HF Metaspace prepend_scheme="first")
RAW = ["hello world", " leading space", "\n\nx", "a<|im_end|>b c", "<think>\nok</think>\n\nans", "함수 $f(x)=x^2$의 최댓값"]
n_raw_bad = 0
probe = subprocess.run([TOOL, GGUF, "x", "raw"], capture_output=True, text=True).stdout
gguf_adds_bos = int(probe.split("IDS", 1)[1].split()[0]) == tok.bos_token_id
for t in RAW:
    hf = ([tok.bos_token_id] if gguf_adds_bos else []) + tok(t, add_special_tokens=False)["input_ids"]
    out = subprocess.run([TOOL, GGUF, t, "raw"], capture_output=True, text=True).stdout
    lc = [int(x) for x in out.split("IDS", 1)[1].split()]
    ok = lc == hf
    n_raw_bad += not ok
    print(("ok  " if ok else "BAD ") + repr(t), "hf", hf, "lc", lc)
print(f"raw text (GGUF add_bos={gguf_adds_bos}): {len(RAW) - n_raw_bad}/{len(RAW)} identical")
