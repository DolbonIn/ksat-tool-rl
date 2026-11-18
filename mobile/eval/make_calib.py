"""Calibration text for llama-imatrix: chat-formatted solutions to *training* problems (never the 2026
held-out set used for evaluation). Output stays under /raid (problem text is not redistributable)."""
import json, random, sys
from transformers import AutoTokenizer
sys.path.insert(0, "/rl")
from ksat_rl.make_parquet import instruction

tok = AutoTokenizer.from_pretrained("/ksat/models/math_v1_step40")
probs = {p["id"]: p for p in map(json.loads, open("/ksat/data/ksat_math/train.jsonl"))}
rolls = [r for r in map(json.loads, open("/ksat/runs/eval/passrate_train.jsonl")) if r["correct"] and r["finish"] == "stop"]
random.Random(1).shuffle(rolls)
seen, out, n_tok = set(), [], 0
for r in rolls:
    if r["id"] in seen or r["id"] not in probs:
        continue
    seen.add(r["id"])
    p = probs[r["id"]]
    msgs = [{"role": "user", "content": f"{instruction(p)}\n\n{p['text']}"}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    s += r["reasoning"].strip() + "\n</think>\n\n" + r["content"].strip() + "<|im_end|>\n"
    ids = tok(s, add_special_tokens=False)["input_ids"][:2500]   # cap per solution for problem diversity
    out.append(tok.decode(ids) + "\n")
    n_tok += len(ids)
    if n_tok > 160_000:
        break
open("/ksat/mobile/eval/calib_train_math.txt", "w").write("".join(out))
print(len(out), "solutions,", n_tok, "tokens")
