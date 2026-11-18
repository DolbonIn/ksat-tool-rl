"""Chat prompts (generation prompt only, thinking on) for the short greedy-generation smoke test."""
import json, sys
from transformers import AutoTokenizer
sys.path.insert(0, "/repo/eval")
from make_seqs import CHAT_Q

Q2 = (r"다음은 수학 객관식 문제입니다. 단계적으로 풀이한 뒤, 마지막에 정답 선택지 번호(1~5 중 하나)만 \boxed{} 안에 넣으세요. 예: \boxed{3}"
      "\n\n" r"$\log_2 8 + \log_3 \frac{1}{9}$의 값은?" "\n" r"① $-1$ ② $0$ ③ $1$ ④ $2$ ⑤ $3$")
tok = AutoTokenizer.from_pretrained("/ksat/models/math_v1_step40")
out = []
for name, q in [("gen_arith", CHAT_Q), ("gen_log", Q2)]:
    p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids = tok(p, add_special_tokens=False)["input_ids"]
    out.append(dict(name=name, n_prompt=len(ids), ids=ids))
with open("/ksat/mobile/eval/seqs/seqs_gen.txt", "w") as f:
    for s in out:
        f.write(f"{s['name']} {s['n_prompt']} " + " ".join(map(str, s["ids"])) + "\n")
json.dump(out, open("/ksat/mobile/eval/seqs/seqs_gen.json", "w"))
print([len(s["ids"]) for s in out], repr(p[-40:]))
