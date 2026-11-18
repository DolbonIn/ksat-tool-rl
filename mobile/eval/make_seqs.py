"""Build the token-id sequences used for the GGUF-vs-HF correctness check and the quant study.

Runs inside ksat-rl/verl-sgl:v4 (transformers tokenizer). Writes, under --out:
  seqs_<set>.txt   one line per sequence: "<name> <n_prompt> <ids...>" (consumed by nb_logprobs)
  seqs_<set>.json  same content as JSON (consumed by hf_logprobs.py)

Sets
  check  : hand-written Korean math with LaTeX (raw text), a chat-templated prompt + thinking-mode answer,
           an English paragraph, and 3 on-distribution rollouts (truncated) -- for HF comparison.
  quant  : N held-out math solutions written by the RL model itself (2026 CSAT eval rollouts, which were
           never trained on), chat-formatted, assistant part scored -- for quant / KV-cache KL.
The rollout and problem text stays under /raid (never copied into the repo).
"""
import argparse, json, os, random
from transformers import AutoTokenizer

KO_MATH = r"""문제. 함수 $f(x)=x^{3}-3x^{2}+2$에 대하여 닫힌구간 $[0, 3]$에서 $f(x)$의 최댓값과 최솟값의 합을 구하시오.

풀이. $f'(x)=3x^{2}-6x=3x(x-2)$이므로 $f'(x)=0$에서 $x=0$ 또는 $x=2$이다. 구간의 양 끝과 임계점에서의 함숫값을 비교하면
$$f(0)=2,\quad f(2)=8-12+2=-2,\quad f(3)=27-27+2=2$$
이다. 따라서 최댓값은 $2$, 최솟값은 $-2$이고 그 합은 $0$이다.

참고로 정적분 $\displaystyle\int_{0}^{3} f(x)\,dx=\left[\frac{x^{4}}{4}-x^{3}+2x\right]_{0}^{3}=\frac{81}{4}-27+6=-\frac{3}{4}$이고,
수열 $a_{n}=\dfrac{1}{n(n+1)}$의 합은 $\displaystyle\sum_{k=1}^{n}a_{k}=1-\frac{1}{n+1}$이므로 $\displaystyle\lim_{n\to\infty}\sum_{k=1}^{n}a_{k}=1$이다."""

CHAT_Q = (r"다음은 수학 단답형 문제입니다. 단계적으로 풀이한 뒤, 마지막에 최종 답(0 이상 999 이하의 정수)만 \boxed{} 안에 넣으세요."
          "\n\n" r"등차수열 $\{a_n\}$에 대하여 $a_3 = 7$, $a_7 = 19$일 때, $\displaystyle\sum_{k=1}^{10} a_k$의 값을 구하시오.")
CHAT_REASONING = (r"공차를 $d$라 하면 $a_7 - a_3 = 4d = 12$이므로 $d = 3$이다. 그러면 $a_1 = a_3 - 2d = 7 - 6 = 1$이다."
                  "\n" r"따라서 $a_k = 1 + 3(k-1) = 3k - 2$이고, $\sum_{k=1}^{10} a_k = \frac{10(a_1 + a_{10})}{2} = 5(1 + 28) = 145$이다."
                  "\n" r"검산: $a_{10} = 28$, 합 $= 1+4+7+\cdots+28$, 항의 개수 10, 평균 $14.5$, 합 $145$. 맞다.")
CHAT_ANSWER = (r"등차수열의 공차는 $d=3$, 첫째항은 $a_1=1$입니다." "\n\n"
               r"$$\sum_{k=1}^{10} a_k = \frac{10(1+28)}{2} = 145$$" "\n\n" r"\boxed{145}")

EN_TEXT = ("The Hexagon tensor processor executes matrix multiplications on a systolic array while vector "
           "threads dequantize weights streamed from DDR into tightly coupled memory. Because a looped "
           "transformer reuses the same physical layers twice, the second pass could in principle find its "
           "weights already resident, but only if the working set fits in the on-chip memory budget.")


def chat_ids(tok, problem_text, reasoning, content, instr=None):
    user = f"{instr}\n\n{problem_text}" if instr else problem_text
    msgs = [{"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    full = prompt + reasoning.strip() + "\n</think>\n\n" + content.strip() + "<|im_end|>"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    assert f_ids[:len(p_ids)] == p_ids, "prompt tokenization is not a prefix"
    return p_ids, f_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/ksat/models/math_v1_step40")
    ap.add_argument("--rollouts", default="/ksat/runs/eval/final_step40_n16.jsonl")
    ap.add_argument("--problems", default="/ksat/data/ksat_math/test_2026.jsonl")
    ap.add_argument("--out", default="/ksat/mobile/eval/seqs")
    ap.add_argument("--n-quant", type=int, default=24)
    ap.add_argument("--max-check", type=int, default=1024)
    ap.add_argument("--max-quant", type=int, default=1536)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    import sys
    sys.path.insert(0, "/rl")
    from ksat_rl.make_parquet import instruction

    probs = {p["id"]: p for p in map(json.loads, open(a.problems))}
    rolls = [r for r in map(json.loads, open(a.rollouts)) if r["finish"] == "stop" and r["correct"]]
    rng = random.Random(0)
    rng.shuffle(rolls)

    def emit(name, seqs):
        with open(f"{a.out}/seqs_{name}.txt", "w") as f:
            for s in seqs:
                f.write(f"{s['name']} {s['n_prompt']} " + " ".join(map(str, s["ids"])) + "\n")
        json.dump(seqs, open(f"{a.out}/seqs_{name}.json", "w"))
        print(name, len(seqs), "seqs,", sum(len(s["ids"]) - max(s["n_prompt"] - 1, 0) - 1 for s in seqs), "scored positions")

    check = []
    ids = tok(KO_MATH, add_special_tokens=False)["input_ids"]
    check.append(dict(name="ko_math_latex", n_prompt=1, ids=ids))
    p, f = chat_ids(tok, CHAT_Q, CHAT_REASONING, CHAT_ANSWER)
    check.append(dict(name="chat_think", n_prompt=len(p), ids=f))
    ids = tok(EN_TEXT, add_special_tokens=False)["input_ids"]
    check.append(dict(name="en_text", n_prompt=1, ids=ids))
    used = set()
    for r in rolls:
        if len([s for s in check if s["name"].startswith("rollout")]) >= 3:
            break
        if r["id"] in used:
            continue
        used.add(r["id"])
        pr = probs[r["id"]]
        p, f = chat_ids(tok, pr["text"], r["reasoning"], r["content"], instruction(pr))
        check.append(dict(name=f"rollout_{len(used)}", n_prompt=len(p), ids=f[:a.max_check]))
    emit("check", check)

    quant, seen = [], {}
    for r in rolls:
        if len(quant) >= a.n_quant:
            break
        if seen.get(r["id"], 0) >= 1:     # one rollout per problem
            continue
        seen[r["id"]] = seen.get(r["id"], 0) + 1
        pr = probs[r["id"]]
        p, f = chat_ids(tok, pr["text"], r["reasoning"], r["content"], instruction(pr))
        quant.append(dict(name=f"q{len(quant):02d}", n_prompt=len(p), ids=f[:len(p) + a.max_quant]))
    emit("quant", quant)

    # long-context probe for KV-cache quantization: a few full-length rollouts (assistant part scored)
    longs = sorted([r for r in rolls if 6000 <= r["tokens"] <= 12000], key=lambda r: r["tokens"])[:4]
    lseqs = []
    for i, r in enumerate(longs):
        pr = probs[r["id"]]
        p, f = chat_ids(tok, pr["text"], r["reasoning"], r["content"], instruction(pr))
        lseqs.append(dict(name=f"long{i}", n_prompt=len(p), ids=f))
    emit("long", lseqs)


if __name__ == "__main__":
    main()
