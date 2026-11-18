"""Rollout/training log-prob agreement: SGLang (rollout engine) vs training model on SGLang-generated tokens.

Generates a few real responses with SGLang (thinking mode, same chat template as RL), then scores the
exact token ids with the training model (packed, FA4 bf16) and (optionally) fp32. Reports per-token
|delta logp| and the distribution of the importance ratio exp(lp_train - lp_rollout), which is what
rollout-correction (TIS/MIS) acts on.
"""
import json, sys, urllib.request, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30500"
MODEL = "/models/Nanbeige4.2-3B-train"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
qs = ["함수 f(x)=x^3-6x^2+9x+1 의 극솟값을 구하시오. 최종 답을 \\boxed{} 안에 쓰시오.",
      "등차수열 {a_n}에서 a_3=7, a_7=19일 때 a_10의 값을 구하시오. 최종 답을 \\boxed{} 안에 쓰시오.",
      "주머니에 흰 공 3개, 검은 공 4개가 있다. 임의로 2개를 꺼낼 때 두 공의 색이 같을 확률은 p/q이다. p+q를 구하시오. 최종 답을 \\boxed{} 안에 쓰시오."]
def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())
samples = []
for q in qs:
    prompt_ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True)
    if not isinstance(prompt_ids, list): prompt_ids = list(prompt_ids["input_ids"])
    r = post("/generate", {"input_ids": prompt_ids, "sampling_params": {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 1500}, "return_logprob": True})
    out_lp = [x[0] for x in r["meta_info"]["output_token_logprobs"]]
    out_ids = [x[1] for x in r["meta_info"]["output_token_logprobs"]]
    samples.append((prompt_ids, out_ids, out_lp))
    print("gen", len(prompt_ids), len(out_ids), repr(tok.decode(out_ids[-60:]))[:120])
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
def score(dtype_tag):
    allp = []
    seqs = [torch.tensor(p + o) for p, o, _ in samples]
    ids = torch.cat(seqs)[None].cuda(); pos = torch.cat([torch.arange(len(s)) for s in seqs])[None].cuda()
    with torch.no_grad():
        logits = model(input_ids=ids, position_ids=pos).logits[0]
    lp = torch.log_softmax(logits.float(), -1)
    off = 0
    for (p, o, _), s in zip(samples, seqs):
        L = len(s); tgt = s[len(p):].cuda()
        allp.append(lp[off + len(p) - 1: off + L - 1].gather(-1, tgt[:, None]).squeeze(-1).cpu()); off += L
    return allp
res = {"bf16_fa4": score("bf16")}
model.float(); model.config._train_attn_backend = "sdpa"; res["fp32_sdpa"] = score("fp32")
for tag, allp in res.items():
    d = torch.cat([a - torch.tensor(s[2]) for a, s in zip(allp, samples)])
    ratio = d.exp()
    print(f"[{tag} vs sglang] tokens {d.numel()} |dlogp| mean {d.abs().mean():.4f} p99 {d.abs().quantile(0.99):.3f} max {d.abs().max():.3f} | "
          f"ratio p1 {ratio.quantile(0.01):.3f} p99 {ratio.quantile(0.99):.3f} frac(ratio>2) {(ratio>2).float().mean():.4f} | seq-level mean dlogp {d.mean():+.4f}")
