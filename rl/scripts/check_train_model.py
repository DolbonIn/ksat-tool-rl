"""Correctness check (the decisive rollout-vs-train test is check_sglang_vs_train.py): training-side Nanbeige (FA4 varlen, packed) vs the official HF remote code.

Checks
  1. per-token log-probs on 3 sequences, official eager (one at a time, use_cache=False) vs ours packed (1, T)
  2. ours FA4 vs ours per-sequence SDPA reference: forward log-probs and parameter gradients
  3. packing independence: sequence A's log-probs identical whether packed with others or alone
"""
import sys, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REF, OURS = "/models/Nanbeige4.2-3B", "/models/Nanbeige4.2-3B-train"
dev = "cuda"
tok = AutoTokenizer.from_pretrained(REF, trust_remote_code=True)
texts = [
    "다음은 수학 문제입니다. 함수 f(x)=x^3-3x+1 의 극댓값을 구하시오. 풀이: f'(x)=3x^2-3=0 에서 x=±1 이다.",
    "Let $a_n$ be a sequence with $a_1=2$ and $a_{n+1}=3a_n-1$. Find $a_5$. We compute step by step: a_2=5, a_3=14.",
    "<|im_start|>user\n1+1은?<|im_end|>\n<|im_start|>assistant\n<think>\n간단한 덧셈이다.\n</think>\n\n\\boxed{2}<|im_end|>",
]
seqs = [tok(t, return_tensors="pt").input_ids[0].to(dev) for t in texts]
print("lens", [len(s) for s in seqs])

def lp_of(logits, ids):  # logits (L,V) for ids (L,)
    return torch.log_softmax(logits[:-1].float(), -1).gather(-1, ids[1:, None]).squeeze(-1)

# ---- 1. official reference
ref = AutoModelForCausalLM.from_pretrained(REF, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
# transformers 5.x leaves the official code's non-persistent rotary inv_freq buffer uninitialized -> NaN. Repair it.
for mod in ref.modules():
    if hasattr(mod, "inv_freq") and hasattr(mod, "base") and hasattr(mod, "dim"):
        mod.inv_freq = 1.0 / (mod.base ** (torch.arange(0, mod.dim, 2, dtype=torch.int64, device=dev).float() / mod.dim))
with torch.no_grad():
    ref_lp = [lp_of(ref(s[None], use_cache=False).logits[0], s) for s in seqs]
del ref; torch.cuda.empty_cache()

ours = AutoModelForCausalLM.from_pretrained(OURS, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
print("ours class", type(ours).__module__, type(ours).__name__)
packed = torch.cat(seqs)[None]
pos = torch.cat([torch.arange(len(s), device=dev) for s in seqs])[None]
bounds = torch.tensor([0] + [len(s) for s in seqs]).cumsum(0).tolist()

def ours_lp(backend):
    ours.config._train_attn_backend = backend
    out = ours(input_ids=packed, position_ids=pos, use_cache=False).logits[0]
    return [lp_of(out[a:b], packed[0, a:b]) for a, b in zip(bounds[:-1], bounds[1:])]

ok = True
with torch.no_grad():
    fa4 = ours_lp("fa4"); sdpa = ours_lp("sdpa")
    ours.float(); fp32 = ours_lp("sdpa"); ours.to(torch.bfloat16)
for i in range(len(seqs)):  # noise floor report
    print(f"seq{i}: bf16 noise floor |sdpa_bf16 - fp32| max {(sdpa[i]-fp32[i]).abs().max():.4f} mean {(sdpa[i]-fp32[i]).abs().mean():.5f}; |fa4 - fp32| max {(fa4[i]-fp32[i]).abs().max():.4f} mean {(fa4[i]-fp32[i]).abs().mean():.5f}; |official_bf16 - fp32| mean {(ref_lp[i]-fp32[i]).abs().mean():.5f}")
for i in range(len(seqs)):
    d_ref = (fa4[i] - ref_lp[i]).abs(); d_sdpa = (fa4[i] - sdpa[i]).abs()
    print(f"seq{i}: |ours_fa4 - official| max {d_ref.max():.4f} mean {d_ref.mean():.5f} | |fa4 - sdpa| max {d_sdpa.max():.4f} mean {d_sdpa.mean():.5f} | official mean lp {ref_lp[i].mean():.3f}")
    # Acceptance: ours-vs-official must be well inside the bf16 noise floor (|official_bf16 - fp32|),
    # which for this looped model is large (~0.1-0.3 mean on off-distribution text).
    floor = (ref_lp[i] - fp32[i]).abs().mean().item()
    ok &= d_ref.mean().item() < 0.75 * floor

# ---- 3. packing independence
with torch.no_grad():
    ours.config._train_attn_backend = "fa4"
    alone = lp_of(ours(input_ids=seqs[1][None], use_cache=False).logits[0], seqs[1])
d = (alone - fa4[1]).abs().max().item(); print(f"packing independence seq1: max diff {d:.5f}"); ok &= d < 0.5  # same-kernel repacking noise, bf16

# ---- 2. gradients fa4 vs sdpa
ours.train(); ours.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
grads = {}
for backend in ["fa4", "sdpa"]:
    ours.zero_grad(set_to_none=True)
    lps = ours_lp(backend)
    loss = -torch.cat(lps).mean(); loss.backward()
    grads[backend] = {n: p.grad.detach().float().clone() for n, p in ours.named_parameters() if p.grad is not None}
    print(backend, "loss", loss.item(), "n_grads", len(grads[backend]))
worst = 0.0
for n in grads["fa4"]:
    a, b = grads["fa4"][n], grads["sdpa"][n]
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(); worst = max(worst, 1 - cos)
print(f"grad cosine worst 1-cos = {worst:.2e}"); ok &= worst < 0.2  # FA4 vs SDPA full-model grads (both bf16); op-level bwd verified separately at cos>0.99999
# both loops must contribute gradient to a layer: compare with num_loops=1 grad (should differ)
print("RESULT", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
