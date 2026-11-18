"""HF reference log-probs for Nanbeige4.2 (CPU), same output format as nb_logprobs.

Uses our clean implementation (modeling_nanbeige_train.py, SDPA path) because the official remote
code is broken on transformers 5.x (uninitialised RoPE inv_freq -> NaN).
  python hf_logprobs.py --model /ksat/models/math_v1_step40 --dtype fp32 --seqs seqs_check.json --out DIR [--full]
"""
import argparse, json, os, sys, time
import numpy as np
import torch


def load(model_dir, dtype):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    cfg._train_attn_backend = "sdpa"
    if "modeling_nanbeige_train" not in json.dumps(getattr(cfg, "auto_map", {})):
        # original checkpoint: its auto_map points at the broken remote code -> use the train dir's code
        cfg.auto_map = {"AutoConfig": "configuration_nanbeige.NanbeigeConfig",
                        "AutoModelForCausalLM": "modeling_nanbeige_train.NanbeigeForCausalLM"}
        code_dir = "/ksat/models/Nanbeige4.2-3B-train"
    else:
        code_dir = model_dir
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cls = get_class_from_dynamic_module("modeling_nanbeige_train.NanbeigeForCausalLM", code_dir)
    m = cls.from_pretrained(model_dir, config=cfg, dtype=dtype)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--threads", type=int, default=64)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    m = load(a.model, dtype)
    for s in json.load(open(a.seqs)):
        t0 = time.time()
        ids = torch.tensor([s["ids"]])
        with torch.no_grad():
            h = m.model(input_ids=ids).last_hidden_state
            first = max(0, s["n_prompt"] - 1)
            logits = m.lm_head(h[0, first:-1]).float()
        lp = torch.log_softmax(logits.double(), dim=-1)
        tgt = ids[0, first + 1:]
        lpt = lp.gather(1, tgt[:, None])[:, 0]
        mx, am = lp.max(dim=-1)
        with open(f"{a.out}/{s['name']}.lp.tsv", "w") as f:
            for j in range(len(tgt)):
                f.write(f"{first + j}\t{int(tgt[j])}\t{float(lpt[j]):.6f}\t{int(am[j])}\t{float(mx[j]):.6f}\n")
        if a.full:
            lp.float().numpy().tofile(f"{a.out}/{s['name']}.full.f32")
        print(f"{s['name']}: {len(s['ids'])} tok, {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
