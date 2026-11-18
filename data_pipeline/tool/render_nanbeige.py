"""Render tir_traces.jsonl with the Nanbeige4.2 chat template -> tir_nanbeige.jsonl (text, token ids, loss mask).

Template call: apply_chat_template(messages, tools=tools_schema, preserve_thinking=True), XML tool calls,
tool outputs as <tool_response> in user turns (see rl/ksat_rl/tools/nanbeige_format.py for the details).
loss_mask: 1 = assistant-generated tokens (reasoning, </think>, content, tool-call XML, <|im_end|>),
           0 = system / user / tool-response tokens and the generation prompt "<|im_start|>assistant\\n<think>\\n".
Run (CPU only):  data_pipeline/tool/run_format.sh render
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.environ.get("KSAT_RL_DIR", "/rl"))
from transformers import AutoTokenizer  # noqa: E402

from ksat_rl.tools.nanbeige_format import build_example  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="/data/tool/tir_traces.jsonl")
    ap.add_argument("--tools", default="/data/tool/tools_schema.json")
    ap.add_argument("--model", default="/models/Nanbeige4.2-3B")
    ap.add_argument("--out", default="/data/tool/tir_nanbeige.jsonl")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    tools = json.load(open(a.tools))
    n, fails, lens = 0, [], []
    with open(a.out, "w") as f:
        for line in open(a.traces):
            r = json.loads(line)
            try:
                ex = build_example(tok, r["messages"], tools)
            except Exception as e:  # noqa: BLE001
                fails.append({"id": r["id"], "error": f"{type(e).__name__}: {e}"})
                continue
            rec = {"id": r["id"], "source": r["source"], "section": r["section"], "kind": r["kind"],
                   "label_source": r["label_source"], "n_calls": r["n_calls"], "model": r["model"],
                   "template": {"tokenizer": "Nanbeige/Nanbeige4.2-3B", "tools": "tools_schema.json",
                                "preserve_thinking": True, "tool_call_format": "xml", "add_special_tokens": False},
                   **ex}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            lens.append((ex["n_tokens"], ex["n_assistant_tokens"]))
    lens.sort()
    print(json.dumps({"rendered": n, "failed": len(fails), "fail_examples": fails[:5],
                      "n_tokens_median": lens[len(lens) // 2][0] if lens else 0,
                      "n_tokens_max": max(l[0] for l in lens) if lens else 0,
                      "assistant_tokens_median": sorted(l[1] for l in lens)[len(lens) // 2] if lens else 0}, indent=1))


if __name__ == "__main__":
    main()
