"""Mock SGLang native /generate endpoint for CPU-testing eval_csat_tool.py (no model, no GPU).

POST /generate {"input_ids": [...], "sampling_params": {...}} -> {"output_ids": [...], "meta_info": {"finish_reason": ...}}
Behaves like SGLang token-in/token-out: in-context tokenization, stops at the first stop token (inclusive),
honours max_new_tokens.  Scripted "policy": find the problem in the prompt; turn 1 calls the python tool to print
the official answer (+1 for ~25% of requests, so accuracy is not trivially 100%); turn 2 boxes whatever number the
REAL sandbox printed in the last <tool_response>.
    python mock_sglang_server.py --problems /data/ksat_math/test_2026.jsonl --port 31000
"""
import argparse
import json
import random
import re
import sys

from aiohttp import web
from transformers import AutoTokenizer

sys.path.insert(0, "/rl")
from ksat_rl.tools.nanbeige_format import encode_continuation  # noqa: E402
from ksat_rl.tools.schema import tool_call_xml  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--problems", required=True)
ap.add_argument("--tokenizer", default="/models/Nanbeige4.2-3B")
ap.add_argument("--port", type=int, default=31000)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
problems = [json.loads(l) for l in open(a.problems) if l.strip()]
rnd = random.Random(0)


async def generate(req):
    body = await req.json()
    prompt = tok.decode(body["input_ids"])
    sp = body.get("sampling_params", {})
    user = prompt.rsplit("<|im_start|>user\n", 1)[0] if "<tool_response>" in prompt else prompt
    p = next((q for q in problems if q["text"][:200] in user), None)
    if p is None:
        text = "no problem found\n</think>\n\n\\boxed{0}<|im_end|>"
    elif "<tool_response>" not in prompt:
        ans = int(p["answer"]) + (1 if rnd.random() < 0.25 else 0)
        text = f"Let me compute it.\n</think>\n\n" + tool_call_xml("python", {"code": f"x = {ans}\nprint(x)"}) + "<|im_end|>"
    else:
        last = prompt.rsplit("<tool_response>\n", 1)[1].split("\n</tool_response>", 1)[0]
        m = re.findall(r"-?\d+", last)
        text = f"The tool printed {last}.\n</think>\n\n계산 결과에 따라 답은 다음과 같다.\n\n\\boxed{{{m[-1] if m else 0}}}<|im_end|>"
    ids = encode_continuation(tok, text)
    stops = set(sp.get("stop_token_ids") or []) | {IM_END}
    finish = {"type": "length"}
    for i, t in enumerate(ids):
        if t in stops:
            ids, finish = ids[: i + 1], {"type": "stop", "matched": t}
            break
    if len(ids) > sp.get("max_new_tokens", 1 << 30):
        ids, finish = ids[: sp["max_new_tokens"]], {"type": "length"}
    return web.json_response({"text": tok.decode(ids), "output_ids": ids, "meta_info": {"finish_reason": finish}})


app = web.Application(client_max_size=64 << 20)
app.router.add_post("/generate", generate)
web.run_app(app, host="127.0.0.1", port=a.port, print=lambda *x: print("mock sglang ready", flush=True))
