"""Format checks for the Nanbeige TIR data (CPU only, inside ksat-rl/verl-sgl:v4).

(a) SGLang parsers: for every rendered assistant turn, the `nanbeige` reasoning parser (Qwen3Detector,
    force_reasoning since "<think>\\n" is in the prompt) and the `nanbeige` tool-call parser (Qwen3CoderDetector)
    recover exactly the intended reasoning, content and tool calls (name + arguments), non-streaming and
    streaming; final turns yield no calls.  Round trip: rebuilding the message from the parsed pieces and
    re-rendering reproduces the identical conversation text.
(b) Prefix preservation: rendering msgs[:k] (+generation prompt) is a text prefix of every later rendering;
    whole-string token ids of each turn-by-turn prefix equal the first ids of the full conversation; and
    rollout-style incremental ids (prompt ids + appended turn/tool-response deltas) equal the full ids when
    deltas are encoded with encode_continuation (naive standalone encoding is measured too: Metaspace quirk).
(c) Loss masks: mask==1 exactly on assistant-generated text (decoded spans match), every tool response,
    system and user token has mask 0, one masked <|im_end|> per assistant turn.
Writes data/tool/format_check_report.json and format_check_failed_ids.txt.
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.environ.get("KSAT_RL_DIR", "/rl"))
from transformers import AutoTokenizer  # noqa: E402

from ksat_rl.tools.nanbeige_format import encode, encode_continuation, render, segments, to_template_messages  # noqa: E402

from sglang.srt.entrypoints.openai.protocol import Tool  # noqa: E402
from sglang.srt.function_call.function_call_parser import FunctionCallParser  # noqa: E402
from sglang.srt.parser.reasoning_parser import ReasoningParser  # noqa: E402

IM_END = "<|im_end|>"


def parse_assistant(seg: str, tools_pyd, stream_chunk=None):
    """seg = generated assistant text incl. trailing <|im_end|>. Returns (reasoning, content, calls)."""
    gen = seg[:-len(IM_END)] if seg.endswith(IM_END) else seg      # SGLang strips the EOS token
    rp = ReasoningParser(model_type="nanbeige", stream_reasoning=False, force_reasoning=True)
    reasoning, normal = rp.parse_non_stream(gen)
    if stream_chunk is None:
        fcp = FunctionCallParser(tools=tools_pyd, tool_call_parser="nanbeige")
        normal_text, items = fcp.parse_non_stream(normal or "")
        calls = [(it.name, json.loads(it.parameters)) for it in items]
        return reasoning, normal_text, calls
    fcp = FunctionCallParser(tools=tools_pyd, tool_call_parser="nanbeige")
    text, acc, names = normal or "", collections.defaultdict(str), {}
    normal_out = ""
    i = 0
    while i < len(text):
        n = stream_chunk()
        nt, items = fcp.parse_stream_chunk(text[i:i + n])
        normal_out += nt or ""
        for it in items:
            if it.name:
                names[it.tool_index] = it.name
            if it.parameters:
                acc[it.tool_index] += it.parameters
        i += n
    calls = [(names.get(k), json.loads(acc[k]) if acc[k] else {}) for k in sorted(set(names) | set(acc))]
    return reasoning, normal_out, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="/data/tool/tir_traces.jsonl")
    ap.add_argument("--nanbeige", default="/data/tool/tir_nanbeige.jsonl")
    ap.add_argument("--tools", default="/data/tool/tools_schema.json")
    ap.add_argument("--model", default="/models/Nanbeige4.2-3B")
    ap.add_argument("--report", default="/data/tool/format_check_report.json")
    ap.add_argument("--failed", default="/data/tool/format_check_failed_ids.txt")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    tools = json.load(open(a.tools))
    tools_pyd = [Tool(**t) for t in tools]
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(a.nanbeige)}
    rnd = random.Random(0)
    C = collections.Counter()
    fails = collections.defaultdict(list)
    quirk = collections.Counter()

    # global tokenizer facts
    probe_msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}]
    t_txt = tok.apply_chat_template(probe_msgs, tools=tools, tokenize=False, add_generation_prompt=True)
    t_ids = tok.apply_chat_template(probe_msgs, tools=tools, tokenize=True, add_generation_prompt=True)
    t_ids = list(t_ids["input_ids"] if hasattr(t_ids, "keys") else t_ids)
    facts = {
        "apply_chat_template_tokenize_equals_add_special_tokens_False": t_ids == encode(tok, t_txt),
        "add_special_tokens_True_prepends_extra_im_start": tok(t_txt)["input_ids"][:2] == [tok.convert_tokens_to_ids("<|im_start|>")] * 2,
        "single_token_markers": {s: len(encode(tok, s)) == 1 for s in ["<think>", "</think>", "<tool_call>", "</tool_call>",
                                                                        "<|im_start|>", "<|im_end|>", "<tool_response>"]},
        "standalone_vs_in_context_example": {
            "standalone('\\n</think>')": tok.convert_ids_to_tokens(encode(tok, "\n</think>")),
            "continuation('\\n</think>')": tok.convert_ids_to_tokens(encode_continuation(tok, "\n</think>"))},
        "pre_tokenizer": str(tok.backend_tokenizer.pre_tokenizer),
    }

    for line in open(a.traces):
        tr = json.loads(line)
        tid = tr["id"]
        ex = rows.get(tid)
        if ex is None:
            fails[tid].append("not_rendered")
            continue
        C["trajectories"] += 1
        msgs = tr["messages"]
        tmsgs = to_template_messages(msgs)
        segs = segments(tok, msgs, tools)
        text = "".join(s for s, _ in segs)
        if text != ex["text"]:
            fails[tid].append("text_mismatch_vs_rendered_file")
        asst_msgs = [m for m in tmsgs if m["role"] == "assistant"]
        asst_segs = [s for s, is_a in segs if is_a]
        assert len(asst_msgs) == len(asst_segs)

        # ---------------- (a) parsers + round trip
        rebuilt = []
        for m, seg in zip(asst_msgs, asst_segs):
            C["a_assistant_turns"] += 1
            intended = [(tc["function"]["name"], tc["function"]["arguments"]) for tc in m.get("tool_calls") or []]
            reasoning, content, calls = parse_assistant(seg, tools_pyd)
            ok_calls = calls == intended
            ok_reason = (reasoning or "").strip() == m["reasoning_content"]
            ok_content = (content or "").strip() == m["content"].strip()
            if intended:
                C["a_turns_with_calls"] += 1
                C["a_calls_intended"] += len(intended)
                C["a_calls_parsed_exact"] += sum(1 for x, y in zip(calls, intended) if x == y) if len(calls) == len(intended) else 0
                if m["content"].strip():
                    quirk["content_before_tool_call(template inserts \\n\\n before <tool_call>)"] += 1
                if len(intended) > 1:
                    quirk["parallel_tool_calls_in_one_turn"] += 1
            else:
                C["a_final_turns"] += 1
                C["a_final_turns_no_calls"] += int(calls == [])
            if not ok_calls:
                fails[tid].append("a_calls_mismatch")
            if not ok_reason:
                fails[tid].append("a_reasoning_mismatch")
            if not ok_content:
                fails[tid].append("a_content_mismatch")
            # streaming parser with random chunk sizes
            _, _, s_calls = parse_assistant(seg, tools_pyd, stream_chunk=lambda: rnd.randint(1, 12))
            if s_calls != intended:
                fails[tid].append("a_stream_calls_mismatch")
            else:
                C["a_stream_turns_exact"] += 1
            mm = {"role": "assistant", "content": (content or "").strip(), "reasoning_content": (reasoning or "").strip()}
            if calls:
                mm["tool_calls"] = [{"type": "function", "function": {"name": n_, "arguments": a_}} for n_, a_ in calls]
            rebuilt.append(mm)
        it = iter(rebuilt)
        rt_msgs = [next(it) if m["role"] == "assistant" else m for m in tmsgs]
        rt_text = render(tok, rt_msgs, tools, template_ready=True)
        if rt_text[:-1] == text:
            C["a_roundtrip_identical"] += 1
        else:
            fails[tid].append("a_roundtrip_changed_text")

        # ---------------- (b) prefix preservation
        full_ids = encode(tok, text)
        if full_ids != ex["input_ids"]:
            fails[tid].append("b_ids_mismatch_vs_rendered_file")
        if tok.decode(full_ids, skip_special_tokens=False) != text:
            fails[tid].append("b_decode_not_lossless")
        ok_prefix = True
        prev_render = None
        for k, m in enumerate(tmsgs):
            if m["role"] != "assistant":
                continue
            pre = render(tok, tmsgs[:k], tools, add_generation_prompt=True, template_ready=True)
            if prev_render is not None and not pre.startswith(prev_render):
                ok_prefix = False
            pids = encode(tok, pre)
            if full_ids[:len(pids)] != pids:
                ok_prefix = False
            post = render(tok, tmsgs[:k + 1], tools, template_ready=True)[:-1]   # up to <|im_end|>
            qids = encode(tok, post)
            if full_ids[:len(qids)] != qids:
                ok_prefix = False
            prev_render = post
            C["b_prefix_points"] += 1
        if ok_prefix:
            C["b_turn_by_turn_prefix_preserved"] += 1
        else:
            fails[tid].append("b_prefix_not_preserved")
        inc = encode(tok, segs[0][0])
        naive = list(inc)
        for s, _ in segs[1:]:
            inc += encode_continuation(tok, s)
            naive += encode(tok, s)
        if inc == full_ids:
            C["b_incremental_continuation_equals_full"] += 1
        else:
            fails[tid].append("b_incremental_mismatch")
        if naive == full_ids:
            C["b_naive_standalone_equals_full"] += 1
        else:
            quirk["naive_standalone_segment_tokenization_differs (Metaspace dummy prefix)"] += 1

        # ---------------- (c) loss masks
        ids, mask, spans = ex["input_ids"], ex["loss_mask"], ex["spans"]
        ok_mask = len(ids) == len(mask)
        seg_iter = iter(segs)
        for (s, e, role), (stext, is_a) in zip(spans, seg_iter):
            if (role == "assistant") != is_a:
                ok_mask = False
            if tok.decode(ids[s:e], skip_special_tokens=False) != stext:
                ok_mask = False
            if any(mask[i] != (1 if is_a else 0) for i in range(s, e)):
                ok_mask = False
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        # every tool response / system / user content region -> all overlapping tokens mask 0
        regions = []
        start = 0
        for m in tmsgs:
            if m["role"] in ("tool", "system", "user") and m["content"]:
                # tool outputs are located by their exact wrapper (a short output like "840" can also occur
                # inside earlier assistant text); system/user occur once, in order, at the top
                pre_ = "<tool_response>\n" if m["role"] == "tool" else ""
                needle = pre_ + m["content"] + ("\n</tool_response>" if m["role"] == "tool" else "")
                j = text.find(needle, start)
                if j < 0:
                    ok_mask = False
                    continue
                j += len(pre_)
                regions.append((j, j + len(m["content"]), m["role"]))
                start = j + len(m["content"])
        for (cs, ce, role) in regions:
            for i, (ts, te) in enumerate(offs):
                if ts < ce and te > cs and mask[i]:
                    ok_mask = False
                    C[f"c_masked_tokens_in_{role}"] += 1
        im_end = tok.convert_tokens_to_ids(IM_END)
        n_masked_end = sum(1 for i, t in enumerate(ids) if t == im_end and mask[i])
        if n_masked_end != len(asst_msgs):
            ok_mask = False
        decoded_masked = "".join(tok.decode(ids[s:e], skip_special_tokens=False) for s, e, r in spans if r == "assistant")
        if "<tool_response>" in decoded_masked or "<|im_start|>" in decoded_masked:
            ok_mask = False
        C["c_tool_responses_checked"] += sum(1 for r in regions if r[2] == "tool")
        if ok_mask:
            C["c_mask_ok"] += 1
        else:
            fails[tid].append("c_mask_error")

    report = {
        "n_trajectories": C["trajectories"],
        "a_sglang_nanbeige_parser": {
            "assistant_turns": C["a_assistant_turns"], "turns_with_tool_calls": C["a_turns_with_calls"],
            "calls_intended": C["a_calls_intended"], "calls_parsed_exact": C["a_calls_parsed_exact"],
            "final_turns": C["a_final_turns"], "final_turns_with_no_spurious_call": C["a_final_turns_no_calls"],
            "streaming_turns_exact": C["a_stream_turns_exact"],
            "roundtrip_parse_rerender_identical": C["a_roundtrip_identical"]},
        "b_prefix_preservation": {
            "trajectories_turn_by_turn_prefix_preserved": C["b_turn_by_turn_prefix_preserved"],
            "prefix_points_checked": C["b_prefix_points"],
            "incremental_ids_with_encode_continuation_equal_full": C["b_incremental_continuation_equals_full"],
            "incremental_ids_with_naive_standalone_encode_equal_full": C["b_naive_standalone_equals_full"]},
        "c_loss_mask": {"trajectories_mask_ok": C["c_mask_ok"], "tool_responses_checked": C["c_tool_responses_checked"],
                        **{k: v for k, v in C.items() if k.startswith("c_masked_tokens_in_")}},
        "template_quirks_counts": dict(quirk),
        "tokenizer_facts": facts,
        "n_failed_trajectories": len(fails),
        "failure_kinds": dict(collections.Counter(x for v in fails.values() for x in set(v))),
        "failed_examples": dict(list(fails.items())[:10]),
    }
    json.dump(report, open(a.report, "w"), ensure_ascii=False, indent=1)
    with open(a.failed, "w") as f:
        for k in fails:
            f.write(k + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
