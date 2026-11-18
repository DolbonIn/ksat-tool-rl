"""Render TIR trajectories with the Nanbeige4.2 chat template -> text, token ids, loss mask.

Template facts this relies on (tokenizer_config.json of Nanbeige/Nanbeige4.2-3B):
  * tools go in the system turn: system content + "\\n\\n# Tools ... <tools>{json}</tools> ..." (the default
    Chinese tool system prompt is used only when no system message is given).
  * assistant turn: "<|im_start|>assistant\\n<think>\\n" + reasoning.trim() + "\\n</think>\\n\\n" + content
    [+ "\\n\\n" if content] + "<tool_call>\\n<function=NAME>\\n<parameter=ARG>\\nVALUE\\n</parameter>\\n</function>\\n</tool_call>"
    (consecutive calls joined by "\\n") + "<|im_end|>\\n".
  * tool results: consecutive tool messages share ONE user turn:
    "<|im_start|>user\\n<tool_response>\\nOBS\\n</tool_response>[\\n<tool_response>...]<|im_end|>\\n".
  * generation prompt: "<|im_start|>assistant\\n<think>\\n" (part of the prompt, mask 0).
  * tool_call.arguments must be a dict (template iterates `arguments|items`); OpenAI JSON strings are decoded.
  * preserve_thinking: earlier reasoning is dropped only if preserve_thinking is False AND the assistant turn
    precedes the last real user query; tool turns are not queries, so in a single-problem episode all
    reasoning is kept either way.  We pass preserve_thinking=True as the model card recommends.
  * bos_token is "<|im_start|>" and add_bos_token=True: tokenize rendered text with add_special_tokens=False,
    otherwise a duplicate "<|im_start|>" is prepended (apply_chat_template(tokenize=True) already does this).
  * the tokenizer's pre-tokenizer is Metaspace(prepend_scheme="first"): a string tokenized ON ITS OWN gets a
    leading "▁" (e.g. "\n</think>" -> ["▁", "<0x0A>", "</think>"], "We" -> "▁We"), but the same text inside
    the conversation (after a special token or "\n") does not.  So never tokenize a turn/delta standalone and
    concatenate; tokenize the whole conversation, or use encode_continuation() (anchored after a special token).

Loss mask: 1 on tokens the policy generates in each assistant turn = reasoning + "\\n</think>\\n\\n" + content +
tool-call XML + "<|im_end|>"; 0 on system/user/tool-response tokens, on each generation prompt
"<|im_start|>assistant\\n<think>\\n" and on the "\\n" after "<|im_end|>" (the model stops at <|im_end|>).
Token ids come from ONE whole-string tokenization of the rendered conversation; the mask is derived from
character offsets, and every assistant boundary is verified to fall on a token boundary.  The format checks
compare this with turn-by-turn (rollout-style) tokenization.
"""
import json

IM_END = "<|im_end|>"


def to_template_messages(messages: list) -> list:
    """OpenAI-style messages (as in tir_traces.jsonl) -> messages for the Nanbeige chat template."""
    out = []
    for m in messages:
        role = m["role"]
        if role == "assistant":
            mm = {"role": "assistant", "content": (m.get("content") or "").strip(),
                  "reasoning_content": (m.get("reasoning_content") or "").strip()}
            if m.get("tool_calls"):
                calls = []
                for tc in m["tool_calls"]:
                    f = tc.get("function", tc)
                    args = f["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    calls.append({"type": "function", "function": {"name": f["name"], "arguments": args}})
                mm["tool_calls"] = calls
            out.append(mm)
        elif role == "tool":
            out.append({"role": "tool", "content": m.get("content") or "", "tool_call_id": m.get("tool_call_id"),
                        "name": m.get("name", "python")})
        else:
            out.append({"role": role, "content": m.get("content") or ""})
    return out


def render(tok, messages: list, tools: list, add_generation_prompt: bool = False, template_ready: bool = False) -> str:
    msgs = messages if template_ready else to_template_messages(messages)
    return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=add_generation_prompt,
                                   preserve_thinking=True)


def encode(tok, text: str) -> list:
    return tok(text, add_special_tokens=False)["input_ids"]


_ANCHOR = "<|im_end|>"


def encode_continuation(tok, text: str) -> list:
    """Ids of `text` as they appear when it FOLLOWS earlier conversation text that ends in a special token or a
    byte token such as "\n" (e.g. a tool-response delta after "<|im_end|>", or an assistant turn after
    "<think>\n").  Avoids the Metaspace dummy-prefix "▁" that standalone tokenization adds."""
    anchor = tok(_ANCHOR, add_special_tokens=False)["input_ids"]
    ids = tok(_ANCHOR + text, add_special_tokens=False)["input_ids"]
    assert ids[:len(anchor)] == anchor
    return ids[len(anchor):]


def segments(tok, messages: list, tools: list) -> list:
    """[(text, is_assistant)] whose concatenation is the full conversation (without the final '\\n').
    Raises ValueError if the template is not prefix-stable (render(msgs[:k], gen) not a prefix of render(msgs[:k+1]))."""
    msgs = to_template_messages(messages)
    segs, pos = [], 0
    prev_full = None
    for k, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prefix = render(tok, msgs[:k], tools, add_generation_prompt=True, template_ready=True)
        full = render(tok, msgs[:k + 1], tools, template_ready=True)
        if not full.startswith(prefix):
            raise ValueError(f"template not prefix-stable at message {k}")
        if prev_full is not None and not prefix.startswith(prev_full):
            raise ValueError(f"earlier turns changed when rendering message {k}")
        if not full.endswith(IM_END + "\n"):
            raise ValueError(f"assistant message {k} does not end with <|im_end|>\\n")
        segs.append((prefix[pos:], False))                       # system/user/tool delta + generation prompt
        segs.append((full[len(prefix):-1], True))                # generated: ... <|im_end|>
        pos = len(full) - 1                                      # '\n' after <|im_end|> belongs to next prompt
        prev_full = full[:-1]
    return segs


def build_example(tok, messages: list, tools: list) -> dict:
    segs = segments(tok, messages, tools)
    text = "".join(s for s, _ in segs)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc["input_ids"]), list(enc["offset_mapping"])
    bounds, pos = [], 0                  # char ranges of every segment
    for s, is_asst in segs:
        bounds.append((pos, pos + len(s), is_asst))
        pos += len(s)
    cuts = {b[0] for b in bounds[1:]}
    tok_start = {}                       # char cut -> first token index starting at/after it
    for i, (cs, ce) in enumerate(offs):
        for c in cuts:
            if cs < c < ce:
                raise ValueError(f"token {i} {tok.convert_ids_to_tokens(ids[i])!r} straddles a turn boundary at char {c}")
    mask, spans, j = [], [], 0
    for (cs, ce, is_asst) in bounds:
        start = j
        while j < len(ids) and offs[j][0] < ce:   # offsets are monotone; straddling was ruled out above
            j += 1
        spans.append([start, j, "assistant" if is_asst else "prompt"])
        mask += [1 if is_asst else 0] * (j - start)
    if j != len(ids):
        raise ValueError("unassigned tokens at the end")
    return {"text": text, "input_ids": ids, "loss_mask": mask, "spans": spans,
            "n_tokens": len(ids), "n_assistant_tokens": sum(mask), "n_prompt_tokens": spans[0][1] if spans else 0}
