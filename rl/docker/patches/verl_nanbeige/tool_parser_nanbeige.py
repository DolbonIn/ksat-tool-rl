# [ksat-patch-nanbeige] installed into verl/experimental/agent_loop/tool_parser_nanbeige.py by
# rl/docker/patches/patch_verl_nanbeige_tools.py (registered on import of tool_parser.py).
"""Tool parser for Nanbeige4.x: multi_turn.format=nanbeige.

Nanbeige's chat template renders tool calls in the Qwen3-Coder XML format

    <tool_call>
    <function=NAME>
    <parameter=ARG>
    VALUE
    </parameter>
    </function>
    </tool_call>

and SGLang serves the model with `--tool-call-parser nanbeige` = Qwen3CoderDetector.  This parser mirrors
that detector so RL rollouts and SGLang serving agree, with three deliberate RL-side choices:

  * stop_token_ids = [</tool_call>]: generation halts right after the first complete call, so the model can
    never write a hallucinated <tool_response> (the real one is appended by the agent loop) and makes at
    most one call per assistant turn.  "</tool_call>" is a single special token in Nanbeige's vocabulary;
    SGLang keeps the matched stop token in output_ids.  NanbeigeContinuousTokenBuilder then inserts the
    template's "<|im_end|>\\n" (mask 0) before the tool response.
  * Only COMPLETE <tool_call>...</tool_call> blocks are executed (veRL's qwen3_coder parser also runs a call
    cut off by the length limit).
  * Parameter values end at </parameter>, at the next <parameter=, or at </function> (SGLang's regex), with
    exactly one leading and one trailing "\\n" stripped, so `code` round-trips byte-exactly.
"""

import json
import logging
import os

import regex

from verl.experimental.agent_loop.tool_parser import FunctionCall, Qwen3XMLToolParser, ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ToolParser.register("nanbeige")
class NanbeigeToolParser(Qwen3XMLToolParser):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        self.tool_call_complete_regex = regex.compile(r"<tool_call>(.*?)</tool_call>", regex.DOTALL)
        self.tool_call_function_regex = regex.compile(r"<function=(.*?)</function>|<function=(.*)$", regex.DOTALL)
        # SGLang Qwen3CoderDetector.tool_call_parameter_regex
        self.tool_call_parameter_regex = regex.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", regex.DOTALL
        )
        tid = tokenizer.convert_tokens_to_ids("</tool_call>")
        if not isinstance(tid, int) or tid < 0 or tid == getattr(tokenizer, "unk_token_id", None):
            raise ValueError("Nanbeige tokenizer must define </tool_call> as a single token")
        self._stop_ids = [tid]

    @property
    def stop_token_ids(self) -> list[int]:
        return list(self._stop_ids)

    def _parse_xml_function_call(self, function_call_str, tools):
        # Same as Qwen3XMLToolParser, but the SGLang parameter regex yields plain strings (no alternation tuple).
        tools = tools or []
        end_index = function_call_str.find(">")
        if end_index == -1:
            return None
        function_name = function_call_str[:end_index]
        param_config = {}
        for config in tools:
            if config.type == "function" and config.function.name == function_name:
                param_config = {k: v.model_dump() for k, v in config.function.parameters.properties.items()}
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        for match_text in self.tool_call_parameter_regex.findall(parameters):
            idx = match_text.find(">")
            if idx == -1:
                continue
            name, value = match_text[:idx], match_text[idx + 1 :]
            if value.startswith("\n"):
                value = value[1:]
            if value.endswith("\n"):
                value = value[:-1]
            ptype = str((param_config.get(name) or {}).get("type", "string")).lower()
            if value.lower() == "null":
                param_dict[name] = None
            elif ptype in ("string", "str", "text"):
                param_dict[name] = value
            else:  # non-string parameters: JSON, else raw string (the python tool only has a string `code`)
                try:
                    param_dict[name] = json.loads(value)
                except Exception:
                    param_dict[name] = value
        return FunctionCall(name=function_name, arguments=json.dumps(param_dict, ensure_ascii=False))

    @rollout_trace_op
    async def extract_tool_calls(
        self, responses_ids: list[int], tools: list[OpenAIFunctionToolSchema] = None
    ) -> tuple[str, list[FunctionCall]]:
        loop = get_event_loop()
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)
        if self.tool_call_start_token not in text or self.tool_call_end_token not in text:
            return text, []
        try:
            calls = []
            for block in self.tool_call_complete_regex.findall(text):
                for m in self.tool_call_function_regex.findall(block):
                    fc = self._parse_xml_function_call(m[0] or m[1], tools)
                    if fc is not None:
                        calls.append(fc)
            content = text[: text.find(self.tool_call_start_token)]
            return content, calls
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Error in extracting nanbeige tool call: {e}")
            return text, []
