"""Run veRL's real tool agent loop (KsatToolAgentLoop) outside the trainer: CPU tests and tool-mode evaluation.

    loop = build_agent_loop(tokenizer, server_manager, prompt_length=8192, response_length=65536)
    out = await loop.run(sampling_params, raw_prompt=messages, extra_info={}, tools_kwargs={})

`server_manager` only needs `async generate(request_id, prompt_ids, sampling_params, **kw) -> TokenOutput`
(what veRL's AgentLoopWorker passes in).  Provided here:
  * SGLangGenerateClient: token-in/token-out against an SGLang server's native /generate endpoint, with the
    same max_new_tokens rule as veRL's SGLang server adapter (eval_csat_tool.py),
  * the CPU tests use a scripted fake (tests/test_verl_tool.py).
Requires the veRL patch (rl/docker/patches/patch_verl_nanbeige_tools.py) for the nanbeige parser/builder.
The multi_turn settings mirror rl/scripts/run_math_tool_v1.sh so eval and training see the same environment.
"""
import os
from typing import Any

from omegaconf import OmegaConf

TOOL_CONFIG = os.environ.get("KSAT_TOOL_CONFIG", "/rl/configs/tool/python_tool.yaml")


def make_config(prompt_length: int, response_length: int, max_user_turns: int = 6, max_assistant_turns: int = 8,
                max_parallel_calls: int = 1, max_tool_response_length: int = 1000, truncate_side: str = "middle",
                tool_format: str = "nanbeige", tool_config_path: str = TOOL_CONFIG,
                chat_template_kwargs: dict | None = None):
    return OmegaConf.create({
        "actor_rollout_ref": {"rollout": {
            "prompt_length": prompt_length, "response_length": response_length, "full_determinism": False,
            "multi_turn": {"enable": True, "max_user_turns": max_user_turns, "max_assistant_turns": max_assistant_turns,
                           "max_parallel_calls": max_parallel_calls, "max_tool_response_length": max_tool_response_length,
                           "tool_response_truncate_side": truncate_side, "format": tool_format,
                           "tool_config_path": tool_config_path},
        }},
        "data": {"apply_chat_template_kwargs": chat_template_kwargs if chat_template_kwargs is not None
                 else {"preserve_thinking": True}},
    })


def build_agent_loop(tokenizer, server_manager, prompt_length: int, response_length: int, tools=None,
                     mask_infra_faults: bool = True, **cfg_kw):
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.tools.tool_registry import load_all_tools
    from verl.utils.dataset.rl_dataset import RLHFDataset

    from ksat_rl.tools.verl_agent_loop import KsatToolAgentLoop

    cfg = make_config(prompt_length, response_length, **cfg_kw)
    if tools is None:
        tools = load_all_tools(cfg.actor_rollout_ref.rollout.multi_turn.tool_config_path, None)
    return KsatToolAgentLoop(trainer_config=DictConfigWrap(cfg), server_manager=server_manager, tokenizer=tokenizer,
                             processor=None, dataset_cls=RLHFDataset, data_config=DictConfigWrap(cfg.data),
                             hf_model_type="nanbeige", tools=ToolListWrap(tools), mask_infra_faults=mask_infra_faults)


class SGLangGenerateClient:
    """Minimal veRL server_manager over HTTP: POST {url}/generate with input_ids (token-in/token-out)."""

    def __init__(self, url: str, prompt_length: int, response_length: int, max_model_len: int, timeout: float = 6 * 3600):
        self.url = url.rstrip("/")
        self.prompt_length, self.response_length, self.max_model_len = prompt_length, response_length, max_model_len
        self.timeout = timeout
        self._session = None

    async def _sess(self):
        import aiohttp
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout),
                                                  connector=aiohttp.TCPConnector(limit=0))
        return self._session

    async def close(self):
        if self._session is not None:
            await self._session.close()

    async def generate(self, request_id: str, prompt_ids: list[int], sampling_params: dict[str, Any], **kwargs):
        from verl.workers.rollout.replica import TokenOutput

        sp = dict(sampling_params)
        sp.pop("logprobs", None)
        # same budget rule as verl/workers/rollout/sglang_rollout/async_sglang_server.py
        max_new = min(self.response_length, self.prompt_length + self.response_length - len(prompt_ids))
        max_new = max(0, min(max_new, self.max_model_len - len(prompt_ids) - 1))
        sp["max_new_tokens"] = max_new
        sp.setdefault("skip_special_tokens", False)
        s = await self._sess()
        # no "rid": the agent loop reuses one request id for all turns of a trajectory
        async with s.post(f"{self.url}/generate", json={"input_ids": list(prompt_ids), "sampling_params": sp}) as r:
            r.raise_for_status()
            out = await r.json()
        if isinstance(out, list):
            out = out[0]
        ids = out.get("output_ids")
        if ids is None:
            raise RuntimeError("SGLang /generate response has no output_ids")
        fr = (out.get("meta_info") or {}).get("finish_reason") or {}
        stop = {"stop": "completed", "length": "length", "abort": "aborted"}.get(fr.get("type") if isinstance(fr, dict) else fr)
        return TokenOutput(token_ids=list(ids), stop_reason=stop)
