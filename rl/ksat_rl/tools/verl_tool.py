"""veRL tool wrapping the KSAT Python sandbox (BaseTool subclass for multi_turn.tool_config_path).

    rl/configs/tool/python_tool.yaml:
      tools:
        - class_name: ksat_rl.tools.verl_tool.KsatPythonTool
          config: {type: native, timeout: 10, max_chars: 1000, max_calls_per_trajectory: 6}
          (tool_schema omitted -> ksat_rl.tools.schema.PYTHON_TOOL, the schema used for the TIR data)

Each call runs the code in a fresh sandboxed process (python_sandbox.arun_python) and returns
format_observation(...) (<= 1000 chars) with chat/tool markup defanged (trace_utils.defang_markup), so code
output can never forge turns or a "</think> \\boxed{}" in the conversation.

Per-trajectory counters are kept in agent_data.extra_fields (-> AgentLoopOutput.extra_fields -> reward
extra_info and the rollout batch):
  ksat_tool_calls, ksat_tool_ok, ksat_tool_error, ksat_tool_timeout, ksat_tool_memory, ksat_tool_killed,
  ksat_tool_invalid (bad/missing `code`), ksat_tool_over_budget, ksat_tool_sandbox_error.
Infrastructure failures (status "sandbox_error": service unreachable after retries, worker crash) are counted
separately from the model's own errors.  KsatToolAgentLoop turns any of them into ksat_infra_fault=1 and
zeroes that trajectory's response_mask, so an infra fault is never trained on as the policy's mistake
(ksat_tool_rl_v2/README.md §5).  The model still sees "[sandbox error: the code was not executed]".
"""
import asyncio
import logging
import os
from typing import Any, Optional

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from ksat_rl.tools.python_sandbox import DEFAULT_URL, arun_python, format_observation
from ksat_rl.tools.schema import MAX_TOOL_CALLS, PYTHON_TOOL
from ksat_rl.tools.trace_utils import defang_markup

logger = logging.getLogger(__name__)

STAT_KEYS = ("ksat_tool_calls", "ksat_tool_ok", "ksat_tool_error", "ksat_tool_timeout", "ksat_tool_memory",
             "ksat_tool_killed", "ksat_tool_invalid", "ksat_tool_over_budget", "ksat_tool_sandbox_error")


def tool_stats(agent_data) -> dict:
    """Counters for one trajectory, stored in agent_data.extra_fields (created on first use)."""
    ef = agent_data.extra_fields if agent_data is not None else {}
    for k in STAT_KEYS:
        ef.setdefault(k, 0)
    return ef


class KsatPythonTool(BaseTool):
    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema.model_validate(PYTHON_TOOL)
        super().__init__(config, tool_schema)
        self.url = config.get("sandbox_url") or os.environ.get("KSAT_SANDBOX_URL", DEFAULT_URL)
        self.timeout = float(config.get("timeout", 10.0))
        self.max_chars = int(config.get("max_chars", 1000))
        self.retries = int(config.get("retries", 4))
        self.max_calls = int(config.get("max_calls_per_trajectory", MAX_TOOL_CALLS))
        self.http_timeout = float(config.get("http_timeout", 120.0))  # includes queueing behind busy sandbox workers
        self._sessions: dict[int, Any] = {}

    def _session(self):
        """One aiohttp session per event loop (AgentLoopWorker actors each run their own loop)."""
        try:
            import aiohttp
        except ImportError:
            return None
        loop = asyncio.get_running_loop()
        s = self._sessions.get(id(loop))
        if s is None or s.closed:
            s = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=512, force_close=True))
            self._sessions[id(loop)] = s
        return s

    async def aclose(self):
        """Close HTTP sessions (eval/test scripts; training workers keep theirs for the process lifetime)."""
        for s in self._sessions.values():
            if not s.closed:
                await s.close()
        self._sessions.clear()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        stats = tool_stats(kwargs.get("agent_data"))
        stats["ksat_tool_calls"] += 1
        if stats["ksat_tool_calls"] > self.max_calls:   # belt and braces; max_user_turns normally ends the episode
            stats["ksat_tool_over_budget"] += 1
            return ToolResponse(text=f"[tool budget exhausted: at most {self.max_calls} python calls per problem]"), 0.0, \
                {"status": "over_budget"}
        code = parameters.get("code") if isinstance(parameters, dict) else None
        if not isinstance(code, str) or not code.strip():
            stats["ksat_tool_invalid"] += 1
            return ToolResponse(text="Error: the python tool needs a non-empty string argument `code`."), 0.0, \
                {"status": "invalid"}
        res = await arun_python(code, timeout=self.timeout, url=self.url, retries=self.retries,
                                max_chars=self.max_chars, session=self._session(), http_timeout=self.http_timeout)
        status = res.get("status") or "sandbox_error"
        key = f"ksat_tool_{status}"
        stats[key if key in stats else "ksat_tool_error"] += 1
        if status == "sandbox_error":
            logger.warning("sandbox infrastructure failure: %s", (res.get("stderr") or "")[:200])
        text = defang_markup(format_observation(res, self.max_chars))
        return ToolResponse(text=text), 0.0, {"status": status, "wall_ms": res.get("wall_ms")}
