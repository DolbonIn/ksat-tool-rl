"""KSAT tool agent loop for veRL: ToolAgentLoop + infra-fault masking + per-trajectory tool stats.

Registered as `ksat_tool_agent` (rl/configs/tool/agent_loop.yaml via rollout.agent.agent_loop_config_path; the
tool parquet sets agent_name=ksat_tool_agent).  Everything else is veRL's ToolAgentLoop:
  * prompt = chat template(messages, tools=schemas) through the Continuous Token builder
    (NanbeigeContinuousTokenBuilder after rl/docker/patches/patch_verl_nanbeige_tools.py),
  * generation stops at </tool_call> (nanbeige parser) or <|im_end|>,
  * tool responses are appended as in-context token deltas with response_mask 0,
  * limits: multi_turn.max_user_turns (= tool calls, one call per turn), max_assistant_turns,
    max_parallel_calls, max_tool_response_length, response_length.
Added here:
  * every trajectory's extra_fields carries all KsatPythonTool counters (0 if unused) plus
    ksat_infra_fault, so batches have a uniform schema and the reward/metrics can see them;
  * if a sandbox infrastructure failure happened (ksat_tool_sandbox_error > 0) and mask_infra_faults is on,
    response_mask is zeroed: the trajectory contributes no policy gradient (it is not the model's fault).
"""
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

from ksat_rl.tools.verl_tool import STAT_KEYS


@register("ksat_tool_agent")
class KsatToolAgentLoop(ToolAgentLoop):
    def __init__(self, *args, mask_infra_faults: bool = True, **kwargs):
        kwargs.pop("name", None)  # agent_loop.yaml entries carry their registry name
        super().__init__(*args, **kwargs)
        self.mask_infra_faults = bool(mask_infra_faults)

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        output = await super().run(sampling_params, priority=priority, **kwargs)
        ef = output.extra_fields
        for k in STAT_KEYS:
            ef[k] = int(ef.get(k) or 0)
        ef["ksat_infra_fault"] = int(ef["ksat_tool_sandbox_error"] > 0)
        if ef["ksat_infra_fault"] and self.mask_infra_faults:
            output.response_mask = [0] * len(output.response_mask)
        return output
