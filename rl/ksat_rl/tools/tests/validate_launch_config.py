"""Stand-in for `python3 -m verl.trainer.main_ppo <overrides>` in the CPU launch dry-run (dryrun_tool_launch.sh).

Receives the exact hydra overrides train_grpo.sh builds from run_math_tool_v1.sh and checks, without GPUs, that
the run would get past config composition and worker set-up for the tool path:
  config composes (ppo_trainer + overrides) and the rollout / multi_turn dataclasses validate;
  RLHFDataset loads the tool parquet with the tool schemas and keeps every row at data.max_prompt_length;
  rows route to ksat_tool_agent and carry system+user messages;
  tools load from multi_turn.tool_config_path; the `nanbeige` parser resolves with </tool_call> stop;
  agent_loop_config_path registers ksat_tool_agent (merges context -> CT check runs);
  the Continuous Token builder for model_type=nanbeige passes the strict sanity check with
  data.apply_chat_template_kwargs, exactly as AgentLoopWorker does;
  the reward function path loads and scores a sample.
"""
import importlib.util
import json
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

overrides = sys.argv[1:]
RESULTS = []


def check(name, cond, extra=""):
    RESULTS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}".rstrip(), flush=True)


with initialize_config_dir(config_dir="/opt/verl/verl/trainer/config", version_base=None):
    cfg = compose(config_name="ppo_trainer", overrides=overrides)
OmegaConf.resolve(cfg)
ro, mt = cfg.actor_rollout_ref.rollout, cfg.actor_rollout_ref.rollout.multi_turn
check("hydra config composes with all overrides", True, f"({len(overrides)} overrides)")
print("  key settings:", json.dumps({
    "train_files": str(cfg.data.train_files), "val_files": str(cfg.data.val_files),
    "max_prompt_length": cfg.data.max_prompt_length, "max_response_length": cfg.data.max_response_length,
    "rollout.prompt_length": ro.prompt_length, "rollout.response_length": ro.response_length,
    "multi_turn": {k: mt[k] for k in ("enable", "format", "tool_config_path", "max_user_turns", "max_assistant_turns",
                                      "max_parallel_calls", "max_tool_response_length", "tokenization_sanity_check_mode")},
    "agent": {"default_agent_loop": ro.agent.default_agent_loop, "agent_loop_config_path": ro.agent.agent_loop_config_path},
    "apply_chat_template_kwargs": OmegaConf.to_container(cfg.data.apply_chat_template_kwargs),
    "reward_fn": cfg.reward.custom_reward_function.path,
    "rollout_correction": {"rollout_is": cfg.algorithm.rollout_correction.rollout_is,
                           "threshold": cfg.algorithm.rollout_correction.rollout_is_threshold},
    "calculate_log_probs": ro.calculate_log_probs, "n": ro.n, "train_batch_size": cfg.data.train_batch_size,
    "model": cfg.actor_rollout_ref.model.path}, ensure_ascii=False, indent=1))
check("multi_turn: format nanbeige, 6 tool turns, 8 assistant turns, 1 call/turn, 1000-char guard",
      mt.enable and mt.format == "nanbeige" and mt.max_user_turns == 6 and mt.max_assistant_turns == 8
      and mt.max_parallel_calls == 1 and mt.max_tool_response_length == 1000)
check("rollout correction on (token IS, cap 2.0) with rollout log-probs",
      cfg.algorithm.rollout_correction.rollout_is == "token" and ro.calculate_log_probs)
check("rollout prompt/response lengths follow data.*", ro.prompt_length == cfg.data.max_prompt_length
      and ro.response_length == cfg.data.max_response_length)
# Ray actors (AgentLoopWorker) import ksat_rl.tools.* by module name: PYTHONPATH must reach them (main_ppo merges this)
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env  # noqa: E402

rt = OmegaConf.merge(get_ppo_ray_runtime_env(cfg), cfg.ray_kwargs.get("ray_init", {}).get("runtime_env", {}))
check("Ray runtime_env forwards PYTHONPATH=/rl to actors", rt.env_vars.get("PYTHONPATH") == "/rl")
try:
    from verl.utils.config import omega_conf_to_dataclass
    rc = omega_conf_to_dataclass(ro)
    check("rollout config validates as dataclass", rc.multi_turn.format == "nanbeige", type(rc).__name__)
except Exception as e:  # noqa: BLE001
    check("rollout config validates as dataclass", False, f"{type(e).__name__}: {e}")

from transformers import AutoTokenizer, GenerationConfig  # noqa: E402

tok = AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path, trust_remote_code=True)
from verl.utils.dataset.rl_dataset import RLHFDataset  # noqa: E402

for split, files in (("train", cfg.data.train_files), ("val", cfg.data.val_files)):
    import pandas as pd
    n_raw = sum(len(pd.read_parquet(f)) for f in (files if not isinstance(files, str) else [files]))
    ds = RLHFDataset(data_files=files, tokenizer=tok, processor=None, config=cfg.data)
    row = ds[0]
    check(f"{split} dataset: tool schemas used for length filter, no row dropped at max_prompt_length={cfg.data.max_prompt_length}",
          ds.tool_schemas is not None and len(ds) == n_raw, f"{len(ds)}/{n_raw} rows")
    check(f"{split} dataset row: agent_name=ksat_tool_agent, system+user raw_prompt",
          row.get("agent_name") == "ksat_tool_agent" and [m["role"] for m in row["raw_prompt"]] == ["system", "user"])

from verl.experimental.agent_loop.agent_loop import _agent_loop_registry, _any_context_merging_agent_loop  # noqa: E402
from verl.experimental.agent_loop.tool_parser import ToolParser  # noqa: E402
from verl.tools.tool_registry import load_all_tools  # noqa: E402

tools = load_all_tools(mt.tool_config_path, mt.function_tool_path)
check("tools load from tool_config_path", [t.name for t in tools] == ["python"], type(tools[0]).__name__)
p = ToolParser.get_tool_parser(mt.format, tok)
check("tool parser resolves; stops at </tool_call>", p.stop_token_ids == [tok.convert_tokens_to_ids("</tool_call>")], type(p).__name__)
for c in OmegaConf.load(ro.agent.agent_loop_config_path):
    _agent_loop_registry[c.name] = c
check("ksat_tool_agent registered and merges context (CT sanity check will run)",
      ro.agent.default_agent_loop in _agent_loop_registry and _any_context_merging_agent_loop({ro.agent.default_agent_loop}))
from verl.utils.tokenizer.continuous_token_check import verify_continuous_token_builder  # noqa: E402
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder  # noqa: E402

hf_type = json.load(open(f"{cfg.actor_rollout_ref.model.path}/config.json"))["model_type"]
b = create_continuous_token_builder(tok, hf_model_type=hf_type, chat_template_kwargs=cfg.data.get("apply_chat_template_kwargs", {}))
try:
    verify_continuous_token_builder(b, mode=mt.tokenization_sanity_check_mode, hf_model_type=hf_type,
                                    generation_config=GenerationConfig.from_pretrained(cfg.actor_rollout_ref.model.path))
    check("Continuous Token builder passes the strict sanity check (as AgentLoopWorker runs it)", True, f"{type(b).__name__}, model_type={hf_type}")
except Exception as e:  # noqa: BLE001
    check("Continuous Token builder passes the strict sanity check (as AgentLoopWorker runs it)", False, str(e)[:300])
spec = importlib.util.spec_from_file_location("rf", cfg.reward.custom_reward_function.path)
rf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf)
fn = getattr(rf, cfg.reward.custom_reward_function.name)
s = fn("ksat_math_tool", "r\n</think>\n\n<tool_call>x</tool_call> user\n<tool_response>\n7\n</tool_response> assistant\nok\n</think>\n\n\\boxed{7}",
       json.dumps({"kind": "sa", "answer": 7}), extra_info={"ksat_tool_calls": 1})
check("reward function loads from the configured path and grades the final turn", s["score"] == 1.0, json.dumps(s))
print(json.dumps({"passed": sum(RESULTS), "failed": len(RESULTS) - sum(RESULTS)}))
sys.exit(0 if all(RESULTS) else 1)
