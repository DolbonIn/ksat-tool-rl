# ksat_rl.tools: Python tool for KSAT math tool-calling RL

As of 2026-11-18. This is the execution environment and data format for the `python` tool from `ksat_tool_rl_v2/README.md` §1, §2 and §5. Model-generated code runs **only** in the sandbox container.

| File | Contents |
|---|---|
| `sandbox/server.py`, `sandbox/Dockerfile` | Sandbox HTTP service (python 3.12 + sympy 1.13.3 + numpy 2.1.3, stdlib only) |
| `sandbox/sandbox.sh` | build / start / stop / status / connect |
| `python_sandbox.py` | Client: `run_python(code)`, `arun_python(code)`, `format_observation(res)` |
| `schema.py` | Tool schema (`TOOLS`), policy system prompt (`TOOL_SYSTEM_PROMPT`), few-shot curriculum prompts |
| `nanbeige_format.py` | Nanbeige chat-template rendering, token ids, loss mask, `encode_continuation` |
| `trace_utils.py` | Checks whether a tool output was used, and detects forbidden markup |
| `tests/run_sandbox_tests.sh` | Security and throughput tests (50 items) |

## Sandbox start/stop

```bash
rl/ksat_rl/tools/sandbox/sandbox.sh build      # ksat-tool/sandbox:v1
rl/ksat_rl/tools/sandbox/sandbox.sh start      # ksat-tool-sandbox on the internal network ksat-tool-sbx-net
rl/ksat_rl/tools/sandbox/sandbox.sh status
rl/ksat_rl/tools/sandbox/sandbox.sh connect <client-container>   # e.g. attach the RL trainer container
rl/ksat_rl/tools/sandbox/sandbox.sh stop
WORKERS=64 CPUS=64 rl/ksat_rl/tools/sandbox/sandbox.sh restart   # more throughput (container memory = WORKERS x 1 GB + 2 GB)
rl/ksat_rl/tools/tests/run_sandbox_tests.sh    # runs the tests from a throwaway client container
```

Clients reach it at `http://ksat-tool-sandbox:8080` (override with `KSAT_SANDBOX_URL`), from containers attached to `ksat-tool-sbx-net`. The network is `--internal`, so the sandbox has no route outward. Give a client internet access, as with the generation container, by attaching it to the default bridge as well; it does not need any ports opened.

## Isolation

- **Container**: internal network only, read-only root FS, 512 MB tmpfs `/tmp` (noexec), uid 10999 (non-root), `--cap-drop ALL`, `no-new-privileges`, CPU, memory and pids limits, **no volumes**, and no keys in the environment.
- **Per call**: each request runs in a **new process forked** from a prefork worker that has already imported sympy and numpy.
  - rlimits: address space 1 GiB, CPU (timeout+1) s, file size 4 MB, 64 open files, core 0.
  - seccomp (no network sockets of any kind, no fork/vfork/clone/exec, so no subprocesses or threads; no signals to other processes; no ptrace or process_vm_*; no namespace, keyring, bpf or io_uring calls).
  - The process is non-dumpable, so other processes cannot read its /proc. The environment is emptied, the cwd is a private temporary directory, and all inherited fds are closed.
  - The parent enforces wall time (default and maximum 10 s) with SIGKILL.
- Output is truncated to 1,000 characters each for stdout (head) and stderr (tail). `format_observation` keeps the combined text within 1,000 characters.
- If the last line is an expression, its repr is printed, as in a notebook. **State is not kept between calls**, and the tool description tells the model so.
- `status`: `ok | error | timeout | memory | killed | sandbox_error`. `sandbox_error` means the infrastructure failed (the code did not run). Do not count it as the policy's mistake; re-collect the rollout (§5).
- Residual risk: concurrent calls run under the same uid and share `/tmp`. Run directories get random names, and a janitor cleans `/tmp` entries older than 60 s. Because of the seccomp filter, a call cannot reach other containers on the internal network either.

Test results (16 workers / 16 CPUs): all 50 items pass. Covered: infinite loop and sleep killed at 10 s; 4 GiB allocations, a growing memory bomb and a numpy 3.2 GB array all give `memory`; HTTP, raw socket, DNS and unix socket all fail; writes to /srv, /etc, /home and the stdlib fail; file size is capped; fork, subprocess, threads and kill(-1) or killing the parent are blocked; output floods are truncated. The service stays healthy after every attack.

Throughput (64 concurrent clients):

| Workload | calls/s |
|---|---|
| Trivial print | ~1,290 |
| sympy factor+integrate | ~172 |
| Enumerating 8! permutations | ~352 |

It scales roughly linearly with CPUs and WORKERS.

## Tool schema / prompt

- `data/tool/tools_schema.json` = `schema.TOOLS`: a single tool, `python(code: str)`, described in Korean and English.
- In a tool episode, only the system message (`TOOL_SYSTEM_PROMPT`, plus 0–2 few-shot demos) and `tools=` differ from the no-tool RL prompt. The user turn is the same `make_parquet.instruction(p) + "\n\n" + text`.
- ICRL curriculum: `schema.build_messages(p, demos, k_demos=2→1→0)`, with demos from `data/tool/fewshot_demos.json`.

## Nanbeige template quirks (verified by the format checks)

1. If you pass `tools=`, the system message content goes in front of `# Tools`. Without a system message, the template inserts a Chinese default tool prompt.
2. Assistant turn: `<think>\n{reasoning.trim()}\n</think>\n\n{content}` + (`\n\n` if content is non-empty) + XML `<tool_call>` + `<|im_end|>`. Reasoning is **trimmed**, so a generated turn with leading or trailing whitespace changes when re-rendered.
3. `tool_call.arguments` must be a dict (the template uses `|items`), so decode OpenAI JSON strings first. Consecutive tool messages are grouped into **one** user turn.
4. preserve_thinking only removes reasoning from assistant turns before the last real user query. Tool turns don't count as queries, so in a single-problem episode reasoning is always kept.
5. `bos_token=<|im_start|>` and `add_bos_token=True`: tokenizing rendered text with `add_special_tokens=True` duplicates `<|im_start|>`. Use `add_special_tokens=False`; `apply_chat_template(tokenize=True)` is fine.
6. **Metaspace(prepend_scheme=first)**: tokenizing a turn or tool-response delta on its own adds a leading `▁` (for example, `"\n</think>"` becomes `▁ <0x0A> </think>`). When you build multi-turn ids incrementally, encode the delta with `encode_continuation()` (encode it after a special token, then drop that token). Naive per-segment encoding does not match whole-string tokenization in any of the trajectories.
7. `<think>`, `</think>`, `<tool_call>` and `</tool_call>` are single tokens. `<tool_response>`, `<function=` and `<parameter=` are several ordinary tokens.
8. The SGLang `nanbeige` parser (Qwen3CoderDetector) strips exactly one `\n` from each end of a parameter value, so code round-trips exactly. If the code contains `</parameter>`, `<parameter=`, `</function>` or `</tool_call>`, the parse breaks; trajectories containing these strings are excluded during generation.

## TIR data (`/raid/cache/b300-07/ksat-rl/data/tool/`, internal only, not committed)

Generation scripts live in `data_pipeline/tool/`, in this order: `build_pool.py`, `run_gen.sh` (runs `gen_tir.py`), `finalize_tir.py`, `run_format.sh render|check|fewshot`.

| File | Contents |
|---|---|
| `tools_schema.json` | The `tools=` schema |
| `pool.jsonl`, `pool_pass2.jsonl` | Problem pool. Excludes the 2026 수능, the 2026 9월 모평 and synthetic items from those families, as well as figure items. Real items are 4-point only. |
| `tir_traces.jsonl` | **624** verified trajectories (OpenAI messages plus real sandbox outputs and metadata) |
| `tir_traces_meta_extra.jsonl` | 493 correct trajectories set aside because the teacher's reasoning says it uses the tool only because the prompt tells it to ("not necessary but instructions encourage") |
| `tir_rejects.jsonl` | Rejected problems with their reasons |
| `tir_nanbeige.jsonl`, `tir_nanbeige_meta_extra.jsonl` | Nanbeige template text, input_ids and loss_mask (assistant = 1, everything else = 0) |
| `format_check_report*.json` | Results of (a) SGLang parser, (b) prefix preservation and (c) mask checks; all trajectories pass |
| `fewshot_demos.json` | 3 ICRL demos (real train items only) and system prompts for k = 2, 1, 0 (4,006, 2,608 and 444 tokens) |
| `tir_gen_summary.json`, `sandbox_test_log.txt` | Generation statistics and token usage; sandbox test log |
| `gen/*.jsonl` | Raw log of every attempt (`pilot_raw` is the gen_v1 prompt and is not used) |

Teacher: `deepseek/deepseek-v4.1-flash-eco`, called through OpenAI function calling (`tools=`), temperature 0.6, at most 32 concurrent requests. The teacher's system prompt is identical to `TOOL_SYSTEM_PROMPT`, the prompt the policy sees.

A trajectory is kept only if all of the following hold:
- `reward.grade` marks it correct (official answer, or consensus answer for synthetic items);
- it makes at least one tool call;
- the output of at least one call reappears in the reasoning that follows, as a number of two or more digits, a fraction or a symbol;
- no chat or tool markup appears in reasoning or code;
- no turn was truncated.

## veRL multi-turn tool RL (Nanbeige)

| File | Role |
|---|---|
| `docker/patches/patch_verl_nanbeige_tools.py` (+ `verl_nanbeige/`) | Idempotent veRL patch. Registers a `nanbeige` tool parser and a `NanbeigeContinuousTokenBuilder` for model_type `nanbeige`. Applied at container start by `train_grpo_tool.sh` / `docker_eval_tool.sh`, so the v4 image works; also baked in when the Dockerfile is rebuilt |
| `ksat_rl/tools/verl_tool.py` | `KsatPythonTool(BaseTool)`. Defangs markup in sandbox output and keeps per-trajectory counters in `extra_fields` |
| `ksat_rl/tools/verl_agent_loop.py` | `ksat_tool_agent` = ToolAgentLoop + counters + infra-fault masking (`response_mask=0`) |
| `ksat_rl/tools/reward_tool.py` | Grades only the final assistant turn; an unexecuted tool call counts as unfinished = 0 |
| `ksat_rl/tools/make_tool_parquet.py` | Parquet variants `--tool-demos 2,1,0` (same rows, same order) |
| `configs/tool/python_tool.yaml`, `configs/tool/agent_loop.yaml` | veRL tool config and agent-loop config |
| `scripts/train_grpo_tool.sh`, `scripts/run_math_tool_v1.sh` | Training entry point (patch, sandbox check, then the existing train_grpo.sh) and the launch script |
| `scripts/eval_csat_tool.py`, `scripts/docker_eval_tool.sh` | Tool-mode 2026 수능 eval. Uses veRL's agent loop over SGLang `/generate`, so the environment matches training |
| `ksat_rl/tools/agent_harness.py` | Runs the real agent loop outside Ray (shared by the tests and eval) |

How it works:
- The parser stops generation at `</tool_call>` (a single special token, kept in SGLang `output_ids`). The model therefore cannot write a fake `<tool_response>`, and it makes one call per turn.
- `max_user_turns=6` is therefore the tool-call cap. After the 6th response the 7th turn is the last; if it contains a call, the episode is unfinished and scores 0.
- The CT builder appends a tool delta, i.e. a token-level suffix of a full-template render, after inserting `\n` after `<|im_end|>`, or `<|im_end|>\n` after `</tool_call>` (mask 0). The resulting rollout tokens equal a single-pass template render, and equal the `tir_nanbeige.jsonl` token ids.
- In the SFT data, `<|im_end|>` after a tool call is an assistant token (1). In RL it is an inserted token (0). That one token per tool turn is the only mask difference.
- An infra fault (`sandbox_error`) zeroes `response_mask`, so the trajectory gets no gradient. Its reward still enters the GRPO group statistics; this is a known limitation.

Data:

```bash
docker run --rm --network none -e PYTHONPATH=/rl -v $PWD/rl:/rl:ro -v /raid/cache/b300-07/ksat-rl/models:/models:ro \
  -v /raid/cache/b300-07/ksat-rl/data:/data --entrypoint python ksat-rl/verl-sgl:v4 -m ksat_rl.tools.make_tool_parquet \
  --inp /data/ksat_math/rl/rl_train.parquet --split train --tool-demos 2,1,0 --out /data/ksat_math/rl_tool/rl_train_tool_k{k}.parquet
```

Maximum prompt length: k2 6,723, k1 5,325, k0 3,161 tokens. `run_math_tool_v1.sh` sets `MAX_PROMPT` to 7,168 / 5,632 / 3,584 for each phase.

CPU tests (no GPU):
- `ksat_rl/tools/tests/run_verl_tool_tests.sh`: 39 checks on veRL's real agent loop with a scripted SGLang and the real sandbox.
- `ksat_rl/tools/tests/dryrun_tool_launch.sh`: replays the launch path with a fake docker and validates the Hydra config, dataset, CT check and reward.
- `ksat_rl/tools/tests/run_eval_tool_mock.sh`: runs `eval_csat_tool.py` against a mock SGLang `/generate`.
