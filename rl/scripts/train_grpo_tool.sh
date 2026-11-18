#!/usr/bin/env bash
# In-container entry for tool-calling GRPO (run_math_tool_v1.sh -> docker_train.sh ENTRY=...).
#  1. patch veRL for Nanbeige tool calls (nanbeige tool parser + Continuous Token builder); idempotent, so the
#     v4 image needs no rebuild (images built with the patch in the Dockerfile skip it),
#  2. check the sandbox service is reachable from this container (fail fast instead of masking every rollout),
#  3. hand over to the unchanged train_grpo.sh; tool settings arrive as hydra overrides in "$@"
#     (later overrides win, e.g. reward.custom_reward_function.path).
set -euo pipefail
python3 /rl/docker/patches/patch_verl_nanbeige_tools.py /opt/verl
python3 -c "
import sys; sys.path.insert(0, '/rl')
from ksat_rl.tools.python_sandbox import health, run_python
h = health(); r = run_python('import sympy; print(sympy.factor(\"x**2-1\"))')
assert r['status'] == 'ok', r
print('sandbox OK:', h, r['stdout'].strip())
"
exec bash /rl/scripts/train_grpo.sh "$@"
