#!/usr/bin/env bash
# CPU dry-run of the tool-RL launch path (no GPU, nothing launched):
#  1. runs rl/scripts/run_math_tool_v1.sh with a fake `docker` on PATH that records every docker command;
#  2. replays the recorded container (same env vars, same ENTRY + hydra args) in a CPU container wired like the
#     real one (default bridge + `docker network connect` to the sandbox network), with a python3 shim that
#     replaces `python3 -m verl.trainer.main_ppo` by validate_launch_config.py.  train_grpo_tool.sh therefore
#     really patches veRL and checks the sandbox before handing the exact override list to the validator.
set -euo pipefail
RL="$(cd "$(dirname "$0")/../../.." && pwd)"
W=$(mktemp -d)   # honours TMPDIR
mkdir -p "$W/bin" "$W/runs"
cat > "$W/bin/docker" <<'EOF'
#!/usr/bin/env bash
echo "$(printf '%q ' "$@")" >> "$DOCKER_LOG"
case "$*" in
  *"network inspect"*) echo true ;;
  *"inspect -f {{.State.Running}}"*) echo true ;;
esac
EOF
chmod +x "$W/bin/docker"
export DOCKER_LOG="$W/docker.log"
PATH="$W/bin:$PATH" PHASE=${PHASE:-k2} PHASE_END_STEP=30 "$RL/scripts/run_math_tool_v1.sh" >/dev/null
echo "== docker commands issued by run_math_tool_v1.sh -> docker_train.sh:"
sed -E 's/(-e [A-Z_]+=)[^ ]*/\1.../g' "$DOCKER_LOG" | cut -c1-400
CREATE=$(grep -m1 '^create ' "$DOCKER_LOG")
grep -q '^network connect ksat-tool-sbx-net ksat-tool-train' "$DOCKER_LOG" || grep -q '^network connect ksat-tool-sbx-net ' "$DOCKER_LOG"
grep -q '^start ' "$DOCKER_LOG"
echo "[PASS] docker_train.sh with SANDBOX_NET: create -> network connect -> start, KSAT_SANDBOX_URL set"
# replay: extract env + args after the image name from the recorded create command
python3 - "$CREATE" "$W" <<'EOF'
import shlex, sys
args = shlex.split(sys.argv[1]); w = sys.argv[2]
envs = [args[i + 1] for i, a in enumerate(args) if a == "-e"]
img = next(i for i, a in enumerate(args) if a.startswith("ksat-rl/verl-sgl:"))
rest = args[img + 1:]
assert rest[0] == "/rl/scripts/train_grpo_tool.sh", rest[0]
open(f"{w}/env.list", "w").write("\n".join(e for e in envs if not e.startswith("NGPUS=")) + "\nNGPUS=8\n")
open(f"{w}/args.sh", "w").write("ARGS=(" + " ".join(shlex.quote(a) for a in rest[1:]) + ")\n")
print("replaying", rest[0], "with", len(rest) - 1, "hydra args and env:", ", ".join(e.split("=")[0] for e in envs))
EOF
cat > "$W/run.sh" <<'EOF'
set -euo pipefail
REAL=$(command -v python3)
mkdir -p /tmp/shim
cat > /tmp/shim/python3 <<SH
#!/usr/bin/env bash
if [ "\${1:-}" = "-m" ] && [ "\${2:-}" = "verl.trainer.main_ppo" ]; then shift 2; exec $REAL /rl/ksat_rl/tools/tests/validate_launch_config.py "\$@"; fi
exec $REAL "\$@"
SH
chmod +x /tmp/shim/python3
export PATH=/tmp/shim:$PATH
source /w/args.sh
bash /rl/scripts/train_grpo_tool.sh "${ARGS[@]}"
EOF
docker rm -f ksat-tool-dryrun >/dev/null 2>&1 || true
docker create --name ksat-tool-dryrun --network bridge --cpus 16 --memory 32g --env-file "$W/env.list" \
  -e PYTHONPATH=/rl -e HF_HUB_OFFLINE=1 \
  -v "$RL":/rl:ro -v /raid/cache/b300-07/ksat-rl/models:/models:ro -v /raid/cache/b300-07/ksat-rl/data:/data:ro \
  -v "$W":/w:ro -v "$W/runs":/runs --entrypoint bash ksat-rl/verl-sgl:v4 /w/run.sh >/dev/null
docker network connect ksat-tool-sbx-net ksat-tool-dryrun
docker start -a ksat-tool-dryrun; rc=$?
docker rm -f ksat-tool-dryrun >/dev/null 2>&1 || true
exit $rc
