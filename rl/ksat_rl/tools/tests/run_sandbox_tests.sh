#!/usr/bin/env bash
# Runs test_sandbox.py in a throwaway client container on the sandbox network, plus container-level checks.
set -euo pipefail
RL="$(cd "$(dirname "$0")/../../.." && pwd)"
NAME="${NAME:-ksat-tool-sandbox}"; NET="${NET:-ksat-tool-sbx-net}"
echo "== container-level: the sandbox container itself has no route to the internet"
if docker exec "$NAME" python3 -c "import socket; socket.create_connection(('1.1.1.1', 443), 3)" 2>/dev/null; then
  echo "[FAIL] sandbox container reached the internet"; exit 1
else echo "[PASS] sandbox container cannot reach 1.1.1.1:443"; fi
if docker exec "$NAME" python3 -c "import socket; socket.getaddrinfo('example.com', 80)" 2>/dev/null; then
  echo "[FAIL] sandbox container resolved example.com"; exit 1
else echo "[PASS] sandbox container cannot resolve external DNS"; fi
docker inspect "$NAME" --format 'user={{.Config.User}} readonly={{.HostConfig.ReadonlyRootfs}} capdrop={{.HostConfig.CapDrop}} secopt={{.HostConfig.SecurityOpt}} pids={{.HostConfig.PidsLimit}} cpus={{.HostConfig.NanoCpus}} mem={{.HostConfig.Memory}} mounts={{len .Mounts}} gpus={{.HostConfig.DeviceRequests}}'
echo "== API tests from a client container on $NET"
docker run --rm --name ksat-tool-sbxtest --network "$NET" -v "$RL":/rl:ro -e PYTHONDONTWRITEBYTECODE=1 \
  python:3.12-slim python3 /rl/ksat_rl/tools/tests/test_sandbox.py
