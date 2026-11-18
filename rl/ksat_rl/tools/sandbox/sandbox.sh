#!/usr/bin/env bash
# Start / stop / inspect the KSAT Python sandbox service (CPU only, never --gpus).
#   rl/ksat_rl/tools/sandbox/sandbox.sh build|start|stop|restart|status|logs|connect <container>|disconnect <container>
# Env: NAME (ksat-tool-sandbox), NET (ksat-tool-sbx-net), WORKERS (16), CPUS (16), MEM_MB per call (1024), IMAGE.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="${NAME:-ksat-tool-sandbox}"
NET="${NET:-ksat-tool-sbx-net}"
IMAGE="${IMAGE:-ksat-tool/sandbox:v1}"
WORKERS="${WORKERS:-16}"
CPUS="${CPUS:-16}"
MEM_MB="${MEM_MB:-1024}"                      # per-call address-space limit (<= 1024)
CONTAINER_MEM_MB=$(( WORKERS * MEM_MB + 2048 )) # all workers at their limit + server overhead
PIDS=$(( WORKERS * 2 + 32 ))

case "${1:-}" in
  build)
    docker build -t "$IMAGE" "$HERE" ;;
  start)
    docker network inspect "$NET" >/dev/null 2>&1 || docker network create --internal "$NET" >/dev/null
    [ "$(docker network inspect -f '{{.Internal}}' "$NET")" = "true" ] || { echo "network $NET is not --internal" >&2; exit 1; }
    docker run -d --name "$NAME" --hostname sandbox \
      --network "$NET" \
      --read-only --tmpfs /tmp:rw,nosuid,nodev,noexec,size=512m,mode=1777 \
      --user 10999:10999 --cap-drop ALL --security-opt no-new-privileges \
      --cpus "$CPUS" --memory "${CONTAINER_MEM_MB}m" --memory-swap "${CONTAINER_MEM_MB}m" \
      --pids-limit "$PIDS" --ulimit core=0 --ulimit nofile=4096:4096 \
      -e SBX_WORKERS="$WORKERS" -e SBX_MEM_MB="$MEM_MB" \
      --restart unless-stopped \
      "$IMAGE" >/dev/null
    for _ in $(seq 1 60); do
      docker logs "$NAME" 2>&1 | grep -q '"event": "start"' && break; sleep 0.5
    done
    docker logs "$NAME" 2>&1 | tail -1
    echo "url: http://$NAME:8080 (reachable only from containers attached to $NET)" ;;
  stop)
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    echo "stopped $NAME" ;;
  restart)
    "$0" stop; "$0" start ;;
  status)
    docker ps -a --filter "name=^${NAME}$" --format '{{.Names}}\t{{.Status}}\t{{.Image}}'
    docker network inspect -f '{{.Name}} internal={{.Internal}} containers={{range .Containers}}{{.Name}} {{end}}' "$NET" ;;
  logs)
    docker logs --tail 50 "$NAME" ;;
  connect)     # attach a client container (e.g. the RL trainer) to the sandbox network
    docker network connect "$NET" "$2" ;;
  disconnect)
    docker network disconnect "$NET" "$2" ;;
  *)
    sed -n 2,4p "$0"; exit 1 ;;
esac
