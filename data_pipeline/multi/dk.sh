#!/usr/bin/env bash
# CPU-only docker helper: dk.sh <name-suffix> [--net] <cmd...>   (container name ksat-data2-<suffix>, no GPUs)
name=$1; shift
net=""; env=""
if [[ ${1:-} == --net ]]; then net="--network host --env-file /home/b300-07/ksat-tool-rl/.env"; shift; fi
exec docker run --rm --name "ksat-data2-$name" $net -u "$(id -u):$(id -g)" \
  -v /home/b300-07/ksat-tool-rl/data_pipeline/multi:/code:ro -v /raid/cache/b300-07/ksat-rl/data:/data \
  -w /tmp ksat-data-tools:local "$@"
