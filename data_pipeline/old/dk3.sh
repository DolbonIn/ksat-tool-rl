#!/usr/bin/env bash
# CPU-only docker helper for the 2017-2021 math (가/나형) pipeline: dk3.sh <name-suffix> [--net] <cmd...>
# container name ksat-data3-<suffix>; never requests GPUs. Code dirs mounted read-only, data at /data.
name=$1; shift
net=""
if [[ ${1:-} == --net ]]; then net="--network host --env-file /home/b300-07/ksat-tool-rl/.env"; shift; fi
exec docker run --rm --name "ksat-data3-$name" $net -u "$(id -u):$(id -g)" \
  -v /home/b300-07/ksat-tool-rl/data_pipeline/old:/code:ro -v /home/b300-07/ksat-tool-rl/data_pipeline:/pipe:ro \
  -v /raid/cache/b300-07/ksat-rl/data:/data -w /tmp ksat-data-tools:local "$@"
