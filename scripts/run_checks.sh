#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m unittest discover -s ksat_reasoning_blueprint/tests -p 'test_*.py' -v
python3 -m unittest discover -s checks -p 'test_*.py' -v
